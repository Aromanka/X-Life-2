"""Five participant 80/20 splits, seen/unseen evaluation and controlled baselines."""
import argparse
from copy import deepcopy
import json
import hashlib
from pathlib import Path
import random
import time
import torch
from torch.utils.data import DataLoader
from dataset import XLifeCleanedDataset
from train import collate,device_batch,curve_metrics,physiological_loss
from world_model import MetabolicWorldModel
from experiments.baselines import DLinear,persistence,linear_history
from experiments.tasks import task_window


def protocol(rows,seed):
    ids=sorted({r['subject_id'] for r in rows});random.Random(seed).shuffle(ids)
    if len(ids)<2: raise ValueError('80/20 requires at least two participants')
    split=max(1,min(len(ids)-1,int(.8*len(ids))));seen_ids=ids[:split]
    train,seen,unseen=[],[],[]
    for subject in ids:
        samples=sorted([r for r in rows if r['subject_id']==subject],key=lambda r:r['origin'])
        if subject not in seen_ids: unseen.extend(samples);continue
        cut=max(1,int(len(samples)*.8))
        # Purge overlapping +/-3h input/target windows at the seen holdout boundary.
        holdout=samples[cut:]
        boundary=min((r['origin'] for r in holdout),default=None)
        from datetime import datetime,timedelta
        train.extend([r for r in samples[:cut] if boundary is None or datetime.fromisoformat(r['origin'])+timedelta(hours=6)<datetime.fromisoformat(boundary)])
        seen.extend(holdout)
    if not train or not unseen: raise ValueError('Insufficient purged training/unseen records')
    return {'train':train,'seen':seen,'unseen':unseen}, {'train_subjects':seen_ids,'unseen_subjects':ids[split:]}


def stratified(rows):
    report=curve_metrics(rows)
    report['modes']={mode:curve_metrics([r for r in rows if r['mode']==mode]) for mode in ('diet','exercise','joint')}
    report['missing_history']={key:curve_metrics([r for r in rows if ('complete' if r['history_points']==36 else 'partial')==key]) for key in ('complete','partial')}
    report['missing_metadata']={key:curve_metrics([r for r in rows if ('complete' if r['metadata_complete'] else 'partial')==key]) for key in ('complete','partial')}
    report['missing_action']={key:curve_metrics([r for r in rows if ('complete' if r['action_complete'] else 'partial')==key]) for key in ('complete','partial')}
    return report


def experiment_batch(rows,vocabulary):
    batch=collate(rows,vocabulary)
    batch["modes"]=[r["mode"] for r in rows]
    return batch


def run(args):
    torch.set_num_threads(1)
    if args.task_data:
        rows=[task_window(r) for r in json.loads(Path(args.task_data).read_text(encoding='utf-8'))]
        output={'status':'validated_only','tasks':len(rows),'measured_points':sum(int(r['target_mask'].sum()) for r in rows),
                'reason':'Task data validated; task-specific action/context semantics and training remain explicit prerequisites'}
        Path(args.output).mkdir(parents=True,exist_ok=True)
        (Path(args.output)/'task_validation.json').write_text(json.dumps(output),encoding='utf-8')
        print(json.dumps(output));return
    if not args.data: raise ValueError('--data or --task-data required')
    dataset=XLifeCleanedDataset(args.data,args.max_subjects,args.max_samples)
    data_identity={'path':str(Path(args.data).resolve()),'sha256':hashlib.sha256(Path(args.data).read_bytes()).hexdigest()}
    config=json.loads(Path(args.config).read_text(encoding='utf-8')) if args.config else {}
    runs=[]
    for seed in args.seeds:
        groups,splits=protocol(dataset.samples,seed)
        for kind in args.models:
            torch.manual_seed(seed);start=time.perf_counter()
            if args.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
            if kind=='dlinear':
                model=DLinear(action_correction=args.action_correction);model.fit_statistics(groups['train'])
            elif kind=='world':
                model=MetabolicWorldModel(**config.get('model',{}));model.fit(groups['train']);model.configure_phase('single')
            else: model=None
            vocabulary={s:i+1 for i,s in enumerate(sorted({r['exercise_type'] for r in groups['train']}))}
            if len(vocabulary)>127: raise ValueError('Exercise vocabulary exceeds model capacity')
            if model is not None:
                model.to(args.device)
                optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.learning_rate)
                for _ in range(args.epochs):
                    model.train()
                    for batch in DataLoader(groups['train'],batch_size=args.batch_size,shuffle=True,collate_fn=lambda r:experiment_batch(r,vocabulary)):
                        batch=device_batch(batch,args.device);optimizer.zero_grad()
                        if kind=='world':
                            squared=[]
                            for mode in ('diet','exercise'):
                                indices=[i for i,v in enumerate(batch['modes']) if v==mode]
                                if not indices: continue
                                part={k:v[indices] if isinstance(v,torch.Tensor) else [v[i] for i in indices] for k,v in batch.items()}
                                squared.append(physiological_loss(model,part,mode)*len(indices))
                            loss=torch.stack(squared).sum()/len(batch['modes'])
                        else:
                            prediction=model(batch)
                            mask=batch['target_mask'];loss=((prediction[mask]-batch['target'][mask])/model.scale).square().mean()
                        if not torch.isfinite(loss): raise ValueError('Nonfinite baseline loss')
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                model.eval()
            reports={}
            for split in ('seen','unseen'):
                predictions=[]
                for row in groups[split]:
                    batch=device_batch(collate([row],vocabulary),args.device)
                    batch['modes']=[row['mode']]
                    with torch.inference_mode():
                        p=(persistence(batch['history'],batch['history_mask']) if kind=='persistence' else
                           linear_history(batch['history'],batch['history_mask']) if kind=='linear' else
                           model.simulate(batch,row['mode'])['predicted_cgm'] if kind=='world' else model(batch))[0].cpu()
                    if not torch.isfinite(p).all(): raise ValueError('Nonfinite evaluation')
                    predictions.append(dict(mode=row['mode'],predicted=p.tolist(),observed=row['target'].tolist(),mask=row['target_mask'].tolist(),
                        last_observed=float(row['history'][row['history_mask']][-1]),history_points=int(row['history_mask'].sum()),
                        metadata_complete=bool(row['metadata']['fields']) and all(f['value'] is not None for f in row['metadata']['fields']),action_complete=bool(torch.isfinite(row[row['mode']]).all())))
                reports[split]=stratified(predictions)
            output=Path(args.output)/f'seed_{seed}'/kind;output.mkdir(parents=True,exist_ok=True)
            result=dict(seed=seed,model=kind,action_correction=args.action_correction,protocol='participant80_unseen20_purged_seen_holdout',
                adaptation='none_zero_shot',splits=splits,counts={k:len(v) for k,v in groups.items()},metrics=reports,
                elapsed_seconds=time.perf_counter()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0,
                model_config=config.get('model',{}),epochs=args.epochs if model else 0)
            result.update(data=data_identity,training_settings={'batch_size':args.batch_size,'learning_rate':args.learning_rate},
                event_splits={k:[{'subject_id':r['subject_id'],'origin':r['origin'],'mode':r['mode']} for r in v] for k,v in groups.items()},
                test_only=bool(args.max_subjects or args.max_samples or (kind=='world' and model.llm_actions and
                               (model.diet_vlm.test_backend or model.exercise_vlm.test_backend))))
            if args.adapt_epochs and kind=='world':
                adaptation_start=time.perf_counter()
                result['participant_fine_tuning']=adapt_unseen(model,groups['unseen'],vocabulary,args)
                result['adaptation_elapsed_seconds']=time.perf_counter()-adaptation_start
            (output/'metrics.json').write_text(json.dumps(result,allow_nan=False,indent=2),encoding='utf-8')
            if model is not None: torch.save({'state_dict':model.state_dict(),'config':result},output/'baseline.pt')
            runs.append(result);print(json.dumps({'seed':seed,'model':kind,'unseen_mae':reports['unseen']['mae']}))
    aggregate={}
    for kind in args.models:
        aggregate[kind]={}
        for split in ('seen','unseen'):
            values=[r['metrics'][split]['mae'] for r in runs if r['model']==kind and r['metrics'][split]['mae'] is not None]
            aggregate[kind][split]={'mae_mean':sum(values)/len(values) if values else None,
                'mae_std':float(torch.tensor(values).std(unbiased=False)) if values else None,'runs':len(values)}
    summary={'runs':runs,'aggregate':aggregate,'full_protocol_executed':len(set(args.seeds))==5 and args.max_subjects is None and args.max_samples is None,
             'fine_tuning':'not performed; zero-shot unseen results only'}
    if args.adapt_epochs: summary['fine_tuning']='Separate per-participant adaptation reports; base unseen metrics remain zero-shot'
    (Path(args.output)/'summary.json').write_text(json.dumps(summary,allow_nan=False,indent=2),encoding='utf-8')


def adapt_unseen(model,rows,vocabulary,args):
    """Separate factual adaptation and purged temporal evaluation for each new participant."""
    from datetime import datetime,timedelta
    results=[]
    for subject in sorted({r['subject_id'] for r in rows}):
        samples=sorted([r for r in rows if r['subject_id']==subject],key=lambda r:r['origin'])
        cut=max(1,int(.8*len(samples)));evaluation=samples[cut:]
        if not evaluation:
            results.append({'subject_id':subject,'status':'insufficient_holdout'});continue
        boundary=datetime.fromisoformat(evaluation[0]['origin'])
        factual=[r for r in samples[:cut] if datetime.fromisoformat(r['origin'])+timedelta(hours=6)<boundary]
        if not factual:
            results.append({'subject_id':subject,'status':'insufficient_purged_adaptation'});continue
        adapted=deepcopy(model);adapted.configure_phase('single');adapted.train()
        optimizer=torch.optim.AdamW([p for p in adapted.parameters() if p.requires_grad],lr=args.learning_rate)
        for _ in range(args.adapt_epochs):
            for row in factual:
                batch=device_batch(collate([row],vocabulary),args.device);optimizer.zero_grad()
                loss=physiological_loss(adapted,batch,row['mode']);loss.backward()
                torch.nn.utils.clip_grad_norm_(adapted.parameters(),1.);optimizer.step()
        from train import evaluate
        metrics,_=evaluate(adapted,evaluation,vocabulary,args.device,args.batch_size)
        results.append({'subject_id':subject,'adaptation_records':len(factual),'evaluation_records':len(evaluation),
                        'epochs':args.adapt_epochs,'metrics':metrics,'status':'adapted'})
    return results


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data');p.add_argument('--task-data');p.add_argument('--config');p.add_argument('--output',required=True)
    p.add_argument('--models',nargs='+',choices=['persistence','linear','dlinear','world'],default=['persistence','linear','dlinear'])
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46]);p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--learning-rate',type=float,default=.001)
    p.add_argument('--max-subjects',type=int);p.add_argument('--max-samples',type=int);p.add_argument('--device',default='cpu')
    p.add_argument('--action-correction',action='store_true')
    p.add_argument('--adapt-epochs',type=int,default=0)
    args=p.parse_args(argv)
    if args.epochs<1 or args.batch_size<1 or args.adapt_epochs<0: p.error('Invalid epoch or batch count')
    run(args)


if __name__=='__main__': main()
