"""Offline candidate generation -> world simulation -> ranking -> safety -> Top K."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import time
import torch
from collections import Counter
from alignment import batch_candidates
from backends.http import HTTPBackend
from dataset import XLifeCleanedDataset
from generation import prepare
from knowledge import KnowledgeGraph
from safety import assess
from world_model import MetabolicWorldModel, llm_actions_enabled
from dataset_metadata import prepare_metadata
from runtime_schema import load_schema, retrieve


def finite_json(value):
    if isinstance(value,torch.Tensor): return finite_json(value.detach().cpu().tolist())
    if isinstance(value,float) and not math.isfinite(value): return None
    if isinstance(value,dict): return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [finite_json(v) for v in value]
    return value


def masked_context(sample):
    # Deliberate allowlist: original diet, exercise, texts, onset and future targets never escape.
    return finite_json({k:sample[k] for k in ('subject_id','origin','history','history_mask','metadata')})


def model_candidate(context,action,vocabulary):
    """Tensor serialization of remotely prepared inputs, without prescription policy."""
    sample={key:torch.tensor([float('nan') if v is None else v for v in values],dtype=torch.float32)
            for key,values in {**{k:context[k] for k in ('history',)},
                               **{k:action[k] for k in ('diet','exercise','onset')}}.items()}
    sample['history_mask']=torch.tensor(context['history_mask'],dtype=torch.bool)
    sample['history']=sample['history'].masked_fill(~sample['history_mask'],0.)
    sample['type_id']=torch.tensor(vocabulary.get(action.get('exercise_type',''),0))
    for key in ('diet_text','exercise_text'): sample[key]=action.get(key,'')
    sample['diet_images'] = action.get('diet_images', [])
    sample['metadata'] = prepare_metadata(context['metadata'])
    return sample


def load_model(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('architecture_version') != 3: raise ValueError('Expected architecture_version 3 checkpoint')
    if checkpoint['llm_actions'] != llm_actions_enabled(): raise ValueError('Checkpoint action backend mismatch')
    model = MetabolicWorldModel(**checkpoint['model_config'])
    model.load_checkpoint(checkpoint)
    return model.to(device).eval(), checkpoint


def recommend(context, model, checkpoint, backend, graph, mode='diet', count=5, seed=42,
              ranking='learned', constraints=None, requirement=None, candidates=None, top_k=3,
              device='cpu', batch_size=8, extract_actions=False):
    if ranking not in ('learned','random'): raise ValueError('Unknown ranking backend')
    if set(context)!={'subject_id','origin','history','history_mask','metadata'}:
        raise ValueError('Only masked physiological context may enter recommendation')
    if count<1 or top_k<1 or batch_size<1: raise ValueError('Positive counts required')
    constraints = constraints or {}
    if requirement is None:
        requirement=dict(mode=mode,preference_seed=seed,onset=[0,-15] if mode=='joint' else [0,0])
        if mode in ('diet','joint'): requirement['energy_kcal']=random.Random(seed).randint(350,650)
        if mode in ('exercise','joint'): requirement['duration_min']=random.Random(seed+1).randint(10,30)
    context = deepcopy(context)
    identity = hashlib.sha256(json.dumps(context,sort_keys=True,allow_nan=False).encode()).hexdigest()
    output = dict(schema_version=2, context=context, context_hash=identity, requirement=requirement,
        masked_original_actions=True, ranking_backend=ranking, test_only=ranking=='random',
        knowledge={'source':graph.source,'triples':len(graph.triples),'provenance':graph.provenance}, candidates=[], top_plans=[], failures=[],
        method_status='pending')
    output['test_only'] |= any(p.get('test_backend',False) for p in graph.provenance)
    output['test_only'] |= bool(checkpoint.get('test_backend',False))
    sources = dict(context=context, requirement=requirement, constraints=constraints)
    try:
        schema = load_schema(backend)
        generation_knowledge, provenance = retrieve(graph, schema['retrieval']['generation'], sources, backend)
    except Exception as error:
        output['method_status'] = 'retrieval_failed'
        output['failures'].append(dict(stage='retrieval', reason=str(error)))
        return output
    output['retrieval_schema'] = schema['retrieval']
    output['retrieval_provenance'] = list(provenance)
    output['test_only'] |= any(p['test_backend'] for p in provenance)
    output['generation_knowledge']=generation_knowledge
    if candidates is None:
        try:
            response = backend.call('generate',dict(context=context, requirement=requirement,mode=mode,
                count=count, seed=seed, constraints=constraints, knowledge=generation_knowledge))
            output['generation_provenance'] = response['provenance']
            output['test_only'] |= response['provenance']['test_backend']
            candidates = response['result']['candidates']
            if not isinstance(candidates,list): raise ValueError('Generator must return a candidate list')
        except Exception as error:
            output['method_status']='generation_failed'
            output['failures'].append(dict(stage='generation',reason=str(error)))
            return output
    else:
        output['generation_provenance'] = {'backend':'prestructured','test_backend':False}
    valid, ranking_logits = [], {}
    for i, raw in enumerate(candidates):
        row = dict(id=i+1, candidate=deepcopy(raw), status='pending')
        output['candidates'].append(row)
        try:
            if extract_actions:
                row['generation_candidate']=deepcopy(raw)
                response = backend.call('action/explicit', {'action':raw})
                raw = response['result']['action']
                row['candidate'], row['extraction_provenance'] = raw, response['provenance']
                output['test_only'] |= response['provenance']['test_backend']
            row['raw_candidate']=deepcopy(raw)
            prepared=prepare(raw,backend)
            row['preparation_provenance']=prepared['provenance']
            output['test_only'] |= prepared['provenance']['test_backend']
            row['candidate']=prepared['result']['candidate']
            action=prepared['result']['action']
            if action['mode'] == 'joint' and checkpoint['phase'] not in ('fusion','professional','physiological'):
                raise ValueError('Joint simulation requires a fusion-trained checkpoint')
            trained = checkpoint.get('trained_domains', [])
            if action['mode'] in ('diet','exercise') and action['mode'] not in trained:
                raise ValueError('Checkpoint has no trained dynamics for requested domain')
            if ranking == 'learned' and not model.ranked_modes[('diet','exercise','joint').index(action['mode'])]:
                raise ValueError('No trained ranking for mode; use --ranking random explicitly for smoke tests')
            sample = model_candidate(context,action,checkpoint['vocabulary'])
            valid.append((row,action,sample))
        except Exception as error:
            row.update(status='invalid',reason=str(error))
    # Every candidate sees the same measured history. Batch only compatible domains.
    for domain in ('diet','exercise','joint'):
        group = [r for r in valid if r[1]['mode']==domain]
        for start in range(0,len(group),batch_size):
            chunk = group[start:start+batch_size]
            try:
                batch = batch_candidates([r[2] for r in chunk],device)
                with torch.inference_mode(): simulation = model.simulate(batch,domain)
                if not torch.isfinite(simulation['predicted_cgm']).all() or not torch.isfinite(simulation['trajectory']).all():
                    raise ValueError('Nonfinite simulation')
                for j,(row,action,_) in enumerate(chunk):
                    score = float(simulation['ranking_logit'][j]) if ranking=='learned' else random.Random(f'{seed}:{row["id"]}').random()
                    if not math.isfinite(score): raise ValueError('Nonfinite ranking score')
                    ranking_logits[row['id']] = score
                    row.update(status='simulated', simulation=finite_json({
                        'predicted_cgm':simulation['predicted_cgm'][j], 'trajectory':simulation['trajectory'][j],
                        'gates':simulation['gates'][j] if simulation['gates'] is not None else None,
                        'time_minutes':list(range(5,181,5)), 'onset':action['onset'],
                        'architecture_version':3, 'glucose_unit':constraints.get('glucose_unit','dataset_native_unknown')}))
            except Exception as error:
                for row,_,_ in chunk: row.update(status='simulation_failed',reason=str(error))
    for row in sorted(output['candidates'],key=lambda r:ranking_logits.get(r['id'],float('-inf')),reverse=True):
        if row['status'] != 'simulated': continue
        raw = row['candidate']
        try:
            row['safety_knowledge'], provenance = retrieve(graph, schema['retrieval']['safety'], dict(sources, candidate=raw, simulation=row['simulation']), backend)
            output['retrieval_provenance'].extend(provenance)
            output['test_only'] |= any(p['test_backend'] for p in provenance)
            row['assessment'] = assess(raw,context,row['simulation'],constraints,row['safety_knowledge'],backend)
        except Exception as error:
            row.update(status='safety_failed',reason=str(error))
            continue
        row['status'] = 'safe' if row['assessment']['is_safe'] else 'rejected'
        output['test_only'] |= row['assessment']['provenance']['test_backend']
    if model.llm_actions:
        output['test_only'] |= model.diet_vlm.test_backend or model.exercise_vlm.test_backend
    output['test_only'] |= model.metadata_encoder.test_backend
    output['method_status'] = 'test_substitutes' if output['test_only'] else 'knowledge_missing' if not graph.triples else 'configured_backends'
    output['top_plans'] = sorted([r for r in output['candidates'] if r['status']=='safe'],
                                key=lambda r:ranking_logits[r['id']],reverse=True)[:top_k]
    if not output['top_plans']: output['failures'].append(dict(stage='selection',reason='No candidates passed validation, simulation and safety'))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--mode',choices=['diet','exercise','joint'],default='diet')
    parser.add_argument('--sample-index',type=int,default=0)
    parser.add_argument('--subject-id')
    parser.add_argument('--source-mode',choices=['diet','exercise'])
    parser.add_argument('--ranking',choices=['learned','random'],default='learned')
    parser.add_argument('--backend-url')
    parser.add_argument('--knowledge')
    parser.add_argument('--constraints')
    parser.add_argument('--requirement')
    parser.add_argument('--candidates')
    parser.add_argument('--extract-actions',action='store_true')
    parser.add_argument('--count',type=int,default=5)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',default='cpu')
    args = parser.parse_args(argv)
    if args.count<1 or args.batch_size<1 or args.sample_index<0: parser.error('Counts must be positive and index nonnegative')
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    started=time.perf_counter()
    model,checkpoint=load_model(args.checkpoint,args.device)
    rows=XLifeCleanedDataset(args.data).samples
    rows=[r for r in rows if (not args.subject_id or r['subject_id']==args.subject_id) and (not args.source_mode or r['mode']==args.source_mode)]
    if args.sample_index>=len(rows): raise ValueError('Sample index outside filtered real events')
    read=lambda path:json.loads(Path(path).read_text(encoding='utf-8')) if path else None
    result=recommend(masked_context(rows[args.sample_index]),model,checkpoint,HTTPBackend(args.backend_url),
        KnowledgeGraph.load(args.knowledge),args.mode,args.count,args.seed,args.ranking,read(args.constraints),
        read(args.requirement),read(args.candidates),device=args.device,batch_size=args.batch_size,extract_actions=args.extract_actions)
    result.update(checkpoint=str(Path(args.checkpoint).resolve()),source_mode=rows[args.sample_index]['mode'],
                  elapsed_seconds=time.perf_counter()-started)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    statuses=Counter(row['status'] for row in result['candidates'])
    reasons=Counter()
    for row in result['candidates']:
        assessment=row.get('assessment') or {}
        semantic=assessment.get('semantic') or {}
        for reason in [row.get('reason'),assessment.get('error')]+assessment.get('hard_vetoes',[])+semantic.get('reasons',[]):
            if reason: reasons[str(reason)]+=1
    print(json.dumps(dict(output=str(path),candidates=len(result['candidates']),safe_top=len(result['top_plans']),
                         test_only=result['test_only'],status_counts=dict(statuses),failure_reasons=dict(reasons),failures=result['failures'])))


if __name__=='__main__': main()
