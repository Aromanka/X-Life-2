import argparse
import json
import random
from pathlib import Path
from datetime import datetime, timedelta
import torch
from torch.utils.data import DataLoader
from dataset import XLifeCleanedDataset, SampleNumericDataset, split_samples, temporal_overlap
from world_model import MetabolicWorldModel, llm_actions_enabled
from runtime_schema import load_schema, supervision_loss
from configuration import settings, parse_configured, optimizer_settings


def collate(rows, vocabulary):
    batch = {key: torch.stack([row[key] for row in rows]) for key in ('diet', 'exercise', 'onset', 'history', 'history_mask', 'target', 'target_mask')}
    batch['metadata'] = [row['metadata'] for row in rows]
    batch['type_id'] = torch.tensor([vocabulary.get(row['exercise_type'], 0) for row in rows])
    for key in ('diet_text', 'exercise_text', 'exercise_type', 'subject_id', 'origin'): batch[key] = [row[key] for row in rows]
    batch['intensity_scale'] = [row.get('intensity_scale', 'unknown') for row in rows]
    batch['diet_images'] = [row.get('diet_images', []) for row in rows]
    if 'supervision' in rows[0]: batch['supervision'] = [row['supervision'] for row in rows]
    return batch


def device_batch(batch, device): return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def physiological_loss(model, batch, mode):
    output = model(batch, mode, horizon=batch['target'].size(1))
    return supervision_loss(model, batch['supervision'], output, mode)


def curve_metrics(rows):
    errors, points, correlations, invalid, baseline = 0., 0, [], 0, 0.
    horizons = {str(n):{'absolute':0., 'points':0, 'persistence_absolute':0.} for n in settings('data', 'metric_horizons_minutes')}
    for row in rows:
        p,y,mask = torch.tensor(row['predicted']),torch.tensor(row['observed']),torch.tensor(row['mask'],dtype=torch.bool)
        error = (p-y).abs(); constant = (y-row['last_observed']).abs()
        errors += float(error[mask].sum()); baseline += float(constant[mask].sum()); points += int(mask.sum())
        if mask.sum()>1 and p[mask].std()>1e-8 and y[mask].std()>1e-8: correlations.append(float(torch.corrcoef(torch.stack([p[mask],y[mask]]))[0,1]))
        else: invalid += 1
        for end,stats in horizons.items():
            m = mask[:int(end)//settings('data', 'interval_minutes')]
            stats['absolute'] += float(error[:len(m)][m].sum()); stats['persistence_absolute'] += float(constant[:len(m)][m].sum()); stats['points'] += int(m.sum())
    return dict(mae=errors/points if points else None, pearson=sum(correlations)/len(correlations) if correlations else None, valid_points=points, records=len(rows), valid_correlations=len(correlations), invalid_correlations=invalid, persistence_mae=baseline/points if points else None, horizons={end:dict(mae=s['absolute']/s['points'] if s['points'] else None,persistence_mae=s['persistence_absolute']/s['points'] if s['points'] else None,valid_points=s['points']) for end,s in horizons.items()})


def evaluate(model, samples, vocabulary, device, batch_size, mode=None, joint_only=False):
    model.eval()
    predictions = []
    for domain in ([mode] if mode else ['diet', 'exercise']):
        rows = [s for s in samples if s['joint']] if domain == 'joint' else [s for s in samples if s['mode'] == domain]
        if joint_only: rows = [s for s in rows if s['joint']]
        for batch in DataLoader(rows, batch_size=batch_size, collate_fn=lambda rows: collate(rows, vocabulary)):
            data = device_batch(batch, device)
            with torch.no_grad(): predicted = model(data, domain)['predicted_cgm'].cpu()
            if not torch.isfinite(predicted).all(): raise FloatingPointError('Nonfinite prediction')
            for i, (p, y, mask) in enumerate(zip(predicted, batch['target'], batch['target_mask'])):
                last = float(batch['history'][i][batch['history_mask'][i]][-1])
                predictions.append(dict(subject_id=batch['subject_id'][i], origin=batch['origin'][i], mode=domain, predicted=p.tolist(), observed=y.tolist(), mask=mask.tolist(),last_observed=last))
    metrics = curve_metrics(predictions)
    metrics['domains'] = {d:curve_metrics([r for r in predictions if r['mode']==d]) for d in sorted({r['mode'] for r in predictions})}
    return metrics, predictions


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--dataset', choices=['xlife_cleaned', 'sample'])
    parser.add_argument('--config')
    parser.add_argument('--phase', choices=['single', 'fusion', 'professional', 'physiological'])
    parser.add_argument('--replay-data')
    parser.add_argument('--rank-only', action='store_true', default=None)
    parser.add_argument('--domain', choices=['both', 'diet', 'exercise'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--init-checkpoint')
    parser.add_argument('--resume')
    parser.add_argument('--predict-only', action='store_true', default=None)
    parser.add_argument('--mode', choices=['diet', 'exercise', 'joint'])
    parser.add_argument('--split', choices=['train', 'validation', 'test'])
    parser.add_argument('--split-strategy', choices=['subject','within_subject'])
    parser.add_argument('--joint-only', action='store_true', default=None)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--max-subjects', type=int)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--device')
    parser.add_argument('--seed', type=int)
    args = parse_configured(parser, argv, 'training',
        ('dataset', 'phase', 'rank_only', 'domain', 'predict_only', 'split', 'joint_only', 'epochs', 'batch_size', 'device', 'seed'))
    if args.epochs < 1 or args.batch_size < 1 or any(v is not None and v < 1 for v in (args.max_steps, args.max_subjects, args.max_samples)): raise ValueError('Counts must be positive')
    if args.resume and args.init_checkpoint: raise ValueError('Choose resume or initialization')
    if args.rank_only and args.predict_only: raise ValueError('Choose ranking or physiological evaluation')
    random.seed(args.seed); torch.manual_seed(args.seed)
    config = settings()
    model_config = dict(config['model'], data_config=config['data'])
    training = config['training']
    checkpoint_path = args.resume or args.init_checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False) if checkpoint_path else None
    restore_split = bool(checkpoint and (args.resume or args.predict_only))
    if checkpoint:
        if checkpoint.get('architecture_version') != 3: raise ValueError('Expected architecture_version 3')
        if checkpoint['llm_actions'] != llm_actions_enabled(): raise ValueError('Checkpoint action backend differs from environment')
        model_config.update(checkpoint['model_config'])
        if not restore_split:
            model_config['data_config'] = dict(model_config['data_config'], **{key:config['data'][key] for key in ('train_fraction', 'validation_fraction', 'within_subject_train_fraction')})
        if model_config['data_config'] != config['data']:
            raise ValueError('Checkpoint data configuration differs from configuration file')
    if args.rank_only or (args.phase in {'professional', 'physiological'} and not args.predict_only):
        if not checkpoint: raise ValueError('Alignment and ranking require a checkpoint')
        from alignment import run_alignment, rank_records
        model = MetabolicWorldModel(**model_config)
        model.load_checkpoint(checkpoint); model.to(args.device)
        output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
        if args.rank_only:
            if checkpoint['phase'] not in {'professional', 'physiological'}: raise ValueError('Ranking requires an aligned checkpoint')
            rank_records(args, model, checkpoint, output)
        else: run_alignment(args, model, checkpoint, config, output)
        return
    if (args.predict_only or args.phase == 'fusion') and not checkpoint: raise ValueError('A checkpoint is required')
    schema = load_schema()
    if checkpoint and (model_config.get('schema') or {}).get('supervision') != schema['supervision']: raise ValueError('Server supervision schema differs from checkpoint; start new training')
    model_config = dict(model_config, schema=schema)
    dataset = {'xlife_cleaned': XLifeCleanedDataset, 'sample': SampleNumericDataset}[args.dataset](args.data, args.max_subjects, args.max_samples, schema=schema)
    if not dataset.samples: raise ValueError('No usable records')
    strategy = args.split_strategy or (checkpoint['split_strategy'] if restore_split else training['split_strategy'])
    if restore_split and strategy != checkpoint['split_strategy']: raise ValueError('Checkpoint split strategy mismatch; start a new experiment')
    groups,splits = split_samples(dataset.samples,strategy,args.seed,checkpoint['splits'] if restore_split else None)
    if not groups['train']: raise ValueError('No training samples in selected data')
    vocabulary = checkpoint['vocabulary'] if checkpoint else {v:i+1 for i,v in enumerate(sorted({s['exercise_type'] for s in groups['train'] if s['exercise_type']}))}
    if len(vocabulary) >= model_config['type_capacity']: raise ValueError('Too many exercise categories')
    model = MetabolicWorldModel(**model_config)
    if checkpoint: model.load_checkpoint(checkpoint)
    else: model.fit(groups['train'])
    if args.phase == 'fusion' and not args.resume and not args.predict_only:
        if checkpoint['phase'] != 'single' or set(checkpoint['trained_domains']) != {'diet','exercise'}: raise ValueError('Fusion requires both pretrained domains')
        model.initialize_fusion()
    model.to(args.device)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    print(json.dumps(dict(records=len(dataset), skipped=dataset.skipped, splits={k:len(v) for k,v in groups.items()}, device=args.device)))
    if strategy == 'within_subject': print(json.dumps(dict(split_strategy=strategy,validation_windows_overlapping_train_windows=temporal_overlap(groups))))
    if args.predict_only:
        if args.mode == 'joint' and checkpoint['phase'] not in {'fusion', 'professional', 'physiological'}: raise ValueError('Joint prediction requires fusion training')
        metrics, predictions = evaluate(model, groups[args.split], vocabulary, args.device, args.batch_size, args.mode or ('joint' if checkpoint['phase'] in {'fusion', 'professional', 'physiological'} else None),args.joint_only)
        if not predictions: raise ValueError('No prediction samples in selected split/mode')
        (output/'predictions.json').write_text(json.dumps(predictions, allow_nan=False), encoding='utf-8')
        (output/'metrics.json').write_text(json.dumps(metrics, allow_nan=False), encoding='utf-8')
        print(json.dumps(metrics)); return
    model.configure_phase(args.phase)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=training['learning_rate'], **optimizer_settings())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', **training['scheduler'])
    best_mae, bad_epochs = float('inf'), 0
    history, global_step, start_epoch = [], 0, 0
    trained_domains = set(checkpoint.get('trained_domains', [])) if checkpoint else set()
    if args.resume:
        if checkpoint['phase'] != args.phase: raise ValueError('Resume phase mismatch')
        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint.get('scheduler'): scheduler.load_state_dict(checkpoint['scheduler'])
        best_mae, bad_epochs = checkpoint.get('best_mae',float('inf')), checkpoint.get('bad_epochs',0)
        torch.set_rng_state(checkpoint['rng_state'])
        random.setstate(checkpoint['random_state'])
        if torch.cuda.is_available() and checkpoint.get('cuda_rng_state'): torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
        history, global_step, start_epoch = checkpoint['history'], checkpoint['global_step'], checkpoint['epoch']
    domains = ['joint'] if args.phase == 'fusion' else ['diet', 'exercise'] if args.domain == 'both' else [args.domain]
    loaders = {}
    for domain in domains:
        rows = [s for s in groups['train'] if s['joint']] if domain == 'joint' else [s for s in groups['train'] if s['mode'] == domain]
        if not rows: raise ValueError(f'No {domain} training records')
        loaders[domain] = DataLoader(rows, batch_size=args.batch_size, shuffle=True, collate_fn=lambda rows: collate(rows, vocabulary))
    local_steps = 0
    for epoch in range(start_epoch, start_epoch+args.epochs):
        model.train(); iterators = {d:iter(loader) for d,loader in loaders.items()}
        totals, counts = 0., 0
        for _ in range(max(map(len, loaders.values()))):
            for domain, loader in loaders.items():
                try: batch = next(iterators[domain])
                except StopIteration: iterators[domain] = iter(loader); batch = next(iterators[domain])
                batch = device_batch(batch, args.device)
                optimizer.zero_grad(set_to_none=True)
                loss = physiological_loss(model, batch, domain)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), training['gradient_clip']); optimizer.step()
                global_step += 1; local_steps += 1; totals += float(loss.detach()); counts += 1
                if domain != 'joint': trained_domains.add(domain)
                if args.max_steps and local_steps >= args.max_steps: break
            if args.max_steps and local_steps >= args.max_steps: break
        eval_mode = 'joint' if args.phase=='fusion' else None if args.domain=='both' else args.domain
        train_metrics, _ = evaluate(model, groups['train'], vocabulary, args.device, args.batch_size, eval_mode)
        metrics, _ = evaluate(model, groups['validation'], vocabulary, args.device, args.batch_size, eval_mode)
        improved = metrics['mae'] is not None and metrics['mae']<best_mae
        if metrics['mae'] is not None:
            scheduler.step(metrics['mae'])
            if improved: best_mae, bad_epochs = metrics['mae'],0
            else: bad_epochs += 1
        entry = dict(epoch=epoch+1, step=global_step, loss=totals/counts, learning_rate=optimizer.param_groups[0]['lr'], best_validation_mae=best_mae if best_mae<float('inf') else None, bad_epochs=bad_epochs, train=train_metrics, validation=metrics)
        history.append(entry); print(json.dumps(entry))
        state = dict(architecture_version=3, schema_version=1, phase=args.phase, trained_domains=sorted(trained_domains), model_config=model_config, llm_actions=model.llm_actions, model_state_dict=model.state_dict(), optimizer=optimizer.state_dict(), splits=splits, vocabulary=vocabulary, epoch=epoch+1, global_step=global_step, history=history, rng_state=torch.get_rng_state(), random_state=random.getstate(), cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, parent=checkpoint_path, data=str(Path(args.data).resolve()), config=config)
        state.update(ranking_version=2,split_strategy=strategy,scheduler=scheduler.state_dict(),best_mae=best_mae,bad_epochs=bad_epochs)
        state['test_backend'] = model.metadata_encoder.test_backend or bool(model.llm_actions and (model.diet_vlm.test_backend or model.exercise_vlm.test_backend))
        state['test_backend'] |= any(v['test_backend'] for row in dataset.samples for v in row.get('supervision', {}).values())
        torch.save(state, output/'checkpoint.pt')
        if improved: torch.save(state,output/'best.pt')
        (output/'history.json').write_text(json.dumps(history, allow_nan=False), encoding='utf-8')
        if args.max_steps and local_steps >= args.max_steps: break
        if bad_epochs >= training['early_stop_patience']:
            print(json.dumps(dict(early_stop=True,epoch=epoch+1,best_validation_mae=best_mae))); break


if __name__ == '__main__': main()
