import argparse
import json
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from dataset import XLifeCleanedDataset, SampleNumericDataset
from world_model import MetabolicWorldModel, llm_actions_enabled


def collate(rows, vocabulary):
    batch = {key: torch.stack([row[key] for row in rows]) for key in ('diet', 'exercise', 'metadata', 'onset', 'history', 'history_mask', 'target', 'target_mask')}
    batch['type_id'] = torch.tensor([vocabulary.get(row['exercise_type'], 0) for row in rows])
    for key in ('diet_text', 'exercise_text', 'subject_id', 'origin'): batch[key] = [row[key] for row in rows]
    return batch


def device_batch(batch, device): return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def evaluate(model, samples, vocabulary, device, batch_size, mode=None):
    model.eval()
    absolute, points, correlations, invalid, predictions = 0., 0, [], 0, []
    for domain in ([mode] if mode else ['diet', 'exercise']):
        rows = [s for s in samples if s['joint']] if domain == 'joint' else [s for s in samples if s['mode'] == domain]
        for batch in DataLoader(rows, batch_size=batch_size, collate_fn=lambda rows: collate(rows, vocabulary)):
            data = device_batch(batch, device)
            with torch.no_grad(): predicted = model(data, domain)['predicted_cgm'].cpu()
            if not torch.isfinite(predicted).all(): raise FloatingPointError('Nonfinite prediction')
            for i, (p, y, mask) in enumerate(zip(predicted, batch['target'], batch['target_mask'])):
                absolute += float((p[mask]-y[mask]).abs().sum()); points += int(mask.sum())
                if mask.sum() > 1 and p[mask].std() > 1e-8 and y[mask].std() > 1e-8:
                    correlations.append(float(torch.corrcoef(torch.stack([p[mask], y[mask]]))[0, 1]))
                else: invalid += 1
                predictions.append(dict(subject_id=batch['subject_id'][i], origin=batch['origin'][i], mode=domain, predicted=p.tolist(), observed=y.tolist(), mask=mask.tolist()))
    return dict(mae=absolute/points if points else None, pearson=sum(correlations)/len(correlations) if correlations else None, valid_points=points, valid_correlations=len(correlations), invalid_correlations=invalid), predictions


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--dataset', choices=['xlife_cleaned', 'sample'], default='xlife_cleaned')
    parser.add_argument('--config')
    parser.add_argument('--phase', choices=['single', 'fusion'], default='single')
    parser.add_argument('--domain', choices=['both', 'diet', 'exercise'], default='both')
    parser.add_argument('--output', required=True)
    parser.add_argument('--init-checkpoint')
    parser.add_argument('--resume')
    parser.add_argument('--predict-only', action='store_true')
    parser.add_argument('--mode', choices=['diet', 'exercise', 'joint'])
    parser.add_argument('--split', choices=['train', 'validation', 'test'], default='test')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--max-subjects', type=int)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or any(v is not None and v < 1 for v in (args.max_steps, args.max_subjects, args.max_samples)): raise ValueError('Counts must be positive')
    if args.resume and args.init_checkpoint: raise ValueError('Choose resume or initialization')
    random.seed(args.seed); torch.manual_seed(args.seed)
    config = json.loads(Path(args.config).read_text(encoding='utf-8')) if args.config else {}
    model_config = config.get('model', {})
    settings = config.get('training', {})
    checkpoint_path = args.resume or args.init_checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False) if checkpoint_path else None
    if checkpoint:
        if checkpoint.get('architecture_version') != 3: raise ValueError('Expected architecture_version 3')
        if checkpoint['llm_actions'] != llm_actions_enabled(): raise ValueError('Checkpoint action backend differs from environment')
        model_config = checkpoint['model_config']
    if (args.predict_only or args.phase == 'fusion') and not checkpoint: raise ValueError('A checkpoint is required')
    dataset = {'xlife_cleaned': XLifeCleanedDataset, 'sample': SampleNumericDataset}[args.dataset](args.data, args.max_subjects, args.max_samples)
    if not dataset.samples: raise ValueError('No usable records')
    ids = sorted({s['subject_id'] for s in dataset.samples})
    if checkpoint: splits = checkpoint['splits']
    else:
        random.shuffle(ids)
        train_count = max(1, int(len(ids)*.8))
        val_count = max(1, int(len(ids)*.1)) if len(ids) >= 3 else 0
        splits = dict(train=ids[:train_count], validation=ids[train_count:train_count+val_count], test=ids[train_count+val_count:])
    groups = {key: [s for s in dataset.samples if s['subject_id'] in subjects] for key, subjects in splits.items()}
    if not groups['train']: raise ValueError('No training subjects from checkpoint in selected data')
    vocabulary = checkpoint['vocabulary'] if checkpoint else {v:i+1 for i,v in enumerate(sorted({s['exercise_type'] for s in groups['train'] if s['exercise_type']}))}
    if len(vocabulary) >= 128: raise ValueError('Too many exercise categories')
    model = MetabolicWorldModel(**model_config)
    if checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
    else: model.fit(groups['train'])
    if args.phase == 'fusion' and not args.resume and not args.predict_only:
        if checkpoint['phase'] != 'single' or set(checkpoint['trained_domains']) != {'diet','exercise'}: raise ValueError('Fusion requires both pretrained domains')
        model.initialize_fusion()
    model.to(args.device)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    print(json.dumps(dict(records=len(dataset), skipped=dataset.skipped, splits={k:len(v) for k,v in groups.items()}, device=args.device)))
    if args.predict_only:
        if args.mode == 'joint' and checkpoint['phase'] != 'fusion': raise ValueError('Joint prediction requires fusion training')
        metrics, predictions = evaluate(model, groups[args.split], vocabulary, args.device, args.batch_size, args.mode or ('joint' if checkpoint['phase']=='fusion' else None))
        if not predictions: raise ValueError('No prediction samples in selected split/mode')
        (output/'predictions.json').write_text(json.dumps(predictions, allow_nan=False), encoding='utf-8')
        (output/'metrics.json').write_text(json.dumps(metrics, allow_nan=False), encoding='utf-8')
        print(json.dumps(metrics)); return
    model.configure_phase(args.phase)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=settings.get('learning_rate', 1e-4))
    history, global_step, start_epoch = [], 0, 0
    trained_domains = set(checkpoint.get('trained_domains', [])) if checkpoint else set()
    if args.resume:
        if checkpoint['phase'] != args.phase: raise ValueError('Resume phase mismatch')
        optimizer.load_state_dict(checkpoint['optimizer'])
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
                predicted = model(batch, domain)['predicted_cgm']
                valid = batch['target_mask']
                loss = ((predicted[valid]-batch['target'][valid])/model.glucose_scale).square().mean()
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
                global_step += 1; local_steps += 1; totals += float(loss.detach()); counts += 1
                if domain != 'joint': trained_domains.add(domain)
                if args.max_steps and local_steps >= args.max_steps: break
            if args.max_steps and local_steps >= args.max_steps: break
        metrics, _ = evaluate(model, groups['validation'], vocabulary, args.device, args.batch_size, 'joint' if args.phase=='fusion' else None)
        entry = dict(epoch=epoch+1, step=global_step, loss=totals/counts, validation=metrics)
        history.append(entry); print(json.dumps(entry))
        state = dict(architecture_version=3, schema_version=1, phase=args.phase, trained_domains=sorted(trained_domains), model_config=model_config, llm_actions=model.llm_actions, model_state_dict=model.state_dict(), optimizer=optimizer.state_dict(), splits=splits, vocabulary=vocabulary, epoch=epoch+1, global_step=global_step, history=history, rng_state=torch.get_rng_state(), random_state=random.getstate(), cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, parent=checkpoint_path, data=str(Path(args.data).resolve()), config=config)
        torch.save(state, output/'checkpoint.pt')
        (output/'history.json').write_text(json.dumps(history, allow_nan=False), encoding='utf-8')
        if args.max_steps and local_steps >= args.max_steps: break


if __name__ == '__main__': main()
