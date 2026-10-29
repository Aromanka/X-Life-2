import hashlib
import json
import math
import random
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from train_data import read_records, ReplayPairs


MODES = ('diet', 'exercise', 'joint')


def candidate(context, action, vocabulary):
    if set(context) != {'history', 'history_mask', 'metadata', 'subject_id', 'origin'}: raise ValueError('Context requires history, history_mask, metadata, subject_id and origin only')
    if set(action) - {'mode', 'diet', 'exercise', 'exercise_type', 'onset', 'diet_text', 'exercise_text'}: raise ValueError('Candidates may contain actions only, never context or future labels')
    mode = action['mode']
    if mode not in MODES: raise ValueError('Unknown candidate mode')
    sample = {key:torch.tensor([float('nan') if v is None else v for v in context[key]], dtype=torch.float32) for key in ('history', 'metadata')}
    if torch.isinf(sample['metadata']).any(): raise ValueError('Invalid metadata')
    if any(value not in (True, False, 0, 1) for value in context['history_mask']): raise ValueError('Invalid history mask')
    sample['history_mask'] = torch.tensor(context['history_mask'], dtype=torch.bool)
    if sample['history'].shape != (36,) or sample['history_mask'].shape != (36,) or sample['metadata'].shape != (84,): raise ValueError('Expected 36 history points and 84 metadata values')
    if not sample['history_mask'].any() or not torch.isfinite(sample['history'][sample['history_mask']]).all(): raise ValueError('Measured history must be finite and nonempty')
    sample['history'] = sample['history'].masked_fill(~sample['history_mask'], 0.)
    for name, size in (('diet', 6), ('exercise', 3)):
        values = action.get(name, [None]*size)
        sample[name] = torch.tensor([float('nan') if v is None else v for v in values], dtype=torch.float32)
        if sample[name].shape != (size,) or torch.isinf(sample[name]).any(): raise ValueError('Invalid action values')
        if (mode == name or mode == 'joint') and not torch.isfinite(sample[name]).any(): raise ValueError('Active action requires observed values')
    sample['onset'] = torch.tensor(action['onset'], dtype=torch.float32)
    if sample['onset'].shape != (2,) or not torch.isfinite(sample['onset']).all(): raise ValueError('Two finite onset offsets are required')
    sample['type_id'] = torch.tensor(vocabulary.get(action.get('exercise_type', ''), 0))
    for key in ('diet_text', 'exercise_text'): sample[key] = action.get(key, '')
    return sample


class NumericPreferencePairs(Dataset):
    def __init__(self, records, vocabulary, stage, split='train'):
        self.rows = []
        contexts = {}
        for row in records:
            if not row.get('state_id') or row.get('split') not in {'train', 'validation', 'test'}: raise ValueError('Pairs require state_id and explicit split')
            identity = (row['split'], json.dumps(row['context'], sort_keys=True))
            for key in (row['state_id'], (row['context']['subject_id'], row['context']['origin'])):
                if key in contexts and contexts[key] != identity: raise ValueError('A state must have one immutable context and split')
                contexts[key] = identity
            if row.get('source_stage', stage) != stage: raise ValueError('Pair source stage mismatch')
            if stage == 'physiological':
                evidence = row.get('evidence', {})
                if not all(isinstance(evidence.get(k), str) and evidence[k].strip() for k in ('source', 'preferred_observation_id', 'rejected_observation_id')): raise ValueError('Physiological pairs require source and two factual observation IDs')
                if evidence['preferred_observation_id'] == evidence['rejected_observation_id']: raise ValueError('Two alternatives cannot share one factual observation')
            weight = float(row.get('pair_weight', 1.))
            if not math.isfinite(weight) or weight <= 0: raise ValueError('Pair weight must be finite and positive')
            pair = {side:candidate(row['context'], row[side], vocabulary) for side in ('preferred', 'rejected')}
            pair.update(preferred_mode=row['preferred']['mode'], rejected_mode=row['rejected']['mode'], pair_weight=weight, source_stage=stage)
            if row['split'] == split: self.rows.append(pair)

    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index]


def batch_candidates(rows, device):
    return {key:torch.stack([r[key] for r in rows]).to(device) if isinstance(rows[0][key], torch.Tensor) else [r[key] for r in rows] for key in rows[0]}


class FrozenRankingReference(nn.Module):
    def __init__(self, model):
        super().__init__()
        if any(p.requires_grad for name,p in model.named_parameters() if not name.startswith('ranking_head.')): raise ValueError('Freeze the physiological backbone first')
        self.ranking_head = deepcopy(model.ranking_head).eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, trajectory, mode): return self.ranking_head(trajectory.detach(), mode)


def pair_scores(model, pairs, side, reference, device):
    scores, refs = [], []
    for mode in MODES:
        indices = [i for i,row in enumerate(pairs) if row[side+'_mode'] == mode]
        if not indices: continue
        batch = batch_candidates([pairs[i][side] for i in indices], device)
        with torch.no_grad(): trajectory = model(batch, mode)['trajectory']
        scores.extend(zip(indices, model.ranking_head(trajectory, mode).unbind()))
        refs.extend(zip(indices, reference(trajectory, mode).unbind()))
    return torch.stack([s for _,s in sorted(scores)]), torch.stack([s for _,s in sorted(refs)])


def preference_objective(model, pairs, reference, beta, device):
    preferred, ref_preferred = pair_scores(model, pairs, 'preferred', reference, device)
    rejected, ref_rejected = pair_scores(model, pairs, 'rejected', reference, device)
    weights = preferred.new_tensor([row['pair_weight'] for row in pairs])
    margin = preferred-rejected-(ref_preferred-ref_rejected)
    return (-F.logsigmoid(beta*margin)*weights).sum()/weights.sum(), (preferred > rejected).float().mean()


def run_alignment(args, model, checkpoint, config, output):
    if model.llm_actions: raise ValueError('Offline numeric alignment currently requires numeric action encoders')
    settings = config.get('alignment', {})
    beta, fraction = float(settings.get('beta', 2.)), float(settings.get('replay_fraction', .25))
    if not math.isfinite(beta) or beta <= 0: raise ValueError('Bounded-score beta must be finite and positive')
    if args.phase == 'physiological' and not .2 <= fraction <= .3: raise ValueError('Physiological replay must be 20–30 percent')
    expected = 'professional' if args.phase == 'physiological' else 'fusion'
    if checkpoint['phase'] != (args.phase if args.resume else expected): raise ValueError(f'{args.phase} requires {expected} initialization or same-phase resume')
    records = read_records(args.data)
    current = NumericPreferencePairs(records, checkpoint['vocabulary'], args.phase)
    if not len(current): raise ValueError('No training preference pairs')
    previous_records = read_records(args.replay_data) if args.replay_data else []
    if args.phase == 'physiological' and not previous_records: raise ValueError('Physiological alignment requires expert replay data')
    if args.phase == 'professional' and previous_records: raise ValueError('Expert alignment does not use replay')
    contexts = {}
    for row in records+previous_records:
        identity = (row['split'], json.dumps(row['context'], sort_keys=True))
        for key in (row['state_id'], (row['context']['subject_id'], row['context']['origin'])):
            if key in contexts and contexts[key] != identity: raise ValueError('Replay context or split mismatch')
            contexts[key] = identity
    previous = NumericPreferencePairs(previous_records, checkpoint['vocabulary'], 'professional')
    data = ReplayPairs(current, [previous] if previous_records else [], fraction if previous_records else 0., args.seed)
    if previous_records and not .2 <= data.replay_count/len(data) <= .3: raise ValueError('Too few current pairs to realize 20–30 percent replay; supply more pairs')
    signature = hashlib.sha256(json.dumps([records, previous_records], sort_keys=True).encode()).hexdigest()
    alignment = dict(beta=beta, replay_fraction=fraction if previous_records else 0., data_signature=signature, seed=args.seed)
    model.configure_phase(args.phase); model.eval()
    reference = FrozenRankingReference(model).to(args.device)
    optimizer = torch.optim.AdamW(model.ranking_head.parameters(), lr=settings.get('learning_rate', 1e-4))
    history, step, start = [], 0, 0
    if args.resume:
        if checkpoint['alignment'] != alignment: raise ValueError('Resume alignment settings or data differ')
        reference.load_state_dict(checkpoint['reference_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['rng_state']); random.setstate(checkpoint['random_state'])
        if torch.cuda.is_available() and checkpoint.get('cuda_rng_state'): torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
        history, step, start = checkpoint['history'], checkpoint['global_step'], checkpoint['epoch']
    validation = NumericPreferencePairs(records, checkpoint['vocabulary'], args.phase, 'validation')
    local_steps = 0
    for epoch in range(start, start+args.epochs):
        data.set_epoch(epoch)
        total, count, replay_seen = 0., 0, 0
        for pairs in DataLoader(data, batch_size=args.batch_size, collate_fn=list):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = preference_objective(model, pairs, reference, beta, args.device)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite preference loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.ranking_head.parameters(), 1.); optimizer.step()
            for row in pairs:
                for side in ('preferred', 'rejected'): model.ranked_modes[MODES.index(row[side+'_mode'])] = True
            total += float(loss.detach())*len(pairs); count += len(pairs)
            replay_seen += sum(row['source_stage'] != args.phase for row in pairs)
            step += 1; local_steps += 1
            if args.max_steps and local_steps >= args.max_steps: break
        correct, valid_count = 0., 0
        with torch.no_grad():
            for pairs in DataLoader(validation, batch_size=args.batch_size, collate_fn=list):
                _, accuracy = preference_objective(model, pairs, reference, beta, args.device)
                correct += float(accuracy)*len(pairs); valid_count += len(pairs)
        entry = dict(epoch=epoch+1, step=step, loss=total/count, pairs=count, replay_pairs=replay_seen, replay_fraction=replay_seen/count, validation_accuracy=correct/valid_count if valid_count else None)
        history.append(entry); print(json.dumps(entry))
        state = dict(checkpoint)
        state.update(ranking_version=1, phase=args.phase, model_state_dict=model.state_dict(), optimizer=optimizer.state_dict(), reference_state_dict=reference.state_dict(), reference_parent=checkpoint.get('reference_parent') if args.resume else args.init_checkpoint, parent=args.resume or args.init_checkpoint, alignment=alignment, config=config, epoch=epoch+1, global_step=step, history=history, data=str(Path(args.data).resolve()), rng_state=torch.get_rng_state(), random_state=random.getstate(), cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        torch.save(state, output/'checkpoint.pt')
        (output/'history.json').write_text(json.dumps(history, allow_nan=False), encoding='utf-8')
        if args.max_steps and local_steps >= args.max_steps: break


def rank_records(args, model, checkpoint, output):
    results = []
    for record in read_records(args.data):
        actions = record['candidates']
        if not actions: raise ValueError('No candidates')
        scores = []
        for start in range(0, len(actions), args.batch_size):
            chunk = actions[start:start+args.batch_size]
            batch = batch_candidates([candidate(record['context'], action, checkpoint['vocabulary']) for action in chunk], args.device)
            scores.extend(model.rank(batch, [a['mode'] for a in chunk]).cpu().tolist())
        if not all(math.isfinite(s) for s in scores): raise FloatingPointError('Nonfinite ranking score')
        results.append(dict(state_id=record['state_id'], checkpoint=str(Path(args.resume or args.init_checkpoint).resolve()), scores=scores, order=sorted(range(len(scores)), key=lambda i:scores[i], reverse=True)))
    (output/'rankings.json').write_text(json.dumps(results, allow_nan=False), encoding='utf-8')
