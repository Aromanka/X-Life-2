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
from dataset_metadata import prepare_metadata
from configuration import settings, optimizer_settings
from world_model import ranking_probabilities, llm_actions_enabled


MODES = ('diet', 'exercise', 'joint')


def candidate(context, action, vocabulary):
    if set(context) != {'history', 'history_mask', 'metadata', 'subject_id', 'origin'}: raise ValueError('Context requires history, history_mask, metadata, subject_id and origin only')
    if set(action) - {'mode', 'diet', 'exercise', 'exercise_type', 'intensity_scale', 'onset', 'diet_text', 'exercise_text', 'diet_images'}: raise ValueError('Candidates may contain actions only, never context or future labels')
    mode = action['mode']
    if mode not in MODES: raise ValueError('Unknown candidate mode')
    sample = {'history':torch.tensor([float('nan') if v is None else v for v in context['history']], dtype=torch.float32),
              'metadata':prepare_metadata(context['metadata'])}
    if any(value not in (True, False, 0, 1) for value in context['history_mask']): raise ValueError('Invalid history mask')
    sample['history_mask'] = torch.tensor(context['history_mask'], dtype=torch.bool)
    shape = (settings('data', 'history_steps'),)
    if sample['history'].shape != shape or sample['history_mask'].shape != shape: raise ValueError(f'Expected {shape[0]} history points')
    if not sample['history_mask'].any() or not torch.isfinite(sample['history'][sample['history_mask']]).all(): raise ValueError('Measured history must be finite and nonempty')
    sample['history'] = sample['history'].masked_fill(~sample['history_mask'], 0.)
    for name, size in (('diet', 6), ('exercise', 3)):
        values = action.get(name, [None]*size)
        sample[name] = torch.tensor([float('nan') if v is None else v for v in values], dtype=torch.float32)
        if sample[name].shape != (size,) or torch.isinf(sample[name]).any(): raise ValueError('Invalid action values')
        if not llm_actions_enabled() and mode in (name, 'joint') and not torch.isfinite(sample[name]).any(): raise ValueError('Active action requires observed values')
    sample['onset'] = torch.tensor(action['onset'], dtype=torch.float32)
    if sample['onset'].shape != (2,) or not torch.isfinite(sample['onset']).all(): raise ValueError('Two finite onset offsets are required')
    sample['type_id'] = torch.tensor(vocabulary.get(action.get('exercise_type', ''), 0))
    sample.update({key:action.get(key, default) for key,default in (
        ('diet_text',''), ('exercise_text',''), ('diet_images',[]), ('exercise_type',''), ('intensity_scale','unknown'))})
    return sample


class NumericPreferencePairs(Dataset):
    def __init__(self, records, vocabulary, stage, split=None):
        split = settings('alignment', 'split') if split is None else split
        self.rows = []
        contexts = {}
        for row in records:
            if not row.get('state_id') or row.get('split') not in {'train', 'validation', 'test'}: raise ValueError('Pairs require state_id and explicit split')
            identity = (row['split'], json.dumps(row['context'], sort_keys=True))
            for key in (row['state_id'], (row['context']['subject_id'], row['context']['origin'])):
                if key in contexts and contexts[key] != identity: raise ValueError('A state must have one immutable context and split')
                contexts[key] = identity
            source_stage = row.get('source_stage', stage or 'professional')
            if stage is not None and source_stage != stage: raise ValueError('Pair source stage mismatch')
            if source_stage == 'physiological':
                evidence = row.get('evidence', {})
                if not all(isinstance(evidence.get(k), str) and evidence[k].strip() for k in ('source', 'preferred_observation_id', 'rejected_observation_id')): raise ValueError('Physiological pairs require source and two factual observation IDs')
                if evidence['preferred_observation_id'] == evidence['rejected_observation_id']: raise ValueError('Two alternatives cannot share one factual observation')
            weight = float(row['pair_weight'] if 'pair_weight' in row else settings('alignment', 'pair_weight'))
            if not math.isfinite(weight) or weight <= 0: raise ValueError('Pair weight must be finite and positive')
            pair = {side:candidate(row['context'], row[side], vocabulary) for side in ('preferred', 'rejected')}
            pair.update(preferred_mode=row['preferred']['mode'], rejected_mode=row['rejected']['mode'], pair_weight=weight, source_stage=source_stage)
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
    # These are unbounded logits. Pairwise log-softmax differences equal logit
    # differences, since the common candidate-set normalizer cancels in DPO.
    preferred, ref_preferred = pair_scores(model, pairs, 'preferred', reference, device)
    rejected, ref_rejected = pair_scores(model, pairs, 'rejected', reference, device)
    weights = preferred.new_tensor([row['pair_weight'] for row in pairs])
    policy = F.log_softmax(torch.stack([preferred,rejected], -1), dim=-1)
    reference_policy = F.log_softmax(torch.stack([ref_preferred,ref_rejected], -1), dim=-1)
    log_ratio = policy-reference_policy
    margin = log_ratio[:,0]-log_ratio[:,1]
    return (-F.logsigmoid(beta*margin)*weights).sum()/weights.sum(), (preferred > rejected).float().mean()


def run_alignment(args, model, checkpoint, config, output):
    alignment_settings = config['alignment']
    beta, fraction = float(alignment_settings['beta']), float(alignment_settings['replay_fraction'])
    if not math.isfinite(beta) or beta <= 0: raise ValueError('DPO beta must be finite and positive')
    expected = 'professional' if args.phase == 'physiological' else 'fusion'
    if checkpoint['phase'] != (args.phase if args.resume else expected): raise ValueError(f'{args.phase} requires {expected} initialization or same-phase resume')
    records = read_records(args.data)
    current = NumericPreferencePairs(records, checkpoint['vocabulary'], args.phase)
    if not len(current): raise ValueError('No training preference pairs')
    previous_records = read_records(args.replay_data) if args.replay_data else []
    contexts = {}
    for row in records+previous_records:
        identity = (row['split'], json.dumps(row['context'], sort_keys=True))
        for key in (row['state_id'], (row['context']['subject_id'], row['context']['origin'])):
            if key in contexts and contexts[key] != identity: raise ValueError('Replay context or split mismatch')
            contexts[key] = identity
    previous = NumericPreferencePairs(previous_records, checkpoint['vocabulary'], None)
    data = ReplayPairs(current, [previous] if previous_records else [], fraction if previous_records else 0., args.seed)
    signature = hashlib.sha256(json.dumps([records, previous_records], sort_keys=True).encode()).hexdigest()
    alignment = dict(beta=beta, replay_fraction=fraction if previous_records else 0., data_signature=signature, seed=args.seed)
    model.configure_phase(args.phase); model.eval()
    reference = FrozenRankingReference(model).to(args.device)
    optimizer = torch.optim.AdamW(model.ranking_head.parameters(), lr=alignment_settings['learning_rate'], **optimizer_settings())
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
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.ranking_head.parameters(), alignment_settings['gradient_clip']); optimizer.step()
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
        state['test_backend'] = model.metadata_encoder.test_backend or checkpoint.get('test_backend', False)
        state.update(ranking_version=2, phase=args.phase, model_state_dict=model.state_dict(), optimizer=optimizer.state_dict(), reference_state_dict=reference.state_dict(), reference_parent=checkpoint.get('reference_parent') if args.resume else args.init_checkpoint, parent=args.resume or args.init_checkpoint, alignment=alignment, config=config, epoch=epoch+1, global_step=step, history=history, data=str(Path(args.data).resolve()), rng_state=torch.get_rng_state(), random_state=random.getstate(), cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        torch.save(state, output/'checkpoint.pt')
        (output/'history.json').write_text(json.dumps(history, allow_nan=False), encoding='utf-8')
        if args.max_steps and local_steps >= args.max_steps: break


def rank_records(args, model, checkpoint, output):
    results = []
    for record in read_records(args.data):
        actions = record['candidates']
        if not actions: raise ValueError('No candidates')
        logits = []
        for start in range(0, len(actions), args.batch_size):
            chunk = actions[start:start+args.batch_size]
            batch = batch_candidates([candidate(record['context'], action, checkpoint['vocabulary']) for action in chunk], args.device)
            logits.extend(model.rank_logits(batch, [a['mode'] for a in chunk]).cpu().tolist())
        scores = ranking_probabilities(torch.tensor(logits, dtype=torch.float64)).tolist()
        results.append(dict(state_id=record['state_id'], checkpoint=str(Path(args.resume or args.init_checkpoint).resolve()), scores=scores,
                            order=sorted(range(len(logits)), key=lambda i:logits[i], reverse=True)))
    (output/'rankings.json').write_text(json.dumps(results, allow_nan=False), encoding='utf-8')
