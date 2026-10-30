"""Deployment personalisation; see USER_FEEDBACK.md for factual measurement input."""
import argparse
from collections import Counter, defaultdict
from configuration import settings, parse_configured, optimizer_settings
import hashlib
import json
import math
from itertools import zip_longest
from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader
from alignment import MODES, NumericPreferencePairs, FrozenRankingReference, preference_objective, candidate, batch_candidates
from runtime_schema import supervision_loss
from train_data import read_records, ReplayPairs
from world_model import MetabolicWorldModel, llm_actions_enabled


def feedback_records(records):
    unique, events, contexts = {}, {}, {}
    for row in records:
        for key in ('feedback_id', 'preferred_id', 'rejected_id'):
            if not isinstance(row.get(key), str) or not row[key].strip(): raise ValueError(f'Feedback requires {key}')
        if row['preferred_id'] == row['rejected_id'] or row['preferred'] == row['rejected']: raise ValueError('Feedback alternatives must differ')
        subject = row['context']['subject_id']
        event = (subject, row['feedback_id'])
        identity = (row['state_id'], row['split'], row['context'], row['preferred_id'], row['preferred'])
        if event in events and events[event] != identity: raise ValueError('Conflicting feedback event')
        events[event] = identity
        key = (*event, row['rejected_id'])
        if key in unique and unique[key] != row: raise ValueError('Conflicting duplicate feedback pair')
        unique[key] = row
        for state in (row['state_id'], (subject, row['context']['origin'])):
            value = (row['split'], row['context'])
            if state in contexts and contexts[state] != value: raise ValueError('Feedback context or split mismatch')
            contexts[state] = value
    return list(unique.values())


def observation_records(records, vocabulary, schema, horizon):
    """Validate factual executed actions; preference alternatives are never labels."""
    if not schema or not any(s.get('prediction') == 'predicted_cgm' for s in schema['supervision']):
        raise ValueError('Feedback simulation updates require a glucose prediction supervision schema')
    unique, rows = {}, dict(train=[], validation=[], test=[])
    for row in records:
        if not isinstance(row.get('observation_id'), str) or not row['observation_id'].strip():
            raise ValueError('Measurements require a factual observation_id')
        if not row.get('state_id') or row.get('split') not in rows: raise ValueError('Observations require state_id and explicit split')
        key = (row['context']['subject_id'], row['observation_id'])
        if key in unique:
            if unique[key] != row: raise ValueError('Conflicting duplicate observation')
            continue
        unique[key] = row
        action = candidate(row['context'], row['action'], vocabulary)
        target = torch.tensor([float('nan') if v is None else v for v in row['target']], dtype=torch.float32)
        if any(v not in (True, False, 0, 1) for v in row['target_mask']): raise ValueError('Invalid observation mask')
        mask = torch.tensor(row['target_mask'], dtype=torch.bool)
        if target.shape != (horizon,) or mask.shape != target.shape: raise ValueError('Observation target and mask must match the prediction horizon')
        if not mask.any() or not torch.isfinite(target[mask]).all(): raise ValueError('Measured glucose must be finite and nonempty')
        target = target.masked_fill(~mask, 0.)
        supervision = {}
        for spec in schema['supervision']:
            if spec.get('prediction') == 'predicted_cgm':
                if spec['shape'] != [horizon]: raise ValueError('Glucose supervision shape differs from observation horizon')
                supervision[spec['name']] = dict(target=target, mask=mask)
            else:
                supervision[spec['name']] = dict(target=torch.zeros(spec['shape']), mask=torch.zeros(spec['shape'], dtype=torch.bool))
        rows[row['split']].append(dict(action=action, mode=row['action']['mode'], supervision=supervision))
    return rows


def observation_objective(model, rows, device):
    losses = []
    for mode in MODES:
        selected = [row for row in rows if row['mode'] == mode]
        if not selected: continue
        batch = batch_candidates([row['action'] for row in selected], device)
        output = model(batch, mode)
        losses.append(supervision_loss(model, [row['supervision'] for row in selected], output, mode)*len(selected))
    return torch.stack(losses).sum()/len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Participant feedback DPO and factual glucose simulation-head updates.',
        epilog='JSON/JSONL rows use the existing preference schema: state_id, split, context, preferred, rejected, optional pair_weight; '
        'add feedback_id (one user selection event), preferred_id and rejected_id (stable prescription IDs), source_stage=user_feedback. '
        'Supply explicit chosen/rejected pairs, not merely unselected candidates. IDs are scoped by context.subject_id. '
        'context uses configured history/history_mask, metadata, subject_id and origin; actions use mode, diet (6 factors), '
        'exercise (3 factors), onset (2 offsets), and optional exercise_type, diet_text, exercise_text, diet_images. '
        'Training keeps one pair per user/selected prescription each epoch and weights it by 1/sqrt(selection count). '
        'Replay uses the configured fraction, with at least one pair per prior stage (the fraction can be higher for small inputs). '
        'Both prior-stage replay files require training pairs; physiological replay requires factual observation IDs. '
        '--observation-data contains factual observation_id, state_id, split, context, executed action, target and target_mask. '
        'Only measured glucose supervises the simulation head; unexecuted alternatives never supply targets. '
        'Use one user per run for a personal checkpoint. Resume requires the same data and settings, starts at the next epoch, '
        'and does not restore unconsumed batches after --max-steps. Use this script for feedback resume; phase remains physiological for inference compatibility.')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--init-checkpoint')
    source.add_argument('--resume')
    parser.add_argument('--data', required=True)
    parser.add_argument('--observation-data', required=True)
    parser.add_argument('--expert-replay-data', required=True)
    parser.add_argument('--physiological-replay-data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device')
    args = parse_configured(parser, argv, 'user_feedback',
        ('epochs', 'batch_size', 'learning_rate', 'beta', 'seed', 'device'))
    feedback = settings('user_feedback')
    if args.epochs < 1 or args.batch_size < 1 or (args.max_steps is not None and args.max_steps < 1): raise ValueError('Counts must be positive')
    if any(not math.isfinite(v) or v <= 0 for v in (args.beta, args.learning_rate)): raise ValueError('Beta and learning rate must be finite and positive')
    random.seed(args.seed); torch.manual_seed(args.seed)
    parent = args.resume or args.init_checkpoint
    checkpoint = torch.load(parent, map_location='cpu', weights_only=False)
    if checkpoint.get('architecture_version') != 3 or checkpoint.get('ranking_version') != 2: raise ValueError('Expected architecture v3 and ranking v2')
    if checkpoint.get('phase') != 'physiological': raise ValueError('User feedback requires physiological alignment')
    if checkpoint['llm_actions'] != llm_actions_enabled(): raise ValueError('Checkpoint action backend mismatch')
    if args.resume and checkpoint.get('alignment_stage') != 'user_feedback': raise ValueError('Resume requires a user-feedback checkpoint')
    if args.init_checkpoint and checkpoint.get('alignment_stage') == 'user_feedback': raise ValueError('Use --resume for a user-feedback checkpoint')
    records = feedback_records(read_records(args.data))
    measured_records = read_records(args.observation_data)
    subjects = {r['context']['subject_id'] for r in records + measured_records}
    if len(subjects) != 1: raise ValueError('Personal feedback and observations must belong to exactly one participant')
    expert_records, physiological_records = read_records(args.expert_replay_data), read_records(args.physiological_replay_data)
    contexts = {}
    for row in records + measured_records + expert_records + physiological_records:
        identity = (row['split'], json.dumps(row['context'], sort_keys=True, allow_nan=False))
        for key in (row['state_id'], (row['context']['subject_id'], row['context']['origin'])):
            if key in contexts and contexts[key] != identity: raise ValueError('Replay context or split mismatch')
            contexts[key] = identity
    vocabulary = checkpoint['vocabulary']
    if checkpoint['model_config']['data_config'] != settings('data'): raise ValueError('Checkpoint data configuration differs from configuration file')
    observations = observation_records(measured_records, vocabulary, checkpoint['model_config'].get('schema'), settings('data', 'horizon_steps'))
    if not observations['train']: raise ValueError('Simulation updates require training observations')
    current = NumericPreferencePairs(records, vocabulary, 'user_feedback')
    validation = NumericPreferencePairs(records, vocabulary, 'user_feedback', 'validation')
    expert = NumericPreferencePairs(expert_records, vocabulary, 'professional')
    physiological = NumericPreferencePairs(physiological_records, vocabulary, 'physiological')
    if not len(current) or not len(expert) or not len(physiological): raise ValueError('Training requires feedback and both prior-stage replay datasets')
    groups, selections = defaultdict(list), defaultdict(set)
    for index, row in enumerate(r for r in records if r['split'] == 'train'):
        key = (row['context']['subject_id'], row['preferred_id'])
        groups[key].append(index); selections[key].add(row['feedback_id'])
    for key, indices in groups.items():
        for index in indices: current.rows[index]['pair_weight'] /= math.sqrt(len(selections[key]))
    feedback_settings = dict(beta=args.beta, learning_rate=args.learning_rate, replay_fraction=feedback['replay_fraction'], seed=args.seed, batch_size=args.batch_size,
        simulation_updates=True, participant_id=next(iter(subjects)), gradient_clip=feedback['gradient_clip'], optimizer=optimizer_settings(),
        data_signature=hashlib.sha256(json.dumps([records, measured_records, expert_records, physiological_records], sort_keys=True, allow_nan=False).encode()).hexdigest())
    previous = checkpoint.get('user_feedback', {})
    if args.resume and previous.get('settings') != feedback_settings: raise ValueError('Resume feedback data or settings differ')
    model = MetabolicWorldModel(**checkpoint['model_config'])
    model.load_checkpoint(checkpoint); model.to(args.device)
    model.configure_phase('physiological'); model.eval()
    reference = FrozenRankingReference(model).to(args.device)
    model.configure_phase('user_feedback')
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, **optimizer_settings())
    history, step, start = [], 0, 0
    if args.resume:
        reference.load_state_dict(previous['reference_state_dict'])
        optimizer.load_state_dict(previous['optimizer'])
        torch.set_rng_state(previous['rng_state']); random.setstate(previous['random_state'])
        if torch.cuda.is_available() and previous.get('cuda_rng_state'): torch.cuda.set_rng_state_all(previous['cuda_rng_state'])
        history, step, start = list(previous['history']), previous['global_step'], previous['epoch']
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    if (output/'checkpoint.pt').resolve() == Path(parent).resolve() and not args.resume: raise ValueError('Output must not overwrite the input checkpoint')
    local_steps = 0
    for epoch in range(start, start + args.epochs):
        rng = random.Random(args.seed + epoch)
        selected = [current[rng.choice(indices)] for indices in groups.values()]
        fraction = feedback['replay_fraction']
        if not 0 < fraction < 1: raise ValueError('Feedback replay fraction must be between zero and one')
        replay_count = max(2, round(len(selected)*fraction/(1-fraction)))
        old = [expert, physiological]
        balanced_replay = [old[i % 2][rng.randrange(len(old[i % 2]))] for i in range(replay_count)]
        data = ReplayPairs(selected, [balanced_replay], 0., args.seed)
        data.indices.extend(('old', 0, i) for i in range(replay_count)); rng.shuffle(data.indices)
        total, count, replay_seen = 0., 0, Counter()
        prediction_total, observed_count = 0., 0
        pair_loader = DataLoader(data, batch_size=args.batch_size, collate_fn=list)
        observation_loader = DataLoader(observations['train'], batch_size=args.batch_size, shuffle=True, collate_fn=list)
        for pairs, measured in zip_longest(pair_loader, observation_loader):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            if pairs:
                preference_loss, _ = preference_objective(model, pairs, reference, args.beta, args.device)
                preference_loss = preference_loss * (sum(row['pair_weight'] for row in pairs)/len(pairs))
                losses.append(preference_loss)
                total += float(preference_loss.detach())*len(pairs); count += len(pairs)
            if measured:
                prediction_loss = observation_objective(model, measured, args.device)
                losses.append(prediction_loss)
                prediction_total += float(prediction_loss.detach())*len(measured); observed_count += len(measured)
            loss = torch.stack(losses).sum()
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite feedback loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, feedback['gradient_clip'], error_if_nonfinite=True); optimizer.step()
            if any(not torch.isfinite(p).all() for p in parameters): raise FloatingPointError('Nonfinite feedback weights')
            for row in pairs or []:
                for side in ('preferred', 'rejected'): model.ranked_modes[MODES.index(row[side+'_mode'])] = True
                if row['source_stage'] != 'user_feedback': replay_seen[row['source_stage']] += 1
            step += 1; local_steps += 1
            if args.max_steps and local_steps >= args.max_steps: break
        correct, valid_count, validation_prediction, validation_observed = 0., 0, 0., 0
        with torch.no_grad():
            for pairs in DataLoader(validation, batch_size=args.batch_size, collate_fn=list):
                _, accuracy = preference_objective(model, pairs, reference, args.beta, args.device)
                correct += float(accuracy)*len(pairs); valid_count += len(pairs)
            for measured in DataLoader(observations['validation'], batch_size=args.batch_size, collate_fn=list):
                validation_prediction += float(observation_objective(model, measured, args.device))*len(measured)
                validation_observed += len(measured)
        entry = dict(epoch=epoch+1, step=step, loss=total/count, physiological_loss=prediction_total/observed_count,
            observations=observed_count, validation_physiological_loss=validation_prediction/validation_observed if validation_observed else None,
            pairs=count, feedback_pairs=len(current), selected_pairs=len(selected),
            replay_pairs=dict(replay_seen), replay_fraction=sum(replay_seen.values())/count, validation_accuracy=correct/valid_count if valid_count else None)
        history.append(entry); print(json.dumps(entry, allow_nan=False))
        state = dict(checkpoint)
        state.update(model_state_dict=model.state_dict(), alignment_stage='user_feedback', parent=str(Path(parent).resolve()),
            test_backend=checkpoint.get('test_backend', False) or model.metadata_encoder.test_backend or bool(model.llm_actions and (model.diet_vlm.test_backend or model.exercise_vlm.test_backend)),
            user_feedback=dict(settings=feedback_settings, optimizer=optimizer.state_dict(), reference_state_dict=reference.state_dict(),
                reference_parent=previous['reference_parent'] if args.resume else str(Path(parent).resolve()), epoch=epoch+1, global_step=step, history=history,
                selection_counts=[dict(subject_id=k[0], preferred_id=k[1], count=len(v)) for k,v in selections.items()],
                rng_state=torch.get_rng_state(), random_state=random.getstate(), cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
        temporary = output/'checkpoint.pt.tmp'
        torch.save(state, temporary); temporary.replace(output/'checkpoint.pt')
        (output/'history.json').write_text(json.dumps(history, allow_nan=False), encoding='utf-8')
        if args.max_steps and local_steps >= args.max_steps: break


if __name__ == '__main__': main()
