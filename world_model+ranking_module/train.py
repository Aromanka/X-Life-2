"""Train X-Life encoders, world model and sequential preference alignment."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import get_tokenizer
from model import CGMPredictionModelWithImage
from training import (LossWeights, configure_training_phase, set_training_mode,
                      encoder_pretraining_objective, world_model_objective,
                      preference_objective, FrozenRankingReference, to_device)
from train_data import (event_records, event_dataset, PreferencePairs, ReplayPairs,
                        read_records, collate_events, collate_pairs)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='JSON with model/data/training settings and local Qwen paths.')
    parser.add_argument('--phase', required=True, choices=['encoder_pretraining', 'world_model', 'preference_alignment', 'participant_update'])
    parser.add_argument('--data', required=True, help='Annotated events or factual observed CGM events, JSON/JSONL.')
    parser.add_argument('--pairs', help='Preferred/rejected actions sharing one context, JSON/JSONL.')
    parser.add_argument('--stage', choices=['professional', 'physiological', 'user'])
    parser.add_argument('--replay', nargs='*', default=[], help='Previous-stage pair files.')
    parser.add_argument('--init-checkpoint', help='Previous training-phase/stage checkpoint.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--participant-id', help='Restrict participant adaptation to this ID.')
    parser.add_argument('--device', default=None)
    parser.add_argument('--export-inference', action='store_true', help='Export core weights without training-only heads.')
    return parser.parse_args(argv)


def cycle_next(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def main(argv=None):
    args = parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    model_config, data_config, settings = config['model'], config.get('data', {}), config.get('training', {})
    seed = int(settings.get('seed', 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    annotation_dim = int(model_config.get('annotation_dim', 4096))
    epochs = int(args.epochs or settings.get('epochs', 1))
    batch_size = int(settings.get('batch_size', 1))
    accumulation = int(settings.get('gradient_accumulation_steps', 1))
    if epochs < 1 or batch_size < 1 or accumulation < 1:
        raise ValueError("Epochs, batch size and accumulation must be positive.")
    weights = LossWeights(**settings.get('loss_weights', {}))
    alignment = args.phase in {'preference_alignment', 'participant_update'}
    if alignment and (not args.stage or not args.pairs):
        raise ValueError("Preference alignment needs --stage and --pairs.")
    if alignment and args.stage != 'professional' and not args.participant_id:
        raise ValueError("Physiological/user alignment is participant-specific; supply --participant-id.")
    if args.phase == 'participant_update' and args.stage == 'professional':
        raise ValueError("Participant updates use physiological or user preference stages.")
    if args.phase == 'participant_update' and not args.participant_id:
        raise ValueError("Participant updates require --participant-id.")
    if args.phase != 'encoder_pretraining' and not args.init_checkpoint:
        raise ValueError("Provide the previous phase/stage checkpoint; frozen action encoders must be pretrained.")
    records = event_records(args.data, data_config.get('image_dir'))
    if args.participant_id:
        records = [row for row in records if str(row['subject_id']) == args.participant_id]
    if args.phase == 'encoder_pretraining':
        records = [row for row in records if any(row.get('training_targets', {}).get(kind) is not None
                                                for kind in ('diet', 'exercise'))]
        if not records:
            raise ValueError("First encode supplied diet/exercise annotations with encode_annotations.py.")
    if args.phase == 'world_model' and weights.omics > 0 and not any(
            row.get('training_targets', {}).get('omics') is not None for row in records):
        raise ValueError("No omics supervision is present. Supply cached omics targets, or explicitly set omics=0 for an unaugmented run.")
    tokenizer = get_tokenizer(model_path=model_config.get('qwen_model_path'))
    model = CGMPredictionModelWithImage(**model_config)
    model.to(device)
    initial = None
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)
        if initial.get('architecture_version') != 2:
            raise ValueError("Checkpoint architecture differs; use checkpoints from this implementation.")
        if initial.get('test_backend', False):
            raise ValueError("Tiny test checkpoints must not be used with research encoders.")
        model.load_state_dict(initial['model_state_dict'], strict=True)
        if args.phase == 'world_model' and initial.get('phase') != 'encoder_pretraining':
            raise ValueError("World-model training must initialize from encoder pretraining.")
        if args.phase == 'participant_update':
            if initial.get('stage') not in {'professional', 'physiological', 'user'}:
                raise ValueError("Participant adaptation requires a previously aligned checkpoint.")
            if initial.get('stage') != 'professional' and initial.get('participant_id') != args.participant_id:
                raise ValueError("Participant adaptation must use this participant's aligned checkpoint.")
        previous = {'professional': None, 'physiological': 'professional', 'user': 'physiological'}
        if alignment and args.phase != 'participant_update':
            expected = previous[args.stage]
            if (expected is None and initial.get('phase') != 'world_model') or (expected and initial.get('stage') != expected):
                raise ValueError("Run professional, physiological and user alignment sequentially.")
            if args.stage == 'user' and initial.get('participant_id') != args.participant_id:
                raise ValueError("User alignment must initialize from this participant's physiological checkpoint.")
    configure_training_phase(model, args.phase)
    factual_dataset = event_dataset(records, tokenizer, data_config, annotation_dim,
                                     require_cgm=args.phase != 'encoder_pretraining')
    if not len(factual_dataset):
        raise ValueError("No usable events remain after filtering.")
    if args.phase == 'encoder_pretraining':
        if not any(row.get('training_targets', {}).get(kind) is not None for row in records for kind in ('diet', 'exercise')):
            raise ValueError("First encode the structured annotations with encode_annotations.py.")
    generator = torch.Generator().manual_seed(seed)
    factual_loader = DataLoader(factual_dataset, batch_size=batch_size, shuffle=True,
                                 generator=generator, collate_fn=collate_events)
    reference, training_dataset, pair_loader = None, None, None
    if alignment:
        pair_records = read_records(args.pairs)
        if args.participant_id:
            pair_records = [row for row in pair_records if str(row['context']['subject_id']) == args.participant_id]
        current = PreferencePairs(pair_records, tokenizer, data_config, annotation_dim, args.stage)
        previous_datasets = []
        for path in args.replay:
            previous_rows = read_records(path)
            if args.participant_id:
                previous_rows = [row for row in previous_rows if row.get('source_stage') == 'professional'
                                 or str(row['context']['subject_id']) == args.participant_id]
            previous_datasets.append(PreferencePairs(previous_rows, tokenizer, data_config, annotation_dim, 'replay'))
        if not len(current):
            raise ValueError("Preference pair dataset is empty.")
        fraction = float(settings.get('replay_fraction', 0.25)) if args.stage != 'professional' else 0.0
        training_dataset = ReplayPairs(current, previous_datasets, fraction, seed)
        pair_loader = DataLoader(training_dataset, batch_size=batch_size, collate_fn=collate_pairs)
        reference = FrozenRankingReference(model).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=float(settings.get('learning_rate', 1e-4)),
                                  weight_decay=float(settings.get('weight_decay', 1e-5)))
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    history = []
    for epoch in range(epochs):
        set_training_mode(model)
        if training_dataset is not None:
            training_dataset.set_epoch(epoch)
        loader = pair_loader if alignment else factual_loader
        factual_iterator = iter(factual_loader)
        optimizer.zero_grad(set_to_none=True)
        total, count = 0.0, 0
        for i, batch in enumerate(loader):
            batch = to_device(batch, device)
            precision = settings.get('mixed_precision', 'no')
            if precision not in {'no', 'bf16'}:
                raise ValueError("mixed_precision must be no or bf16.")
            context = torch.autocast(device.type, dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()
            with context:
                if alignment:
                    observed, factual_iterator = cycle_next(factual_loader, factual_iterator)
                    loss, components = preference_objective(model, batch, to_device(observed, device),
                                                            reference, weights, args.stage)
                elif args.phase == 'encoder_pretraining':
                    loss, components = encoder_pretraining_objective(model, batch, weights)
                else:
                    loss, components = world_model_objective(model, batch, weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training produced a nonfinite loss.")
            group_start = i // accumulation * accumulation
            group_size = min(accumulation, len(loader) - group_start)
            (loss / group_size).backward()
            if (i + 1) % accumulation == 0 or i + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings.get('gradient_clip_norm', 1.0)))
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach()); count += 1
        entry = {'epoch': epoch + 1, 'loss': total / count,
                 'components_last_batch': {key: float(value) for key, value in components.items()},
                 'replay_pairs': training_dataset.replay_count if training_dataset else 0,
                 'total_pairs': len(training_dataset) if training_dataset else 0}
        history.append(entry); print(json.dumps(entry))
    checkpoint = {'architecture_version': 2, 'phase': args.phase, 'stage': args.stage,
                  'participant_id': args.participant_id,
                  'model_state_dict': model.state_dict(), 'model_config': model_config,
                  'config': config, 'loss_weights': asdict(weights), 'history': history}
    torch.save(checkpoint, output / 'checkpoint.pt')
    (output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    if args.export_inference:
        inference_config = dict(model_config, include_training_heads=False)
        torch.save({'architecture_version': 2, 'model_config': inference_config,
                    'phase': args.phase, 'stage': args.stage, 'participant_id': args.participant_id,
                    'model_state_dict': model.inference_state_dict()},
                   output / 'inference.pt')
    return checkpoint


if __name__ == '__main__':
    main()
