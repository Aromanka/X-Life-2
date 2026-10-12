"""Event and preference-pair adapters for X-Life training."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset

from dataset import SampleImageDataset
from main import SampleImagePredictionDataset, CGMPredictionTrainerWithImage


def read_records(path):
    path = Path(path)
    if path.suffix.lower() == '.jsonl':
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    records = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(records, list):
        raise ValueError("Training JSON must be a list of events or patient objects.")
    return records


def normalize_event(record):
    if not record.get('subject_id'):
        raise ValueError("Every training event needs a pseudonymous subject_id.")
    result = deepcopy(record)
    for key in ('metadata', 'sleep_data', 'food_data', 'exercise_data', 'training_annotations', 'training_targets'):
        result.setdefault(key, {})
    for key in ('cgm_pre', 'cgm_post'):
        result.setdefault(key, [])
    result.setdefault('cgm_type', 'exercise' if result['exercise_data'] and not result['food_data'] else 'meal')
    result.setdefault('subject_name', '')
    if 'event_time' not in result:
        if result['cgm_type'] == 'exercise':
            time = SampleImagePredictionDataset._meal_event_time('', result['exercise_data'].get('recorded_at', ''))
        else:
            food = result['food_data']; details = food.get('data', food)
            time = SampleImagePredictionDataset._meal_event_time(food.get('date', ''), details.get('time', ''))
        result['event_time'] = time.isoformat() if time is not None else None
    return result


def event_records(path, image_dir=None):
    records = read_records(path)
    if records and ('meals' in records[0] or 'exercise_events' in records[0]):
        # Read patient-object JSON through the patient event adapter.
        if Path(path).suffix.lower() == '.jsonl':
            raise ValueError("Use JSON for bundled patient objects and JSONL for normalized events.")
        records = SampleImageDataset(str(path), image_dir=image_dir).samples
    return [normalize_event(record) for record in records]


def event_dataset(records, tokenizer, data_config, annotation_dim, require_cgm=True):
    image_dir = data_config.get('image_dir')
    if image_dir:
        image_dir = str(Path(image_dir).expanduser().resolve())
    original = SampleImageDataset.from_samples([normalize_event(row) for row in records], image_dir)
    return SampleImagePredictionDataset(original, tokenizer,
        max_seq_length=int(data_config.get('max_text_length', 512)),
        cgm_seq_length=int(data_config.get('cgm_seq_length', 36)), image_dir=image_dir,
        annotation_dim=annotation_dim, require_cgm=require_cgm,
        sampling_interval_minutes=float(data_config.get('sampling_interval_minutes', 5)))


def collate_events(rows):
    return CGMPredictionTrainerWithImage._collate_fn(None, rows)


class PreferencePairs(Dataset):
    """Each preferred/rejected action is evaluated from one immutable context."""
    action_fields = {'food_data', 'exercise_data'}

    def __init__(self, records, tokenizer, data_config, annotation_dim, stage):
        self.rows, preferred, rejected = [], [], []
        for row in records:
            if stage == 'replay' and row.get('source_stage') not in {'professional', 'physiological', 'user'}:
                raise ValueError("Replay records must declare their original source_stage.")
            if not row.get('state_id'):
                raise ValueError("Preference pairs need a state_id for a shared decision context.")
            context = normalize_event(row['context'])
            for key in ('preferred', 'rejected'):
                action = row[key]
                if set(action) - self.action_fields:
                    raise ValueError("Candidates may replace actions only; metadata and pre-event CGM belong to context.")
                sample = deepcopy(context)
                sample.update(deepcopy(action))
                sample['cgm_post'] = []  # No observed label for an unexecuted alternative.
                (preferred if key == 'preferred' else rejected).append(sample)
            self.rows.append(row)
        self.preferred = event_dataset(preferred, tokenizer, data_config, annotation_dim, False)
        self.rejected = event_dataset(rejected, tokenizer, data_config, annotation_dim, False)
        self.stage = stage

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        weight = float(row.get('pair_weight', 1.0))
        count = float(row.get('selection_count', 1.0))
        diversity = float(row.get('diversity_weight', 1.0))
        if not all(torch.isfinite(torch.tensor(value)) for value in (weight, count, diversity)):
            raise ValueError("Pair weights and selection counts must be finite.")
        if weight < 0 or count < 1 or not 0 < diversity <= 1:
            raise ValueError("Invalid preference weight, repeat count or diversity weight.")
        return {'preferred': self.preferred[index], 'rejected': self.rejected[index],
                'pair_weight': weight, 'selection_count': count, 'diversity_weight': diversity,
                'source_stage': row.get('source_stage', self.stage), 'state_id': row['state_id']}


def collate_pairs(rows):
    return {'preferred': collate_events([row['preferred'] for row in rows]),
            'rejected': collate_events([row['rejected'] for row in rows]),
            **{key: torch.tensor([row[key] for row in rows], dtype=torch.float32)
               for key in ('pair_weight', 'selection_count', 'diversity_weight')},
            'source_stage': [row['source_stage'] for row in rows],
            'state_id': [row['state_id'] for row in rows]}


class ReplayPairs(Dataset):
    """Shuffle all new pairs and a configured proportion of prior-stage pairs per epoch."""
    def __init__(self, current, previous, fraction=0.25, seed=42):
        if not 0 <= fraction < 1:
            raise ValueError("Replay fraction must be in [0, 1).")
        self.current, self.previous, self.fraction, self.seed = current, list(previous), fraction, seed
        self.previous_indices = [(j, i) for j, dataset in enumerate(previous) for i in range(len(dataset))]
        if fraction and not self.previous_indices:
            raise ValueError("Supply previous-stage pairs when replay_fraction is positive.")
        self.set_epoch(0)

    def set_epoch(self, epoch):
        rng = random.Random(self.seed + epoch)
        self.indices = [('new', 0, i) for i in range(len(self.current))]
        count = round(len(self.current) * self.fraction / (1.0 - self.fraction))
        self.indices.extend(('old', *rng.choice(self.previous_indices)) for _ in range(count))
        rng.shuffle(self.indices)
        self.replay_count = count

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        kind, dataset, item = self.indices[index]
        return self.current[item] if kind == 'new' else self.previous[dataset][item]
