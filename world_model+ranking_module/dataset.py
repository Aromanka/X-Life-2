import json
import math
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime
from pathlib import Path


def numeric_value(value):
    try: value = float(value)
    except (TypeError, ValueError): return float('nan')
    return value if math.isfinite(value) else float('nan')


def event_timestamp(date, time):
    if not time: return None
    try: return datetime.fromisoformat(str(time) if 'T' in str(time) or ' ' in str(time) else f'{date}T{time}')
    except ValueError: return None


def cgm_window(points, origin, future=False):
    values, mask = torch.zeros(36), torch.zeros(36, dtype=torch.bool)
    for point in points or []:
        if not isinstance(point, (list, tuple)) or len(point) != 2: continue
        timestamp, value = event_timestamp('', point[0]), numeric_value(point[1])
        if timestamp is None or not math.isfinite(value): continue
        minutes = (timestamp - origin).total_seconds() / 60
        slot = math.ceil(minutes / 5) - 1 if future else math.floor(minutes / 5) + 36
        if (0 < minutes <= 180 if future else -180 <= minutes < 0) and 0 <= slot < 36:
            values[slot], mask[slot] = value, True
    return values, mask


class XLifeCleanedDataset(Dataset):
    """Numeric action-response records."""
    def __init__(self, path, max_subjects=None, max_samples=None):
        with open(path, encoding='utf-8') as handle: patients = json.load(handle)
        self.root = Path(path).resolve().parent
        self.samples, self.skipped = [], 0
        for patient in patients[:max_subjects]:
            for row in patient.get('meals', []): self.add_record(patient, row)
            for row in patient.get('exercise_events', []): self.add_record(patient, row, True)
        if max_samples:
            groups = [[s for s in self.samples if s['mode'] == mode] for mode in ('diet', 'exercise')]
            self.samples = [g[i] for i in range(max(map(len, groups))) for g in groups if i < len(g)][:max_samples]

    def add_record(self, patient, row, standalone=False):
        food = row.get('food_data') or {}
        details = food.get('data') or {}
        exercise = row if standalone else row.get('exercise_data_after_meal') or {}
        is_exercise = 'cgm_pre_exercise' in row
        origin = event_timestamp(exercise.get('date', ''), exercise.get('start_time') or exercise.get('recorded_at')) if is_exercise else event_timestamp(food.get('date', ''), details.get('time'))
        if origin is None: self.skipped += 1; return
        prefix = ('cgm_pre_exercise', 'cgm_post_exercise') if is_exercise else ('cgm_preprandial', 'cgm_postprandial')
        history, history_mask = cgm_window(row.get(prefix[0]), origin)
        target, target_mask = cgm_window(row.get(prefix[1]), origin, True)
        if not history_mask.any() or not target_mask.any(): self.skipped += 1; return
        diet = []
        for key in ('carbs', 'protein', 'fat', 'fiber', 'mass', 'calories'):
            totals = []
            for item in details.get('food_items') or []:
                total = item.get('total_nutrition') or {}
                value = numeric_value(total.get(key))
                if not math.isfinite(value):
                    parts = [numeric_value(p.get(key)) for p in item.get('nutrition') or []]
                    value = sum(parts) if parts and all(math.isfinite(p) for p in parts) else float('nan')
                totals.append(value)
            diet.append(sum(totals) if totals and all(math.isfinite(v) for v in totals) else float('nan'))
        exer = [numeric_value(exercise.get('duration_min', exercise.get('duration'))), numeric_value(exercise.get('energy_cal', exercise.get('calories'))), numeric_value(exercise.get('intensity'))]
        metadata = patient.get('metadata') or {}
        meta = []
        for key, length in (('physical', 7), ('blood', 77)):
            source = metadata.get(key) or []
            meta.extend([numeric_value(source[i]) if i < len(source) else float('nan') for i in range(length)])
        meal_time = event_timestamp(food.get('date', ''), details.get('time'))
        exercise_time = event_timestamp(exercise.get('date', ''), exercise.get('start_time') or exercise.get('recorded_at'))
        joint = bool(is_exercise and meal_time and exercise_time and meal_time <= origin and any(math.isfinite(v) for v in diet))
        if not any(math.isfinite(v) for v in (exer if is_exercise else diet)): self.skipped += 1; return
        self.samples.append(dict(subject_id=str(patient['subject_id']), origin=origin.isoformat(), mode='exercise' if is_exercise else 'diet', joint=joint, diet=torch.tensor(diet), exercise=torch.tensor(exer), metadata=torch.tensor(meta), exercise_type=str(exercise.get('type') or ''), onset=torch.tensor([(origin-meal_time).total_seconds()/60 if meal_time else 0., (origin-exercise_time).total_seconds()/60 if exercise_time else 0.]), history=history, history_mask=history_mask, target=target, target_mask=target_mask, diet_text=json.dumps(details, ensure_ascii=False), exercise_text=json.dumps(exercise, ensure_ascii=False)))

        images = details.get('images', [])
        if not isinstance(images, list) or not all(isinstance(p, str) for p in images): raise ValueError('Meal images must be a list of paths')
        self.samples[-1]['diet_images'] = [str((self.root/p).resolve()) for p in images]
        self.samples[-1]['diet_text'] = json.dumps(details, ensure_ascii=False) if details else ''
        self.samples[-1]['exercise_text'] = json.dumps(exercise, ensure_ascii=False) if exercise else ''

    def __len__(self): return len(self.samples)
    def __getitem__(self, index): return self.samples[index]


class SampleNumericDataset(XLifeCleanedDataset):
    """Legacy patient sample adapter."""
    pass


