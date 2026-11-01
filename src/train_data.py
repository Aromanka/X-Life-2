"""Event and preference-pair adapters for X-Life training."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset
from configuration import configured

def read_records(path):
    path = Path(path)
    if path.suffix.lower() == '.jsonl':
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    records = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(records, list):
        raise ValueError("Training JSON must be a list of events or patient objects.")
    return records


class ReplayPairs(Dataset):
    """Shuffle all new pairs and a configured proportion of prior-stage pairs per epoch."""
    def __init__(self, current, previous, fraction=None, seed=None):
        fraction = configured('alignment', 'replay_fraction', fraction)
        seed = configured('training', 'seed', seed)
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
