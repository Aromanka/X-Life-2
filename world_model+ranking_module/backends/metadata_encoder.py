import torch
from torch import nn


class SemanticMetadataEncoder(nn.Module):
    """Semantic feature alignment."""
    def __init__(self, dimension):
        super().__init__()
        self.projection = nn.LazyLinear(dimension)
        self.backend_id = None
        self.test_backend = False

    def get_extra_state(self): return {'backend_id':self.backend_id, 'test_backend':self.test_backend}
    def set_extra_state(self, state): self.backend_id, self.test_backend = state['backend_id'], state['test_backend']

    def forward(self, records):
        vectors = []
        for record in records:
            if self.backend_id is None: self.backend_id = record['backend_id']
            if record['backend_id'] != self.backend_id: raise ValueError('Metadata encoder artifact differs from checkpoint')
            vector = torch.as_tensor(record['embedding'], device=self.projection.weight.device, dtype=self.projection.weight.dtype)
            if vector.ndim != 1 or not vector.numel() or not torch.isfinite(vector).all(): raise ValueError('Invalid semantic metadata embedding')
            self.test_backend |= record['test_backend']
            vectors.append(vector)
        return self.projection(torch.stack(vectors))
