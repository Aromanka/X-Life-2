import torch
from torch import nn
from .http import HTTPBackend


class HTTPActionEncoder(nn.Module):
    """Frozen VLM factors and action projection."""
    def __init__(self, domain, dimension, backend_id):
        super().__init__()
        self.domain, self.backend_id = domain, backend_id
        scales = [100., 50., 50., 20., 500., 1000.] if domain == 'diet' else [60., 500., 10.]
        self.register_buffer('scale', torch.tensor(scales))
        self.projection = nn.Sequential(nn.Linear(len(scales), dimension), nn.GELU(), nn.Linear(dimension, dimension))
        self.test_backend = False

    def forward(self, texts, images=None):
        if images is None: images = [[] for _ in texts]
        if len(images) != len(texts): raise ValueError('One image list per action is required')
        vectors = []
        for text, pictures in zip(texts, images):
            if not text.strip() and not pictures:
                vectors.append(self.scale.new_zeros(self.projection[-1].out_features)); continue
            envelope = HTTPBackend().call('action/factors', dict(domain=self.domain, text=text, images=pictures))
            result = envelope['result']
            if result.get('backend_id') != self.backend_id: raise ValueError('Action encoder artifact differs from checkpoint')
            factors = self.scale.new_tensor(result['factors'])
            if factors.shape != self.scale.shape or not torch.isfinite(factors).all() or (factors < 0).any(): raise ValueError('Invalid physiological action factors')
            if self.domain == 'exercise' and result.get('intensity_scale') != 'MET': raise ValueError('Exercise factors require MET intensity')
            self.test_backend |= envelope['provenance']['test_backend']
            vectors.append(self.projection(factors / self.scale))
        return torch.stack(vectors)
