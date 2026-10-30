from copy import deepcopy
import torch
from torch import nn
from .http import HTTPBackend


class HTTPActionEncoder(nn.Module):
    """Joint explicit and latent action projection."""
    def __init__(self, domain, dimension, backend_id):
        super().__init__()
        self.domain, self.backend_id = domain, backend_id
        self.factor_schema, self.latent_dimension = None, None
        self.projection = nn.Sequential(nn.LazyLinear(dimension), nn.GELU(), nn.Linear(dimension, dimension))
        self.test_backend = False

    def get_extra_state(self): return dict(factor_schema=self.factor_schema, latent_dimension=self.latent_dimension, backend_id=self.backend_id)

    def set_extra_state(self, state):
        if state['backend_id'] != self.backend_id: raise ValueError('Action encoder artifact differs from checkpoint')
        self.factor_schema, self.latent_dimension = deepcopy(state['factor_schema']), state['latent_dimension']

    def forward(self, texts, images=None):
        if images is None: images = [[] for _ in texts]
        if len(images) != len(texts): raise ValueError('One image list per action is required')
        vectors = []
        for text, pictures in zip(texts, images):
            if not text.strip() and not pictures:
                raise ValueError(f'{self.domain}: active action requires text or images')
            envelope = HTTPBackend().call('action/factors', dict(domain=self.domain, text=text, images=pictures))
            result = envelope['result']
            if result.get('backend_id') != self.backend_id: raise ValueError('Action encoder artifact differs from checkpoint')
            schema = result.get('factor_schema')
            if not isinstance(schema, list) or not schema or any(not isinstance(s, dict) or not isinstance(s.get('name'), str) or not s['name'] or not isinstance(s.get('unit'), str) for s in schema): raise ValueError('Action service requires an ordered factor_schema with names, units and scales')
            if len({s['name'] for s in schema}) != len(schema): raise ValueError('Duplicate action factor names')
            factors = self.projection[-1].weight.new_tensor(result['factors'])
            scales = factors.new_tensor([s['scale'] for s in schema])
            latent = factors.new_tensor(result['latent'])
            if factors.shape != (len(schema),) or not torch.isfinite(factors).all() or (factors < 0).any(): raise ValueError('Invalid physiological action factors')
            if scales.shape != factors.shape or not torch.isfinite(scales).all() or (scales <= 0).any(): raise ValueError('Invalid action factor scales')
            if latent.ndim != 1 or not latent.numel() or not torch.isfinite(latent).all(): raise ValueError('Invalid latent effector')
            if self.factor_schema is not None and (schema != self.factor_schema or latent.numel() != self.latent_dimension): raise ValueError('Action representation schema differs from checkpoint')
            self.factor_schema, self.latent_dimension = deepcopy(schema), latent.numel()
            self.test_backend |= envelope['provenance']['test_backend']
            vectors.append(self.projection(torch.cat([factors / scales, latent])))
        return torch.stack(vectors)
