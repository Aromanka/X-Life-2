import os
import torch
from torch import nn
from .http import HTTPBackend


class HTTPActionEncoder(nn.Module):
    """Frozen remote latent extraction plus a trainable local projection."""
    def __init__(self, domain, dimension, latent_dimension, backend_id):
        super().__init__()
        self.domain,self.latent_dimension,self.backend_id=domain,latent_dimension,backend_id
        self.projection=nn.Linear(latent_dimension,dimension)
        self.test_backend=False

    def forward(self, texts):
        vectors=[]
        for text in texts:
            envelope=HTTPBackend().call('action/latent',dict(domain=self.domain,text=text,dimension=self.latent_dimension))
            result=envelope['result']
            if result.get('backend_id')!=self.backend_id: raise ValueError('Remote action encoder artifact identity differs from checkpoint')
            vector=torch.tensor(result['vector'],device=self.projection.weight.device,dtype=self.projection.weight.dtype)
            if vector.shape!=(self.latent_dimension,) or not torch.isfinite(vector).all(): raise ValueError('Invalid remote latent')
            self.test_backend |= envelope['provenance']['test_backend']
            vectors.append(vector)
        return self.projection(torch.stack(vectors))
