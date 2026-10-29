import torch
from torch import nn
from torch.nn import functional as F


def fill_history(values, mask):
    """Forward-fill within measured history only, including its initial gap."""
    if not mask.any(1).all(): raise ValueError('Empty baseline history')
    positions=torch.arange(values.size(1),device=values.device).expand_as(values)
    first=positions.masked_fill(~mask,values.size(1)).min(1).values
    indices=positions.masked_fill(~mask,-1).cummax(1).values
    indices=torch.where(indices<0,first[:,None],indices)
    return values.gather(1,indices)


def persistence(values,mask,horizon=36):
    return fill_history(values,mask)[:,-1:].expand(-1,horizon)


def linear_history(values,mask,horizon=36):
    x=torch.arange(values.size(1),device=values.device,dtype=values.dtype).expand_as(values)
    weights=mask.to(values.dtype); n=weights.sum(1,keepdim=True).clamp_min(1)
    mx=(x*weights).sum(1,keepdim=True)/n
    my=(values*weights).sum(1,keepdim=True)/n
    slope=((x-mx)*(values-my)*weights).sum(1,keepdim=True)/((x-mx).square()*weights).sum(1,keepdim=True).clamp_min(1e-8)
    future=torch.arange(values.size(1),values.size(1)+horizon,device=values.device)
    return my+slope*(future[None]-mx)


class DLinear(nn.Module):
    """Independent standard moving-average decomposition, scalar CGM implementation.

    No external source copied. Action correction is an explicit optional ablation.
    """
    def __init__(self,history=36,horizon=36,kernel=7,action_correction=False):
        super().__init__()
        if kernel%2!=1: raise ValueError('Use an odd moving-average kernel')
        self.kernel,self.action_correction=kernel,action_correction
        self.trend,self.residual=nn.Linear(history,horizon),nn.Linear(history,horizon)
        self.register_buffer('mean',torch.tensor(0.));self.register_buffer('scale',torch.tensor(1.))
        if action_correction:
            self.action=nn.Linear(18,horizon)
            self.register_buffer('action_mean',torch.zeros(9));self.register_buffer('action_scale',torch.ones(9))

    def fit_statistics(self,rows):
        values=torch.cat([r['history'][r['history_mask']] for r in rows])
        self.mean.copy_(values.mean());self.scale.copy_(values.std(unbiased=False).clamp_min(1e-3))
        if self.action_correction:
            actions=torch.stack([torch.cat([r['diet'] if r['mode']=='diet' else torch.full_like(r['diet'],float('nan')),
                r['exercise'] if r['mode']=='exercise' else torch.full_like(r['exercise'],float('nan'))]) for r in rows])
            mask=torch.isfinite(actions);count=mask.sum(0).clamp_min(1)
            mean=actions.masked_fill(~mask,0).sum(0)/count
            scale=((actions-mean).square().masked_fill(~mask,0).sum(0)/count).sqrt().clamp_min(1e-3)
            self.action_mean.copy_(mean);self.action_scale.copy_(scale)

    def forward(self,batch):
        values=(fill_history(batch['history'],batch['history_mask'])-self.mean)/self.scale
        trend=F.avg_pool1d(F.pad(values[:,None],(self.kernel//2,)*2,mode='replicate'),self.kernel,stride=1)[:,0]
        result=self.trend(trend)+self.residual(values-trend)
        if self.action_correction:
            action=torch.cat([batch['diet'],batch['exercise']],-1)
            if 'modes' not in batch: raise ValueError('Action correction requires explicit domain routing')
            for i,mode in enumerate(batch['modes']):
                if mode=='diet': action[i,6:]=float('nan')
                elif mode=='exercise': action[i,:6]=float('nan')
                else: raise ValueError('This baseline only supports single-domain action correction')
            valid=torch.isfinite(action)
            result=result+self.action(torch.cat([torch.where(valid,(action-self.action_mean)/self.action_scale,0),valid.to(action.dtype)],-1))
        return result*self.scale+self.mean
