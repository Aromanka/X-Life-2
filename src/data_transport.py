"""JSON transport for prepared samples; no dataset policy lives here."""
import math
import torch


def pack(value):
    if isinstance(value, torch.Tensor):
        return {'__tensor__':str(value.dtype).split('.')[-1], 'values':pack(value.tolist())}
    if isinstance(value, dict): return {k:pack(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [pack(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def unpack(value):
    if isinstance(value, dict):
        if set(value) == {'__tensor__', 'values'}:
            dtype = {'float32':torch.float32, 'float64':torch.float64, 'bool':torch.bool, 'int64':torch.int64}[value['__tensor__']]
            def restore(v):
                if isinstance(v, list): return [restore(x) for x in v]
                return float('nan') if v is None else v
            return torch.tensor(restore(value['values']), dtype=dtype)
        return {k:unpack(v) for k,v in value.items()}
    if isinstance(value, list): return [unpack(v) for v in value]
    return value
