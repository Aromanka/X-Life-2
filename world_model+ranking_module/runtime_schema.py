import json
import math
import torch
from backends.http import HTTPBackend


def read_path(value, path):
    for key in path:
        if isinstance(value, dict): value = value.get(key)
        elif isinstance(value, (list, tuple)) and isinstance(key, int) and 0 <= key < len(value): value = value[key]
        else: return None
    return value


def load_schema(backend=None):
    schema = (backend or HTTPBackend()).call('schema', {})['result']
    if not isinstance(schema, dict) or not {'supervision', 'retrieval'} <= set(schema) or set(schema) - {'supervision', 'retrieval', 'actions'}: raise ValueError('Schema requires supervision and retrieval, with optional actions')
    if not isinstance(schema['supervision'], list) or not isinstance(schema['retrieval'], dict): raise ValueError('Invalid schema sections')
    if set(schema['retrieval']) != {'generation', 'safety'}: raise ValueError('Retrieval schema requires generation and safety')
    names = set()
    for spec in schema['supervision']:
        name = spec['name']
        if not isinstance(name, str) or not name or '.' in name or name in names: raise ValueError('Invalid supervision name')
        names.add(name)
        if spec['loss'] not in ('mse', 'l1') or not math.isfinite(spec['weight']) or spec['weight'] <= 0: raise ValueError('Invalid supervision loss')
        if not spec['shape'] or any(type(n) is not int or n < 1 for n in spec['shape']): raise ValueError('Invalid supervision shape')
        if spec['readout'] not in ('sequence', 'pooled'): raise ValueError('Invalid supervision readout')
        if not isinstance(spec['path'], list) or not spec['path']: raise ValueError('Supervision path must be a nonempty list')
        if spec.get('mask') is not None and not isinstance(spec['mask'], list): raise ValueError('Mask path must be a list')
        if not isinstance(spec['scale'], str) and (not math.isfinite(spec['scale']) or spec['scale'] <= 0): raise ValueError('Invalid supervision scale')
    for spec in schema['retrieval'].values():
        if spec['strategy'] not in ('vector', 'exact', 'substring') or type(spec['limit']) is not int or spec['limit'] < 1: raise ValueError('Invalid retrieval strategy')
        if not isinstance(spec['paths'], list) or any(not isinstance(p, list) for p in spec['paths']): raise ValueError('Retrieval paths must be lists')
        if spec['strategy'] == 'vector' and (type(spec['dimension']) is not int or spec['dimension'] < 1): raise ValueError('Invalid embedding dimension')
    return schema


def supervision_values(schema, sources):
    result = {}
    for spec in schema['supervision']:
        value = read_path(sources, spec['path'])
        mask = read_path(sources, spec['mask']) if spec.get('mask') else None
        test_backend = False
        if value is not None and spec.get('encode'):
            encoder = spec['encode']
            response = HTTPBackend().call(encoder['role'], {encoder['input']:value})
            value, test_backend = response['result'][encoder['output']], response['provenance']['test_backend']
        if value is None: target = torch.full(spec['shape'], float('nan'))
        else:
            def numbers(v):
                if isinstance(v, (list, tuple)): return [numbers(x) for x in v]
                return float('nan') if v is None else v
            target = torch.as_tensor(numbers(value), dtype=torch.float32)
        if list(target.shape) != spec['shape']: raise ValueError('Supervision shape differs from schema: '+spec['name'])
        valid = torch.isfinite(target)
        if mask is not None:
            mask = torch.as_tensor(mask)
            if not ((mask == 0) | (mask == 1)).all(): raise ValueError('Invalid observation mask')
            while mask.ndim < target.ndim: mask = mask.unsqueeze(-1)
            valid &= torch.broadcast_to(mask.bool(), target.shape)
        result[spec['name']] = {'target':torch.nan_to_num(target), 'mask':valid, 'test_backend':test_backend}
    return result


def supervision_loss(model, rows, output, mode):
    losses = []
    for spec in model.schema['supervision']:
        name = spec['name']
        target = torch.stack([r[name]['target'] for r in rows]).to(output['trajectory'].device)
        mask = torch.stack([r[name]['mask'] for r in rows]).to(target.device)
        if not mask.any(): continue
        if spec.get('prediction'): prediction = output[spec['prediction']]
        else:
            trajectory = output['trajectory']
            if spec['readout'] == 'pooled': trajectory = trajectory.mean(1)
            elif trajectory.size(1) != spec['shape'][0]: raise ValueError('Sequence target length differs from trajectory')
            prediction = model.supervision_heads[name][mode](trajectory).reshape_as(target)
        if prediction.shape != target.shape: raise ValueError('Prediction shape differs from supervision')
        scale = getattr(model, spec['scale']) if isinstance(spec['scale'], str) else spec['scale']
        error = (prediction[mask] - target[mask]) / scale
        losses.append(spec['weight'] * (error.square() if spec['loss'] == 'mse' else error.abs()).mean())
    if not losses: raise ValueError('No observed supervision targets')
    loss = torch.stack(losses).sum()
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
    return loss


def retrieve(graph, spec, sources, backend):
    values = [read_path(sources, path) for path in spec['paths']]
    values = [v for v in values if v is not None]
    if not values or not graph.triples: return [], []
    if spec['strategy'] == 'vector':
        query = json.dumps(values, ensure_ascii=False, sort_keys=True, allow_nan=False)
        return graph.retrieve_vector(query, backend, spec['dimension'], spec['limit'])
    terms = []
    def collect(value):
        if isinstance(value, dict):
            for child in value.values(): collect(child)
        elif isinstance(value, list):
            for child in value: collect(child)
        elif value is not None: terms.append(str(value))
    for value in values: collect(value)
    return graph.retrieve(terms, 'safety' if spec['strategy'] == 'exact' else 'generation', spec['limit']), []
