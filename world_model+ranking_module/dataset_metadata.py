import math
from backends.http import HTTPBackend


def metadata_fields(value):
    """Canonical metadata fields."""
    fields = []
    def visit(item, path):
        if isinstance(item, dict):
            if 'value' in item:
                field = {'name':item.get('name', path), 'value':item['value'], 'unit':item.get('unit', '')}
                fields.append(field)
            else:
                for key in sorted(item): visit(item[key], f'{path}.{key}' if path else key)
        elif isinstance(item, list):
            for i, child in enumerate(item): visit(child, f'{path}[{i}]')
        else: fields.append({'name':path, 'value':item, 'unit':''})
    if isinstance(value, dict) and set(value) == {'fields'}:
        fields = [dict(field) for field in value['fields']]
    else: visit(value or {}, '')
    names = set()
    for field in fields:
        if set(field) != {'name','value','unit'} or not isinstance(field['name'], str) or not field['name'] or field['name'] in names:
            raise ValueError('Metadata requires unique field names, values and units')
        names.add(field['name'])
        if not isinstance(field['unit'], str) or (field['value'] is not None and not isinstance(field['value'], (str, int, float, bool))):
            raise ValueError('Metadata values must be scalar or null')
        if isinstance(field['value'], float) and not math.isfinite(field['value']): field['value'] = None
    return {'fields':sorted(fields, key=lambda field:field['name'])}


def prepare_metadata(value):
    """Semantic metadata envelope."""
    if isinstance(value, dict) and 'embedding' in value:
        if set(value) != {'fields','embedding','backend_id','test_backend'}: raise ValueError('Invalid encoded metadata envelope')
        metadata_fields({'fields':value['fields']})
        embedding = value['embedding']
        if not isinstance(embedding,list) or not embedding or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in embedding):
            raise ValueError('Metadata embedding must be a finite nonempty vector')
        if not isinstance(value['backend_id'],str) or not value['backend_id'] or type(value['test_backend']) is not bool:
            raise ValueError('Metadata encoder identity required')
        return value
    fields = metadata_fields(value)
    response = HTTPBackend().call('metadata/encode', fields)
    return prepare_metadata(dict(fields, embedding=response['result']['embedding'], backend_id=response['result']['backend_id'],
                                 test_backend=response['provenance']['test_backend']))
