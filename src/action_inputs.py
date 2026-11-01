import json
import math


def action_inputs(action):
    mode = action['mode']
    if mode not in ('diet', 'exercise', 'joint'): raise ValueError('Unknown action mode')
    images = action.get('diet_images', [])
    if not isinstance(images, list) or not all(isinstance(p, str) and p.strip() for p in images): raise ValueError('Meal images must be nonempty paths')
    result = {'diet_images':images if mode in ('diet', 'joint') else []}
    for domain, units in (('diet', ['g']*5+['kcal']), ('exercise', ['min', 'kcal', 'scale'])):
        text = action.get(domain+'_text') or ''
        if not isinstance(text, str): raise ValueError('Action text must be a string')
        if mode not in (domain, 'joint'):
            result[domain+'_text'] = ''
            continue
        values = action.get(domain, [None]*len(units))
        values = [v if v is not None and math.isfinite(v) else None for v in values]
        exercise_type = action.get('exercise_type', '') if domain == 'exercise' else ''
        if not text.strip() and not (domain == 'diet' and images) and not any(v is not None for v in values) and not exercise_type.strip():
            raise ValueError(f'{domain}: active action requires text, images or structured records')
        fields = ['carbohydrate', 'protein', 'fat', 'fiber', 'mass', 'energy'] if domain == 'diet' else ['duration', 'energy', 'intensity']
        record = {domain:values, 'fields':fields, 'units':units, 'onset_min':action.get('onset'), 'description':text}
        if domain == 'exercise': record.update(exercise_type=exercise_type, intensity_scale=action.get('intensity_scale', 'unknown'))
        result[domain+'_text'] = json.dumps(record, ensure_ascii=False, allow_nan=False)
    return result
