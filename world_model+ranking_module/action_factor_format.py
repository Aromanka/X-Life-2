"""Normalize common VLM JSON formats without guessing missing factors or units."""
import json
import math
import re


DIET = (
    ('carbohydrate_g', 'carbohydrates_g', 'carbohydrate', 'carbohydrates', 'carbs', '碳水化合物'),
    ('protein_g', 'protein', '蛋白质'),
    ('fat_g', 'fat', '脂肪'),
    ('fiber_g', 'fibre_g', 'fiber', 'fibre', '膳食纤维'),
    ('mass_g', 'mass', 'weight_g', 'portion_g', 'total_mass_g', '重量'),
    ('energy_kcal', 'calories', 'calories_kcal', 'energy', '热量'),
)
EXERCISE = (
    ('duration_min', 'duration', 'duration_minutes', '时长'),
    ('energy_kcal', 'calories', 'calories_kcal', 'energy', '热量'),
    ('intensity_MET', 'intensity', 'MET', 'METs'),
)


def key(value):
    return re.sub(r'[\s_()\-]', '', str(value)).lower()


def number(value, unit):
    if isinstance(value, dict):
        supplied_unit = value.get('unit', unit)
        if key(supplied_unit) != key(unit):
            raise ValueError(f'Expected unit {unit}, got {supplied_unit!r}')
        value = value.get('value')
    if isinstance(value, str):
        match = re.fullmatch(r'\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Z]*)\s*', value)
        if not match or (match[2] and key(match[2]) != key(unit)):
            raise ValueError(f'Expected a number in {unit}, got {value!r}')
        value = float(match[1])
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('Action factors must be finite nonnegative numbers')
    return float(value)


def normalize(value, domain, depth=0, schema=None):
    if depth > 8:
        raise ValueError('Action JSON nesting exceeds 8 levels')
    aliases = DIET if domain == 'diet' else EXERCISE
    units = ['g'] * 5 + ['kcal'] if domain == 'diet' else ['min', 'kcal', 'MET']
    if schema is not None:
        known = {names[0]:names for names in aliases}
        aliases = [known.get(spec['name'], (spec['name'],)) for spec in schema]
        units = [spec['unit'] for spec in schema]
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], (list, dict)):
            return normalize(value[0], domain, depth+1, schema)
        if len(value) == len(aliases) and all(not isinstance(v, (dict, list)) for v in value):
            return [number(v, unit) for v, unit in zip(value, units)]
        # Lists of named factors are order independent. Arbitrary lists of food
        # objects are not summed: they may be alternatives or duplicate images.
        if value and all(isinstance(v, dict) for v in value):
            named = {}
            for row in value:
                if 'name' in row and 'value' in row:
                    items = [(row['name'], {'value':row['value'], 'unit':row['unit']} if 'unit' in row else row['value'])]
                elif len(row) == 1:
                    items = list(row.items())
                else:
                    raise ValueError('Ambiguous list of objects; return one aggregate factor vector')
                for name, item in items:
                    name = key(name)
                    if name in named:
                        raise ValueError('Duplicate named factor: ' + name)
                    named[name] = item
            return normalize(named, domain, depth+1, schema)
        raise ValueError(f'Expected {len(aliases)} factors, got list length {len(value)}')
    if isinstance(value, dict):
        if domain == 'exercise' and 'intensity_scale' in value and key(value['intensity_scale']) != 'met':
            raise ValueError('Exercise intensity must use MET')
        indexed = {key(k):v for k,v in value.items()}
        if len(indexed) != len(value):
            raise ValueError('Duplicate normalized factor names')
        for wrapper in ('factors', 'total_nutrition', 'nutrition', 'nutrients', 'result', 'data', 'output'):
            if wrapper in value:
                return normalize(value[wrapper], domain, depth+1, schema)
        if domain == 'diet' and 'food_items' in value:
            # Only an explicit food_items container denotes components of one
            # meal. Every component must provide all factors in known units.
            rows = value['food_items']
            if not isinstance(rows, list) or not rows:
                raise ValueError('food_items must be a nonempty list')
            vectors = [normalize(row, domain, depth+1, schema) for row in rows]
            return [sum(v[i] for v in vectors) for i in range(len(aliases))]
        factors = []
        for names, unit in zip(aliases, units):
            # Several aliases can normalize to the same key (e.g. intensity_MET).
            matches = [indexed[k] for k in dict.fromkeys(key(n) for n in names) if k in indexed]
            if len(matches) != 1:
                raise ValueError(f'Missing or ambiguous factor {names[0]}')
            factors.append(number(matches[0], unit))
        return factors
    raise ValueError(f'Expected factor object or array, got {type(value).__name__}')


def parse_action_factors(answer, domain, schema=None):
    text = answer.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*```$', '', text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        if text.startswith(('[', '{')):
            raise ValueError('Incomplete or invalid JSON; return a complete factor object')
        # Accept prose around one JSON value; never regex-extract numeric values.
        decoder = json.JSONDecoder()
        values, position = [], 0
        while position < len(text):
            if text[position] not in '[{':
                position += 1
                continue
            try:
                value, end = decoder.raw_decode(text, position)
            except json.JSONDecodeError:
                position += 1
                continue
            values.append(value)
            position = end
        if len(values) != 1:
            raise ValueError(f'Expected one complete JSON value; found {len(values)}')
        value = values[0]
    factors = normalize(value, domain, schema=schema)
    # Revalidate sums as well as directly decoded vectors.
    if any(not math.isfinite(v) or v < 0 for v in factors):
        raise ValueError('Action factors must be finite nonnegative numbers')
    return factors
