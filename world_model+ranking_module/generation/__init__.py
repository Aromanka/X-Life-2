"""Quantitative action validation shared by generation, simulation and safety."""
import math

UNITS = {'diet':['g','g','g','g','g','kcal'], 'exercise':['min','kcal','scale'], 'onset':'min'}
ACTION_KEYS = ('mode','diet','exercise','exercise_type','onset','diet_text','exercise_text')


def validate_candidate(value):
    if not isinstance(value, dict): raise ValueError('Candidate must be an object')
    mode = value.get('mode')
    if mode not in ('diet','exercise','joint'): raise ValueError('Unknown candidate mode')
    if not isinstance(value.get('description'),str) or not value['description'].strip(): raise ValueError('Candidate description required')
    if not isinstance(value.get('ingredients',[]),list) or not all(isinstance(v,str) for v in value.get('ingredients',[])):
        raise ValueError('Ingredients must be a list of strings')
    if not isinstance(value.get('exercise_type',''),str): raise ValueError('Exercise type must be a string')
    if value.get('units') != UNITS: raise ValueError('Missing or ambiguous units')
    action = {'mode':mode}
    for domain, size in (('diet',6),('exercise',3)):
        active = mode in (domain,'joint')
        values = value.get(domain, [None]*size) if active else [None]*size
        if not isinstance(values, list) or len(values) != size: raise ValueError(f'{domain}: invalid factor count')
        for number in values:
            if number is not None and (isinstance(number, bool) or not isinstance(number,(int,float)) or not math.isfinite(number) or number < 0):
                raise ValueError(f'{domain}: factors must be finite nonnegative numbers or null')
        if active:
            required = range(6) if domain == 'diet' else (0,)
            if any(values[i] is None for i in required): raise ValueError(f'{domain}: essential quantitative factors missing')
            if domain == 'diet' and (values[4] <= 0 or values[5] <= 0): raise ValueError('Diet mass and energy must be positive')
            if domain == 'exercise':
                if values[0] <= 0 or not value.get('exercise_type'): raise ValueError('Exercise duration and type required')
                if values[2] is not None and value.get('intensity_scale') not in ('MET','Borg6_20','Borg0_10'):
                    raise ValueError('Explicit exercise intensity scale required')
        action[domain] = values
    onset = value.get('onset')
    if not isinstance(onset,list) or len(onset)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in onset):
        raise ValueError('Two finite onset offsets in minutes required')
    action.update(onset=onset, exercise_type=value.get('exercise_type',''))
    # Numeric checkpoint intensity has no universal scale: preserve explicit scale in audit text.
    if mode in ('exercise','joint') and value.get('intensity_scale') not in ('unknown', None):
        action['exercise'][2] = None
    for key in ('diet_text','exercise_text'):
        text=value.get(key,value['description'])
        if not isinstance(text,str): raise ValueError('Action text must be a string')
        action[key]=text
    return action
