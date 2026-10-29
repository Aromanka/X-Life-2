"""Deterministic vetoes and an independent HTTP semantic review."""
import math
from .rules import SafeguardRules


def migrated_checks(candidate, constraints):
    """Adapt canonical factors to existing rules without guessing clinical array meanings."""
    rules=SafeguardRules();checks=[];risks=[]
    profile=dict(constraints.get('profile',{}))
    profile.setdefault('medical_conditions',constraints.get('conditions',[]))
    for domain in ('diet','exercise'):
        if candidate['mode'] not in (domain,'joint'): continue
        if domain=='diet':
            diet=candidate['diet']
            plan={'total_calories':diet[5], 'items':[{'food':str(x)} for x in candidate.get('ingredients',[])],
                  'macro_nutrients':{'protein_ratio':diet[1]*4/diet[5],'fat_ratio':diet[2]*9/diet[5]}}
            c,r=rules._check_diet_safety(plan,profile)
        else:
            plan={'total_duration_minutes':candidate['exercise'][0],
                  'sessions':{'event':{'exercises':[{'name':candidate['exercise_type'],
                               'exercise_type':candidate['exercise_type'],'instructions':candidate.get('description','')}]}}}
            # The old daily-duration rule needs an explicit fitness category; do not invent one.
            c,r=rules._check_exercise_safety(plan,profile) if 'fitness_level' in profile else ([],[])
        checks.extend(c);risks.extend(r)
        c,r=rules._check_condition_restrictions(plan,domain,profile);checks.extend(c);risks.extend(r)
        c,r=rules._check_environment_safety(plan,domain,constraints.get('environment',{}));checks.extend(c);risks.extend(r)
    return [c.model_dump(mode='json') for c in checks],[r.model_dump(mode='json') for r in risks]


def hard_checks(candidate, simulation, constraints):
    failures = []
    mode = candidate['mode']
    if not all(math.isfinite(v) for v in simulation['predicted_cgm']): failures.append('nonfinite_simulation')
    ingredients = candidate.get('ingredients', [])
    if not isinstance(ingredients,list): failures.append('invalid_ingredients')
    allergies = constraints.get('allergies', [])
    if allergies:
        if not ingredients: failures.append('allergy_ingredients_missing')
        elif any(str(a).casefold() in ' '.join(map(str,ingredients)).casefold() for a in allergies): failures.append('allergy_veto')
    for active, index, key in (('diet',0,'max_carbs_g'),('diet',5,'max_energy_kcal'),('exercise',0,'max_duration_min')):
        if mode in (active,'joint') and key in constraints:
            if candidate[active][index] > float(constraints[key]): failures.append(key)
    if mode in ('exercise','joint') and candidate['exercise_type'] in constraints.get('forbidden_exercise_types',[]):
        failures.append('forbidden_exercise_type')
    if mode == 'joint' and 'min_exercise_delay_min' in constraints:
        delay = candidate['onset'][0]-candidate['onset'][1]
        if delay < constraints['min_exercise_delay_min']: failures.append('joint_exercise_timing')
    bounds = constraints.get('predicted_glucose_bounds')
    if bounds:
        if bounds.get('unit') != constraints.get('glucose_unit') or bounds.get('unit') not in ('mg/dL','mmol/L'):
            failures.append('glucose_unit_unknown')
        else:
            values = simulation['predicted_cgm']
            if min(values) < bounds['min'] or max(values) > bounds['max']: failures.append('predicted_glucose_bounds')
    return failures


def assess(candidate, context, simulation, constraints, knowledge, backend):
    try:
        rule_checks,rule_risks=migrated_checks(candidate,constraints)
        vetoes = hard_checks(candidate, simulation, constraints)
        vetoes += [c['check_name'] for c in rule_checks if not c['passed']]
        vetoes += [r['factor'] for r in rule_risks]
    except Exception as error:
        return dict(is_safe=False,hard_vetoes=['rule_evaluation_failed'],error=str(error),
                    provenance={'role':'safety','test_backend':False,'status':'failed'})
    try:
        envelope = backend.call('safety', dict(candidate=candidate, context=context, simulation=simulation,
                               constraints=constraints, knowledge=knowledge, hard_vetoes=vetoes))
        review = envelope['result']
        if not isinstance(review,dict) or type(review.get('is_safe')) is not bool or not isinstance(review.get('reasons'),list) or not all(isinstance(r,str) for r in review['reasons']):
            raise ValueError('Malformed semantic review')
        checks = review.get('checks')
        if not isinstance(checks,list) or not checks or any(not isinstance(c,dict) or not isinstance(c.get('name'),str) or type(c.get('passed')) is not bool for c in checks):
            raise ValueError('Semantic review requires explicit checks')
        passed = review['is_safe'] and all(c['passed'] for c in checks)
        return dict(is_safe=not vetoes and passed, hard_vetoes=vetoes, rule_checks=rule_checks,rule_risks=rule_risks,
                    semantic=review, provenance=envelope['provenance'])
    except Exception as error:
        return dict(is_safe=False, hard_vetoes=vetoes, semantic=None, error=str(error),
                    provenance={'role':'safety','test_backend':False,'status':'failed'})
