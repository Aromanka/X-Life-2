"""Abstract request for external prescription safety assessment."""
def assess(candidate, context, simulation, constraints, knowledge, backend):
    response=backend.call('safety/evaluate', dict(candidate=candidate,context=context,
        simulation=simulation,constraints=constraints,knowledge=knowledge))
    inputs = response['result'].get('inputs', {})
    if response['result'].get('is_safe') and (not isinstance(inputs,dict) or inputs.get('schema_version') != 1 or not isinstance(inputs.get('facts'), list)):
        raise ValueError('Safety approval requires normalized safety input schema v1')
    return dict(response['result'],provenance=response['provenance'])
