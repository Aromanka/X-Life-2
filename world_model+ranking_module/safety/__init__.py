"""Abstract request for external prescription safety assessment."""
def assess(candidate, context, simulation, constraints, knowledge, backend):
    response=backend.call('safety/evaluate', dict(candidate=candidate,context=context,
        simulation=simulation,constraints=constraints,knowledge=knowledge))
    return dict(response['result'],provenance=response['provenance'])
