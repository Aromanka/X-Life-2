"""Abstract request for external prescription preparation."""
def prepare(candidate, backend):
    return backend.call('candidate/prepare', {'candidate':candidate})
