"""Client adapters for server-owned metadata preparation."""
from backends.http import HTTPBackend


def metadata_fields(value):
    return HTTPBackend().call('metadata/fields', {'value':value})['result']


def prepare_metadata(value):
    return HTTPBackend().call('metadata/prepare', {'value':value})['result']
