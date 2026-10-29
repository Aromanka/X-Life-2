import json
import os
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


class BackendError(RuntimeError):
    pass


class HTTPBackend:
    """Role-based POST contract; prompts and model selection belong to the server."""
    def __init__(self, url=None, timeout=180):
        self.url = (url or os.environ.get('XLIFE_LLM_URL', 'http://127.0.0.1:8765')).rstrip('/')
        self.timeout = timeout

    def call(self, role, payload):
        request = Request(self.url + '/v1/' + role, data=json.dumps(payload, allow_nan=False).encode(),
                          headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as error:
            try: detail=json.loads(error.read(4096)).get('error',str(error))
            except (ValueError,AttributeError): detail=str(error)
            raise BackendError(f'{role} service failed ({error.code}): {detail}') from error
        except (URLError, TimeoutError, ValueError) as error:
            raise BackendError(f'{role} service failed: {error}') from error
        if not isinstance(result, dict) or not isinstance(result.get('provenance'), dict) or 'result' not in result:
            raise BackendError(f'{role}: invalid response envelope')
        if type(result['provenance'].get('test_backend')) is not bool:
            raise BackendError(f'{role}: missing backend identity')
        return result
