"""Private configuration, resolved independently of the working directory."""
import json
import os
from pathlib import Path

_config = None
_config_source = None


def load_config(config_path):
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError('Create kg_agents/config.json from config.example.json, or set XLIFE_CONFIG.')
    return json.loads(path.read_text(encoding='utf-8'))


def get_config(config_path=None):
    global _config, _config_source
    path = Path(config_path or os.environ.get('XLIFE_CONFIG') or Path(__file__).with_name('config.json')).expanduser().resolve()
    if _config is None or _config_source != path:
        _config, _config_source = load_config(path), path
    return _config


def get_neo4j_config():
    config = dict(get_config()['neo4j'])
    config['password'] = os.environ.get('XLIFE_NEO4J_PASSWORD', config.get('password', ''))
    return config


def get_deepseek_config():
    # OpenAI-compatible local DeepSeek-R1 endpoint.
    source = get_config()
    config = dict(source.get('api_model', source.get('deepseek', {})))
    config['api_key'] = os.environ.get('XLIFE_LLM_API_KEY', config.get('api_key', 'local'))
    return config


NEO4J_URI = lambda: get_neo4j_config()['uri']
NEO4J_AUTH = lambda: (get_neo4j_config()['username'], get_neo4j_config()['password'])
DEEPSEEK_API_KEY = lambda: get_deepseek_config()['api_key']
DEEPSEEK_BASE_URL = lambda: get_deepseek_config()['base_url']
DEEPSEEK_MODEL = lambda: get_deepseek_config().get('model', 'DeepSeek-R1')
