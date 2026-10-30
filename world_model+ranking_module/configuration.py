import json
import os
from copy import deepcopy
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parents[1] / 'example_config.json'
_config = None


def configure(path=None):
    global _config
    source = Path(path or os.environ.get('XLIFE_CONFIG') or CONFIG_PATH).resolve()
    try:
        config = json.loads(source.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ValueError(f'Cannot read configuration {source}: {error}') from error
    if not isinstance(config, dict):
        raise ValueError('Configuration must be a JSON object')
    _config = config
    return deepcopy(config)


def settings(*keys):
    global _config
    if _config is None:
        configure()
    value = _config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError('Missing configuration field: ' + '.'.join(keys))
        value = value[key]
    return deepcopy(value)


def configured(section, key, value):
    return settings(section, key) if value is None else value


def parse_configured(parser, argv, section, names):
    if not any(action.dest == 'config' for action in parser._actions):
        parser.add_argument('--config')
    args = parser.parse_args(argv)
    configure(args.config)
    for name in names:
        if getattr(args, name, None) is None:
            setattr(args, name, settings(section, name))
    return args


def optimizer_settings():
    values = settings('training', 'optimizer')
    values['betas'] = tuple(values['betas'])
    return values
