"""A small TOML contract shared by generation, detection and segmentation."""
import json
import math
from pathlib import Path
import tomllib

from .generation import build_parser
from .validation import positive, validate_generation


DEFAULTS = {
    'experiment': {'name': 'experiment', 'output_root': '../results', 'seed': 1, 'threads': 1},
    'generation': {
        'model_root': '../models', 'dataset_root': '../data', 'model': 'facebook/opt-1.3b',
        'method': 'gumbel', 'device': 'cpu', 'dataset_source': 'local',
        'number_of_experiments': 1, 'batch_size': 1, 'prompt_tokens': 50,
        'tokens_count': 64, 'buffer_tokens': 20, 'watermark_key_length': 1000,
        'truncate_vocab': 8, 'offset': False, 'max_scan_examples': 100000,
        'substitution_blocks_start': '0', 'substitution_blocks_end': '0',
        'insertion_blocks_start': '0', 'insertion_blocks_length': '0',
    },
    'detection': {'rolling_window_size': 20, 'permutation_count': 999, 'gamma': 0.4},
    'segmentation': {'backend': 'r', 'minimum_length': 20, 'permutation_count': 999, 'block_size': 10,
                     'threshold': 0.01},
}


def load_config(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        supplied = tomllib.load(stream)
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown configuration sections: {sorted(unknown)}')
    config = {}
    for section, defaults in DEFAULTS.items():
        values = supplied.get(section, {})
        if not isinstance(values, dict) or set(values) - set(defaults):
            raise ValueError(f'Unknown keys or invalid table in [{section}]. Allowed: {sorted(defaults)}')
        config[section] = defaults | values
        for key, default in defaults.items():
            value = config[section][key]
            if type(default) is float:
                valid = type(value) in (float, int) and math.isfinite(value)
            else:
                valid = type(value) is type(default)
            if not valid:
                raise ValueError(f'{section}.{key} has wrong type; expected {type(default).__name__}.')
    for section, key in [('experiment', 'output_root'), ('generation', 'model_root'), ('generation', 'dataset_root')]:
        location = Path(config[section][key]).expanduser()
        config[section][key] = str((path.parent/location).resolve())
    name = config['experiment']['name']
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in name):
        raise ValueError('experiment.name must contain only letters, digits, hyphens and underscores.')
    positive('threads', config['experiment']['threads'])
    for section, names in [('detection', ['rolling_window_size', 'permutation_count']),
                           ('segmentation', ['minimum_length', 'permutation_count', 'block_size'])]:
        for key in names:
            positive(f'{section}.{key}', config[section][key])
    if config['detection']['gamma'] < 0:
        raise ValueError('detection.gamma must be nonnegative.')
    if not 0 < config['segmentation']['threshold'] < 1:
        raise ValueError('segmentation.threshold must be between 0 and 1.')
    if config['segmentation']['backend'] not in ('r', 'python'):
        raise ValueError('segmentation.backend must be r (original implementation) or python (experimental adapter).')
    return config


def generation_args(config, prefix):
    args = build_parser().parse_args([])
    for key, value in config['generation'].items():
        setattr(args, key, value)
    args.seed = config['experiment']['seed']
    args.save = str(prefix)
    return args


def preflight(config, stages, prefix):
    args = generation_args(config, prefix)
    if 'generate' in stages:
        validate_generation(args)
    if any(stage in stages for stage in ('detect', 'segment')):
        if args.method not in ('gumbel', 'transform'):
            raise ValueError('The local detection pipeline supports gumbel and transform only.')
        if sum(int(x) for x in args.insertion_blocks_length.split(',')):
            raise ValueError('Pipeline detection/segmentation currently requires zero insertion length; standalone generation supports insertion.')
        window = config['detection']['rolling_window_size']
        if not 0 < window < args.tokens_count:
            raise ValueError('rolling_window_size must be smaller than tokens_count.')
        if 'segment' in stages:
            if window % 2:
                raise ValueError('Segmentation requires an even rolling_window_size.')
            seg = config['segmentation']
            if args.tokens_count - window - 1 < seg['minimum_length']:
                raise ValueError('No seeded interval can satisfy minimum_length at this token count.')
            if seg['block_size'] > seg['minimum_length']//2:
                raise ValueError('block_size must be <= minimum_length // 2.')
            if seg['backend'] == 'r':
                if not 0 <= config['experiment']['seed'] <= 2147483647:
                    raise ValueError('Original R segmentation requires a seed in [0, 2147483647].')
                if seg['block_size'] != 10:
                    raise ValueError('The unchanged original R significance function uses block_size=10.')
                import shutil
                if shutil.which('Rscript') is None:
                    raise ValueError('Original R segmentation requires Rscript on PATH. '
                                     'Generation/detection can run separately. The Python backend is an explicit experimental alternative.')
    model_config = Path(args.model_root)/args.model/'model/config.json'
    try:
        metadata = json.loads(model_config.read_text())
        vocab = metadata['vocab_size']
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f'Cannot read vocabulary size from {model_config}: {exc}') from exc
    positive('model vocab_size', vocab)
    if not 0 <= args.truncate_vocab < vocab:
        raise ValueError('truncate_vocab must be smaller than model vocabulary size.')
    context = metadata.get('max_position_embeddings', metadata.get('n_positions'))
    if context and args.prompt_tokens + args.tokens_count + args.buffer_tokens > context:
        raise ValueError(f'Requested generation exceeds model context length {context}.')
    return vocab
