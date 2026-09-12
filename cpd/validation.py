"""Cheap checks; this module deliberately imports no ML dependencies."""
from pathlib import Path


def positive(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")


def integers(value):
    try:
        values = [int(part) for part in value.split(',')]
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"Expected comma-separated integer positions, got {value!r}.") from exc
    if any(v < 0 for v in values):
        raise ValueError("Attack positions and lengths must be nonnegative.")
    return values


def generation_paths(args):
    return (Path(args.model_root) / args.model / 'model',
            Path(args.model_root) / args.model / 'tokenizer',
            Path(args.dataset_root) / 'allenai/c4/realnewslike/train')


def validate_generation(args):
    for name in ('number_of_experiments', 'batch_size', 'prompt_tokens',
                 'tokens_count', 'watermark_key_length', 'max_scan_examples'):
        positive(name, getattr(args, name))
    for name in ('buffer_tokens', 'truncate_vocab', 'seed'):
        value = getattr(args, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")
    if args.seed >= 2**63:
        raise ValueError('seed must be smaller than 2**63.')
    if args.prompt_tokens + args.tokens_count + args.buffer_tokens > 2048:
        raise ValueError('prompt_tokens + tokens_count + buffer_tokens must be <= 2048.')
    if args.method not in ('gumbel', 'transform', 'kirchenbauer'):
        raise ValueError(f'Unsupported method: {args.method}')
    if args.device not in ('cpu', 'cuda', 'auto'):
        raise ValueError(f'Unsupported device: {args.device}')
    if args.dataset_source not in ('local', 'online'):
        raise ValueError('dataset_source must be local or online.')
    if Path(args.model).is_absolute() or '..' in Path(args.model).parts:
        raise ValueError('model must be a relative model ID under model_root.')
    starts = integers(args.substitution_blocks_start)
    ends = integers(args.substitution_blocks_end)
    insert = integers(args.insertion_blocks_start)
    lengths = integers(args.insertion_blocks_length)
    size = args.tokens_count + args.buffer_tokens
    if len(starts) != len(ends) or any(not 0 <= a <= b <= size for a, b in zip(starts, ends)):
        raise ValueError('Substitution starts/ends must be paired and inside generated tokens.')
    if len(insert) != len(lengths) or insert != sorted(insert) or any(x > size for x in insert):
        raise ValueError('Insertion starts/lengths must be paired, ordered, and within generated tokens.')
    if any(x > size for x in lengths):
        raise ValueError('An insertion block cannot exceed the original generated length.')
    model, tokenizer, dataset = generation_paths(args)
    for label, path in [('model', model), ('tokenizer', tokenizer)]:
        if not path.is_dir():
            raise FileNotFoundError(f'Missing {label} directory: {path}')
    if not (model / 'config.json').is_file():
        raise FileNotFoundError(f'Missing model config: {model / "config.json"}')
    if not any(model.glob('*.safetensors')) and not any(model.glob('pytorch_model*.bin')):
        raise FileNotFoundError(f'Missing model weights in {model}')
    if args.dataset_source == 'local' and not (dataset / 'state.json').is_file():
        raise FileNotFoundError(f'Missing saved Dataset: {dataset}. Use Dataset.save_to_disk; no network fallback.')
    prefix = Path(args.save) if args.save else Path(args.output_dir) / 'smoke'
    suffixes = ('seeds', 'prompt', 'tokens-before-attack', 'attacked-tokens', 'pi')
    for suffix in suffixes:
        path = Path(f'{prefix}-{suffix}.csv')
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}; choose a new output prefix.')
