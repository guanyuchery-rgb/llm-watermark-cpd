"""Explicit input preparation; model downloads and local flow-test data are separate commands."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cpd.config import load_config

MODEL = 'facebook/opt-1.3b'


def model_inputs(config, revision, cache_dir):
    """Download only the pinned PyTorch checkpoint; do not load weights or run inference."""
    if config['generation']['model'] != MODEL:
        raise ValueError(f'This first-deployment helper only prepares {MODEL}.')
    if not re.fullmatch(r'[0-9a-fA-F]{40}', revision):
        raise ValueError('Use a full 40-character model commit SHA, not main or a short revision.')
    target = Path(config['generation']['model_root'])/MODEL
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite {target}; use a new model root.')
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    patterns = ['config.json', 'generation_config.json', 'pytorch_model.bin',
                'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt',
                'LICENSE.md', 'README.md']
    print(f'Explicit download: {MODEL}@{revision}; cache={cache_dir}', flush=True)
    snapshot = Path(snapshot_download(repo_id=MODEL, revision=revision, cache_dir=str(cache_dir),
                                      allow_patterns=patterns))
    for name in ['config.json', 'pytorch_model.bin', 'vocab.json', 'merges.txt', 'tokenizer_config.json']:
        if not (snapshot/name).is_file():
            raise FileNotFoundError(f'Pinned snapshot is missing {name}: {snapshot}')
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    # Copy the existing from_pretrained-compatible checkpoint; no dtype conversion.
    target.mkdir(parents=True, exist_ok=False)
    (target/'model').mkdir()
    for name in ['config.json', 'generation_config.json', 'pytorch_model.bin']:
        if (snapshot/name).is_file():
            shutil.copy2(snapshot/name, target/'model'/name)
    tokenizer.save_pretrained(target/'tokenizer')
    for name in ['LICENSE.md', 'README.md']:
        if (snapshot/name).is_file():
            shutil.copy2(snapshot/name, target/name)
    (target/'preparation.json').write_text(json.dumps({
        'model': MODEL, 'revision': revision, 'source': f'https://huggingface.co/{MODEL}',
        'format': 'Unmodified PyTorch checkpoint copied; tokenizer.save_pretrained',
    }, indent=2)+'\n')
    print(f'Prepared {target}; no inference was run.')


def synthetic_dataset(config):
    """Three locally authored records, clearly labelled as flow-test data, not C4."""
    gen = config['generation']
    destination = Path(gen['dataset_root'])/'allenai/c4/realnewslike/train'
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}; select a new dataset root.')
    tokenizer_path = Path(gen['model_root'])/gen['model']/'tokenizer'
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f'Missing local tokenizer: {tokenizer_path}')
    from transformers import AutoTokenizer
    from datasets import Dataset
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    paragraphs = [
        'The research team keeps a written record of each experiment. They compare the inputs, '
        'software versions, and output files before interpreting any differences in the results.',
        'A small library opens every morning beside the river. Readers discuss books, share notes, '
        'and carefully return each volume to its original place at the end of the day.',
        'An engineer measures the temperature of a room at regular intervals. The observations are '
        'saved with timestamps so another person can repeat the analysis and check the conclusions.',
    ]
    texts = [' '.join([paragraph]*24) for paragraph in paragraphs]
    encoded = [tokenizer.encode(text, truncation=True, max_length=2048-gen['buffer_tokens']) for text in texts]
    required = gen['prompt_tokens']+gen['tokens_count']
    if gen['number_of_experiments'] > len(texts) or any(len(ids) < required for ids in encoded):
        raise ValueError(f'Three synthetic records cannot satisfy this configuration; required length={required}.')
    # Reserve the target before saving, so reruns cannot silently replace data.
    destination.mkdir(parents=True, exist_ok=False)
    Dataset.from_dict({'text': texts}).save_to_disk(str(destination))
    manifest = {
        'purpose': 'synthetic flow test ONLY; not C4 and not paper replication data',
        'source': 'Locally authored paragraphs in scripts/prepare_autodl.py; repeated 24 times',
        'text_sha256': [hashlib.sha256(t.encode()).hexdigest() for t in texts],
        'encoded_lengths': [len(ids) for ids in encoded],
        'first_token_ids': [ids[0] for ids in encoded],
        'tokenizer': str(tokenizer_path), 'tokenizer_vocab_size': len(tokenizer),
        'max_token_id': max(tokenizer.get_vocab().values()),
        'bos_token_id': tokenizer.bos_token_id, 'eos_token_id': tokenizer.eos_token_id,
        'pad_token_id': tokenizer.pad_token_id,
    }
    (destination/'preparation.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))
    print(f'Saved synthetic flow-test Dataset: {destination}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    download = commands.add_parser('model', help='Explicit NETWORK download of pinned OPT weights, only when invoked')
    download.add_argument('--config', type=Path, required=True)
    download.add_argument('--revision', required=True, help='Full 40-character Hugging Face commit SHA')
    download.add_argument('--cache-dir', type=Path, required=True)
    dataset = commands.add_parser('synthetic-data', help='OFFLINE artificial data for flow testing, not C4')
    dataset.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == 'model':
            model_inputs(config, args.revision, args.cache_dir)
        else:
            synthetic_dataset(config)
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(2, f'Preparation failed: {exc}\n')


if __name__ == '__main__':
    main()
