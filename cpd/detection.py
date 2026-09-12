"""Detection adapters. Statistics are retained from the original 3-detect.py."""
from pathlib import Path
import csv
import numpy as np
import torch

from watermarking.detection import sliding_permutation_test, phi
from watermarking.gumbel.key import gumbel_key_func
from watermarking.gumbel.score import gumbel_score, gumbel_edit_score
from watermarking.transform.key import transform_key_func
from watermarking.transform.score import transform_score, transform_edit_score

def make_statistics(method, gamma):
    if method == "transform":
        test_stats = []
        def dist1(x, y): return transform_edit_score(x, y, gamma=gamma)

        def test_stat1(
            tokens, watermark_key_length, rolling_window_size,
            generator, vocab_size, null=False
        ):
            return phi(
                tokens, watermark_key_length, rolling_window_size, generator,
                vocab_size, transform_key_func, dist1, null=False, normalize=True
            )
        test_stats.append(test_stat1)
        def dist2(x, y): return transform_score(x, y)

        def test_stat2(
            tokens, watermark_key_length, rolling_window_size,
            generator, vocab_size, null=False
        ):
            return phi(
                tokens, watermark_key_length, rolling_window_size, generator,
                vocab_size, transform_key_func, dist2, null=False, normalize=True
            )
        test_stats.append(test_stat2)

    elif method == "gumbel":
        test_stats = []
        def dist1(x, y): return gumbel_edit_score(x, y, gamma=gamma)

        def test_stat1(
            tokens, watermark_key_length, rolling_window_size,
            generator, vocab_size, null=False
        ):
            return phi(
                tokens, watermark_key_length, rolling_window_size, generator,
                vocab_size, gumbel_key_func, dist1, null=null, normalize=False
            )
        test_stats.append(test_stat1)
        def dist2(x, y): return gumbel_score(x, y)

        def test_stat2(
            tokens, watermark_key_length, rolling_window_size,
            generator, vocab_size, null=False
        ):
            return phi(
                tokens, watermark_key_length, rolling_window_size, generator,
                vocab_size, gumbel_key_func, dist2, null=null, normalize=False
            )
        test_stats.append(test_stat2)
    else:
        raise ValueError(f"Unsupported detection method: {method}")

    return test_stats


def read_tokens(prefix, vocab_size):
    def read(suffix):
        path = Path(f'{prefix}-{suffix}.csv')
        try:
            with path.open(newline='') as stream:
                rows = [[int(value) for value in row] for row in csv.reader(stream)]
        except (OSError, ValueError) as exc:
            raise ValueError(f'Cannot read integer CSV {path}: {exc}') from exc
        if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError(f'Empty or ragged CSV: {path}')
        return np.array(rows, dtype=np.int64)
    seeds = read('seeds')
    tokens = read('attacked-tokens')
    if seeds.shape != (1, tokens.shape[0]):
        raise ValueError(f'Seed shape {seeds.shape} does not match {tokens.shape[0]} token rows.')
    if np.any(seeds < 0) or np.any(seeds >= 2**32):
        raise ValueError('Watermark seeds must be in [0, 2**32).')
    if np.any(tokens < 0) or np.any(tokens >= vocab_size):
        raise ValueError(f'Token IDs must be in [0, {vocab_size}).')
    return seeds[0], tokens


def detect(prefix, output_dir, *, vocab_size, method, watermark_key_length,
           rolling_window_size, permutation_count, seed=1, gamma=0.4,
           sample_index=None, window_index=None):
    from .validation import positive
    for name, value in [('vocab_size', vocab_size), ('watermark_key_length', watermark_key_length),
                        ('rolling_window_size', rolling_window_size), ('permutation_count', permutation_count)]:
        positive(name, value)
    seeds, tokens = read_tokens(prefix, vocab_size)
    if rolling_window_size >= tokens.shape[1]:
        raise ValueError('rolling_window_size must be smaller than the token row length.')
    if sample_index is not None and not 0 <= sample_index < len(tokens):
        raise ValueError(f'sample index must be in [0, {len(tokens)}); got {sample_index}.')
    if window_index is not None and not 0 <= window_index < tokens.shape[1]:
        raise ValueError(f'window index must be in [0, {tokens.shape[1]}); got {window_index}.')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = make_statistics(method, gamma)
    samples = range(len(tokens)) if sample_index is None else [sample_index]
    windows = range(tokens.shape[1]) if window_index is None else [window_index]
    from time import perf_counter
    for i in samples:
        for j in windows:
            path = output_dir / f'{i}-{j}.csv'
            if path.exists():
                raise FileExistsError(f'Refusing to overwrite {path}')
            # A stable seed per task makes task order / separate processes reproducible.
            torch.manual_seed((seed + i * tokens.shape[1] + j) % (2**63))
            start = perf_counter()
            values = sliding_permutation_test(tokens[i], vocab_size, watermark_key_length,
                                               rolling_window_size, permutation_count,
                                               seeds[i], j, stats)
            np.savetxt(path, values, delimiter=',')
            (output_dir / f'{i}-{j}-time.txt').write_text(str(perf_counter()-start))
        print(f'Detected sample {i + 1}/{len(tokens)}', flush=True)


def main(argv=None):
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Local rolling-window detection; sample/window indices are zero-based.')
    parser.add_argument('--token_file', required=True)
    parser.add_argument('--model', default='facebook/opt-1.3b')
    parser.add_argument('--vocab_size', type=int)
    parser.add_argument('--method', choices=['gumbel', 'transform'], default='transform')
    parser.add_argument('--watermark_key_length', type=int, default=256)
    parser.add_argument('--rolling_window_size', type=int, required=True)
    parser.add_argument('--permutation_count', type=int, default=999)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.4)
    parser.add_argument('--Tindex', type=int, default=0)
    parser.add_argument('--rolling_window_index', type=int, default=-1,
                        help='-1 processes all windows for the requested sample.')
    parser.add_argument('--output_dir')
    args = parser.parse_args(argv)
    known = {'facebook/opt-1.3b': 50272, 'openai-community/gpt2': 50257,
             'meta-llama/Meta-Llama-3-8B': 128256}
    vocab = args.vocab_size or known.get(args.model)
    if vocab is None:
        parser.error('Unknown model; pass --vocab_size from its local config.json. No model download is needed.')
    output = args.output_dir or f'{args.token_file}-{args.rolling_window_size}-{args.permutation_count}-detect'
    try:
        detect(args.token_file, output, vocab_size=vocab, method=args.method,
               watermark_key_length=args.watermark_key_length, rolling_window_size=args.rolling_window_size,
               permutation_count=args.permutation_count, seed=args.seed, gamma=args.gamma,
               sample_index=args.Tindex,
               window_index=None if args.rolling_window_index == -1 else args.rolling_window_index)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'Detection failed: {exc}\n')
