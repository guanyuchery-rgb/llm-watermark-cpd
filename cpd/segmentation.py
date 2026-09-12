"""Python SeedBS + NOT adapter; original R analysis remains available as reference."""
import csv
import random
from pathlib import Path

import numpy as np

from .seedbs import get_seeded_intervals, segment_significance


FIELDS = ['sample', 'metric', 'interval', 'from', 'to', 'segment_length',
          'index_within_segment', 'change_point_index', 'significance']


def select_not(candidates, threshold):
    """Selection rule from 5-not.R:348-371; stable first minimum breaks ties."""
    pending = [row for row in candidates if row['significance'] <= threshold]
    selected = []
    while pending:
        point = min(pending, key=lambda row: row['segment_length'])
        selected.append(point)
        pending = [row for row in pending
                   if row['from'] > point['change_point_index']
                   or row['to'] < point['change_point_index']]
    return sorted(selected, key=lambda row: row['change_point_index'])


def write_rows(path, rows, fields):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def segment(detection_dir, output_dir, *, sample_count, token_count,
            rolling_window_size, minimum_length, permutation_count,
            block_size, threshold, seed):
    if rolling_window_size % 2:
        raise ValueError('Segmentation requires an even rolling_window_size for integral coordinates.')
    intervals = get_seeded_intervals(token_count - rolling_window_size, unique_int=True)
    intervals = intervals[intervals[:, 1] - intervals[:, 0] >= minimum_length]
    if not len(intervals):
        raise ValueError('No seeded intervals remain; decrease minimum_length or increase token_count.')
    intervals = intervals + rolling_window_size // 2
    # Preflight every expected input before doing any permutation work.
    matrices = []
    for sample in range(sample_count):
        values = []
        for position in range(token_count):
            path = Path(detection_dir) / f'{sample}-{position}.csv'
            try:
                row = np.loadtxt(path, delimiter=',', ndmin=2)
            except (ValueError, OSError) as exc:
                raise ValueError(f'Cannot read detection result {path}: {exc}') from exc
            if row.shape != (1, 2):
                raise ValueError(f'Expected two detection metrics in {path}, got {row.shape}.')
            values.append(row[0])
        matrix = np.array(values)
        for start, end in intervals:
            subset = matrix[int(start)-1:int(end)]
            if not np.isfinite(subset).all() or np.any(subset < 0) or np.any(subset > 1):
                raise ValueError(f'Invalid p-values within sample {sample}, interval [{start}, {end}].')
            n = len(subset)
            if block_size > n or (n + block_size - 1)//block_size > n-block_size+1:
                raise ValueError(f'block_size={block_size} cannot be sampled within interval length {n}.')
        matrices.append(matrix)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates, changes = [], []
    for sample, matrix in enumerate(matrices):
        for metric in range(matrix.shape[1]):
            current = []
            for index, (start, end) in enumerate(intervals):
                # Stable task seeds also permit changing execution order later.
                task_seed = seed + sample*len(intervals)*matrix.shape[1] + metric*len(intervals) + index
                random.seed(task_seed)
                np.random.seed(task_seed % 2**32)
                split, pvalue = segment_significance(
                    matrix[int(start)-1:int(end), metric].tolist(), permutation_count, block_size)
                row = dict(sample=sample, metric=metric, interval=index,
                           **{'from': int(start), 'to': int(end)},
                           segment_length=int(end-start), index_within_segment=int(split),
                           change_point_index=int(split+start-1), significance=float(pvalue))
                current.append(row)
            candidates.extend(current)
            changes.extend(select_not(current, threshold))
        print(f'Segmented sample {sample+1}/{sample_count}', flush=True)
    write_rows(output_dir/'seedbs.csv', candidates, FIELDS)
    write_rows(output_dir/'changepoints.csv', changes, FIELDS)
    write_rows(output_dir/'intervals.csv',
               [{'interval': i, 'from': int(a), 'to': int(b)} for i, (a, b) in enumerate(intervals)],
               ['interval', 'from', 'to'])
