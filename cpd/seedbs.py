"""Numerical functions extracted from 4.1-seedbs.py; R parity is not assumed."""
from numpy import ceil as np_ceil, floor, array, linspace, unique, mean
from math import ceil, log, sqrt
from random import sample
from scipy.stats import ks_2samp

def get_seeded_intervals(n, decay=sqrt(2), unique_int=False):
    n = int(n)
    depth = log(n, decay)
    depth = ceil(depth)

    boundary_mtx = []
    boundary_mtx.append((1, n))

    for i in range(2, depth + 1):
        int_length = n * (1 / decay) ** (i - 1)
        n_int = ceil(round(n / int_length, 14)) * 2 - 1

        starts = floor(linspace(
            1, n - int_length, int(n_int))).astype(int)
        ends = np_ceil(linspace(int_length, n, int(n_int))).astype(int)
        for st, end in zip(starts, ends):
            boundary_mtx.append((st, end))

    boundary_mtx = array(boundary_mtx)

    if unique_int:
        boundary_mtx = unique(boundary_mtx, axis=0)

    return boundary_mtx


def ks_statistic(pvalues):
    result = []
    n = len(pvalues)
    for k in range(1, n):
        segment_before = pvalues[:k]
        segment_after = pvalues[k:]
        if len(segment_before) == 0 or len(segment_after) == 0:
            continue
        ks_test_stat = ks_2samp(segment_before, segment_after).statistic
        value = k * (n - k) / (n ** 1.5) * ks_test_stat
        result.append((k, value))
    if not result:
        return (None, None)
    max_k, max_val = max(result, key=lambda x: x[1])
    return (max_k, max_val)


def permute_pvalues(pvalues, block_size=1):
    n = len(pvalues)
    pvalue_indices = list(range(n - block_size + 1))
    sampled_size = ceil(n / block_size)
    sampled_indices = sample(pvalue_indices, k=sampled_size)

    permuted_pvalues = []
    for idx in sampled_indices:
        permuted_pvalues.extend(pvalues[idx:idx + block_size])

    # Truncate to original length
    return permuted_pvalues[:n]


def segment_significance(pvalues, significance_permutation_count=999, block_size=10):
    original_ks_statistic = ks_statistic(pvalues)
    if original_ks_statistic[1] is None:
        return (None, None)
    p_tilde = [1]
    for _ in range(significance_permutation_count):
        pvalues_permuted = permute_pvalues(pvalues, block_size=block_size)
        ks_statistic_permuted = ks_statistic(pvalues_permuted)
        if ks_statistic_permuted[1] is None:
            p_tilde.append(0)
        else:
            p_tilde.append(
                int(original_ks_statistic[1] <= ks_statistic_permuted[1]))
    return (original_ks_statistic[0], mean(p_tilde))
