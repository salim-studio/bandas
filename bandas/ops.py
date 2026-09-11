"""Fast kernels with optional Numba acceleration. Pure-NumPy fallback always works."""
from __future__ import annotations

import numpy as np

try:
    from numba import njit  # type: ignore
    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco


@njit(cache=True, fastmath=True)
def _group_sum_mean_codes(values, codes, n_groups):
    sums = np.zeros(n_groups, dtype=np.float64)
    counts = np.zeros(n_groups, dtype=np.int64)
    for i in range(values.shape[0]):
        c = codes[i]
        if c >= 0:
            v = values[i]
            if v == v:  # skip NaN
                sums[c] += v
                counts[c] += 1
    return sums, counts


@njit(cache=True, fastmath=True)
def _group_min_max_codes(values, codes, n_groups):
    mins = np.empty(n_groups, dtype=np.float64)
    maxs = np.empty(n_groups, dtype=np.float64)
    for g in range(n_groups):
        mins[g] = np.inf
        maxs[g] = -np.inf
    for i in range(values.shape[0]):
        c = codes[i]
        if c >= 0:
            v = values[i]
            if v == v:
                if v < mins[c]:
                    mins[c] = v
                if v > maxs[c]:
                    maxs[c] = v
    return mins, maxs


def group_sum(values: np.ndarray, codes: np.ndarray, n_groups: int):
    """Fast group sum/count. Returns (sums, counts). NaN-aware."""
    vals = np.asarray(values, dtype=np.float64)
    cd = np.asarray(codes, dtype=np.int64)
    if _HAS_NUMBA:
        return _group_sum_mean_codes(vals, cd, n_groups)
    # numpy fallback: still fast via bincount (mask NaN out)
    mask = ~np.isnan(vals)
    sums = np.bincount(cd[mask & (cd >= 0)], weights=vals[mask & (cd >= 0)], minlength=n_groups)
    counts = np.bincount(cd[mask & (cd >= 0)], minlength=n_groups)
    return sums.astype(np.float64), counts.astype(np.int64)


def group_min_max(values: np.ndarray, codes: np.ndarray, n_groups: int):
    vals = np.asarray(values, dtype=np.float64)
    cd = np.asarray(codes, dtype=np.int64)
    if _HAS_NUMBA:
        return _group_min_max_codes(vals, cd, n_groups)
    mins = np.full(n_groups, np.inf)
    maxs = np.full(n_groups, -np.inf)
    # pure python loop fallback is slow; use pandas-free segmented approach via sorting
    order = np.argsort(cd, kind="stable")
    sc = cd[order]
    sv = vals[order]
    # iterate groups (n_groups usually small)
    for g in range(n_groups):
        m = (sc == g)
        v = sv[m]
        v = v[~np.isnan(v)]
        if v.size:
            mins[g] = v.min()
            maxs[g] = v.max()
    return mins, maxs


def nansum_fast(a: np.ndarray):
    return np.nansum(a)


def nanmean_fast(a: np.ndarray):
    return np.nanmean(a)
