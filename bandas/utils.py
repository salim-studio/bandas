"""Small shared helpers."""
from __future__ import annotations

import numpy as np


def default_index(n: int) -> np.ndarray:
    return np.arange(n)


def as_index(idx, n: int) -> np.ndarray:
    if idx is None:
        return default_index(n)
    a = np.asarray(idx)
    if a.shape == ():
        a = np.full(n, a)
    if len(a) != n:
        raise ValueError(f"index length {len(a)} != data length {n}")
    return a


def as_array(data, dtype=None) -> np.ndarray:
    if isinstance(data, np.ndarray) and dtype is None:
        return data
    try:
        return np.asarray(data, dtype=dtype)
    except Exception:
        return np.array(data, dtype=dtype, copy=False)


def isna_mask(a: np.ndarray) -> np.ndarray:
    if a.dtype.kind in "f":
        return np.isnan(a)
    if a.dtype.kind in "c":
        return np.isnan(a.real) | np.isnan(a.imag)
    if a.dtype.kind in "M":
        return np.isnat(a)
    # object / str / int / bool
    if a.dtype.kind == "O":
        out = np.zeros(len(a), dtype=bool)
        for i, v in enumerate(a):
            try:
                if v is None:
                    out[i] = True
                elif isinstance(v, float) and np.isnan(v):
                    out[i] = True
                elif v is np.nan:
                    out[i] = True
            except Exception:
                pass
        # also pandas-style NA
        try:
            import pandas as pd  # local import
            out |= pd.isna(a)
        except Exception:
            pass
        return out
    try:
        import pandas as pd
        return pd.isna(a)
    except Exception:
        return np.zeros(len(a), dtype=bool)


def factorize(keys: np.ndarray):
    """Return (codes, uniques) like pandas.factorize.
    Fast path: pandas hash-based factorize (Cython). Fallback: np.unique."""
    k = np.asarray(keys)
    # numpy '<U' unicode arrays factorize slowly -> convert to object (5x faster)
    if k.dtype.kind in "US":
        k = k.astype(object)
    try:
        import pandas as pd
        codes, uniq = pd.factorize(k, use_na_sentinel=True)
        codes = np.asarray(codes, dtype=np.int64)
        return codes, np.asarray(uniq)
    except Exception:
        uniq, codes = np.unique(k, return_inverse=True)
        return codes.astype(np.int64, copy=False), uniq


def take_rows(a: np.ndarray, pos: np.ndarray) -> np.ndarray:
    return np.asarray(a)[np.asarray(pos)]
