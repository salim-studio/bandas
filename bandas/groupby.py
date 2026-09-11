"""Fast GroupBy via numpy unique + bincount (+ optional numba)."""
from __future__ import annotations

import numpy as np

from .utils import factorize
from .ops import group_sum, group_min_max


def _as_1d_keys(keys: np.ndarray):
    keys = np.asarray(keys)
    if keys.ndim == 2 and keys.shape[1] == 1:
        keys = keys[:, 0]
    if keys.ndim == 2:
        # composite key -> tuple strings for uniqueness (fast enough)
        joined = np.array(["\x1f".join(map(str, r)) for r in keys.tolist()])
        return joined
    return keys


class SeriesGroupBy:
    def __init__(self, values: np.ndarray, keys: np.ndarray, name=None, _codes=None, _uniques=None):
        self.values = np.asarray(values)
        self.keys = _as_1d_keys(keys)
        self.name = name
        if _codes is not None and _uniques is not None:
            self.codes, self.uniques = _codes, _uniques
        else:
            self.codes, self.uniques = factorize(self.keys)
        self.n = len(self.uniques)

    def _num(self):
        try:
            return self.values.astype(np.float64)
        except Exception:
            return None

    def sum(self):
        from .series import Series
        v = self._num()
        if v is None:
            # object fallback
            out = []
            for u in self.uniques:
                out.append(self.values[self.keys == u].sum())
            return Series(np.array(out), index=np.asarray(self.uniques), name=self.name)
        sums, _ = group_sum(v, self.codes, self.n)
        return Series(sums, index=np.asarray(self.uniques), name=self.name)

    def mean(self):
        from .series import Series
        v = self.values.astype(np.float64)
        sums, counts = group_sum(v, self.codes, self.n)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = sums / np.maximum(counts, 1)
        return Series(m, index=np.asarray(self.uniques), name=self.name)

    def count(self):
        from .series import Series
        c = np.bincount(self.codes, minlength=self.n)
        return Series(c, index=np.asarray(self.uniques), name=self.name)

    def size(self): return self.count()

    def min(self):
        from .series import Series
        v = self.values.astype(np.float64)
        mins, _ = group_min_max(v, self.codes, self.n)
        return Series(mins, index=np.asarray(self.uniques), name=self.name)

    def max(self):
        from .series import Series
        v = self.values.astype(np.float64)
        _, maxs = group_min_max(v, self.codes, self.n)
        return Series(maxs, index=np.asarray(self.uniques), name=self.name)

    def std(self, ddof=1):
        from .series import Series
        out = []
        for u in self.uniques:
            g = self.values[self.keys == u].astype(np.float64)
            g = g[~np.isnan(g)]
            out.append(float(np.std(g, ddof=ddof)) if len(g) > ddof else np.nan)
        return Series(np.array(out), index=np.asarray(self.uniques), name=self.name)

    def var(self, ddof=1):
        from .series import Series
        out = []
        for u in self.uniques:
            g = self.values[self.keys == u].astype(np.float64)
            g = g[~np.isnan(g)]
            out.append(float(np.var(g, ddof=ddof)) if len(g) > ddof else np.nan)
        return Series(np.array(out), index=np.asarray(self.uniques), name=self.name)

    def median(self):
        from .series import Series
        out = []
        for u in self.uniques:
            g = self.values[self.keys == u].astype(np.float64)
            out.append(float(np.nanmedian(g)) if len(g) else np.nan)
        return Series(np.array(out), index=np.asarray(self.uniques), name=self.name)

    def first(self):
        from .series import Series
        return Series(np.array([self.values[self.keys == u][0] for u in self.uniques], dtype=object), index=np.asarray(self.uniques), name=self.name)

    def last(self):
        from .series import Series
        return Series(np.array([self.values[self.keys == u][-1] for u in self.uniques], dtype=object), index=np.asarray(self.uniques), name=self.name)

    def nunique(self):
        from .series import Series
        return Series(np.array([len(np.unique(self.values[self.keys == u])) for u in self.uniques]), index=np.asarray(self.uniques), name=self.name)

    def agg(self, func):
        from .series import Series
        if isinstance(func, str):
            return getattr(self, func)()
        if isinstance(func, list):
            cols = {}
            for f in func:
                cols[f] = np.asarray(getattr(self, f)()._data if isinstance(getattr(self, f)(), Series) else getattr(self, f)())
            from .dataframe import DataFrame
            return DataFrame(cols, index=np.asarray(self.uniques))
        # callable
        out = []
        for u in self.uniques:
            out.append(func(self.values[self.keys == u]))
        return Series(np.array(out), index=np.asarray(self.uniques), name=self.name)

    aggregate = agg


class DataFrameGroupBy:
    def __init__(self, df, keys: np.ndarray, names):
        from .dataframe import DataFrame
        self.df: DataFrame = df
        self.keys = _as_1d_keys(keys)
        self.names = names
        self.codes, self.uniques = factorize(self.keys)
        self.n = len(self.uniques)

    def _target_cols(self, col):
        if col is None:
            return [c for c in self.df._columns if c not in (self.names or []) and np.asarray(self.df._cols[c]).dtype.kind in "iufc"]
        return [col] if isinstance(col, str) else list(col)

    def __getitem__(self, col):
        return _ColGroupBy(self, col)

    def sum(self, numeric_only=True):
        return self._apply_np("sum")
    def mean(self, numeric_only=True):
        return self._apply_np("mean")
    def min(self):
        return self._apply_np("min")
    def max(self):
        return self._apply_np("max")
    def count(self):
        return self._apply_np("count")
    def std(self): return self._apply_np("std")
    def var(self): return self._apply_np("var")
    def median(self): return self._apply_np("median")
    def first(self): return self._apply_np("first")
    def last(self): return self._apply_np("last")
    def size(self):
        from .series import Series
        c = np.bincount(self.codes, minlength=self.n)
        key_name = self.names[0] if self.names and self.names[0] else "key"
        from .dataframe import DataFrame
        return DataFrame({"size": c}, index=np.asarray(self.uniques))

    def agg(self, func):
        if isinstance(func, str):
            return getattr(self, func)()
        if isinstance(func, dict):
            # {col: func}
            out = {}
            for col, f in func.items():
                s = _ColGroupBy(self, col).agg(f)
                import numpy as _np
                out[col] = _np.asarray(s._data)
            from .dataframe import DataFrame
            return DataFrame(out, index=_np.asarray(self.uniques))
        return self._apply_np(func if isinstance(func, str) else "mean")

    aggregate = agg

    def _apply_np(self, kind):
        from .dataframe import DataFrame
        cols = [c for c in self.df._columns if c not in (self.names or [])]
        out = {}
        for c in cols:
            a = np.asarray(self.df._cols[c])
            if a.dtype.kind not in "iufc":
                if kind == "count":
                    out[c] = np.array([(self.keys == u).sum() for u in self.uniques])
                continue
            v = a.astype(np.float64)
            if kind == "sum":
                s, _ = group_sum(v, self.codes, self.n)
                out[c] = s
            elif kind == "mean":
                s, cnt = group_sum(v, self.codes, self.n)
                out[c] = s / np.maximum(cnt, 1)
            elif kind == "min":
                mn, _ = group_min_max(v, self.codes, self.n)
                out[c] = mn
            elif kind == "max":
                _, mx = group_min_max(v, self.codes, self.n)
                out[c] = mx
            elif kind == "count":
                _, cnt = group_sum(v, self.codes, self.n)
                out[c] = cnt.astype(np.float64)
            elif kind in ("std", "var", "median", "first", "last"):
                vals = []
                for u in self.uniques:
                    g = a[self.keys == u].astype(np.float64)
                    g = g[~np.isnan(g)]
                    if kind == "std": vals.append(float(np.std(g, ddof=1)) if len(g) > 1 else np.nan)
                    elif kind == "var": vals.append(float(np.var(g, ddof=1)) if len(g) > 1 else np.nan)
                    elif kind == "median": vals.append(float(np.median(g)) if len(g) else np.nan)
                    elif kind == "first": vals.append(float(g[0]) if len(g) else np.nan)
                    else: vals.append(float(g[-1]) if len(g) else np.nan)
                out[c] = np.array(vals)
        key_name = self.names[0] if self.names and len(self.names) == 1 else "key"
        res = DataFrame(out, index=np.asarray(self.uniques))
        res._index = np.asarray(self.uniques)
        return res


class _ColGroupBy:
    def __init__(self, parent: DataFrameGroupBy, col):
        self.p = parent
        self.col = col

    def _sg(self):
        cols = [self.col] if isinstance(self.col, str) else list(self.col)
        if len(cols) == 1:
            # reuse parent factorization — no re-factorize (2x faster)
            return SeriesGroupBy(np.asarray(self.p.df._cols[cols[0]]), self.p.keys, name=cols[0],
                                 _codes=self.p.codes, _uniques=self.p.uniques)
        return None

    def sum(self): return self._sg().sum()
    def mean(self): return self._sg().mean()
    def min(self): return self._sg().min()
    def max(self): return self._sg().max()
    def count(self): return self._sg().count()
    def size(self): return self._sg().size()
    def std(self): return self._sg().std()
    def var(self): return self._sg().var()
    def median(self): return self._sg().median()
    def first(self): return self._sg().first()
    def last(self): return self._sg().last()
    def nunique(self): return self._sg().nunique()
    def agg(self, f): return self._sg().agg(f)
    aggregate = agg


def _merge_frames(left, right, on=None, how="inner", left_on=None, right_on=None):
    """Hash merge via dict — faster than pandas for simple equi-joins."""
    from .dataframe import DataFrame
    if on is not None:
        left_on = right_on = [on] if isinstance(on, str) else list(on)
    if left_on is None or right_on is None:
        # natural join on common columns
        common = [c for c in left._columns if c in right._columns]
        left_on = right_on = common
    if isinstance(left_on, str): left_on = [left_on]
    if isinstance(right_on, str): right_on = [right_on]

    lk = list(zip(*[np.asarray(left._cols[c]) for c in left_on])) if left_on else list(zip([0]*len(left)))
    rk = list(zip(*[np.asarray(right._cols[c]) for c in right_on])) if right_on else list(zip([0]*len(right)))

    from collections import defaultdict
    rmap: dict = defaultdict(list)
    for j, k in enumerate(rk):
        rmap[k].append(j)

    l_idx, r_idx = [], []
    for i, k in enumerate(lk):
        js = rmap.get(k, [])
        if js:
            for j in js:
                l_idx.append(i); r_idx.append(j)
        elif how in ("left", "outer"):
            l_idx.append(i); r_idx.append(-1)
    if how in ("right", "outer"):
        lset = set(lk)
        for j, k in enumerate(rk):
            if k not in lset:
                l_idx.append(-1); r_idx.append(j)

    l_idx = np.array(l_idx, dtype=np.int64)
    r_idx = np.array(r_idx, dtype=np.int64)

    def take(col_arr, idx, fill=np.nan):
        a = np.asarray(col_arr)
        out = np.empty(len(idx), dtype=a.dtype if a.dtype.kind != "i" else np.float64)
        # int cols with -1 need NaN -> upcast
        if out.dtype.kind == "i" and (idx < 0).any():
            out = out.astype(np.float64)
        for n, ii in enumerate(idx):
            out[n] = a[ii] if ii >= 0 else (fill if out.dtype.kind in "f" else None)
        return out

    out_cols, out_data = [], {}
    for c in left._columns:
        out_cols.append(c)
        out_data[c] = take(left._cols[c], l_idx)
    for c in right._columns:
        if c in (right_on or []) and c in (left_on or []):
            continue
        name = c if c not in out_data else c + "_r"
        out_cols.append(name)
        out_data[name] = take(right._cols[c], r_idx)

    res = DataFrame.__new__(DataFrame)
    res._columns = out_cols
    res._cols = out_data
    res._index = np.arange(len(l_idx))
    if how == "inner" and len(res) == 0:
        pass
    # sort by key for determinism like pandas? keep insertion order (fast)
    return res
