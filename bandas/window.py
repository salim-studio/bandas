"""bandas.window — Rolling / Expanding / EWM (pandas-compatible API)."""
from __future__ import annotations
import numpy as np


def _roll_nanmean(a, w, min_periods):
    out = np.full(len(a), np.nan)
    c = np.concatenate([[0.0], np.cumsum(np.where(np.isnan(a), 0, a))])
    n = np.concatenate([[0], np.cumsum((~np.isnan(a)).astype(int))])
    for i in range(len(a)):
        s = max(0, i - w + 1)
        cnt = n[i + 1] - n[s]
        if cnt >= min_periods:
            out[i] = (c[i + 1] - c[s]) / max(cnt, 1)
    return out


class Rolling:
    def __init__(self, obj, window, min_periods=None, center=False):
        self._obj = obj
        self.window = int(window)
        self.min_periods = min_periods or self.window
        self.center = center

    def _apply_series(self, a, fn):
        w, mp = self.window, self.min_periods
        a = np.asarray(a, dtype=np.float64)
        if fn == "mean": return _roll_nanmean(a, w, mp)
        if fn == "sum":
            out = np.full(len(a), np.nan)
            for i in range(len(a)):
                s = max(0, i - w + 1)
                seg = a[s:i + 1]
                seg = seg[~np.isnan(seg)]
                if len(seg) >= mp: out[i] = seg.sum()
            return out
        if fn in ("min", "max", "std", "var", "median"):
            out = np.full(len(a), np.nan)
            for i in range(len(a)):
                s = max(0, i - w + 1)
                seg = a[s:i + 1]
                seg = seg[~np.isnan(seg)]
                if len(seg) >= mp:
                    out[i] = {"min": seg.min(), "max": seg.max(), "std": seg.std(ddof=1) if len(seg) > 1 else np.nan,
                              "var": seg.var(ddof=1) if len(seg) > 1 else np.nan, "median": np.median(seg)}[fn]
            return out
        if fn == "count":
            m = (~np.isnan(a)).astype(float)
            return np.array([m[max(0, i - w + 1):i + 1].sum() for i in range(len(a))])
        raise ValueError(fn)

    def _wrap(self, data):
        from .series import Series
        from .dataframe import DataFrame
        if isinstance(self._obj, Series):
            return Series(data, index=np.asarray(self._obj._index.copy()), name=self._obj._name)
        nd = DataFrame.__new__(DataFrame)
        nd._columns = list(self._obj._columns); nd._cols = dict(zip(nd._columns, [data[c] for c in nd._columns]))
        nd._index = np.asarray(self._obj._index.copy())
        return nd

    def _run(self, fn):
        from .series import Series
        if isinstance(self._obj, Series):
            return self._wrap(self._apply_series(np.asarray(self._obj._data), fn))
        return self._wrap({c: self._apply_series(np.asarray(self._obj._cols[c]), fn) for c in self._obj._columns})

    def mean(self): return self._run("mean")
    def sum(self): return self._run("sum")
    def min(self): return self._run("min")
    def max(self): return self._run("max")
    def std(self): return self._run("std")
    def var(self): return self._run("var")
    def median(self): return self._run("median")
    def count(self): return self._run("count")
    def agg(self, f): return getattr(self, f)() if isinstance(f, str) else f(self)
    aggregate = agg


class Expanding:
    def __init__(self, obj, min_periods=1):
        self._obj = obj; self.min_periods = min_periods

    def _run(self, fn):
        from .series import Series
        def one(a):
            a = np.asarray(a, dtype=np.float64); out = np.full(len(a), np.nan)
            for i in range(len(a)):
                seg = a[:i + 1]; seg = seg[~np.isnan(seg)]
                if len(seg) >= self.min_periods:
                    out[i] = {"mean": seg.mean(), "sum": seg.sum(), "min": seg.min(), "max": seg.max(),
                              "std": seg.std(ddof=1) if len(seg) > 1 else np.nan, "var": seg.var(ddof=1) if len(seg) > 1 else np.nan,
                              "count": float(len(seg)), "median": float(np.median(seg))}[fn]
            return out
        if isinstance(self._obj, Series):
            return Series(one(np.asarray(self._obj._data)), index=np.asarray(self._obj._index.copy()), name=self._obj._name)
        from .dataframe import DataFrame
        nd = DataFrame.__new__(DataFrame); nd._columns = list(self._obj._columns)
        nd._cols = {c: one(np.asarray(self._obj._cols[c])) for c in nd._columns}
        nd._index = np.asarray(self._obj._index.copy()); return nd

    def mean(self): return self._run("mean")
    def sum(self): return self._run("sum")
    def min(self): return self._run("min")
    def max(self): return self._run("max")
    def std(self): return self._run("std")
    def var(self): return self._run("var")
    def count(self): return self._run("count")
    def median(self): return self._run("median")
    def agg(self, f): return getattr(self, f)() if isinstance(f, str) else f(self)
    aggregate = agg


class EWM:
    def __init__(self, obj, span=None, alpha=None, adjust=True, min_periods=0):
        self._obj = obj
        if alpha is None and span is not None: alpha = 2.0 / (span + 1.0)
        self.alpha = float(alpha or 0.5); self.adjust = adjust; self.min_periods = min_periods

    def mean(self):
        from .series import Series
        def one(a):
            a = np.asarray(a, dtype=np.float64); out = np.full(len(a), np.nan)
            # pandas-compatible adjust=True EWM via weights
            coef = (1 - self.alpha)
            num = 0.0; den = 0.0; w = 1.0; n = 0
            for i, v in enumerate(a):
                if v == v:
                    if self.adjust:
                        num = num * coef + v; den = den * coef + 1.0
                        out[i] = num / den
                    else:
                        out[i] = v if i == 0 or np.isnan(out[i - 1]) else self.alpha * v + coef * out[i - 1]
                    n += 1
                else:
                    out[i] = out[i - 1] if i else np.nan
            if self.min_periods:
                c = 0
                for i, v in enumerate(a):
                    if v == v: c += 1
                    if c < self.min_periods: out[i] = np.nan
            return out
        if isinstance(self._obj, Series):
            return Series(one(np.asarray(self._obj._data)), index=np.asarray(self._obj._index.copy()), name=self._obj._name)
        from .dataframe import DataFrame
        nd = DataFrame.__new__(DataFrame); nd._columns = list(self._obj._columns)
        nd._cols = {c: one(np.asarray(self._obj._cols[c])) for c in nd._columns}
        nd._index = np.asarray(self._obj._index.copy()); return nd
