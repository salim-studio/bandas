"""bandas.lazy — out-of-core / chunked processing (LazyFrame)."""
from __future__ import annotations
import numpy as np


class LazyFrame:
    """Lazy/chunked reader: df = LazyFrame.read_csv(path, chunksize=100_000).collect()"""

    def __init__(self, gen_fn, columns=None):
        self._gen_fn = gen_fn
        self._columns = columns
        self._ops = []

    @classmethod
    def read_csv(cls, path, chunksize=100_000, **kw):
        def gen():
            try:
                import pandas as pd
                for chunk in pd.read_csv(path, chunksize=chunksize, **kw):
                    from .dataframe import DataFrame
                    yield DataFrame(chunk)
            except Exception:
                from .io import read_csv
                yield read_csv(path, **kw)
        return cls(gen)

    @classmethod
    def from_frames(cls, frames):
        def gen():
            for f in frames: yield f
        return cls(gen, columns=getattr(frames[0], "_columns", None) if frames else None)

    def filter(self, mask_fn):
        lf = LazyFrame(self._gen_fn, self._columns); lf._ops = self._ops + [("filter", mask_fn)]; return lf

    def select(self, columns):
        def op(df): return df[list(columns)]
        lf = LazyFrame(self._gen_fn, columns); lf._ops = self._ops + [("map", op)]; return lf

    def map(self, fn):
        lf = LazyFrame(self._gen_fn, self._columns); lf._ops = self._ops + [("map", fn)]; return lf

    def _apply(self, df):
        from .series import Series
        for kind, fn in self._ops:
            if kind == "filter":
                m = fn(df)
                if isinstance(m, Series): m = np.asarray(m._data, dtype=bool)
                df = df[np.asarray(m, dtype=bool)]
            else:
                df = fn(df)
        return df

    def __iter__(self):
        for chunk in self._gen_fn():
            yield self._apply(chunk)

    def collect(self):
        from .dataframe import DataFrame
        parts = list(self)
        if not parts: return DataFrame({})
        if len(parts) == 1: return parts[0]
        from . import concat
        return concat(parts, ignore_index=True)

    def sum(self, col):
        tot = 0.0
        for chunk in self: tot += float(chunk[col].sum())
        return tot

    def mean(self, col):
        tot, n = 0.0, 0
        for chunk in self:
            a = np.asarray(chunk._cols[col], dtype=np.float64)
            tot += float(np.nansum(a)); n += int((~np.isnan(a)).sum())
        return tot / n if n else float("nan")

    def count(self):
        return sum(len(c) for c in self)
