"""bandas.preprocessing — imputers / scalers / encoders (sklearn-compatible, zero-dep)."""
from __future__ import annotations
import numpy as np


class SimpleImputer:
    def __init__(self, strategy="mean", fill_value=None):
        assert strategy in ("mean", "median", "most_frequent", "constant")
        self.strategy = strategy; self.fill_value = fill_value; self.statistics_ = {}

    def _stat(self, a):
        from .utils import isna_mask
        m = isna_mask(np.asarray(a)); v = np.asarray(a)[~m]
        if self.strategy == "mean":
            try: return float(np.nanmean(np.asarray(a, dtype=np.float64)))
            except Exception: return self.fill_value
        if self.strategy == "median":
            try: return float(np.nanmedian(np.asarray(a, dtype=np.float64)))
            except Exception: return self.fill_value
        if self.strategy == "most_frequent":
            if len(v) == 0: return self.fill_value
            u, c = np.unique(v, return_counts=True); return u[int(np.argmax(c))]
        return self.fill_value

    def fit(self, df):
        cols = df._columns if hasattr(df, "_columns") else [df.name]
        for c in cols:
            a = np.asarray(df._cols[c] if hasattr(df, "_cols") else df._data)
            self.statistics_[c] = self._stat(a)
        return self

    def transform(self, df):
        from .dataframe import DataFrame
        from .series import Series
        from .utils import isna_mask
        if hasattr(df, "_columns"):
            out = {}
            for c in df._columns:
                a = np.array(np.asarray(df._cols[c]), copy=True)
                m = isna_mask(np.asarray(a))
                if m.any():
                    try: a[m] = self.statistics_[c]
                    except Exception:
                        a = a.astype(object); a[m] = self.statistics_[c]
                out[c] = a
            return DataFrame(out, index=np.asarray(df._index))
        a = np.array(np.asarray(df._data), copy=True)
        m = isna_mask(a)
        if m.any():
            try: a[m] = self.statistics_.get(df.name, self.fill_value)
            except Exception:
                a = a.astype(object); a[m] = self.statistics_.get(df.name, self.fill_value)
        return Series(a, index=np.asarray(df._index.copy()), name=df.name)

    def fit_transform(self, df): return self.fit(df).transform(df)


class StandardScaler:
    def __init__(self): self.mean_ = {}; self.scale_ = {}
    def fit(self, df, columns=None):
        cols = list(columns) if columns else ([c for c in df._columns if np.asarray(df._cols[c]).dtype.kind in "iufc"] if hasattr(df, "_columns") else [None])
        for c in cols:
            a = np.asarray(df._cols[c] if c is not None else df._data, dtype=np.float64)
            self.mean_[c] = float(np.nanmean(a)); s = float(np.nanstd(a))
            self.scale_[c] = s if s else 1.0
        return self
    def transform(self, df):
        from .dataframe import DataFrame
        if not hasattr(df, "_columns"):
            a = (np.asarray(df._data, dtype=np.float64) - self.mean_[None]) / self.scale_[None]
            from .series import Series
            return Series(a, index=np.asarray(df._index.copy()), name=df.name)
        out = dict(df._cols)
        for c in self.mean_:
            out[c] = (np.asarray(df._cols[c], dtype=np.float64) - self.mean_[c]) / self.scale_[c]
        return DataFrame(out, index=np.asarray(df._index))
    def fit_transform(self, df, columns=None): return self.fit(df, columns).transform(df)
    def inverse_transform(self, df):
        from .dataframe import DataFrame
        out = dict(df._cols)
        for c in self.mean_: out[c] = np.asarray(df._cols[c]) * self.scale_[c] + self.mean_[c]
        return DataFrame(out, index=np.asarray(df._index))


class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)): self.range = feature_range; self.min_ = {}; self.max_ = {}
    def fit(self, df, columns=None):
        cols = list(columns) if columns else ([c for c in df._columns if np.asarray(df._cols[c]).dtype.kind in "iufc"] if hasattr(df, "_columns") else [None])
        for c in cols:
            a = np.asarray(df._cols[c] if c is not None else df._data, dtype=np.float64)
            self.min_[c] = float(np.nanmin(a)); self.max_[c] = float(np.nanmax(a))
        return self
    def transform(self, df):
        from .dataframe import DataFrame
        lo, hi = self.range
        def one(a, mn, mx):
            a = np.asarray(a, dtype=np.float64)
            rng = (mx - mn) or 1.0
            return (a - mn) / rng * (hi - lo) + lo
        if not hasattr(df, "_columns"):
            from .series import Series
            return Series(one(np.asarray(df._data), self.min_[None], self.max_[None]), index=np.asarray(df._index.copy()), name=df.name)
        return DataFrame({c: (one(np.asarray(v), self.min_[c], self.max_[c]) if c in self.min_ else np.asarray(v)) for c, v in df._cols.items()}, index=np.asarray(df._index))
    def fit_transform(self, df, columns=None): return self.fit(df, columns).transform(df)


class RobustScaler:
    def __init__(self): self.center_ = {}; self.scale_ = {}
    def fit(self, df, columns=None):
        cols = list(columns) if columns else ([c for c in df._columns if np.asarray(df._cols[c]).dtype.kind in "iufc"] if hasattr(df, "_columns") else [None])
        for c in cols:
            a = np.asarray(df._cols[c] if c is not None else df._data, dtype=np.float64)
            self.center_[c] = float(np.nanmedian(a))
            q75, q25 = np.nanpercentile(a, [75, 25])
            self.scale_[c] = float(q75 - q25) or 1.0
        return self
    def transform(self, df):
        from .dataframe import DataFrame
        if not hasattr(df, "_columns"):
            from .series import Series
            return Series((np.asarray(df._data, dtype=np.float64) - self.center_[None]) / self.scale_[None], index=np.asarray(df._index.copy()), name=df.name)
        return DataFrame({c: ((np.asarray(v, dtype=np.float64) - self.center_[c]) / self.scale_[c] if c in self.center_ else np.asarray(v)) for c, v in df._cols.items()}, index=np.asarray(df._index))
    def fit_transform(self, df, columns=None): return self.fit(df, columns).transform(df)


class LabelEncoder:
    def __init__(self): self.classes_ = None; self._map = {}
    def fit(self, y):
        v = np.asarray(y._data if hasattr(y, "_data") else y)
        self.classes_ = np.asarray(sorted(set(map(str, v)), key=str), dtype=object)
        self._map = {c: i for i, c in enumerate(self.classes_)}
        return self
    def transform(self, y):
        from .series import Series
        v = np.asarray(y._data if hasattr(y, "_data") else y)
        out = np.array([self._map[str(x)] for x in v])
        if hasattr(y, "_data"):
            return Series(out, index=np.asarray(y._index.copy()), name=y.name)
        return out
    def fit_transform(self, y): return self.fit(y).transform(y)
    def inverse_transform(self, y):
        v = np.asarray(y._data if hasattr(y, "_data") else y)
        return np.array([self.classes_[int(i)] for i in v], dtype=object)


class OneHotEncoder:
    def __init__(self, columns=None, dtype=float):
        self.columns = columns; self.dtype = dtype; self.categories_ = {}
    def fit(self, df):
        cols = list(self.columns) if self.columns else ([c for c in df._columns if np.asarray(df._cols[c]).dtype.kind in "OUS"] if hasattr(df, "_columns") else [None])
        for c in cols:
            v = np.asarray(df._cols[c] if c is not None else df._data)
            self.categories_[c] = sorted(set(map(str, v)), key=str)
        return self
    def transform(self, df):
        from .reshape import get_dummies
        return get_dummies(df, columns=list(self.categories_.keys()), dtype=self.dtype)
    def fit_transform(self, df): return self.fit(df).transform(df)
