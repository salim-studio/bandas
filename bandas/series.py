"""bandas.Series — pandas-compatible, NumPy-backed, faster."""
from __future__ import annotations

from typing import Any, Callable
import numpy as np

from .utils import as_array, as_index, isna_mask, factorize


class _ILocS:
    def __init__(self, s: "Series"):
        self._s = s

    def __getitem__(self, key):
        s = self._s
        d = s._data
        idx = s._index
        if isinstance(key, (int, np.integer)):
            k = int(key)
            if k < 0:
                k += len(d)
            return d[k].item() if hasattr(d[k], "item") else d[k]
        res_d = d[key]
        res_i = idx[key] if len(idx) == len(d) else np.arange(len(np.asarray(res_d)))
        if np.ndim(res_d) == 0:
            return res_d.item() if hasattr(res_d, "item") else res_d
        return Series(np.asarray(res_d), index=np.asarray(res_i), name=s._name)


class _LocS:
    def __init__(self, s: "Series"):
        self._s = s

    def __getitem__(self, key):
        s = self._s
        # label-based for non-default index, else positional
        if isinstance(key, slice):
            # try label slice
            try:
                idx = list(s._index)
                start = idx.index(key.start) if key.start is not None else None
                stop = idx.index(key.stop) if key.stop is not None else None
                sl = slice(start, (stop + 1) if stop is not None else None, key.step)
                d = s._data[sl]
                ii = s._index[sl]
                return Series(np.asarray(d), index=np.asarray(ii), name=s._name)
            except Exception:
                d = s._data[key]
                return Series(np.asarray(d), index=np.asarray(s._index[key]), name=s._name)
        if isinstance(key, (list, np.ndarray)):
            ka = np.asarray(key)
            if ka.dtype == bool:
                return Series(s._data[ka], index=s._index[ka], name=s._name)
            # labels?
            try:
                pos = [list(s._index).index(k) for k in ka]
                return Series(s._data[pos], index=s._index[pos], name=s._name)
            except Exception:
                return Series(s._data[ka], index=s._index[np.asarray(ka)], name=s._name)
        # scalar label
        try:
            pos = list(s._index).index(key)
            v = s._data[pos]
            return v.item() if hasattr(v, "item") else v
        except ValueError:
            raise KeyError(key)


class Series:
    def __init__(self, data=None, index=None, name=None, dtype=None):
        if isinstance(data, Series):
            arr = np.asarray(data._data, dtype=dtype) if dtype else np.array(data._data, copy=True)
            idx = np.array(data._index, copy=True) if index is None else as_index(index, len(arr))
            self._name = name if name is not None else data._name
        elif isinstance(data, dict):
            # dict -> values ordered, keys as index if index None
            keys = list(data.keys())
            vals = list(data.values())
            arr = as_array(vals, dtype=dtype)
            idx = as_index(keys if index is None else index, len(arr))
            self._name = name
        else:
            if data is None:
                arr = np.array([], dtype=dtype if dtype else np.float64)
            else:
                # keep pandas Series support
                try:
                    import pandas as pd
                    if isinstance(data, pd.Series):
                        arr = data.to_numpy(dtype=dtype, na_value=np.nan) if dtype else data.to_numpy(na_value=np.nan)
                        idx0 = data.index.to_numpy()
                        self._name = name if name is not None else data.name
                        self._data = np.asarray(arr)
                        self._index = as_index(idx0 if index is None else index, len(arr))
                        return
                except ImportError:
                    pass
                arr = as_array(data, dtype=dtype)
                if arr.shape == ():
                    arr = arr.reshape(1)
                if dtype is None and arr.dtype.kind in "SU":
                    arr = arr.astype(object)
            idx = as_index(index, len(arr))
            self._name = name
        self._data = arr
        self._index = idx

    # ---- basics ----
    @property
    def values(self): return np.asarray(self._data)
    @property
    def index(self): return self._index
    @index.setter
    def index(self, v): self._index = as_index(v, len(self._data))
    @property
    def name(self): return self._name
    @name.setter
    def name(self, v): self._name = v
    @property
    def dtype(self): return self._data.dtype
    @property
    def shape(self): return self._data.shape
    @property
    def size(self): return self._data.size
    @property
    def ndim(self): return 1
    @property
    def iloc(self): return _ILocS(self)
    @property
    def loc(self): return _LocS(self)
    @property
    def str(self):
        from .strings import StringAccessor
        return StringAccessor(self)
    @property
    def dt(self):
        from .datetimes import DatetimeAccessor
        return DatetimeAccessor(self)
    @property
    def T(self): return self
    @property
    def array(self): return np.asarray(self._data)

    def __len__(self): return len(self._data)
    def __iter__(self): return iter(self._data)
    def __array__(self, dtype=None, copy=None):
        a = np.asarray(self._data, dtype=dtype)
        return a.copy() if copy else a

    def __repr__(self):
        n = len(self._data)
        lines = []
        show = min(n, 10)
        for i in range(show if n <= 10 else 5):
            lines.append(f"{self._index[i]}    {self._data[i]}")
        if n > 10:
            lines.append("...")
            for i in range(n - 5, n):
                lines.append(f"{self._index[i]}    {self._data[i]}")
        lines.append(f"Name: {self._name}, dtype: {self._data.dtype}")
        return "\n".join(lines)

    # ---- get/set ----
    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            k = int(key)
            v = self._data[k]
            return v.item() if hasattr(v, "item") else v
        if isinstance(key, slice):
            return Series(self._data[key], index=self._index[key], name=self._name)
        ka = np.asarray(key)
        if ka.dtype == bool:
            return Series(self._data[ka], index=self._index[ka], name=self._name)
        return Series(self._data[ka], index=self._index[ka], name=self._name)

    def __setitem__(self, key, value):
        self._data[key] = value

    # ---- binary ops (positional, vectorized — fastest path, zero-copy index) ----
    def _binop(self, other, op):
        if isinstance(other, Series):
            o = np.asarray(other._data)
        else:
            o = other
        return Series(op(np.asarray(self._data), o), index=self._index, name=self._name)

    def __add__(self, o): return self._binop(o, np.add)
    def __radd__(self, o): return Series(np.add(o, np.asarray(self._data)), index=self._index, name=self._name)
    def __sub__(self, o): return self._binop(o, np.subtract)
    def __rsub__(self, o): return Series(np.subtract(o, np.asarray(self._data)), index=self._index, name=self._name)
    def __mul__(self, o): return self._binop(o, np.multiply)
    def __rmul__(self, o): return Series(np.multiply(o, np.asarray(self._data)), index=self._index, name=self._name)
    def __truediv__(self, o): return self._binop(o, np.true_divide)
    def __floordiv__(self, o): return self._binop(o, np.floor_divide)
    def __mod__(self, o): return self._binop(o, np.remainder)
    def __pow__(self, o): return self._binop(o, np.power)
    def __eq__(self, o): return self._binop(o, np.equal)
    def __ne__(self, o): return self._binop(o, np.not_equal)
    def __lt__(self, o): return self._binop(o, np.less)
    def __le__(self, o): return self._binop(o, np.less_equal)
    def __gt__(self, o): return self._binop(o, np.greater)
    def __ge__(self, o): return self._binop(o, np.greater_equal)
    def __neg__(self): return Series(-np.asarray(self._data), index=self._index.copy(), name=self._name)
    def __invert__(self): return Series(~np.asarray(self._data), index=self._index.copy(), name=self._name)

    # ---- reductions ----
    def sum(self, skipna=True):
        a = self._data
        if a.dtype.kind in "iu":
            return a.sum().item() if hasattr(a.sum(), "item") else a.sum()
        if a.dtype.kind in "b":
            return int(a.sum())
        if a.dtype.kind in "f":
            if skipna:
                # fast path: plain sum is ~2x faster than nansum when no NaNs
                s = float(np.sum(a))
                if s == s:  # not NaN -> no NaN present
                    return s
                return float(np.nansum(a))
            return float(np.sum(a))
        if skipna and a.dtype.kind in "cm":
            return float(np.nansum(a))
        try:
            return a.sum()
        except TypeError:
            # object dtype with None
            s = 0
            for v in a:
                if skipna and v is None:
                    continue
                try:
                    if skipna and isinstance(v, float) and np.isnan(v):
                        continue
                except Exception:
                    pass
                s += v
            return s

    def mean(self, skipna=True):
        a = self._data
        if a.dtype.kind in "iu":
            return float(np.mean(a))
        if a.dtype.kind in "f":
            if skipna:
                m = float(np.mean(a))
                if m == m:
                    return m
                return float(np.nanmean(a))
            return float(np.mean(a))
        if a.dtype.kind in "cm":
            return float(np.nanmean(a)) if skipna else float(np.mean(a))
        import pandas as pd
        return float(pd.Series(np.asarray(a)).mean(skipna=skipna))

    def min(self, skipna=True):
        a = np.asarray(self._data)
        if a.dtype.kind in "f":
            return float(np.nanmin(a)) if skipna else float(np.min(a))
        if skipna:
            m = isna_mask(a)
            a = a[~m]
        return a.min() if len(a) else np.nan

    def max(self, skipna=True):
        a = np.asarray(self._data)
        if a.dtype.kind in "f":
            return float(np.nanmax(a)) if skipna else float(np.max(a))
        if skipna:
            m = isna_mask(a)
            a = a[~m]
        return a.max() if len(a) else np.nan

    def std(self, ddof=1, skipna=True):
        a = np.asarray(self._data, dtype=np.float64)
        return float(np.nanstd(a, ddof=ddof))

    def var(self, ddof=1, skipna=True):
        a = np.asarray(self._data, dtype=np.float64)
        return float(np.nanvar(a, ddof=ddof))

    def count(self):
        return int((~isna_mask(np.asarray(self._data))).sum())

    def any(self): return bool(np.asarray(self._data).any())
    def all(self): return bool(np.asarray(self._data).all())

    # ---- missing ----
    def isna(self): return Series(isna_mask(np.asarray(self._data)), index=self._index.copy(), name=self._name)
    def isnull(self): return self.isna()
    def notna(self): return Series(~isna_mask(np.asarray(self._data)), index=self._index.copy(), name=self._name)
    def notnull(self): return self.notna()

    def fillna(self, value):
        a = np.array(self._data, copy=True)
        m = isna_mask(a)
        if m.any():
            try:
                a[m] = value
            except Exception:
                a = a.astype(object)
                a[m] = value
        return Series(a, index=self._index.copy(), name=self._name)

    def dropna(self):
        m = ~isna_mask(np.asarray(self._data))
        return Series(np.asarray(self._data)[m], index=np.asarray(self._index)[m], name=self._name)

    # ---- transforms ----
    def astype(self, dtype):
        return Series(np.asarray(self._data).astype(dtype), index=self._index.copy(), name=self._name)

    def copy(self, deep=True):
        if deep:
            return Series(np.array(self._data, copy=True), index=np.array(self._index, copy=True), name=self._name)
        return Series(np.asarray(self._data), index=np.asarray(self._index), name=self._name)

    def sort_values(self, ascending=True):
        idx = np.argsort(np.asarray(self._data), kind="quicksort")
        if not ascending:
            idx = idx[::-1]
        return Series(np.asarray(self._data)[idx], index=np.asarray(self._index)[idx], name=self._name)

    def sort_index(self, ascending=True):
        idx = np.argsort(np.asarray(self._index), kind="stable")
        if not ascending:
            idx = idx[::-1]
        return Series(np.asarray(self._data)[idx], index=np.asarray(self._index)[idx], name=self._name)

    def reset_index(self, drop=False):
        if drop:
            return Series(np.asarray(self._data), name=self._name)
        from .dataframe import DataFrame
        return DataFrame({"index": np.asarray(self._index), self._name or 0: np.asarray(self._data)})

    def head(self, n=5): return self.iloc[:n]
    def tail(self, n=5): return self.iloc[len(self) - n:] if len(self) >= n else self.iloc[:]

    def unique(self): return np.unique(np.asarray(self._data))
    def nunique(self, dropna=True):
        a = np.asarray(self._data)
        if dropna:
            a = a[~isna_mask(a)]
        return len(np.unique(a))

    def value_counts(self, ascending=False):
        from .dataframe import DataFrame
        vals, counts = np.unique(np.asarray(self._data), return_counts=True)
        order = np.argsort(counts, kind="stable")
        if not ascending:
            order = order[::-1]
        return Series(counts[order], index=vals[order], name="count")

    def map(self, arg):
        a = np.asarray(self._data)
        if isinstance(arg, dict):
            v = np.array([arg.get(x, np.nan) for x in a], dtype=object)
            # try downcast
            try:
                v = v.astype(np.asarray(list(arg.values())).dtype)
            except Exception:
                pass
            return Series(v, index=self._index.copy(), name=self._name)
        f: Callable = arg
        return Series(np.array([f(x) for x in a], dtype=object), index=self._index.copy(), name=self._name)

    def apply(self, func, **kw):
        return Series(np.array([func(x) for x in np.asarray(self._data)], dtype=object),
                      index=self._index.copy(), name=self._name)

    def to_numpy(self, dtype=None, copy=False, na_value=np.nan):
        a = np.asarray(self._data, dtype=dtype)
        return a.copy() if copy else a

    def to_list(self): return list(np.asarray(self._data))
    def tolist(self): return list(np.asarray(self._data))

    def equals(self, other):
        if not isinstance(other, Series):
            return False
        a = np.asarray(self._data)
        b = np.asarray(other._data)
        if a.shape != b.shape:
            return False
        try:
            return bool(np.array_equal(a, b, equal_nan=True))
        except TypeError:
            # object / string dtypes: isnan not supported
            return bool(np.array_equal(a, b))

    def corr(self, other):
        a = np.asarray(self._data, dtype=np.float64)
        b = np.asarray(other._data if isinstance(other, Series) else other, dtype=np.float64)
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() < 2:
            return np.nan
        return float(np.corrcoef(a[m], b[m])[0, 1])

    def cov(self, other):
        a = np.asarray(self._data, dtype=np.float64)
        b = np.asarray(other._data if isinstance(other, Series) else other, dtype=np.float64)
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() < 2:
            return np.nan
        return float(np.cov(a[m], b[m])[0, 1])

    def autocorr(self, lag=1):
        a = np.asarray(self._data, dtype=np.float64)
        if len(a) <= lag: return np.nan
        x, y = a[:-lag], a[lag:]
        m = ~(np.isnan(x) | np.isnan(y))
        if m.sum() < 2: return np.nan
        return float(np.corrcoef(x[m], y[m])[0, 1])

    def quantile(self, q=0.5):
        a = np.asarray(self._data, dtype=np.float64)
        if np.ndim(q) == 0: return float(np.nanquantile(a, q))
        return Series(np.nanquantile(a, q), index=np.asarray(q))

    def median(self, skipna=True):
        a = np.asarray(self._data, dtype=np.float64)
        return float(np.nanmedian(a)) if skipna else float(np.median(a))

    def skew(self, skipna=True):
        try:
            import pandas as pd
            return float(pd.Series(np.asarray(self._data)).skew(skipna=skipna))
        except Exception:
            a = np.asarray(self._data, dtype=np.float64); a = a[~np.isnan(a)]
            m, s = a.mean(), a.std()
            return float(np.mean(((a - m) / (s or 1)) ** 3)) if len(a) else np.nan

    def kurt(self, skipna=True):
        try:
            import pandas as pd
            return float(pd.Series(np.asarray(self._data)).kurt(skipna=skipna))
        except Exception:
            a = np.asarray(self._data, dtype=np.float64); a = a[~np.isnan(a)]
            m, s = a.mean(), a.std()
            return float(np.mean(((a - m) / (s or 1)) ** 4) - 3) if len(a) else np.nan

    def sem(self, ddof=1):
        a = np.asarray(self._data, dtype=np.float64)
        n = int((~np.isnan(a)).sum())
        return float(np.nanstd(a, ddof=ddof) / np.sqrt(n)) if n else np.nan

    def mode(self):
        vals, counts = np.unique(np.asarray(self._data), return_counts=True)
        return Series(vals[counts == counts.max()], name=self._name)

    def rank(self, ascending=True, method="average"):
        import pandas as pd
        return Series(np.asarray(pd.Series(np.asarray(self._data)).rank(ascending=ascending, method=method)),
                      index=np.asarray(self._index.copy()), name=self._name)

    def clip(self, lower=None, upper=None):
        return Series(np.clip(np.asarray(self._data).astype(np.float64) if np.asarray(self._data).dtype.kind in "iufc" else np.asarray(self._data), lower, upper),
                      index=np.asarray(self._index.copy()), name=self._name)

    def round(self, decimals=0):
        return Series(np.round(np.asarray(self._data, dtype=np.float64), decimals),
                      index=np.asarray(self._index.copy()), name=self._name)

    def replace(self, to_replace, value=None):
        a = np.array(np.asarray(self._data, dtype=object), copy=True)
        if isinstance(to_replace, dict):
            for k, v in to_replace.items(): a[a == k] = v
        else:
            a[a == to_replace] = value
        try: a = a.astype(np.asarray(self._data).dtype)
        except Exception: pass
        return Series(a, index=np.asarray(self._index.copy()), name=self._name)

    def isin(self, values):
        s = set(np.asarray(values._data) if isinstance(values, Series) else np.asarray(values))
        s = set(list(s) + [str(x) for x in s])
        return Series(np.array([x in s or str(x) in s for x in np.asarray(self._data)], dtype=bool),
                      index=np.asarray(self._index.copy()), name=self._name)

    def between(self, left, right, inclusive="both"):
        a = np.asarray(self._data)
        try: a = a.astype(np.float64)
        except Exception: pass
        m = np.ones(len(a), dtype=bool)
        try: m &= (a >= left) if inclusive in ("both", "left") else (a > left)
        except Exception: pass
        try: m &= (a <= right) if inclusive in ("both", "right") else (a < right)
        except Exception: pass
        return Series(m, index=np.asarray(self._index.copy()), name=self._name)

    def duplicated(self, keep="first"):
        import pandas as pd
        return Series(np.asarray(pd.Series(np.asarray(self._data)).duplicated(keep=keep)),
                      index=np.asarray(self._index.copy()), name=self._name)

    def drop_duplicates(self, keep="first"):
        m = ~np.asarray(self.duplicated(keep=keep)._data, dtype=bool)
        return Series(np.asarray(self._data)[m], index=np.asarray(self._index)[m], name=self._name)

    def shift(self, periods=1):
        a = np.asarray(self._data)
        use_float = a.dtype.kind in "iu" and periods != 0
        out = np.empty(len(a), dtype=np.float64 if use_float else a.dtype)
        fill = np.nan if out.dtype.kind in "f" else None
        if use_float: a = a.astype(np.float64)
        if periods >= 0:
            out[:periods] = fill
            out[periods:] = a[:len(a) - periods] if periods else a
        else:
            out[periods:] = fill
            out[:periods] = a[-periods:]
        if not use_float:
            try: out = out.astype(a.dtype)
            except Exception: pass
        return Series(out, index=np.asarray(self._index.copy()), name=self._name)

    def diff(self, periods=1):
        a = np.asarray(self._data, dtype=np.float64)
        out = np.full(len(a), np.nan); out[periods:] = a[periods:] - a[:-periods] if periods >= 0 else a[:periods] - a[-periods:]
        return Series(out, index=np.asarray(self._index.copy()), name=self._name)

    def pct_change(self, periods=1):
        a = np.asarray(self._data, dtype=np.float64)
        out = np.full(len(a), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[periods:] = (a[periods:] - a[:-periods]) / np.where(a[:-periods] == 0, np.nan, a[:-periods])
        return Series(out, index=np.asarray(self._index.copy()), name=self._name)

    def cumsum(self, skipna=True):
        a = np.asarray(self._data, dtype=np.float64)
        if skipna:
            m = np.isnan(a); out = np.cumsum(np.where(m, 0, a)); out[m] = np.nan if m.all() else out[m]
            # propagate correctly: simple nancumsum
            acc, out = 0.0, np.full(len(a), np.nan)
            for i, v in enumerate(a):
                if v == v: acc += v; out[i] = acc
            return Series(out, index=np.asarray(self._index.copy()), name=self._name)
        return Series(np.cumsum(a), index=np.asarray(self._index.copy()), name=self._name)

    def cumprod(self, skipna=True):
        a = np.asarray(self._data, dtype=np.float64)
        acc, out = 1.0, np.full(len(a), np.nan)
        for i, v in enumerate(a):
            if v == v or not skipna: acc *= v; out[i] = acc
        return Series(out, index=np.asarray(self._index.copy()), name=self._name)

    def cummax(self):
        a = np.asarray(self._data, dtype=np.float64)
        return Series(np.maximum.accumulate(np.where(np.isnan(a), -np.inf, a)), index=np.asarray(self._index.copy()), name=self._name)

    def cummin(self):
        a = np.asarray(self._data, dtype=np.float64)
        return Series(np.minimum.accumulate(np.where(np.isnan(a), np.inf, a)), index=np.asarray(self._index.copy()), name=self._name)

    def ffill(self, limit=None):
        a = np.array(np.asarray(self._data), copy=True); last = None; n = 0
        for i, v in enumerate(a):
            isna = v is None or (isinstance(v, float) and np.isnan(v))
            if isna:
                if last is not None and (limit is None or n < limit): a[i] = last; n += 1
            else: last = v; n = 0
        return Series(a, index=np.asarray(self._index.copy()), name=self._name)
    ffill_alias = ffill
    def bfill(self, limit=None):
        return self[::-1].ffill(limit=limit)[::-1]
    def interpolate(self, method="linear", limit_direction="both"):
        import pandas as pd
        return Series(np.asarray(pd.Series(np.asarray(self._data, dtype=np.float64)).interpolate(method=method, limit_direction=limit_direction)),
                      index=np.asarray(self._index.copy()), name=self._name)

    def rolling(self, window, min_periods=None, center=False):
        from .window import Rolling
        return Rolling(self, window, min_periods, center)
    def expanding(self, min_periods=1):
        from .window import Expanding
        return Expanding(self, min_periods)
    def ewm(self, span=None, alpha=None, adjust=True, min_periods=0):
        from .window import EWM
        return EWM(self, span=span, alpha=alpha, adjust=adjust, min_periods=min_periods)

    def nlargest(self, n=5): return self.sort_values(ascending=False).iloc[:n]
    def nsmallest(self, n=5): return self.sort_values(ascending=True).iloc[:n]

    def sample(self, n=None, frac=None, replace=False, random_state=None):
        rng = np.random.default_rng(random_state)
        N = len(self); k = int(frac * N) if frac is not None else (n or 1)
        idx = rng.choice(N, size=k, replace=replace)
        return Series(np.asarray(self._data)[idx], index=np.asarray(self._index)[idx], name=self._name)

    def pipe(self, func, *a, **k): return func(self, *a, **k)
    def to_frame(self, name=None):
        from .dataframe import DataFrame
        return DataFrame({name or self._name or 0: np.asarray(self._data)}, index=np.asarray(self._index))
    def explode(self):
        idx, vals = [], []
        for i, v in zip(np.asarray(self._index), np.asarray(self._data)):
            if isinstance(v, (list, tuple, np.ndarray)):
                for x in v: idx.append(i); vals.append(x)
            else: idx.append(i); vals.append(v)
        return Series(np.array(vals, dtype=object), index=np.array(idx, dtype=object), name=self._name)
    def combine_first(self, other):
        from .utils import isna_mask
        a = np.array(np.asarray(self._data, dtype=object), copy=True)
        b = np.asarray(other._data if isinstance(other, Series) else other)
        m = isna_mask(np.asarray(self._data))
        for i in np.where(m)[0]:
            if i < len(b): a[i] = b[i]
        return Series(a, index=np.asarray(self._index.copy()), name=self._name)
    def update(self, other):
        b = np.asarray(other._data if isinstance(other, Series) else other)
        n = min(len(self._data), len(b))
        self._data[:n] = b[:n]
    def memory_usage(self, deep=True):
        return int(np.asarray(self._data).nbytes + np.asarray(self._index).nbytes)
    def plot(self, kind="line", ax=None, **kw):
        from . import viz as _viz
        return {"line": _viz.line, "bar": _viz.bar, "hist": _viz.hist}[kind](self, ax=ax, **kw)
    def hist(self, bins=30, ax=None): return self.plot(kind="hist", ax=ax)
    def to_torch(self, dtype=None):
        from .dl import to_torch as _t
        return _t(self, dtype=dtype)

    # ---- pandas interop ----
    def to_pandas(self):
        import pandas as pd
        return pd.Series(np.asarray(self._data), index=np.asarray(self._index), name=self._name)

    # groupby on series level -> delegate
    def groupby(self, by):
        from .groupby import SeriesGroupBy
        if isinstance(by, Series):
            keys = np.asarray(by._data)
        else:
            keys = np.asarray(by)
        return SeriesGroupBy(np.asarray(self._data), keys, name=self._name)
