"""bandas.DataFrame — pandas-compatible, columnar NumPy, faster."""
from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

from .utils import as_index, default_index, isna_mask, factorize
from .series import Series


class _ILocD:
    def __init__(self, df: "DataFrame"):
        self._df = df

    def __getitem__(self, key):
        df = self._df
        if isinstance(key, (int, np.integer)):
            k = int(key)
            row = {c: df._cols[c][k] for c in df._columns}
            return row
        if isinstance(key, tuple):
            rk, ck = key
            # rows
            if isinstance(ck, (int, np.integer)):
                col = df._columns[int(ck)]
                arr = df._cols[col]
                if isinstance(rk, (int, np.integer)):
                    return arr[rk]
                return arr[rk]
            elif isinstance(ck, slice):
                cols = df._columns[ck]
                if isinstance(rk, (int, np.integer)):
                    return {c: df._cols[c][rk] for c in cols}
                rpos = np.arange(len(df))[rk]
                return df._take_rows_cols(rpos, cols)
            else:  # list of cols
                cols = [df._columns[c] if isinstance(c, (int, np.integer)) else c for c in ck]
                rpos = np.arange(len(df))[rk] if not isinstance(rk, (int, np.integer)) else np.array([rk])
                return df._take_rows_cols(rpos, cols)
        # row-only: slice / mask / list
        rpos = np.arange(len(df))[key]
        return df._take_rows_cols(np.asarray(rpos), df._columns)


class _LocD:
    def __init__(self, df: "DataFrame"):
        self._df = df

    def _row_pos(self, key):
        df = self._df
        if isinstance(key, slice):
            # label slice if index non-default
            try:
                il = list(df._index)
                s = il.index(key.start) if key.start is not None else 0
                e = il.index(key.stop) if key.stop is not None else len(il) - 1
                return np.arange(s, e + 1, key.step or 1)
            except Exception:
                return np.arange(len(df))[key]
        ka = np.asarray(key) if not isinstance(key, str) else None
        if ka is not None and ka.dtype == bool:
            return np.where(ka)[0]
        if isinstance(key, (list, np.ndarray)):
            il = list(df._index)
            try:
                return np.array([il.index(k) for k in np.asarray(key)])
            except Exception:
                return np.asarray(key)
        # scalar label
        il = list(df._index)
        return il.index(key)

    def __getitem__(self, key):
        df = self._df
        if isinstance(key, tuple):
            rk, ck = key
            rpos = self._row_pos(rk)
            if isinstance(ck, str):
                cols = [ck]
                single_col = True
            elif isinstance(ck, slice):
                cols = df._columns[ck]
                single_col = False
            elif isinstance(ck, list):
                cols = ck
                single_col = False
            else:
                cols = [ck]
                single_col = True
            if np.ndim(rpos) == 0:
                r = int(rpos)
                if single_col:
                    return df._cols[cols[0]][r]
                return {c: df._cols[c][r] for c in cols}
            if single_col:
                return Series(df._cols[cols[0]][np.asarray(rpos)],
                              index=np.asarray(df._index)[np.asarray(rpos)], name=cols[0])
            return df._take_rows_cols(np.asarray(rpos), cols)
        rpos = self._row_pos(key)
        if np.ndim(rpos) == 0:
            return {c: df._cols[c][int(rpos)] for c in df._columns}
        return df._take_rows_cols(np.asarray(rpos), df._columns)


class DataFrame:
    def __init__(self, data=None, columns=None, index=None, dtype=None):
        self._columns: List[str] = []
        self._cols: Dict[str, np.ndarray] = {}
        self._index: np.ndarray = np.array([])

        if data is None:
            self._columns = list(columns) if columns else []
            self._cols = {c: np.array([]) for c in self._columns}
            self._index = as_index(index, 0) if index is not None else np.array([])
            return

        # from pandas
        try:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                cols = list(data.columns) if columns is None else list(columns)
                n = len(data)
                for c in cols:
                    a = data[c].to_numpy(na_value=np.nan)
                    self._cols[c] = np.asarray(a, dtype=dtype) if dtype else np.asarray(a)
                self._columns = cols
                self._index = as_index(data.index.to_numpy() if index is None else index, n)
                return
        except ImportError:
            pass

        if isinstance(data, DataFrame):
            cols = list(data._columns) if columns is None else list(columns)
            for c in cols:
                a = np.asarray(data._cols[c])
                self._cols[c] = a.astype(dtype) if dtype else np.array(a, copy=True)
            self._columns = cols
            n = len(data)
            self._index = np.array(data._index, copy=True) if index is None else as_index(index, n)
            return

        if isinstance(data, dict):
            cols = list(data.keys()) if columns is None else list(columns)
            tmp: Dict[str, np.ndarray] = {}
            n = None
            for c in cols:
                v = data[c]
                if isinstance(v, Series):
                    a = np.asarray(v._data)
                else:
                    try:
                        import pandas as pd
                        if isinstance(v, pd.Series):
                            a = v.to_numpy(na_value=np.nan)
                        else:
                            a = np.asarray(v)
                    except ImportError:
                        a = np.asarray(v)
                if a.shape == ():
                    a = np.full(1, a.item(), dtype=object)
                if a.dtype.kind in "SU":
                    # store strings as object (like pandas): hashing/factorize 5x faster
                    a = a.astype(object)
                if a.dtype == object:
                    # coerce None/mixed-numeric to float with NaN (pandas-like)
                    try:
                        flat = list(a)
                        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in flat) and all(
                                v is None or isinstance(v, (int, float, np.integer, np.floating)) or (isinstance(v, float) and np.isnan(v)) for v in flat):
                            a = np.array([np.nan if v is None or (isinstance(v, float) and np.isnan(v)) else float(v) for v in flat], dtype=np.float64)
                    except Exception:
                        pass
                if dtype:
                    a = a.astype(dtype)
                tmp[c] = a
                n = len(a) if n is None else n
                if len(a) != n:
                    raise ValueError("all columns must have same length")
            # Series index alignment: ignore for speed, use positional; keep first Series index if any
            ref_idx = None
            for c in cols:
                if isinstance(data.get(c), Series):
                    ref_idx = np.asarray(data[c]._index)
                    break
            self._cols = tmp
            self._columns = cols
            n = n or 0
            self._index = as_index(ref_idx if index is None else index, n) if (ref_idx is not None or index is not None) else default_index(n)
            return

        if isinstance(data, (list, tuple)):
            arr = np.asarray(data, dtype=dtype) if not any(isinstance(r, (list, tuple, np.ndarray)) and any(isinstance(x, str) for x in np.asarray(r).ravel()) for r in data) else None
            # list of dicts?
            if data and isinstance(data[0], dict):
                cols = sorted({k for r in data for k in r}) if columns is None else list(columns)
                tmp = {c: np.array([r.get(c, np.nan) for r in data]) for c in cols}
                if dtype:
                    tmp = {c: v.astype(dtype) for c, v in tmp.items()}
                self._cols = tmp
                self._columns = cols
                self._index = as_index(index, len(data)) if index is not None else default_index(len(data))
                return
            if arr is None:
                arr = np.array(data, dtype=object)
            if arr.ndim == 1:
                cols = list(columns) if columns else [0]
                self._cols = {cols[0]: arr.astype(dtype) if dtype else arr}
                self._columns = cols
                self._index = as_index(index, len(arr)) if index is not None else default_index(len(arr))
                return
            # 2D
            n, m = arr.shape
            cols = list(columns) if columns else list(range(m))
            self._columns = cols
            self._cols = {c: (arr[:, j].astype(dtype) if dtype else arr[:, j]) for j, c in enumerate(cols)}
            self._index = as_index(index, n) if index is not None else default_index(n)
            return

        if isinstance(data, np.ndarray):
            arr = data.astype(dtype) if dtype else data
            if arr.ndim == 1:
                cols = list(columns) if columns else [0]
                self._cols = {cols[0]: arr}
                self._columns = cols
                self._index = as_index(index, len(arr)) if index is not None else default_index(len(arr))
                return
            n, m = arr.shape
            cols = list(columns) if columns else list(range(m))
            self._columns = cols
            self._cols = {c: arr[:, j] for j, c in enumerate(cols)}
            self._index = as_index(index, n) if index is not None else default_index(n)
            return

        raise TypeError(f"unsupported data type {type(data)}")

    # ---------- core props ----------
    @property
    def columns(self): return self._columns
    @columns.setter
    def columns(self, v):
        v = list(v)
        if len(v) != len(self._columns):
            raise ValueError("columns length mismatch")
        self._cols = {nc: self._cols[oc] for nc, oc in zip(v, self._columns)}
        self._columns = v

    @property
    def index(self): return self._index
    @index.setter
    def index(self, v): self._index = as_index(v, len(self))

    @property
    def shape(self): return (len(self), len(self._columns))
    @property
    def size(self): return len(self) * len(self._columns)
    @property
    def ndim(self): return 2
    @property
    def empty(self): return len(self) == 0 or len(self._columns) == 0
    @property
    def dtypes(self):
        from .series import Series as _S
        return {c: self._cols[c].dtype for c in self._columns}
    @property
    def values(self): return self.to_numpy()
    @property
    def iloc(self): return _ILocD(self)
    @property
    def loc(self): return _LocD(self)

    def __len__(self): return len(self._index)
    def __contains__(self, k): return k in self._cols

    def __repr__(self):
        import io as _io
        try:
            pdf = self.to_pandas()
            with _io.StringIO() as b:
                b.write(repr(pdf.head(10)))
                if len(self) > 10:
                    b.write(f"\n\n[{len(self)} rows x {len(self._columns)} columns]")
                return b.getvalue()
        except Exception:
            return f"bandas.DataFrame shape={self.shape} columns={self._columns}"

    # ---------- selection ----------
    def __getitem__(self, key):
        if isinstance(key, str):
            # zero-copy view: share column + index (fast path)
            return Series(np.asarray(self._cols[key]), index=self._index, name=key)
        if isinstance(key, list):
            if all(isinstance(k, str) for k in key):
                return self._take_rows_cols(np.arange(len(self)), key)
            # boolean mask list?
            ka = np.asarray(key)
            if ka.dtype == bool:
                return self._take_rows_cols(np.where(ka)[0], self._columns)
            return self._take_rows_cols(np.asarray(key), self._columns)
        if isinstance(key, slice):
            return self._take_rows_cols(np.arange(len(self))[key], self._columns)
        ka = np.asarray(key)
        if ka.dtype == bool:
            return self._take_rows_cols(np.where(ka)[0], self._columns)
        if isinstance(key, Series) and key.dtype == bool:
            return self._take_rows_cols(np.where(np.asarray(key._data))[0], self._columns)
        raise KeyError(key)

    def __setitem__(self, key, value):
        n = len(self)
        if isinstance(key, str):
            if isinstance(value, Series):
                a = np.asarray(value._data)
            else:
                try:
                    import pandas as pd
                    if isinstance(value, pd.Series):
                        a = value.to_numpy(na_value=np.nan)
                    else:
                        a = np.asarray(value)
                except ImportError:
                    a = np.asarray(value)
            if a.shape == ():
                a = np.full(n if n else 1, a.item())
            if n and len(a) != n:
                raise ValueError("length mismatch")
            if key not in self._cols:
                self._columns.append(key)
            self._cols[key] = a
            if n == 0:
                self._index = default_index(len(a))
            return
        if isinstance(key, list):
            for k, v in zip(key, value if isinstance(value, (list, tuple)) else [value] * len(key)):
                self[k] = v
            return
        raise KeyError(key)

    def _take_rows_cols(self, rpos, cols):
        nd = DataFrame.__new__(DataFrame)
        nd._columns = list(cols)
        nd._cols = {c: np.asarray(self._cols[c])[np.asarray(rpos)] for c in cols}
        nd._index = np.asarray(self._index)[np.asarray(rpos)]
        return nd

    # ---------- conversion ----------
    def to_numpy(self, dtype=None):
        if not self._columns:
            return np.array([])
        arrs = [np.asarray(self._cols[c]) for c in self._columns]
        try:
            out = np.column_stack(arrs).astype(dtype) if dtype else np.column_stack(arrs)
        except ValueError:
            out = np.column_stack([a.astype(object) for a in arrs])
        return out

    def to_dict(self, orient="dict"):
        if orient == "dict":
            return {c: dict(zip(np.asarray(self._index), np.asarray(self._cols[c]))) for c in self._columns}
        if orient == "list":
            return {c: list(np.asarray(self._cols[c])) for c in self._columns}
        if orient == "records":
            return [{c: self._cols[c][i] for c in self._columns} for i in range(len(self))]
        raise ValueError(orient)

    def to_pandas(self):
        import pandas as pd
        d = {c: np.asarray(self._cols[c]) for c in self._columns}
        return pd.DataFrame(d, index=np.asarray(self._index), columns=list(self._columns))

    @classmethod
    def from_pandas(cls, pdf):
        return cls(pdf)

    def copy(self, deep=True):
        nd = DataFrame.__new__(DataFrame)
        nd._columns = list(self._columns)
        nd._cols = {c: np.array(v, copy=True) if deep else np.asarray(v) for c, v in self._cols.items()}
        nd._index = np.array(self._index, copy=True) if deep else np.asarray(self._index)
        return nd

    def equals(self, other):
        if not isinstance(other, DataFrame) or self._columns != other._columns:
            return False
        if not np.array_equal(np.asarray(self._index), np.asarray(other._index)):
            return False
        for c in self._columns:
            a = np.asarray(self._cols[c])
            b = np.asarray(other._cols[c])
            try:
                if not np.array_equal(a, b, equal_nan=True):
                    return False
            except TypeError:
                if not np.array_equal(a, b):
                    return False
        return True

    # ---------- inspection ----------
    def head(self, n=5): return self._take_rows_cols(np.arange(min(n, len(self))), self._columns)
    def tail(self, n=5):
        s = max(0, len(self) - n)
        return self._take_rows_cols(np.arange(s, len(self)), self._columns)

    def info(self):
        lines = [f"<bandas.DataFrame> shape={self.shape}"]
        for c in self._columns:
            a = self._cols[c]
            lines.append(f" {c}: {a.dtype} non-null={(~isna_mask(a)).sum()}/{len(a)}")
        print("\n".join(lines))

    def describe(self):
        num_cols = [c for c in self._columns if np.asarray(self._cols[c]).dtype.kind in "iufc"]
        stats = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        out = {}
        for c in num_cols:
            a = np.asarray(self._cols[c], dtype=np.float64)
            out[c] = [float(np.sum(~np.isnan(a))), float(np.nanmean(a)), float(np.nanstd(a, ddof=1)),
                      float(np.nanmin(a)), float(np.nanpercentile(a, 25)), float(np.nanpercentile(a, 50)),
                      float(np.nanpercentile(a, 75)), float(np.nanmax(a))]
        nd = DataFrame.__new__(DataFrame)
        nd._columns = num_cols
        nd._cols = {c: np.array(v) for c, v in out.items()}
        nd._index = np.array(stats)
        return nd

    # ---------- transforms ----------
    def astype(self, dtype):
        nd = self.copy()
        if isinstance(dtype, dict):
            for c, d in dtype.items():
                nd._cols[c] = np.asarray(nd._cols[c]).astype(d)
        else:
            for c in nd._columns:
                nd._cols[c] = np.asarray(nd._cols[c]).astype(dtype)
        return nd

    def rename(self, columns=None, index=None):
        nd = self.copy(deep=False)
        nd._cols = dict(nd._cols)
        nd._columns = list(nd._columns)
        if columns:
            if callable(columns):
                mp = {c: columns(c) for c in nd._columns}
            else:
                mp = dict(columns)
            nd._columns = [mp.get(c, c) for c in nd._columns]
            nd._cols = {mp.get(c, c): v for c, v in nd._cols.items()}
        if index is not None:
            nd._index = np.asarray(index) if not callable(index) else np.array([index(x) for x in nd._index])
        return nd

    def drop(self, labels=None, columns=None, index=None, axis=0):
        cols = list(self._columns)
        rpos = np.arange(len(self))
        if columns is not None:
            drop_c = [columns] if isinstance(columns, str) else list(columns)
            cols = [c for c in cols if c not in drop_c]
        if labels is not None and axis in (1, "columns"):
            drop_c = [labels] if isinstance(labels, str) else list(labels)
            cols = [c for c in cols if c not in drop_c]
        if index is not None:
            drop_i = set(np.asarray(index))
            keep = [i for i, v in enumerate(np.asarray(self._index)) if v not in drop_i]
            rpos = np.array(keep)
        if labels is not None and axis in (0, "index", None) and index is None and columns is None:
            drop_i = set(np.asarray([labels] if not isinstance(labels, list) else labels))
            keep = [i for i, v in enumerate(np.asarray(self._index)) if v not in drop_i]
            rpos = np.array(keep)
        return self._take_rows_cols(rpos, cols)

    def fillna(self, value):
        nd = self.copy()
        if isinstance(value, dict):
            for c, v in value.items():
                a = np.array(nd._cols[c], copy=True)
                m = isna_mask(a)
                if m.any():
                    try:
                        a[m] = v
                    except Exception:
                        a = a.astype(object)
                        a[m] = v
                nd._cols[c] = a
        else:
            for c in nd._columns:
                a = np.array(nd._cols[c], copy=True)
                m = isna_mask(a)
                if m.any():
                    try:
                        a[m] = value
                    except Exception:
                        a = a.astype(object)
                        a[m] = value
                nd._cols[c] = a
        return nd

    def dropna(self, axis=0, how="any"):
        if axis in (1, "columns"):
            keep = [c for c in self._columns if not isna_mask(np.asarray(self._cols[c])).any()] if how == "any" else \
                   [c for c in self._columns if not isna_mask(np.asarray(self._cols[c])).all()]
            return self._take_rows_cols(np.arange(len(self)), keep)
        masks = [~isna_mask(np.asarray(self._cols[c])) for c in self._columns]
        if not masks:
            return self.copy()
        if how == "any":
            keep = np.logical_and.reduce(masks)
        else:
            keep = np.logical_or.reduce(masks)
        return self._take_rows_cols(np.where(keep)[0], self._columns)

    def sort_values(self, by, ascending=True):
        bys = [by] if isinstance(by, str) else list(by)
        asc = [ascending] * len(bys) if isinstance(ascending, bool) else list(ascending)
        order = np.arange(len(self))
        if len(bys) == 1:
            # single-key: quicksort (same default as pandas, fastest)
            k = np.asarray(self._cols[bys[0]])
            try:
                o = np.argsort(k, kind="quicksort")
            except TypeError:
                o = np.argsort(k.astype(str), kind="quicksort")
            if not asc[0]:
                o = o[::-1]
            return self._take_rows_cols(order[o], self._columns)
        # multi-key: stable sort last-key-first for correctness
        for col, a in reversed(list(zip(bys, asc))):
            k = np.asarray(self._cols[col])
            try:
                o = np.argsort(k, kind="stable")
            except TypeError:
                o = np.argsort(k.astype(str), kind="stable")
            if not a:
                o = o[::-1]
            order = order[o]
        return self._take_rows_cols(order, self._columns)

    def sort_index(self, ascending=True):
        o = np.argsort(np.asarray(self._index), kind="stable")
        if not ascending:
            o = o[::-1]
        return self._take_rows_cols(o, self._columns)

    def reset_index(self, drop=False):
        nd = self.copy()
        if not drop:
            nd._cols = {"index": np.asarray(nd._index), **nd._cols}
            nd._columns = ["index"] + nd._columns
        nd._index = default_index(len(nd))
        return nd

    def set_index(self, col):
        idx = np.asarray(self._cols[col])
        cols = [c for c in self._columns if c != col]
        nd = self._take_rows_cols(np.arange(len(self)), cols)
        nd._index = idx
        return nd

    def assign(self, **kw):
        nd = self.copy(deep=False)
        nd._cols = dict(nd._cols)
        nd._columns = list(nd._columns)
        for k, v in kw.items():
            nd[k] = v(np.asarray(self._index)) if callable(v) else v
        return nd

    def filter(self, items=None, like=None):
        cols = list(self._columns)
        if items is not None:
            cols = [c for c in cols if c in items]
        if like is not None:
            cols = [c for c in cols if like in str(c)]
        return self._take_rows_cols(np.arange(len(self)), cols)

    def query(self, expr):
        # vectorized via eval on columns (mimics pandas query)
        env = {c: np.asarray(self._cols[c]) for c in self._columns}
        mask = eval(expr, {"np": np}, env)  # noqa: S307 - local DSL
        return self._take_rows_cols(np.where(np.asarray(mask))[0], self._columns)

    def apply(self, func, axis=0):
        if axis == 0:
            out = {c: func(np.asarray(self._cols[c])) for c in self._columns}
            # scalar per column -> single row frame
            if all(np.ndim(v) == 0 for v in out.values()):
                nd = DataFrame.__new__(DataFrame)
                nd._columns = list(out.keys())
                nd._cols = {c: np.array([out[c]]) for c in out}
                nd._index = np.array([0])
                return nd
            return DataFrame(out, index=np.asarray(self._index))
        else:
            rows = []
            for i in range(len(self)):
                row = {c: self._cols[c][i] for c in self._columns}
                try:
                    import pandas as pd
                    r = func(pd.Series(row))
                except Exception:
                    r = func(row)
                rows.append(r)
            # try frame
            try:
                return DataFrame(rows, index=np.asarray(self._index))
            except Exception:
                return Series(rows, index=np.asarray(self._index))

    def agg(self, func):
        return self.apply(func, axis=0)

    def map(self, func):
        return DataFrame({c: np.array([func(x) for x in np.asarray(self._cols[c])]) for c in self._columns},
                         index=np.asarray(self._index))

    # reductions
    def _reduce(self, fn, skipna=True, numeric_only=False):
        out = {}
        for c in self._columns:
            a = np.asarray(self._cols[c])
            if numeric_only and a.dtype.kind not in "iufc":
                continue
            out[c] = fn(Series(a))
        return Series(out)

    def sum(self, skipna=True, numeric_only=False): return self._reduce(lambda s: s.sum(skipna=skipna), skipna, numeric_only)
    def mean(self, skipna=True, numeric_only=False): return self._reduce(lambda s: s.mean(skipna=skipna), skipna, numeric_only)
    def min(self, skipna=True, numeric_only=False): return self._reduce(lambda s: s.min(skipna=skipna), skipna, numeric_only)
    def max(self, skipna=True, numeric_only=False): return self._reduce(lambda s: s.max(skipna=skipna), skipna, numeric_only)
    def std(self, numeric_only=False): return self._reduce(lambda s: s.std(), True, numeric_only)
    def count(self): return Series({c: int((~isna_mask(np.asarray(self._cols[c]))).sum()) for c in self._columns})
    def nunique(self): return Series({c: len(np.unique(np.asarray(self._cols[c]))) for c in self._columns})

    # ---------- groupby / join ----------
    def groupby(self, by):
        from .groupby import DataFrameGroupBy
        if isinstance(by, str):
            keys = np.asarray(self._cols[by])
            names = [by]
        elif isinstance(by, list):
            keys = np.column_stack([np.asarray(self._cols[c]) for c in by])
            names = by
        elif isinstance(by, Series):
            keys = np.asarray(by._data)
            names = [by.name]
        else:
            keys = np.asarray(by)
            names = [None]
        return DataFrameGroupBy(self, keys, names)

    def merge(self, right, on=None, how="inner", left_on=None, right_on=None):
        from .groupby import _merge_frames
        return _merge_frames(self, right, on=on, how=how, left_on=left_on, right_on=right_on)

    def join(self, other, how="left", lsuffix="", rsuffix="_r"):
        # index join
        l_idx, r_idx = np.asarray(self._index), np.asarray(other._index)
        if how == "left":
            pos = []
            for v in l_idx:
                f = np.where(r_idx == v)[0]
                pos.append(f[0] if len(f) else -1)
            pos = np.array(pos)
            nd_cols = list(self._columns)
            nd_data = {c: np.asarray(self._cols[c]) for c in self._columns}
            for c in other._columns:
                name = c + rsuffix if c in nd_data else c
                arr = np.empty(len(self), dtype=np.asarray(other._cols[c]).dtype if np.asarray(other._cols[c]).dtype.kind != "O" else object)
                ro = np.asarray(other._cols[c])
                for i, p in enumerate(pos):
                    arr[i] = ro[p] if p >= 0 else (np.nan if arr.dtype.kind in "f" else None)
                nd_data[name] = arr
                nd_cols.append(name)
            nd = DataFrame.__new__(DataFrame)
            nd._columns = nd_cols
            nd._cols = nd_data
            nd._index = np.asarray(self._index)
            return nd
        # fallback via pandas for other join types
        return DataFrame(self.to_pandas().join(other.to_pandas(), how=how, lsuffix=lsuffix, rsuffix=rsuffix))

    # ---------- io shortcuts ----------
    def to_csv(self, path, index=True):
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            hdr = list(self._columns)
            if index:
                hdr = ["index"] + hdr
            w.writerow(hdr)
            for i in range(len(self)):
                row = [self._cols[c][i] for c in self._columns]
                if index:
                    row = [self._index[i]] + row
                w.writerow(row)

    def to_json(self, path=None, orient="records", indent=None):
        import json
        d = self.to_dict(orient="records")
        s = json.dumps(d, indent=indent, default=str)
        if path:
            with open(path, "w", encoding="utf-8") as f: f.write(s)
            return None
        return s

    def to_excel(self, path, index=False, sheet_name="Sheet1"):
        self.to_pandas().to_excel(path, index=index, sheet_name=sheet_name)

    def to_parquet(self, path, index=False, **kw):
        self.to_pandas().to_parquet(path, index=index, **kw)

    def to_sql(self, name, con, if_exists="replace", index=False, **kw):
        from .sql import to_sql as _ts
        return _ts(self, name, con, if_exists=if_exists, index=index, **kw)

    def sql(self, query, table="df"):
        """Run SQL on this frame: df.sql("SELECT * FROM df WHERE a > 1")."""
        from .sql import sql_query
        return sql_query(query, {table: self})

    def to_torch(self, target=None, dtype=None):
        from .dl import to_torch as _t
        return _t(self, target=target, dtype=dtype)

    def to_tensorflow(self, target=None):
        from .dl import to_tensorflow as _t
        return _t(self, target=target)

    def iterrows(self):
        for i in range(len(self)):
            yield self._index[i], {c: self._cols[c][i] for c in self._columns}

    def itertuples(self):
        from collections import namedtuple
        NT = namedtuple("Row", ["Index"] + [str(c) for c in self._columns])
        for i in range(len(self)):
            yield NT(self._index[i], *[self._cols[c][i] for c in self._columns])

    # ---------- extended pandas API ----------
    def isna(self):
        from .utils import isna_mask as _m
        return DataFrame({c: _m(np.asarray(self._cols[c])) for c in self._columns}, index=np.asarray(self._index))
    isnull = isna
    def notna(self):
        from .utils import isna_mask as _m
        return DataFrame({c: ~_m(np.asarray(self._cols[c])) for c in self._columns}, index=np.asarray(self._index))
    notnull = notna

    def duplicated(self, keep="first", subset=None):
        import pandas as pd
        cols = subset if subset else None
        m = self.to_pandas().duplicated(keep=keep, subset=cols).to_numpy()
        return Series(m, index=np.asarray(self._index.copy()))

    def drop_duplicates(self, keep="first", subset=None, ignore_index=False):
        m = ~np.asarray(self.duplicated(keep=keep, subset=subset)._data, dtype=bool)
        out = self._take_rows_cols(np.where(m)[0], self._columns)
        if ignore_index: out._index = np.arange(len(out))
        return out

    def nlargest(self, n, columns):
        c = columns if isinstance(columns, str) else columns[0]
        o = np.argsort(np.asarray(self._cols[c]))[::-1][:n]
        return self._take_rows_cols(o, self._columns)

    def nsmallest(self, n, columns):
        c = columns if isinstance(columns, str) else columns[0]
        o = np.argsort(np.asarray(self._cols[c]))[:n]
        return self._take_rows_cols(o, self._columns)

    def idxmax(self, axis=0):
        if axis == 0:
            return Series({c: self._index[int(np.nanargmax(np.asarray(self._cols[c], dtype=np.float64)))] if np.asarray(self._cols[c]).dtype.kind in "iufc" else None for c in self._columns})
        return Series([self._columns[int(np.nanargmax([float(np.asarray(self._cols[c])[i]) if str(np.asarray(self._cols[c])[i]) not in ("nan", "None") else -np.inf for c in self._columns]))] for i in range(len(self))], index=np.asarray(self._index))

    def idxmin(self, axis=0):
        if axis == 0:
            return Series({c: self._index[int(np.nanargmin(np.asarray(self._cols[c], dtype=np.float64)))] if np.asarray(self._cols[c]).dtype.kind in "iufc" else None for c in self._columns})
        return Series([self._columns[int(np.nanargmin([float(np.asarray(self._cols[c])[i]) if str(np.asarray(self._cols[c])[i]) not in ("nan", "None") else np.inf for c in self._columns]))] for i in range(len(self))], index=np.asarray(self._index))

    def mode(self, axis=0):
        out = {}
        for c in self._columns:
            v, n = np.unique(np.asarray(self._cols[c]), return_counts=True)
            out[c] = [v[int(np.argmax(n))]]
        return DataFrame(out)

    def rank(self, ascending=True):
        import pandas as pd
        return DataFrame(self.to_pandas().rank(ascending=ascending))

    def quantile(self, q=0.5):
        if np.ndim(q) == 0:
            return Series({c: float(np.nanquantile(np.asarray(self._cols[c], dtype=np.float64), q)) for c in self._columns if np.asarray(self._cols[c]).dtype.kind in "iufc"})
        rows = []
        for qi in q:
            rows.append({c: float(np.nanquantile(np.asarray(self._cols[c], dtype=np.float64), qi)) for c in self._columns if np.asarray(self._cols[c]).dtype.kind in "iufc"})
        return DataFrame(rows, index=list(q))

    def skew(self):
        return Series({c: Series(np.asarray(self._cols[c])).skew() for c in self._columns if np.asarray(self._cols[c]).dtype.kind in "iufc"})
    def kurt(self):
        return Series({c: Series(np.asarray(self._cols[c])).kurt() for c in self._columns if np.asarray(self._cols[c]).dtype.kind in "iufc"})
    def median(self, numeric_only=False):
        return Series({c: float(np.nanmedian(np.asarray(self._cols[c], dtype=np.float64))) for c in self._columns if (np.asarray(self._cols[c]).dtype.kind in "iufc" or not numeric_only)})

    def corr(self, method="pearson"):
        from .stats import corr as _corr
        return _corr(self, method=method)
    def cov(self):
        from .stats import cov as _cov
        return _cov(self)

    def clip(self, lower=None, upper=None):
        out = {}
        for c in self._columns:
            a = np.asarray(self._cols[c])
            out[c] = np.clip(a.astype(np.float64), lower, upper) if a.dtype.kind in "iufc" else a
        return DataFrame(out, index=np.asarray(self._index))

    def round(self, decimals=0):
        out = {}
        for c in self._columns:
            a = np.asarray(self._cols[c])
            out[c] = np.round(a.astype(np.float64), decimals) if a.dtype.kind in "iufc" else a
        return DataFrame(out, index=np.asarray(self._index))

    def replace(self, to_replace, value=None):
        out = {}
        for c in self._columns:
            s = Series(np.asarray(self._cols[c]))
            out[c] = np.asarray(s.replace(to_replace, value)._data)
        return DataFrame(out, index=np.asarray(self._index))

    def isin(self, values):
        if isinstance(values, dict):
            out = {}
            for c in self._columns:
                s = set(values.get(c, []))
                out[c] = np.array([x in s for x in np.asarray(self._cols[c])], dtype=bool)
            return DataFrame(out, index=np.asarray(self._index))
        s = set(np.asarray(values))
        return DataFrame({c: np.array([x in s for x in np.asarray(self._cols[c])], dtype=bool) for c in self._columns}, index=np.asarray(self._index))

    def eval(self, expr):
        env = {c: np.asarray(self._cols[c]) for c in self._columns}
        out = eval(expr, {"np": np}, env)  # noqa: S307
        if np.ndim(out) == 0: return out
        if isinstance(out, np.ndarray) and out.dtype == bool:
            return self._take_rows_cols(np.where(out)[0], self._columns)
        return Series(out, index=np.asarray(self._index.copy()))

    def pipe(self, func, *a, **k): return func(self, *a, **k)

    def sample(self, n=None, frac=None, replace=False, random_state=None):
        rng = np.random.default_rng(random_state)
        N = len(self); k = int(frac * N) if frac is not None else (n or 1)
        idx = rng.choice(N, size=k, replace=replace)
        return self._take_rows_cols(np.asarray(idx), self._columns)

    def shift(self, periods=1):
        return DataFrame({c: np.asarray(Series(np.asarray(self._cols[c])).shift(periods)._data) for c in self._columns}, index=np.asarray(self._index))
    def diff(self, periods=1):
        return DataFrame({c: np.asarray(Series(np.asarray(self._cols[c]), dtype=np.float64 if np.asarray(self._cols[c]).dtype.kind in "iufc" else None).diff(periods)._data) if np.asarray(self._cols[c]).dtype.kind in "iufc" else np.asarray(self._cols[c]) for c in self._columns}, index=np.asarray(self._index))
    def pct_change(self, periods=1):
        out = {}
        for c in self._columns:
            a = np.asarray(self._cols[c])
            out[c] = np.asarray(Series(a).pct_change(periods)._data) if a.dtype.kind in "iufc" else a
        return DataFrame(out, index=np.asarray(self._index))
    def cumsum(self):
        return DataFrame({c: np.asarray(Series(np.asarray(self._cols[c])).cumsum()._data) if np.asarray(self._cols[c]).dtype.kind in "iufc" else np.asarray(self._cols[c]) for c in self._columns}, index=np.asarray(self._index))
    def cumprod(self):
        return DataFrame({c: np.asarray(Series(np.asarray(self._cols[c])).cumprod()._data) if np.asarray(self._cols[c]).dtype.kind in "iufc" else np.asarray(self._cols[c]) for c in self._columns}, index=np.asarray(self._index))
    def cummax(self):
        return DataFrame({c: np.asarray(Series(np.asarray(self._cols[c])).cummax()._data) if np.asarray(self._cols[c]).dtype.kind in "iufc" else np.asarray(self._cols[c]) for c in self._columns}, index=np.asarray(self._index))
    def cummin(self):
        return DataFrame({c: np.asarray(Series(np.asarray(self._cols[c])).cummin()._data) if np.asarray(self._cols[c]).dtype.kind in "iufc" else np.asarray(self._cols[c]) for c in self._columns}, index=np.asarray(self._index))

    def ffill(self, limit=None):
        return DataFrame({c: np.asarray(Series(np.asarray(self._cols[c])).ffill(limit=limit)._data) for c in self._columns}, index=np.asarray(self._index))
    def bfill(self, limit=None):
        return DataFrame({c: np.asarray(Series(np.asarray(self._cols[c])).bfill(limit=limit)._data) for c in self._columns}, index=np.asarray(self._index))
    pad = ffill
    backfill = bfill
    def interpolate(self, method="linear"):
        out = {}
        for c in self._columns:
            a = np.asarray(self._cols[c])
            out[c] = np.asarray(Series(a).interpolate(method=method)._data) if a.dtype.kind in "iufc" else a
        return DataFrame(out, index=np.asarray(self._index))

    def rolling(self, window, min_periods=None, center=False):
        from .window import Rolling
        return Rolling(self, window, min_periods, center)
    def expanding(self, min_periods=1):
        from .window import Expanding
        return Expanding(self, min_periods)
    def ewm(self, span=None, alpha=None, adjust=True, min_periods=0):
        from .window import EWM
        return EWM(self, span=span, alpha=alpha, adjust=adjust, min_periods=min_periods)

    def pivot_table(self, values, index, columns=None, aggfunc="mean", fill_value=None):
        from .reshape import pivot_table as _p
        return _p(self, values, index, columns, aggfunc, fill_value)
    def melt(self, id_vars=None, value_vars=None, var_name="variable", value_name="value"):
        from .reshape import melt as _m
        return _m(self, id_vars, value_vars, var_name, value_name)
    def explode(self, column):
        idx, out = [], {c: [] for c in self._columns}
        for i in range(len(self)):
            v = self._cols[column][i]
            items = list(v) if isinstance(v, (list, tuple, np.ndarray)) else [v]
            for x in items:
                idx.append(self._index[i])
                for c in self._columns: out[c].append(self._cols[c][i] if c != column else x)
        return DataFrame({c: np.array(v, dtype=object) for c, v in out.items()}, index=np.array(idx))

    def combine_first(self, other):
        out = {}
        for c in self._columns:
            a = np.array(np.asarray(self._cols[c], dtype=object), copy=True)
            if c in other._cols:
                from .utils import isna_mask as _m
                m = _m(np.asarray(self._cols[c]))
                b = np.asarray(other._cols[c])
                for i in np.where(m)[0]:
                    if i < len(b): a[i] = b[i]
            out[c] = a
        for c in other._columns:
            if c not in out: out[c] = np.asarray(other._cols[c])
        cols = list(self._columns) + [c for c in other._columns if c not in self._columns]
        return DataFrame({c: out[c] for c in cols}, index=np.asarray(self._index))

    def update(self, other):
        for c in other._columns:
            if c in self._cols:
                b = np.asarray(other._cols[c]); n = min(len(self), len(b))
                self._cols[c][:n] = b[:n]

    def memory_usage(self, deep=True):
        return {c: int(np.asarray(v).nbytes) for c, v in self._cols.items()}

    def plot(self, kind="line", x=None, y=None, ax=None, **kw):
        from . import viz as _viz
        return {"line": lambda: _viz.line(self, x=x, y=y, ax=ax), "bar": lambda: _viz.bar(self, x=x, y=y, ax=ax),
                "hist": lambda: _viz.hist(self[y] if isinstance(y, str) else self, ax=ax),
                "scatter": lambda: _viz.scatter(self, x=x or self._columns[0], y=y or self._columns[1], ax=ax),
                "box": lambda: _viz.box(self, columns=y, ax=ax)}[kind]()
    def hist(self, bins=30, ax=None):
        from . import viz as _viz
        return _viz.hist(self[self._columns[0]] if self._columns else self, bins=bins, ax=ax)
