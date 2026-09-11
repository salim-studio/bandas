"""bandas — pandas-compatible, faster DataFrame library + universal data platform.

Core (fast NumPy paths) + SQL/DB + time-series + preprocessing + ML/DL + viz + lazy.
"""
from __future__ import annotations

from .series import Series
from .dataframe import DataFrame
from .io import (
    read_csv, read_parquet, read_json, read_excel, read_html,
    read_feather, read_orc, read_sql, read_csv_chunked,
    from_pandas, to_pandas,
)
from .sql import sql_query, to_sql, Database
from .reshape import get_dummies, melt, pivot_table, crosstab, cut, qcut
from .datetimes import to_datetime, date_range
from .lazy import LazyFrame
from . import preprocessing, ml, stats, viz, dl
from .preprocessing import SimpleImputer, StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from .ml import train_test_split, KFold, accuracy_score, mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score, f1_score

__version__ = "0.3.0"
__all__ = [
    "DataFrame", "Series", "Database", "LazyFrame",
    "read_csv", "read_parquet", "read_json", "read_excel", "read_html",
    "read_feather", "read_orc", "read_sql", "read_csv_chunked",
    "from_pandas", "to_pandas", "sql_query", "to_sql",
    "concat", "merge", "merge_ordered", "merge_asof",
    "get_dummies", "melt", "pivot_table", "crosstab", "cut", "qcut",
    "to_datetime", "date_range",
    "SimpleImputer", "StandardScaler", "MinMaxScaler", "RobustScaler",
    "LabelEncoder", "OneHotEncoder",
    "train_test_split", "KFold",
    "accuracy_score", "mean_squared_error", "mean_absolute_error", "r2_score",
    "precision_score", "recall_score", "f1_score",
    "preprocessing", "ml", "stats", "viz", "dl",
    "show_versions", "options",
]


def concat(objs, axis=0, ignore_index=False):
    import numpy as np
    from .series import Series as _S
    if not objs:
        return DataFrame({})
    if isinstance(objs[0], _S):
        arr = np.concatenate([np.asarray(o._data) for o in objs])
        idx = np.concatenate([np.asarray(o._index) for o in objs]) if not ignore_index else np.arange(len(arr))
        return _S(arr, index=idx, name=objs[0]._name)
    if axis == 1:
        out_cols, out_data = [], {}
        n = max(len(o) for o in objs)
        for o in objs:
            for c in o._columns:
                name = c
                k = 0
                while name in out_data:
                    k += 1
                    name = f"{c}_{k}"
                out_cols.append(name)
                a = np.asarray(o._cols[c])
                if len(a) < n:
                    pad = np.full(n - len(a), np.nan, dtype=np.float64 if a.dtype.kind in "iuf" else object)
                    a = np.concatenate([a.astype(pad.dtype) if a.dtype != pad.dtype else a, pad])
                out_data[name] = a
        nd = DataFrame.__new__(DataFrame)
        nd._columns = out_cols
        nd._cols = out_data
        nd._index = np.arange(n) if ignore_index else np.asarray(objs[0]._index)
        return nd
    cols = list(objs[0]._columns)
    for o in objs[1:]:
        for c in o._columns:
            if c not in cols:
                cols.append(c)
    out = {c: [] for c in cols}
    idx_parts = []
    for o in objs:
        for c in cols:
            if c in o._cols:
                out[c].append(np.asarray(o._cols[c]))
            else:
                out[c].append(np.full(len(o), np.nan))
        idx_parts.append(np.asarray(o._index))
    data = {c: np.concatenate(v) if v else np.array([]) for c, v in out.items()}
    idx = np.arange(sum(len(o) for o in objs)) if ignore_index else np.concatenate(idx_parts) if idx_parts else np.array([])
    return DataFrame(data, index=idx)


def merge(left, right, on=None, how="inner", left_on=None, right_on=None):
    return left.merge(right, on=on, how=how, left_on=left_on, right_on=right_on)


def merge_ordered(left, right, on=None, how="outer", **kw):
    out = merge(left, right, on=on, how=how, **kw)
    return out.sort_values(on) if on else out


def merge_asof(left, right, on=None, direction="backward"):
    import pandas as pd
    return DataFrame(pd.merge_asof(left.to_pandas().sort_values(on), right.to_pandas().sort_values(on), on=on, direction=direction))


class _Options(dict):
    def __repr__(self): return f"bandas.options({dict(self)})"

options = _Options({"display.max_rows": 10, "compute.numba": True, "io.pyarrow": True})


def show_versions():
    import sys, numpy, pandas
    info = {"python": sys.version.split()[0], "bandas": __version__, "numpy": numpy.__version__, "pandas": pandas.__version__}
    for mod in ("pyarrow", "numba", "sqlalchemy", "duckdb", "sklearn", "torch", "tensorflow", "matplotlib"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "installed")
        except Exception:
            info[mod] = "not installed"
    for k, v in info.items(): print(f"{k}: {v}")
    return info
