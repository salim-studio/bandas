"""bandas.reshape — pivot / melt / crosstab / get_dummies / cut / qcut."""
from __future__ import annotations
import numpy as np


def get_dummies(df, columns=None, prefix=None, dtype=float):
    from .dataframe import DataFrame
    from .series import Series
    if isinstance(df, Series):
        vals = np.asarray(df._data)
        uniq = sorted(set(str(v) for v in vals), key=str)
        out = {}
        for u in uniq:
            name = f"{df._name}_{u}" if prefix is None else (f"{prefix}_{u}" if isinstance(prefix, str) else u)
            out[name] = np.array([1.0 if str(v) == u else 0.0 for v in vals], dtype=dtype)
        return DataFrame(out, index=np.asarray(df._index))
    cols = list(columns) if columns else [c for c in df._columns if np.asarray(df._cols[c]).dtype.kind in "OUS"]
    # default: object/str columns only (like pandas)
    res_cols = {c: np.asarray(df._cols[c]) for c in df._columns if c not in cols}
    res_order = [c for c in df._columns if c not in cols]
    for c in cols:
        vals = np.asarray(df._cols[c])
        for u in sorted(set(map(str, vals)), key=str):
            name = f"{c}_{u}" if prefix is None else (f"{prefix}_{u}" if isinstance(prefix, str) else f"{c}_{u}")
            k = 0; base = name
            while name in res_cols: k += 1; name = f"{base}_{k}"
            res_cols[name] = np.array([1.0 if str(v) == u else 0.0 for v in vals], dtype=dtype)
            res_order.append(name)
    nd = DataFrame.__new__(DataFrame); nd._columns = res_order; nd._cols = res_cols
    nd._index = np.asarray(df._index.copy()); return nd


def melt(df, id_vars=None, value_vars=None, var_name="variable", value_name="value"):
    from .dataframe import DataFrame
    id_vars = [id_vars] if isinstance(id_vars, str) else list(id_vars or [])
    value_vars = [value_vars] if isinstance(value_vars, str) else (list(value_vars) if value_vars else [c for c in df._columns if c not in id_vars])
    out = {c: [] for c in id_vars} | {var_name: [], value_name: []}
    for v in value_vars:
        for i in range(len(df)):
            for c in id_vars: out[c].append(df._cols[c][i])
            out[var_name].append(v); out[value_name].append(df._cols[v][i])
    return DataFrame({k: np.array(val, dtype=object) for k, val in out.items()})


def pivot_table(df, values, index, columns=None, aggfunc="mean", fill_value=None):
    import pandas as pd
    pdf = df.to_pandas()
    out = pdf.pivot_table(values=values, index=index, columns=columns, aggfunc=aggfunc, fill_value=fill_value)
    from .dataframe import DataFrame
    return DataFrame(out)


def crosstab(a, b):
    from .series import Series
    from .dataframe import DataFrame
    av = np.asarray(a._data if isinstance(a, Series) else a)
    bv = np.asarray(b._data if isinstance(b, Series) else b)
    ra, ca = sorted(set(map(str, av)), key=str), sorted(set(map(str, bv)), key=str)
    mat = np.zeros((len(ra), len(ca)), dtype=np.int64)
    ri = {v: i for i, v in enumerate(ra)}; ci = {v: i for i, v in enumerate(ca)}
    for x, y in zip(av, bv): mat[ri[str(x)], ci[str(y)]] += 1
    nd = DataFrame.__new__(DataFrame); nd._columns = list(ca)
    nd._cols = {c: mat[:, j] for j, c in enumerate(ca)}; nd._index = np.array(ra, dtype=object)
    return nd


def cut(x, bins, labels=None, include_lowest=False):
    from .series import Series
    v = np.asarray(x._data if isinstance(x, Series) else x, dtype=np.float64)
    edges = np.asarray(bins, dtype=np.float64)
    idx = np.digitize(v, edges, right=False) - 1
    if include_lowest: idx[(v == edges[0])] = 0
    if labels is not None:
        out = np.array([labels[i] if 0 <= i < len(labels) else None for i in idx], dtype=object)
    else:
        out = np.array([f"({edges[i]}, {edges[i+1]}]" if 0 <= i < len(edges) - 1 else None for i in idx], dtype=object)
    if isinstance(x, Series):
        return Series(out, index=np.asarray(x._index.copy()), name=x._name)
    return out


def qcut(x, q, labels=None):
    from .series import Series
    v = np.asarray(x._data if isinstance(x, Series) else x, dtype=np.float64)
    qs = np.linspace(0, 100, q + 1) if isinstance(q, int) else np.asarray(q) * 100
    edges = np.percentile(v[~np.isnan(v)], qs)
    edges[0] -= 1e-9; edges[-1] += 1e-9
    return cut(x, edges, labels=labels, include_lowest=True)
