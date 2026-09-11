"""bandas.viz — one-line plotting (matplotlib backend, optional)."""
from __future__ import annotations
import numpy as np


def _ax(ax):
    if ax is not None: return ax
    try:
        import matplotlib.pyplot as plt
        _, a = plt.subplots(); return a
    except ImportError as e:
        raise ImportError(f"plotting needs matplotlib: {e}")


def line(df, x=None, y=None, ax=None):
    a = _ax(ax)
    if hasattr(df, "_columns"):
        xs = np.asarray(df._index) if x is None else np.asarray(df._cols[x])
        for c in ([y] if isinstance(y, str) else (y or df._columns)):
            if c == x: continue
            a.plot(xs, np.asarray(df._cols[c]), label=c)
        a.legend()
    else:
        a.plot(np.asarray(df._index), np.asarray(df._data), label=df.name); a.legend()
    return a


def bar(df, x=None, y=None, ax=None):
    a = _ax(ax)
    if hasattr(df, "_columns"):
        xs = np.arange(len(df)) if x is None else np.asarray(df._cols[x])
        cols = [y] if isinstance(y, str) else (y or df._columns)
        w = 0.8 / max(len(cols), 1)
        for i, c in enumerate(cols):
            if c == x: continue
            a.bar(np.arange(len(xs)) + i * w, np.asarray(df._cols[c]), width=w, label=c)
        a.set_xticks(np.arange(len(xs))); a.set_xticklabels([str(v) for v in xs], rotation=45, ha="right")
        a.legend()
    else:
        a.bar(np.arange(len(df)), np.asarray(df._data)); a.set_title(df.name or "")
    return a


def hist(s, bins=30, ax=None):
    a = _ax(ax)
    v = np.asarray(s._data if hasattr(s, "_data") else s._cols[s._columns[0]], dtype=np.float64)
    a.hist(v[~np.isnan(v)], bins=bins); return a


def scatter(df, x, y, c=None, ax=None):
    a = _ax(ax)
    cc = None if c is None else np.asarray(df._cols[c])
    a.scatter(np.asarray(df._cols[x]), np.asarray(df._cols[y]), c=cc)
    a.set_xlabel(x); a.set_ylabel(y); return a


def box(df, columns=None, ax=None):
    a = _ax(ax)
    cols = list(columns) if columns else list(df._columns)
    a.boxplot([np.asarray(df._cols[c], dtype=np.float64) for c in cols], labels=cols)
    return a


def heatmap_corr(df, ax=None, annot=True):
    from .stats import corr
    C = corr(df)
    a = _ax(ax)
    m = np.column_stack([np.asarray(C._cols[c]) for c in C._columns])
    im = a.imshow(m, vmin=-1, vmax=1)
    a.set_xticks(range(len(C._columns)), C._columns, rotation=45, ha="right")
    a.set_yticks(range(len(C._columns)), C._columns)
    if annot:
        for i in range(len(C._columns)):
            for j in range(len(C._columns)):
                a.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7)
    try:
        import matplotlib.pyplot as plt
        plt.colorbar(im, ax=a)
    except Exception: pass
    return a
