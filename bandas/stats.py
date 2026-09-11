"""bandas.stats — correlations, distributions, outliers, sampling helpers."""
from __future__ import annotations
import numpy as np


def corr(df, method="pearson"):
    from .dataframe import DataFrame
    cols = [c for c in df._columns if np.asarray(df._cols[c]).dtype.kind in "iufc"]
    if len(cols) <= 1:
        nd = DataFrame.__new__(DataFrame); nd._columns = list(cols)
        nd._cols = {c: np.array([1.0]) for c in cols} if cols else {}
        nd._index = np.array(cols, dtype=object)
        return nd
    m = np.column_stack([np.asarray(df._cols[c], dtype=np.float64) for c in cols])
    if method == "pearson":
        # NaN-aware pairwise correlation (pandas-like)
        C = np.eye(len(cols))
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                x, y = m[:, i], m[:, j]
                mk = ~(np.isnan(x) | np.isnan(y))
                v = float(np.corrcoef(x[mk], y[mk])[0, 1]) if mk.sum() >= 2 else np.nan
                C[i, j] = C[j, i] = v
    elif method == "spearman":
        try:
            from scipy.stats import spearmanr
            C, _ = spearmanr(m, nan_policy="omit")
            C = np.asarray(C)
        except Exception:
            ranks = np.argsort(np.argsort(m, axis=0), axis=0).astype(float)
            C = np.corrcoef(ranks, rowvar=False)
    else:  # kendall fallback via pandas
        C = df.to_pandas()[cols].corr(method=method).to_numpy() if cols else np.zeros((0, 0))
    C = np.asarray(C, dtype=np.float64).reshape(len(cols), len(cols))
    nd = DataFrame.__new__(DataFrame); nd._columns = list(cols)
    nd._cols = {c: np.asarray(C[:, j], dtype=np.float64) for j, c in enumerate(cols)} if cols else {}
    from .utils import as_index
    nd._index = np.array(cols, dtype=object)
    return nd


def cov(df):
    from .dataframe import DataFrame
    cols = [c for c in df._columns if np.asarray(df._cols[c]).dtype.kind in "iufc"]
    if len(cols) == 0:
        nd = DataFrame.__new__(DataFrame); nd._columns = []; nd._cols = {}; nd._index = np.array([], dtype=object); return nd
    m = np.column_stack([np.asarray(df._cols[c], dtype=np.float64) for c in cols])
    C = np.cov(m, rowvar=False)
    C = np.asarray(C, dtype=np.float64).reshape(len(cols), len(cols))
    nd = DataFrame.__new__(DataFrame); nd._columns = list(cols)
    nd._cols = {c: C[:, j] for j, c in enumerate(cols)} if cols else {}
    nd._index = np.array(cols, dtype=object)
    return nd


def quantile(s, q):
    a = np.asarray(s._data if hasattr(s, "_data") else s, dtype=np.float64)
    return float(np.nanquantile(a, q))


def zscore(s):
    from .series import Series
    a = np.asarray(s._data, dtype=np.float64)
    mu, sd = float(np.nanmean(a)), float(np.nanstd(a)) or 1.0
    return Series((a - mu) / sd, index=np.asarray(s._index.copy()), name=s._name)


def detect_outliers(s, method="iqr", threshold=3.0):
    a = np.asarray(s._data, dtype=np.float64)
    if method == "iqr":
        q1, q3 = np.nanpercentile(a, [25, 75]); iqr = (q3 - q1) or 1e-9
        return (a < q1 - 1.5 * iqr) | (a > q3 + 1.5 * iqr)
    if method == "zscore":
        mu, sd = float(np.nanmean(a)), float(np.nanstd(a)) or 1e-9
        return np.abs((a - mu) / sd) > threshold
    raise ValueError(method)


def value_distribution(s, bins=10):
    counts, edges = np.histogram(np.asarray(s._data, dtype=np.float64), bins=bins)
    return counts, edges


def summary_stats(df):
    """Extended describe: adds skew/kurtosis/missing%."""
    try:
        pdf = df.to_pandas()
        desc = pdf.describe(include="all").T
        desc["missing_pct"] = pdf.isna().mean().values * 100
        try:
            desc["skew"] = pdf.skew(numeric_only=True).reindex(desc.index).values
            desc["kurt"] = pdf.kurt(numeric_only=True).reindex(desc.index).values
        except Exception:
            pass
        from .dataframe import DataFrame
        return DataFrame(desc)
    except Exception:
        return df.describe()
