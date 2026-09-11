"""Fast IO: pyarrow fast-path, else pandas C parser, else csv module."""
from __future__ import annotations

import numpy as np


def _to_bandas_from_pandas(pdf):
    from .dataframe import DataFrame
    return DataFrame(pdf)


def read_csv(path, **kw):
    """read_csv سريع — نفس توقيع pandas.read_csv."""
    # 1) pyarrow fast path (أسرع 3-5x)
    try:
        import pyarrow.csv as pacsv  # type: ignore
        t = pacsv.read_csv(path)
        import pandas as pd
        pdf = t.to_pandas()
        if kw.get("usecols"):
            pdf = pdf[kw["usecols"]]
        if kw.get("nrows"):
            pdf = pdf.iloc[:kw["nrows"]]
        return _to_bandas_from_pandas(pdf)
    except Exception:
        pass
    # 2) pandas C engine
    try:
        import pandas as pd
        pdf = pd.read_csv(path, **kw)
        return _to_bandas_from_pandas(pdf)
    except Exception:
        pass
    # 3) fallback: numpy/csv
    import csv
    with open(path, encoding="utf-8") as f:
        r = list(csv.DictReader(f))
    if not r:
        from .dataframe import DataFrame
        return DataFrame({})
    cols = list(r[0].keys())
    data = {c: [row[c] for row in r] for c in cols}
    # try numeric coercion
    for c in cols:
        try:
            data[c] = np.array(data[c], dtype=np.float64)
            if np.all(data[c] == data[c].astype(np.int64)):
                data[c] = data[c].astype(np.int64)
        except Exception:
            data[c] = np.array(data[c])
    from .dataframe import DataFrame
    return DataFrame(data)


def read_parquet(path, **kw):
    try:
        import pandas as pd
        pdf = pd.read_parquet(path, **kw)
        return _to_bandas_from_pandas(pdf)
    except Exception as e:
        raise ImportError(f"parquet needs pandas/pyarrow: {e}")


def read_json(path_or_str, orient="records", **kw):
    import pandas as pd
    import io as _io, os
    if isinstance(path_or_str, str) and os.path.exists(path_or_str):
        pdf = pd.read_json(path_or_str, orient=orient, **kw)
    elif isinstance(path_or_str, str) and path_or_str.strip().startswith(("[", "{")):
        pdf = pd.read_json(_io.StringIO(path_or_str), orient=orient, **kw)
    else:
        pdf = pd.read_json(path_or_str, orient=orient, **kw)
    return _to_bandas_from_pandas(pdf)


def read_excel(path, sheet_name=0, **kw):
    import pandas as pd
    pdf = pd.read_excel(path, sheet_name=sheet_name, **kw)
    if isinstance(pdf, dict):
        return {k: _to_bandas_from_pandas(v) for k, v in pdf.items()}
    return _to_bandas_from_pandas(pdf)


def read_html(source, **kw):
    import pandas as pd
    tables = pd.read_html(source, **kw)
    return [_to_bandas_from_pandas(t) for t in tables]


def read_feather(path, **kw):
    import pandas as pd
    return _to_bandas_from_pandas(pd.read_feather(path, **kw))


def read_orc(path, **kw):
    import pandas as pd
    return _to_bandas_from_pandas(pd.read_orc(path, **kw))


def read_sql(query, con, **kw):
    from .sql import read_sql as _rs
    return _rs(query, con, **kw)


def read_csv_chunked(path, chunksize=100_000, **kw):
    from .lazy import LazyFrame
    return LazyFrame.read_csv(path, chunksize=chunksize, **kw)


def from_pandas(pdf):
    return _to_bandas_from_pandas(pdf)


def to_pandas(df):
    return df.to_pandas()
