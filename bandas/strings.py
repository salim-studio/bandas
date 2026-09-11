"""bandas.strings — vectorized string accessor (Series.str.*)."""
from __future__ import annotations
import numpy as np


class StringAccessor:
    def __init__(self, series):
        self._s = series

    def _arr(self):
        return np.asarray(self._s._data, dtype=object)

    def _wrap(self, out):
        from .series import Series
        return Series(np.asarray(out, dtype=object), index=np.asarray(self._s._index.copy()), name=self._s._name)

    def lower(self): return self._wrap(np.array([str(x).lower() if x is not None else None for x in self._arr()], dtype=object))
    def upper(self): return self._wrap(np.array([str(x).upper() if x is not None else None for x in self._arr()], dtype=object))
    def strip(self, to_strip=None): return self._wrap(np.array([str(x).strip(to_strip) if x is not None else None for x in self._arr()], dtype=object))
    def lstrip(self, to_strip=None): return self._wrap(np.array([str(x).lstrip(to_strip) if x is not None else None for x in self._arr()], dtype=object))
    def rstrip(self, to_strip=None): return self._wrap(np.array([str(x).rstrip(to_strip) if x is not None else None for x in self._arr()], dtype=object))
    def len(self): return self._wrap(np.array([len(str(x)) if x is not None and x == x else 0 for x in self._arr()]))
    __len__ = len

    def contains(self, pat, regex=True, na=False):
        import re
        out = []
        for x in self._arr():
            if x is None or (isinstance(x, float) and np.isnan(x)):
                out.append(na); continue
            s = str(x)
            out.append(bool(re.search(pat, s)) if regex else (pat in s))
        return self._wrap(np.array(out, dtype=bool))

    def startswith(self, pat): return self._wrap(np.array([str(x).startswith(pat) for x in self._arr()], dtype=bool))
    def endswith(self, pat): return self._wrap(np.array([str(x).endswith(pat) for x in self._arr()], dtype=bool))
    def replace(self, old, new, regex=False):
        import re
        if regex:
            return self._wrap(np.array([re.sub(old, new, str(x)) for x in self._arr()], dtype=object))
        return self._wrap(np.array([str(x).replace(old, new) for x in self._arr()], dtype=object))
    def split(self, sep=None, expand=False):
        parts = [str(x).split(sep) if x is not None else [] for x in self._arr()]
        if not expand:
            return self._wrap(np.array(parts, dtype=object))
        from .dataframe import DataFrame
        m = max((len(p) for p in parts), default=0)
        cols = {i: [p[i] if i < len(p) else None for p in parts] for i in range(m)}
        return DataFrame(cols, index=np.asarray(self._s._index))
    def cat(self, sep=""):
        return sep.join(str(x) for x in self._arr() if x is not None)
    def get(self, i):
        return self._wrap(np.array([str(x)[i] if x is not None and len(str(x)) > abs(i) else None for x in self._arr()], dtype=object))
    def slice(self, start, stop=None):
        return self._wrap(np.array([str(x)[start:stop] for x in self._arr()], dtype=object))
    def isnumeric(self): return self._wrap(np.array([str(x).isnumeric() for x in self._arr()], dtype=bool))
    def isalpha(self): return self._wrap(np.array([str(x).isalpha() for x in self._arr()], dtype=bool))
    def zfill(self, w): return self._wrap(np.array([str(x).zfill(w) for x in self._arr()], dtype=object))
    def pad(self, width, side="left", fillchar=" "):
        out = [(str(x).rjust(width, fillchar) if side == "left" else str(x).ljust(width, fillchar)) for x in self._arr()]
        return self._wrap(np.array(out, dtype=object))
    def extract(self, pat):
        import re
        rx = re.compile(pat)
        rows = []
        for x in self._arr():
            m = rx.search(str(x))
            rows.append(list(m.groups()) if m else [None] * rx.groups)
        from .dataframe import DataFrame
        if not rows:
            return DataFrame({})
        return DataFrame({i: [r[i] for r in rows] for i in range(len(rows[0]))}, index=np.asarray(self._s._index))
    def count(self, pat):
        return self._wrap(np.array([str(x).count(pat) for x in self._arr()]))
    def findall(self, pat):
        import re
        return self._wrap(np.array([re.findall(pat, str(x)) for x in self._arr()], dtype=object))
