"""bandas.datetimes — Series.dt accessor + to_datetime/date_range helpers."""
from __future__ import annotations
import numpy as np


def to_datetime(values, format=None, errors="raise"):
    import pandas as pd
    arr = pd.to_datetime(np.asarray(values) if not hasattr(values, "_data") else np.asarray(values._data),
                         format=format, errors=errors).to_numpy()
    return arr


def date_range(start, end=None, periods=None, freq="D", tz=None):
    import pandas as pd
    return pd.date_range(start=start, end=end, periods=periods, freq=freq, tz=tz).to_numpy()


class DatetimeAccessor:
    def __init__(self, series):
        self._s = series

    def _as_pandas(self):
        import pandas as pd
        return pd.Series(np.asarray(self._s._data))

    def _prop(self, name):
        import pandas as pd
        s = pd.to_datetime(self._as_pandas(), errors="coerce")
        from .series import Series
        return Series(np.asarray(getattr(s.dt, name)), index=np.asarray(self._s._index.copy()), name=self._s._name)

    @property
    def year(self): return self._prop("year")
    @property
    def month(self): return self._prop("month")
    @property
    def day(self): return self._prop("day")
    @property
    def hour(self): return self._prop("hour")
    @property
    def minute(self): return self._prop("minute")
    @property
    def second(self): return self._prop("second")
    @property
    def dayofweek(self): return self._prop("dayofweek")
    @property
    def dayofyear(self): return self._prop("dayofyear")
    @property
    def quarter(self): return self._prop("quarter")
    @property
    def date(self): return self._prop("date")
    @property
    def time(self): return self._prop("time")

    def strftime(self, fmt):
        import pandas as pd
        s = pd.to_datetime(self._as_pandas(), errors="coerce")
        from .series import Series
        return Series(np.asarray(s.dt.strftime(fmt), dtype=object), index=np.asarray(self._s._index.copy()), name=self._s._name)

    def floor(self, freq):
        import pandas as pd
        s = pd.to_datetime(self._as_pandas(), errors="coerce")
        from .series import Series
        return Series(np.asarray(s.dt.floor(freq)), index=np.asarray(self._s._index.copy()), name=self._s._name)

    def ceil(self, freq):
        import pandas as pd
        s = pd.to_datetime(self._as_pandas(), errors="coerce")
        from .series import Series
        return Series(np.asarray(s.dt.ceil(freq)), index=np.asarray(self._s._index.copy()), name=self._s._name)
