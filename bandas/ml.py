"""bandas.ml — train/test split, CV, metrics, sklearn bridge."""
from __future__ import annotations
import numpy as np


def train_test_split(*arrays, test_size=0.2, random_state=None, shuffle=True):
    rng = np.random.default_rng(random_state)
    n = len(np.asarray(arrays[0]._data if hasattr(arrays[0], "_data") else arrays[0]._cols[arrays[0]._columns[0]] if hasattr(arrays[0], "_columns") else arrays[0]))
    idx = np.arange(n)
    if shuffle: rng.shuffle(idx)
    k = int(n * (1 - test_size)) if isinstance(test_size, float) else n - int(test_size)
    tr, te = idx[:k], idx[k:]
    def part(a):
        from .series import Series
        from .dataframe import DataFrame
        if isinstance(a, Series): return Series(np.asarray(a._data)[tr], index=np.asarray(a._index)[tr], name=a._name), Series(np.asarray(a._data)[te], index=np.asarray(a._index)[te], name=a._name)
        if isinstance(a, DataFrame): return a._take_rows_cols(tr, a._columns), a._take_rows_cols(te, a._columns)
        a = np.asarray(a); return a[tr], a[te]
    out = []
    for a in arrays: out.extend(part(a))
    return out


class KFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits; self.shuffle = shuffle; self.random_state = random_state
    def split(self, X):
        n = len(X)
        idx = np.arange(n)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state); rng.shuffle(idx)
        fold = n // self.n_splits
        for i in range(self.n_splits):
            te = idx[i * fold:(i + 1) * fold if i < self.n_splits - 1 else n]
            tr = np.array([j for j in idx if j not in set(te)])
            yield tr, te


def _v(a): return np.asarray(a._data if hasattr(a, "_data") else (a._cols[a._columns[0]] if hasattr(a, "_columns") and len(a._columns) == 1 else a), dtype=np.float64)


def accuracy_score(y_true, y_pred): return float(np.mean(_v(y_true) == _v(y_pred)))
def mean_squared_error(y_true, y_pred): return float(np.mean((_v(y_true) - _v(y_pred)) ** 2))
def mean_absolute_error(y_true, y_pred): return float(np.mean(np.abs(_v(y_true) - _v(y_pred))))
def r2_score(y_true, y_pred):
    t, p = _v(y_true), _v(y_pred)
    ss = np.sum((t - p) ** 2); tot = np.sum((t - t.mean()) ** 2)
    return float(1 - ss / tot) if tot else 0.0
def precision_score(y_true, y_pred, pos_label=1):
    t, p = _v(y_true), _v(y_pred)
    tp = np.sum((p == pos_label) & (t == pos_label)); fp = np.sum((p == pos_label) & (t != pos_label))
    return float(tp / (tp + fp)) if tp + fp else 0.0
def recall_score(y_true, y_pred, pos_label=1):
    t, p = _v(y_true), _v(y_pred)
    tp = np.sum((p == pos_label) & (t == pos_label)); fn = np.sum((p != pos_label) & (t == pos_label))
    return float(tp / (tp + fn)) if tp + fn else 0.0
def f1_score(y_true, y_pred, pos_label=1):
    pr, rc = precision_score(y_true, y_pred, pos_label), recall_score(y_true, y_pred, pos_label)
    return float(2 * pr * rc / (pr + rc)) if pr + rc else 0.0


def to_sklearn_Xy(df, target):
    X = df.drop(columns=[target]).to_numpy(dtype=np.float64)
    y = np.asarray(df._cols[target])
    return X, y


def make_pipeline(*steps):
    """Minimal sklearn-style pipeline that works with bandas frames."""
    class Pipe:
        def __init__(self, steps): self.steps = steps
        def fit(self, X, y=None):
            Xt = X
            for name, s in self.steps:
                if hasattr(s, "fit_transform"): Xt = s.fit_transform(Xt) if y is None or name == self.steps[-1][0] else s.fit(Xt, y).transform(Xt) if hasattr(s, "fit") else s.fit_transform(Xt)
                elif hasattr(s, "fit"): s.fit(Xt, y)
            self._Xt = Xt; return self
        def predict(self, X):
            Xt = X
            for name, s in self.steps[:-1]:
                if hasattr(s, "transform"): Xt = s.transform(Xt)
            return self.steps[-1][1].predict(np.asarray(Xt.to_numpy(dtype=np.float64)) if hasattr(Xt, "to_numpy") else np.asarray(Xt))
    return Pipe(list(steps))
