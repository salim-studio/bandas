"""bandas.dl — PyTorch / TensorFlow bridges (optional deps)."""
from __future__ import annotations
import numpy as np


def to_torch(df, target=None, dtype=None):
    try:
        import torch
    except ImportError as e:
        raise ImportError(f"to_torch needs torch: {e}")
    import torch as _t
    if hasattr(df, "_columns"):
        if target is None:
            return _t.as_tensor(df.to_numpy(dtype=np.float32 if dtype is None else dtype))
        X = df.drop(columns=[target]).to_numpy(dtype=np.float32)
        y = np.asarray(df._cols[target])
        return _t.as_tensor(X), _t.as_tensor(y)
    return _t.as_tensor(np.asarray(df._data))


def to_tensorflow(df, target=None):
    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError(f"to_tensorflow needs tensorflow: {e}")
    if hasattr(df, "_columns"):
        if target is None:
            return tf.convert_to_tensor(df.to_numpy(dtype=np.float32))
        return tf.convert_to_tensor(df.drop(columns=[target]).to_numpy(dtype=np.float32)), tf.convert_to_tensor(np.asarray(df._cols[target]))
    import tensorflow as tf  # noqa
    return tf.convert_to_tensor(np.asarray(df._data))


def torch_dataset(df, target, batch_size=32, shuffle=True):
    try:
        import torch
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError as e:
        raise ImportError(f"torch_dataset needs torch: {e}")
    X, y = to_torch(df, target)
    if y.dtype != torch.float32:
        try: y = y.float()
        except Exception: pass
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)


def tf_dataset(df, target, batch_size=32, shuffle=True):
    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError(f"tf_dataset needs tensorflow: {e}")
    X, y = df.drop(columns=[target]).to_numpy(dtype=np.float32), np.asarray(df._cols[target])
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle: ds = ds.shuffle(len(X))
    return ds.batch(batch_size)
