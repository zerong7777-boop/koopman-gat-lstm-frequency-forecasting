import numpy as np


def fit_frequency_stats(train_x, train_y):
    merged = np.concatenate(
        [train_x.reshape(-1, train_x.shape[-1]), train_y.reshape(-1, train_y.shape[-1])],
        axis=0,
    )
    return {"mean": merged.mean(axis=0), "std": merged.std(axis=0) + 1e-8}


def apply_frequency_standardization(values, stats):
    values = np.asarray(values, dtype=float)
    return (values - stats["mean"]) / stats["std"]


def normalize_koopman(values):
    values = np.asarray(values, dtype=float)
    mins = values.min(axis=-1, keepdims=True)
    denom = values.max(axis=-1, keepdims=True) - mins
    zeros = np.zeros_like(values)
    return np.divide(values - mins, denom, out=zeros, where=denom != 0)
