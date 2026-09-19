"""Validated numeric arrays and JSON conversion."""

from __future__ import annotations

import numpy as np


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _array(value, shape, name):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite array of shape {shape}")
    return a


def _broadcast(value, count, name):
    a = np.broadcast_to(np.asarray(value, dtype=float), (count,))
    if not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return a


def _maximum(value):
    return max(0.0, float(np.max(np.asarray(value), initial=0.0)))
