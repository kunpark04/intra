"""
zscore.py - Rolling Z-score computation (NumPy + Numba paths).
"""
from __future__ import annotations
import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(fn): return fn
        return args[0] if (len(args) == 1 and callable(args[0])) else decorator


def _zscore_numpy(close: np.ndarray, window: int, ddof: int = 0) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 1 or n < window:
        return out
    shape = (n - window + 1, window)
    strides = (close.strides[0], close.strides[0])
    wins = np.lib.stride_tricks.as_strided(close, shape=shape, strides=strides)
    mean = wins.mean(axis=1)
    std = wins.std(axis=1, ddof=ddof)
    z = np.where(std == 0.0, 0.0, (close[window - 1:] - mean) / std)
    out[window - 1:] = z
    return out


@njit(cache=True)
def _zscore_numba(close: np.ndarray, window: int, ddof: int = 0) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan)
    for t in range(window - 1, n):
        s = close[t - window + 1: t + 1]
        m = 0.0
        for v in s:
            m += v
        m /= window
        var = 0.0
        for v in s:
            diff = v - m
            var += diff * diff
        denom = window - ddof
        std = (var / denom) ** 0.5 if denom > 0 else 0.0
        out[t] = 0.0 if std == 0.0 else (close[t] - m) / std
    return out


def compute_zscore(close: np.ndarray, window: int, ddof: int = 0,
                   use_numba: bool = True) -> np.ndarray:
    """Rolling Z-score. NaN for first (window-1) bars."""
    close = np.asarray(close, dtype=np.float64)
    if use_numba and NUMBA_AVAILABLE:
        return _zscore_numba(close, window, ddof)
    return _zscore_numpy(close, window, ddof)


def compute_volume_zscore(volume: np.ndarray, window: int, ddof: int = 0,
                          use_numba: bool = True) -> np.ndarray:
    """Rolling Z-score on volume. Delegates to compute_zscore."""
    return compute_zscore(np.asarray(volume, dtype=np.float64), window, ddof, use_numba)
