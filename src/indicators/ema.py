"""
ema.py - Exponential Moving Average computation (NumPy + Numba paths).
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


def _ema_numpy(close: np.ndarray, span: int) -> np.ndarray:
    n = len(close)
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    alpha = 2.0 / (span + 1)
    one_minus = 1.0 - alpha
    out[0] = close[0]
    for t in range(1, n):
        out[t] = alpha * close[t] + one_minus * out[t - 1]
    return out


@njit(cache=True)
def _ema_numba(close: np.ndarray, span: int) -> np.ndarray:
    n = len(close)
    out = np.empty(n)
    if n == 0:
        return out
    alpha = 2.0 / (span + 1)
    one_minus = 1.0 - alpha
    out[0] = close[0]
    for t in range(1, n):
        out[t] = alpha * close[t] + one_minus * out[t - 1]
    return out


def compute_ema(close: np.ndarray, span: int, use_numba: bool = True) -> np.ndarray:
    """Exponential moving average. First value = close[0]."""
    close = np.asarray(close, dtype=np.float64)
    if use_numba and NUMBA_AVAILABLE:
        return _ema_numba(close, span)
    return _ema_numpy(close, span)
