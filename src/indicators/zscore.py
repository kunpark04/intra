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
        """No-op stand-in for `numba.njit` when Numba is unavailable."""
        def decorator(fn):
            """Pass-through: return `fn` unchanged."""
            return fn
        return args[0] if (len(args) == 1 and callable(args[0])) else decorator


def _zscore_numpy(close: np.ndarray, window: int, ddof: int = 0) -> np.ndarray:
    """Strided-NumPy rolling Z-score — used when Numba is unavailable.

    Args:
        close: 1-D close-price array.
        window: Rolling window length in bars.
        ddof: Delta degrees of freedom for std. Default 0 (population std).

    Returns:
        Array of same length as `close` with the first `window-1` values NaN.
        Zero-variance windows map to `z = 0.0` rather than NaN.
    """
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
    """Numba-JIT rolling Z-score — equivalent math to `_zscore_numpy`."""
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
    """Rolling Z-score of a price series.

    Formula: `z[t] = (close[t] - rolling_mean) / rolling_std`.

    Args:
        close: 1-D array-like of close prices.
        window: Rolling window length in bars.
        ddof: Delta degrees of freedom for std. Default 0.
        use_numba: Prefer the Numba path when available. Default True.

    Returns:
        Array with the first `window-1` values NaN. Zero-std windows
        return `0.0` instead of NaN to preserve downstream comparisons.
    """
    close = np.asarray(close, dtype=np.float64)
    if use_numba and NUMBA_AVAILABLE:
        return _zscore_numba(close, window, ddof)
    return _zscore_numpy(close, window, ddof)


def compute_volume_zscore(volume: np.ndarray, window: int, ddof: int = 0,
                          use_numba: bool = True) -> np.ndarray:
    """Rolling Z-score on bar volume — thin wrapper over `compute_zscore`.

    Args:
        volume: 1-D array-like of per-bar volume.
        window: Rolling window length in bars.
        ddof: Delta degrees of freedom for std. Default 0.
        use_numba: Prefer the Numba path when available. Default True.

    Returns:
        Volume Z-score array with first `window-1` values NaN.
    """
    return compute_zscore(np.asarray(volume, dtype=np.float64), window, ddof, use_numba)
