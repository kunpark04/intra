"""
atr.py - Average True Range computation (NumPy + Numba paths).
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


def _atr_numpy(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               window: int) -> np.ndarray:
    """Strided-NumPy simple-moving-average ATR.

    Builds the per-bar true range as `max(H-L, |H-prev_close|, |L-prev_close|)`
    then takes a simple moving average over `window` bars.

    Args:
        high: 1-D array of bar highs.
        low: 1-D array of bar lows.
        close: 1-D array of bar closes.
        window: ATR window length in bars.

    Returns:
        Array same length as `close` with first `window-1` values NaN.
    """
    n = len(close)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    hl = high[1:] - low[1:]
    hc = np.abs(high[1:] - close[:-1])
    lc = np.abs(low[1:] - close[:-1])
    tr[1:] = np.maximum(hl, np.maximum(hc, lc))
    out = np.full(n, np.nan, dtype=np.float64)
    shape = (n - window + 1, window)
    strides = (tr.strides[0], tr.strides[0])
    wins = np.lib.stride_tricks.as_strided(tr, shape=shape, strides=strides)
    out[window - 1:] = wins.mean(axis=1)
    return out


@njit(cache=True)
def _atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               window: int) -> np.ndarray:
    """Numba-JIT ATR — identical output to `_atr_numpy`, fewer allocations."""
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for t in range(1, n):
        hl = high[t] - low[t]
        hc = abs(high[t] - close[t - 1])
        lc = abs(low[t] - close[t - 1])
        tr[t] = hl if hl >= hc and hl >= lc else (hc if hc >= lc else lc)
    out = np.full(n, np.nan)
    for t in range(window - 1, n):
        s = tr[t - window + 1: t + 1]
        m = 0.0
        for v in s:
            m += v
        out[t] = m / window
    return out


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                window: int, use_numba: bool = True) -> np.ndarray:
    """Simple-moving-average Average True Range.

    Args:
        high: 1-D array-like of bar highs.
        low: 1-D array-like of bar lows.
        close: 1-D array-like of bar closes.
        window: ATR window length in bars.
        use_numba: Prefer the Numba path when available. Default True.

    Returns:
        Array same length as `close` with first `window-1` values NaN.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    if use_numba and NUMBA_AVAILABLE:
        return _atr_numba(high, low, close, window)
    return _atr_numpy(high, low, close, window)
