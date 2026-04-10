"""
indicators.py - Core indicator computations for the MNQ 1-minute backtest system.

All core computation functions operate on NumPy arrays (Numba-friendly).
Pandas is used only in the high-level add_indicators() helper.

Numba dispatch pattern:
  - If numba is installed AND cfg.USE_NUMBA is True -> use @njit path.
  - Otherwise -> use pure-numpy path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Numba optional import ─────────────────────────────────────────────────────
try:
    import numba  # noqa: F401
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Provide a no-op decorator so the numba-decorated functions can still be
    # defined in the module even when numba is absent.
    def njit(*args, **kwargs):  # type: ignore[misc]
        """No-op replacement for numba.njit when numba is not installed."""
        def decorator(fn):
            return fn
        # Support both @njit and @njit(cache=True)
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Z-score
# ─────────────────────────────────────────────────────────────────────────────

def _zscore_numpy(close: np.ndarray, window: int, ddof: int = 0) -> np.ndarray:
    """Pure-numpy vectorised rolling z-score.

    Uses numpy stride tricks to build a sliding-window view and compute
    mean/std over the entire array without a Python loop.
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
    """Numba-JIT bar-by-bar rolling z-score."""
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
        if denom <= 0:
            std = 0.0
        else:
            std = (var / denom) ** 0.5
        if std == 0.0:
            out[t] = 0.0
        else:
            out[t] = (close[t] - m) / std
    return out


def compute_zscore(close: np.ndarray, window: int, ddof: int = 0) -> np.ndarray:
    """Rolling z-score.  Dispatches to Numba or NumPy based on availability."""
    close = np.asarray(close, dtype=np.float64)
    if NUMBA_AVAILABLE:
        return _zscore_numba(close, window, ddof)
    return _zscore_numpy(close, window, ddof)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  EMA
# ─────────────────────────────────────────────────────────────────────────────

def _ema_numpy(close: np.ndarray, span: int) -> np.ndarray:
    """Simple-loop EMA (NumPy path).

    A recursive formula cannot be cleanly vectorised without cumulative-product
    tricks that add complexity; a plain loop is readable and fast enough.
    """
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
    """Numba-JIT EMA loop."""
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


def compute_ema(close: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average.  Dispatches to Numba or NumPy."""
    close = np.asarray(close, dtype=np.float64)
    if NUMBA_AVAILABLE:
        return _ema_numba(close, span)
    return _ema_numpy(close, span)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ATR (Average True Range)
# ─────────────────────────────────────────────────────────────────────────────

def _atr_numpy(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               window: int) -> np.ndarray:
    """Pure-numpy ATR using vectorised true-range then rolling mean."""
    n = len(close)
    tr = np.empty(n, dtype=np.float64)

    # Bar 0: no prior close available
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
    """Numba-JIT ATR."""
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for t in range(1, n):
        hl = high[t] - low[t]
        hc = abs(high[t] - close[t - 1])
        lc = abs(low[t] - close[t - 1])
        if hl >= hc and hl >= lc:
            tr[t] = hl
        elif hc >= lc:
            tr[t] = hc
        else:
            tr[t] = lc

    out = np.full(n, np.nan)
    for t in range(window - 1, n):
        s = tr[t - window + 1: t + 1]
        m = 0.0
        for v in s:
            m += v
        out[t] = m / window
    return out


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                window: int) -> np.ndarray:
    """Average True Range.  Dispatches to Numba or NumPy."""
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    if NUMBA_AVAILABLE:
        return _atr_numba(high, low, close, window)
    return _atr_numpy(high, low, close, window)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Volume Z-score
# ─────────────────────────────────────────────────────────────────────────────

def compute_volume_zscore(volume: np.ndarray, window: int,
                          ddof: int = 0) -> np.ndarray:
    """Rolling z-score on volume.  Reuses the same z-score dispatch logic."""
    volume = np.asarray(volume, dtype=np.float64)
    return compute_zscore(volume, window, ddof)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  add_indicators — pandas-level convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Compute all strategy indicators and attach them to *df* as new columns.

    Parameters
    ----------
    df  : DataFrame with at least columns 'close', 'high', 'low', 'volume'.
    cfg : config module (or any object) exposing the required attributes:
          Z_WINDOW, ZSCORE_DDOF, Z_BAND_K, EMA_FAST, EMA_SLOW,
          ATR_WINDOW, VOLUME_ZSCORE_WINDOW, USE_NUMBA.

    Returns
    -------
    A copy of *df* with additional columns:
        zscore, ema_fast, ema_slow, ema_spread, atr, volume_zscore
    """
    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)

    use_numba = getattr(cfg, "USE_NUMBA", True) and NUMBA_AVAILABLE

    def _dispatch_zscore(arr, window, ddof):
        if use_numba:
            return _zscore_numba(arr, window, ddof)
        return _zscore_numpy(arr, window, ddof)

    def _dispatch_ema(arr, span):
        if use_numba:
            return _ema_numba(arr, span)
        return _ema_numpy(arr, span)

    def _dispatch_atr(h, l, c, window):
        if use_numba:
            return _atr_numba(h, l, c, window)
        return _atr_numpy(h, l, c, window)

    z_window = cfg.Z_WINDOW
    zscore_ddof = getattr(cfg, "ZSCORE_DDOF", 0)
    ema_fast_span = cfg.EMA_FAST
    ema_slow_span = cfg.EMA_SLOW
    atr_window = cfg.ATR_WINDOW
    vol_window = getattr(cfg, "VOLUME_ZSCORE_WINDOW", z_window)

    df = df.copy()
    df["zscore"] = _dispatch_zscore(close, z_window, zscore_ddof)
    df["ema_fast"] = _dispatch_ema(close, ema_fast_span)
    df["ema_slow"] = _dispatch_ema(close, ema_slow_span)
    df["ema_spread"] = df["ema_fast"] - df["ema_slow"]
    df["atr"] = _dispatch_atr(high, low, close, atr_window)
    df["volume_zscore"] = _dispatch_zscore(volume, vol_window, zscore_ddof)

    return df
