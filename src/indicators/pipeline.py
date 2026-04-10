"""
pipeline.py - Pandas-level wrapper that attaches all indicators to a DataFrame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ema import compute_ema
from .zscore import compute_zscore, compute_volume_zscore
from .atr import compute_atr


def add_indicators(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Compute all strategy indicators and attach them as new columns.

    Parameters
    ----------
    df  : DataFrame with at least columns 'close', 'high', 'low', 'volume'.
    cfg : config module exposing: Z_WINDOW, ZSCORE_DDOF, EMA_FAST, EMA_SLOW,
          ATR_WINDOW, VOLUME_ZSCORE_WINDOW, USE_NUMBA.

    Added columns
    -------------
    ema_fast, ema_slow, ema_spread   — EMA values and their difference
    ema_cross_up                     — bool: fast crossed above slow this bar
    ema_cross_down                   — bool: fast crossed below slow this bar
    zscore                           — rolling Z-score on close (optional filter)
    atr                              — Average True Range
    volume_zscore                    — rolling Z-score on volume
    """
    use_numba = getattr(cfg, "USE_NUMBA", True)
    ddof = getattr(cfg, "ZSCORE_DDOF", 0)

    close  = df["close"].to_numpy(dtype=np.float64)
    high   = df["high"].to_numpy(dtype=np.float64)
    low    = df["low"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)

    ema_f = compute_ema(close, cfg.EMA_FAST, use_numba)
    ema_s = compute_ema(close, cfg.EMA_SLOW, use_numba)

    df = df.copy()
    df["ema_fast"]   = ema_f
    df["ema_slow"]   = ema_s
    df["ema_spread"] = ema_f - ema_s

    # EMA crossover detection (requires at least 2 bars)
    fast_above = ema_f > ema_s
    df["ema_cross_up"]   = pd.Series(fast_above, index=df.index) & \
                           ~pd.Series(np.concatenate([[False], fast_above[:-1]]), index=df.index)
    df["ema_cross_down"] = ~pd.Series(fast_above, index=df.index) & \
                           pd.Series(np.concatenate([[False], fast_above[:-1]]), index=df.index)

    df["zscore"]       = compute_zscore(close, cfg.Z_WINDOW, ddof, use_numba)
    df["atr"]          = compute_atr(high, low, close, cfg.ATR_WINDOW, use_numba)
    df["volume_zscore"]= compute_volume_zscore(volume, cfg.VOLUME_ZSCORE_WINDOW, ddof, use_numba)

    return df
