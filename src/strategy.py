"""
strategy.py - Signal generation for the MNQ 1-minute EMA crossover strategy.

Primary signal: 8/21 EMA crossover.
  fast crosses above slow  →  1  (long)
  fast crosses below slow  → -1  (short)

Optional Z-score filter (cfg.USE_ZSCORE_FILTER=True):
  long  only when zscore <= -cfg.Z_BAND_K
  short only when zscore >=  cfg.Z_BAND_K

This function emits ALL bars meeting entry conditions (no position-state
awareness). The backtest engine enforces one-position-at-a-time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_signals(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Add 'signal' and 'signal_bar_idx' columns to df and return it.

    Parameters
    ----------
    df  : DataFrame produced by indicators.add_indicators(). Required columns:
          ema_cross_up, ema_cross_down, zscore (optional filter).
    cfg : config module exposing USE_ZSCORE_FILTER, Z_BAND_K.

    Returns
    -------
    df copy with added columns:
        signal          int8  — 1=long, -1=short, 0=flat
        signal_bar_idx  int   — iloc position of the signal bar (-1 if no signal)
    """
    df = df.copy()

    cross_up   = df["ema_cross_up"].to_numpy(dtype=bool)
    cross_down = df["ema_cross_down"].to_numpy(dtype=bool)

    use_z_filter = getattr(cfg, "USE_ZSCORE_FILTER", False)

    if use_z_filter:
        z = df["zscore"].to_numpy(dtype=np.float64)
        k = cfg.Z_BAND_K
        long_cond  = cross_up   & (z <= -k) & ~np.isnan(z)
        short_cond = cross_down & (z >=  k) & ~np.isnan(z)
    else:
        long_cond  = cross_up
        short_cond = cross_down

    signal = np.zeros(len(df), dtype=np.int8)
    signal[long_cond]  =  1
    signal[short_cond] = -1

    df["signal"] = signal

    # signal_bar_idx: iloc position (integer) of the signal bar; -1 if none
    bar_positions = np.arange(len(df), dtype=np.int64)
    df["signal_bar_idx"] = np.where(signal != 0, bar_positions, -1)

    return df
