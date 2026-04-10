"""
strategy.py - Signal generation for the MNQ 1-minute strategy.

Two modes (cfg.SIGNAL_MODE):

  "ema_crossover"  — V1/V2 baseline
      Primary:  8/21 EMA crossover fires the signal.
      Filter:   Optional z-score gate (cfg.USE_ZSCORE_FILTER).
        long  only when zscore <= -cfg.Z_BAND_K
        short only when zscore >=  cfg.Z_BAND_K

  "zscore_reversal"  — V3+
      Primary:  z-score crosses back through the band.
        long  when z crosses from <= -k to > -k  (price recovering from stretched low)
        short when z crosses from >=  k to <  k  (price recovering from stretched high)
      Confirming filter: EMA direction must agree at the bar of the signal.
        long  only when fast EMA > slow EMA
        short only when fast EMA < slow EMA

In both modes the function emits all bars meeting entry conditions; the
backtest engine enforces one-position-at-a-time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_signals(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Add 'signal' column to df and return it.

    Parameters
    ----------
    df  : DataFrame produced by indicators.add_indicators().
    cfg : config module exposing SIGNAL_MODE, USE_ZSCORE_FILTER, Z_BAND_K.

    Returns
    -------
    df copy with added columns:
        signal          int8  — 1=long, -1=short, 0=flat
        signal_bar_idx  int   — iloc position of the signal bar (-1 if no signal)
    """
    df = df.copy()
    mode = getattr(cfg, "SIGNAL_MODE", "ema_crossover")

    if mode == "zscore_reversal":
        long_cond, short_cond = _zscore_reversal_signals(df, cfg)
    else:
        long_cond, short_cond = _ema_crossover_signals(df, cfg)

    signal = np.zeros(len(df), dtype=np.int8)
    signal[long_cond]  =  1
    signal[short_cond] = -1

    df["signal"] = signal
    bar_positions = np.arange(len(df), dtype=np.int64)
    df["signal_bar_idx"] = np.where(signal != 0, bar_positions, -1)

    return df


# ── Mode: EMA crossover (V1/V2) ───────────────────────────────────────────────

def _ema_crossover_signals(df, cfg):
    cross_up   = df["ema_cross_up"].to_numpy(dtype=bool)
    cross_down = df["ema_cross_down"].to_numpy(dtype=bool)

    if getattr(cfg, "USE_ZSCORE_FILTER", False):
        z = df["zscore"].to_numpy(dtype=np.float64)
        k = cfg.Z_BAND_K
        long_cond  = cross_up   & (z <= -k) & ~np.isnan(z)
        short_cond = cross_down & (z >=  k) & ~np.isnan(z)
    else:
        long_cond  = cross_up
        short_cond = cross_down

    return long_cond, short_cond


# ── Mode: Z-score reversal (V3+) ──────────────────────────────────────────────

def _zscore_reversal_signals(df, cfg):
    """Z-score crosses back through the band, confirmed by EMA direction.

    Long:  z crosses from <= -k to > -k  AND  fast EMA > slow EMA
    Short: z crosses from >=  k to <  k  AND  fast EMA < slow EMA
    """
    z    = df["zscore"].to_numpy(dtype=np.float64)
    k    = float(cfg.Z_BAND_K)
    n    = len(z)

    z_prev = np.empty(n, dtype=np.float64)
    z_prev[0] = np.nan
    z_prev[1:] = z[:-1]

    # z-score crosses back through the lower band (upward crossing = recovery)
    z_cross_up   = (z > -k) & (z_prev <= -k) & ~np.isnan(z_prev)
    # z-score crosses back through the upper band (downward crossing = recovery)
    z_cross_down = (z < k)  & (z_prev >= k)  & ~np.isnan(z_prev)

    # EMA direction at the bar of the signal
    ema_f = df["ema_fast"].to_numpy(dtype=np.float64)
    ema_s = df["ema_slow"].to_numpy(dtype=np.float64)
    ema_bullish = ema_f > ema_s
    ema_bearish = ema_f < ema_s

    long_cond  = z_cross_up   & ema_bullish
    short_cond = z_cross_down & ema_bearish

    return long_cond, short_cond
