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
    """Attach `signal` and `signal_bar_idx` columns to an indicator frame.

    Dispatches on `cfg.SIGNAL_MODE` to produce raw long/short booleans,
    then optionally applies the V5+ entry filters (volume threshold,
    volatility regime gate, session hours). All filters are AND-combined
    so disabling one just leaves the others in effect.

    Args:
        df: DataFrame produced by `indicators.add_indicators()`. May
            optionally include `atr_pct_rank` (for volatility regime) and
            `bar_hour` (for session filter).
        cfg: Config module exposing `SIGNAL_MODE`, `USE_ZSCORE_FILTER`,
            `Z_BAND_K`, and optionally `VOLUME_ENTRY_THRESHOLD`,
            `VOL_REGIME_LOOKBACK`, `VOL_REGIME_MIN_PCT`,
            `VOL_REGIME_MAX_PCT`, `SESSION_FILTER_MODE`,
            `ZSCORE_CONFIRMATION`.

    Returns:
        A copy of `df` with:

        - ``signal`` (int8): `1` long, `-1` short, `0` flat.
        - ``signal_bar_idx`` (int): iloc position of the signal bar, or
          `-1` if no signal on that bar.
    """
    df = df.copy()
    mode = getattr(cfg, "SIGNAL_MODE", "ema_crossover")

    if mode == "zscore_reversal":
        long_cond, short_cond = _zscore_reversal_signals(df, cfg)
    else:
        long_cond, short_cond = _ema_crossover_signals(df, cfg)

    # ── V5+ entry filters (applied after primary signal logic) ────────────────

    # 1. Volume entry threshold: skip low-participation bars
    vol_thresh = float(getattr(cfg, "VOLUME_ENTRY_THRESHOLD", 0.0))
    if vol_thresh > 0.0 and "volume_zscore" in df.columns:
        vol_z = df["volume_zscore"].to_numpy(dtype=np.float64)
        vol_ok = (vol_z >= vol_thresh) & ~np.isnan(vol_z)
        long_cond  = long_cond  & vol_ok
        short_cond = short_cond & vol_ok

    # 2. Volatility regime gate: skip entries when ATR percentile is out of range
    vol_regime_lb = int(getattr(cfg, "VOL_REGIME_LOOKBACK", 0))
    if vol_regime_lb > 0 and "atr_pct_rank" in df.columns:
        atr_pct = df["atr_pct_rank"].to_numpy(dtype=np.float64)
        min_pct  = float(getattr(cfg, "VOL_REGIME_MIN_PCT", 0.0))
        max_pct  = float(getattr(cfg, "VOL_REGIME_MAX_PCT", 1.0))
        regime_ok = (atr_pct >= min_pct) & (atr_pct <= max_pct) & ~np.isnan(atr_pct)
        long_cond  = long_cond  & regime_ok
        short_cond = short_cond & regime_ok

    # 3. Session filter: restrict entries to allowed hours
    session_mode = int(getattr(cfg, "SESSION_FILTER_MODE", 0))
    if session_mode != 0 and "bar_hour" in df.columns:
        hour = df["bar_hour"].to_numpy(dtype=np.int64)
        if session_mode == 1:    # daytime: 7–19h (inclusive)
            session_ok = (hour >= 7) & (hour < 20)
        elif session_mode == 2:  # core US: 9–15h (inclusive)
            session_ok = (hour >= 9) & (hour < 16)
        else:                    # overnight: 20–6h (wraps midnight)
            session_ok = (hour >= 20) | (hour < 7)
        long_cond  = long_cond  & session_ok
        short_cond = short_cond & session_ok

    signal = np.zeros(len(df), dtype=np.int8)
    signal[long_cond]  =  1
    signal[short_cond] = -1

    df["signal"] = signal
    bar_positions = np.arange(len(df), dtype=np.int64)
    df["signal_bar_idx"] = np.where(signal != 0, bar_positions, -1)

    return df


# ── Mode: EMA crossover (V1/V2) ───────────────────────────────────────────────

def _ema_crossover_signals(df, cfg):
    """Raw long/short masks from the V1/V2 EMA-crossover rule.

    With `cfg.USE_ZSCORE_FILTER=True`, only permits entries when the
    z-score is beyond the `cfg.Z_BAND_K` band in the same direction
    (long on oversold, short on overbought).

    Args:
        df: Indicator frame with `ema_cross_up`, `ema_cross_down`, `zscore`.
        cfg: Config exposing `USE_ZSCORE_FILTER` and `Z_BAND_K`.

    Returns:
        `(long_cond, short_cond)` tuple of bool arrays same length as `df`.
    """
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
    """Raw long/short masks from the V3+ z-score reversal rule.

    Long: z crosses from ≤ -k to > -k and fast EMA > slow EMA.
    Short: z crosses from ≥  k to <  k and fast EMA < slow EMA.
    Optional `cfg.ZSCORE_CONFIRMATION` requires `|z|` to be declining on
    the signal bar (peaked-and-reverting entries only).

    Args:
        df: Indicator frame with `zscore`, `ema_fast`, `ema_slow`.
        cfg: Config exposing `Z_BAND_K` and optional `ZSCORE_CONFIRMATION`.

    Returns:
        `(long_cond, short_cond)` tuple of bool arrays same length as `df`.
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

    # Z-confirmation: only enter when |z| is already declining (peaked and reverting)
    if getattr(cfg, "ZSCORE_CONFIRMATION", False):
        z_abs      = np.abs(z)
        z_abs_prev = np.empty(n, dtype=np.float64)
        z_abs_prev[0] = np.nan
        z_abs_prev[1:] = z_abs[:-1]
        z_declining = (z_abs < z_abs_prev) & ~np.isnan(z_abs_prev)
        long_cond  = long_cond  & z_declining
        short_cond = short_cond & z_declining

    return long_cond, short_cond
