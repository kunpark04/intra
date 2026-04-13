"""
zscore_variants.py — Generalised Z-score computation for parameter sweep.

Supports all combinations of:
  z_input  : what series to score  — "close" | "returns" | "typical_price"
  z_anchor : equilibrium reference — "rolling_mean" | "vwap_session" | "ema_fast" | "ema_slow"
  z_denom  : normalisation         — "rolling_std" | "atr" | "parkinson"
  z_type   : output form           — "parametric" | "quantile_rank"

Compatibility rules
-------------------
- When z_input="returns":
    - z_anchor is forced to "rolling_mean" (EMA/VWAP are in price-space).
    - z_denom is restricted to "rolling_std" or "parkinson" (ATR is in price-points,
      making deviation/ATR ≈ 1e-5 — too small to cross any z_band_k threshold).
  These overrides are applied inside compute_zscore_variant AND enforced at
  sampling time in _sample_combos so that Parquet metadata is always accurate.

- quantile_rank z_type ignores z_denom entirely (it operates only on ranked series).
  z_denom is recorded as "n/a" in the combo dict for quantile_rank combos so the
  ML dataset does not contain misleading categorical values.

Output is always in standard-deviation units so the existing Z_BAND_K threshold
in strategy.py works without modification.  Quantile-rank output is probit-
transformed and normalized to ±3 scale so z_band_k is window-size independent.

Optional second window (z_window_2 > 0)
-----------------------------------------
If z_window_2 > 0 a composite z is returned:
    z = (1 - z_window_2_weight) * z_window1 + z_window_2_weight * z_window2
Both windows use the same z_input / z_anchor / z_denom / z_type.
z_window_2 is sampled from [z_window + 5, 51) to prevent overlap with z_window.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

# ── Parkinson constants ───────────────────────────────────────────────────────
_LOG_PARKINSON_SCALE = math.sqrt(4.0 * math.log(2))          # ≈ 1.6651
_PARKINSON_SCALE     = math.sqrt(1.0 / (4.0 * math.log(2)))  # ≈ 0.6006

# Denominator options that are unit-compatible with returns-space series
_RETURNS_VALID_DENOMS = frozenset({"rolling_std", "parkinson"})


# ── Probit approximation (rational, no scipy dependency) ─────────────────────
# Converts a probability p ∈ (0,1) to its standard-normal quantile.
# Abramowitz & Stegun rational approximation; max error ~4.5e-4.
def _probit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    mask = p < 0.5
    q = np.where(mask, p, 1.0 - p)
    t = np.sqrt(-2.0 * np.log(q))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    num = c0 + c1 * t + c2 * t * t
    den = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t
    z = t - num / den
    return np.where(mask, -z, z)


# ── Rolling helpers (pure NumPy, fast via stride tricks) ──────────────────────

def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 1 or n < window:
        return out
    shape = (n - window + 1, window)
    strides = (arr.strides[0], arr.strides[0])
    wins = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    out[window - 1:] = wins.mean(axis=1)
    return out


def _rolling_std(arr: np.ndarray, window: int, ddof: int = 0) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 1 or n < window:
        return out
    shape = (n - window + 1, window)
    strides = (arr.strides[0], arr.strides[0])
    wins = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    out[window - 1:] = wins.std(axis=1, ddof=ddof)
    return out


def _rolling_rank(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling percentile rank of arr[t] within arr[t-window+1 : t+1].
    Result ∈ (0, 1), NaN for first (window-1) bars.

    Uses pandas rolling().rank() to avoid materialising a full (n, window)
    float64 array via stride tricks, which causes OOM on 2M-bar series
    (e.g. window=50 → 782 MiB contiguous allocation per combo).

    Mid-rank formula: (rank_1based - 0.5) / window keeps result strictly in
    (0, 1) so _probit never returns ±inf.  Mathematically identical to the
    original stride-tricks implementation for the common no-ties case.
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 1 or n < window:
        return out
    ranks_1based = (
        pd.Series(arr)
        .rolling(window, min_periods=window)
        .rank(method="average")
        .to_numpy(dtype=np.float64)
    )
    valid = ~np.isnan(ranks_1based)
    out[valid] = (ranks_1based[valid] - 0.5) / window
    return out


# ── Session VWAP ──────────────────────────────────────────────────────────────

def compute_vwap_session(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    session_break: np.ndarray,
) -> np.ndarray:
    """Intraday session VWAP, reset at each session_break bar.

    Uses typical price = (H + L + C) / 3.
    First bar of each session returns that bar's typical price (single-bar VWAP).
    """
    n = len(close)
    typical = (high + low + close) / 3.0
    out = np.full(n, np.nan, dtype=np.float64)

    cum_tp_vol = 0.0
    cum_vol    = 0.0

    for t in range(n):
        if t == 0 or session_break[t]:
            cum_tp_vol = typical[t] * volume[t]
            cum_vol    = volume[t]
        else:
            cum_tp_vol += typical[t] * volume[t]
            cum_vol    += volume[t]
        out[t] = cum_tp_vol / cum_vol if cum_vol > 0 else typical[t]

    return out


# ── Core variant computation ───────────────────────────────────────────────────

def compute_zscore_variant(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    session_break: np.ndarray,
    ema_fast_arr: np.ndarray,
    ema_slow_arr: np.ndarray,
    atr_arr: np.ndarray,
    window: int,
    z_input: str   = "close",
    z_anchor: str  = "rolling_mean",
    z_denom: str   = "rolling_std",
    z_type: str    = "parametric",
    ddof: int      = 0,
    vwap_arr: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a single z-score array for one (input, anchor, denom, type, window) combo.

    Parameters
    ----------
    close, high, low, volume, session_break : price/volume arrays
    ema_fast_arr, ema_slow_arr              : pre-computed EMA arrays
    atr_arr                                 : pre-computed ATR array
    window   : rolling window length
    z_input  : "close" | "returns" | "typical_price"
    z_anchor : "rolling_mean" | "vwap_session" | "ema_fast" | "ema_slow"
               (forced to "rolling_mean" when z_input="returns")
    z_denom  : "rolling_std" | "atr" | "parkinson" | "n/a"
               ("atr" is invalid with z_input="returns" → treated as "rolling_std";
                "n/a" and quantile_rank combos skip denom computation entirely)
    z_type   : "parametric" | "quantile_rank"
    ddof     : degrees-of-freedom correction for rolling_std
    vwap_arr : pre-computed session VWAP (pass in to avoid recomputation)

    Returns
    -------
    np.ndarray of float64, same length as close, NaN where not yet computable.
    Always in standard-deviation units (quantile_rank normalized to ±3 scale).
    """
    n = len(close)
    close  = np.asarray(close,  dtype=np.float64)
    high   = np.asarray(high,   dtype=np.float64)
    low    = np.asarray(low,    dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)

    # ── 1. Build input series ─────────────────────────────────────────────────
    if z_input == "returns":
        series = np.empty(n, dtype=np.float64)
        series[0] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            series[1:] = np.log(close[1:] / close[:-1])
        # Enforce compatibility: returns are price-space agnostic
        effective_anchor = "rolling_mean"
        effective_denom  = z_denom if z_denom in _RETURNS_VALID_DENOMS else "rolling_std"
    elif z_input == "typical_price":
        series = (high + low + close) / 3.0
        effective_anchor = z_anchor
        effective_denom  = z_denom
    else:  # "close"
        series = close
        effective_anchor = z_anchor
        effective_denom  = z_denom

    # ── 2. Build anchor (equilibrium) — skipped for quantile_rank ────────────
    if z_type != "quantile_rank":
        if effective_anchor == "vwap_session":
            anchor = (vwap_arr if vwap_arr is not None
                      else compute_vwap_session(high, low, close, volume, session_break))
        elif effective_anchor == "ema_fast":
            anchor = np.asarray(ema_fast_arr, dtype=np.float64)
        elif effective_anchor == "ema_slow":
            anchor = np.asarray(ema_slow_arr, dtype=np.float64)
        else:  # "rolling_mean"
            anchor = _rolling_mean(series, window)
        deviation = series - anchor

    # ── 3. Build denominator — skipped for quantile_rank ─────────────────────
    if z_type != "quantile_rank":
        if effective_denom == "atr":
            denom = np.asarray(atr_arr, dtype=np.float64)
        elif effective_denom == "parkinson":
            if z_input == "returns":
                # Return-space Parkinson: ln(H/L) / sqrt(4*ln2)
                with np.errstate(divide="ignore", invalid="ignore"):
                    park_bar = np.where(
                        (low > 0) & (high > 0) & (high >= low),
                        np.log(np.maximum(high, low) / np.maximum(low, 1e-12)) / _LOG_PARKINSON_SCALE,
                        np.nan,
                    )
            else:
                # Point-space range normalizer: (H-L) * scale
                park_bar = np.maximum(high - low, 0.0) * _PARKINSON_SCALE
            denom = _rolling_mean(park_bar, window)
        else:  # "rolling_std"
            denom = _rolling_std(series, window, ddof)

    # ── 4. Compute z-score ────────────────────────────────────────────────────
    if z_type == "quantile_rank":
        ranks = _rolling_rank(series, window)
        z_raw = _probit(ranks)
        # Normalize to ±3 so z_band_k thresholds are window-size independent.
        # max achievable probit = probit((window - 0.5) / window)
        max_rank   = (window - 0.5) / float(window)
        max_probit = float(_probit(np.array([max_rank]))[0])
        scale      = 3.0 / max_probit if max_probit > 0 else 1.0
        z = z_raw * scale
    else:  # "parametric"
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(
                (denom > 1e-12) & ~np.isnan(denom) & ~np.isnan(deviation),
                deviation / denom,
                np.nan,
            )

    return z.astype(np.float64)


# ── Public entry point: single or composite window ───────────────────────────

def compute_zscore_v2(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    session_break: np.ndarray,
    ema_fast_arr: np.ndarray,
    ema_slow_arr: np.ndarray,
    atr_arr: np.ndarray,
    window: int,
    z_input: str   = "close",
    z_anchor: str  = "rolling_mean",
    z_denom: str   = "rolling_std",
    z_type: str    = "parametric",
    ddof: int      = 0,
    vwap_arr: np.ndarray | None = None,
    z_window_2: int        = 0,
    z_window_2_weight: float = 0.3,
) -> np.ndarray:
    """Compute z-score with optional two-window composite.

    If z_window_2 > 0 and z_window_2 != window:
        z = (1 - z_window_2_weight) * z_window1 + z_window_2_weight * z_window2

    All other parameters are passed through to compute_zscore_variant.
    """
    z1 = compute_zscore_variant(
        close, high, low, volume, session_break,
        ema_fast_arr, ema_slow_arr, atr_arr,
        window, z_input, z_anchor, z_denom, z_type, ddof, vwap_arr,
    )

    if z_window_2 > 0 and z_window_2 != window:
        z2 = compute_zscore_variant(
            close, high, low, volume, session_break,
            ema_fast_arr, ema_slow_arr, atr_arr,
            z_window_2, z_input, z_anchor, z_denom, z_type, ddof, vwap_arr,
        )
        w = float(z_window_2_weight)
        return np.where(
            ~np.isnan(z1) & ~np.isnan(z2),
            (1.0 - w) * z1 + w * z2,
            np.nan,
        )

    return z1
