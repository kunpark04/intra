"""
scoring.py — Track A: entry quality scoring (1-3).

Provides:
  compute_entry_score   — vectorised scorer for a trades DataFrame
  score_single_trade    — scalar version called from backtest.py per-trade

Score components (all derived from signal-bar features; zero lookahead):
  zscore    — abs(zscore_entry) / Z_BAND_K   : price stretch at entry
  volume    — volume_zscore                  : participation confirmation
  ema       — abs(ema_spread) / EMA_NORM     : trend strength behind crossover
  body      — bar_body / bar_range           : directional conviction of signal bar
  session   — time_of_day_hhmm              : RTH > extended hours > overnight

Each component is clipped to [0, 1]. Missing (NaN) components are excluded
and the score is renormalised by the sum of present weights, so a trade with
a missing volume_zscore still gets a valid score from the other components.

Session assumes timestamps are in US Central Time (CT):
  RTH       08:30-15:15   score = 1.0
  Extended  06:00-08:30 or 15:15-17:00   score = 0.6
  Overnight 17:00-06:00   score = 0.3
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


# ── Session quality map (time_of_day_hhmm as int → score) ────────────────────

def _session_score_from_hhmm(hhmm: int) -> float:
    """Map an integer like 935 or 1500 to a session quality score."""
    if 830 <= hhmm < 1515:
        return 1.0   # RTH
    if (600 <= hhmm < 830) or (1515 <= hhmm < 1700):
        return 0.6   # extended / pre-open / post-close
    return 0.3       # overnight Globex


def _session_scores_vectorised(hhmm_series: pd.Series) -> pd.Series:
    hhmm = pd.to_numeric(hhmm_series, errors="coerce").fillna(0).astype(int)
    scores = np.where(
        (hhmm >= 830) & (hhmm < 1515), 1.0,
        np.where(
            ((hhmm >= 600) & (hhmm < 830)) | ((hhmm >= 1515) & (hhmm < 1700)),
            0.6, 0.3
        )
    )
    return pd.Series(scores, index=hhmm_series.index)


# ── Vectorised scorer (DataFrame → Series) ────────────────────────────────────

def compute_entry_score(trades: pd.DataFrame, cfg) -> pd.Series:
    """Compute entry_score for every row in a trades DataFrame.

    Parameters
    ----------
    trades : DataFrame containing the LOG_SCHEMA columns produced by backtest.py
    cfg    : config module with SCORE_W_* and SCORE_EMA_NORM attributes

    Returns
    -------
    pd.Series of float64 in [0, 1], same index as trades.
    """
    w_z   = float(cfg.SCORE_W_ZSCORE)
    w_vol = float(cfg.SCORE_W_VOLUME)
    w_ema = float(cfg.SCORE_W_EMA)
    w_bod = float(cfg.SCORE_W_BODY)
    w_ses = float(cfg.SCORE_W_SESSION)
    ema_norm = float(cfg.SCORE_EMA_NORM)
    z_band_k = float(cfg.Z_BAND_K)

    n = len(trades)
    score_num = np.zeros(n, dtype=np.float64)
    score_den = np.zeros(n, dtype=np.float64)

    # ── zscore component ──────────────────────────────────────────────────────
    if "zscore_entry" in trades.columns:
        z_raw = pd.to_numeric(trades["zscore_entry"], errors="coerce")
        z_comp = (z_raw.abs() / z_band_k).clip(0.0, 1.0)
        valid = z_comp.notna()
        score_num[valid] += w_z * z_comp[valid]
        score_den[valid] += w_z

    # ── volume component ──────────────────────────────────────────────────────
    if "volume_zscore" in trades.columns:
        v_raw = pd.to_numeric(trades["volume_zscore"], errors="coerce")
        # Map volume_zscore: (-inf, +inf) → [0, 1] via linear clip on [-2, 2]
        v_comp = ((v_raw + 2.0) / 4.0).clip(0.0, 1.0)
        valid = v_comp.notna()
        score_num[valid] += w_vol * v_comp[valid]
        score_den[valid] += w_vol

    # ── EMA spread component ──────────────────────────────────────────────────
    if "ema_spread" in trades.columns:
        e_raw = pd.to_numeric(trades["ema_spread"], errors="coerce")
        e_comp = (e_raw.abs() / ema_norm).clip(0.0, 1.0)
        valid = e_comp.notna()
        score_num[valid] += w_ema * e_comp[valid]
        score_den[valid] += w_ema

    # ── Bar body ratio component ──────────────────────────────────────────────
    if "bar_body_points" in trades.columns and "bar_range_points" in trades.columns:
        body = pd.to_numeric(trades["bar_body_points"], errors="coerce")
        rng  = pd.to_numeric(trades["bar_range_points"], errors="coerce")
        # Avoid div-by-zero on doji bars (range < 1 tick)
        safe_rng = rng.where(rng >= 0.25, other=np.nan)
        b_comp = (body / safe_rng).clip(0.0, 1.0)
        valid = b_comp.notna()
        score_num[valid] += w_bod * b_comp[valid]
        score_den[valid] += w_bod

    # ── Session component ─────────────────────────────────────────────────────
    if "time_of_day_hhmm" in trades.columns:
        s_comp = _session_scores_vectorised(trades["time_of_day_hhmm"])
        valid = s_comp.notna()
        score_num[valid] += w_ses * s_comp[valid]
        score_den[valid] += w_ses

    # ── Normalise by sum of present weights ───────────────────────────────────
    with np.errstate(invalid="ignore", divide="ignore"):
        final = np.where(score_den > 0, score_num / score_den, np.nan)

    return pd.Series(final, index=trades.index, name="entry_score")


# ── Scalar version (called per-trade in backtest.py) ─────────────────────────

def score_single_trade(
    zscore_entry:      float,
    volume_zscore:     float,
    ema_spread:        float,
    bar_body_points:   float,
    bar_range_points:  float,
    time_of_day_hhmm:  str,
    cfg,
) -> float:
    """Return entry_score for one trade. NaN inputs are handled gracefully."""
    w_z   = cfg.SCORE_W_ZSCORE
    w_vol = cfg.SCORE_W_VOLUME
    w_ema = cfg.SCORE_W_EMA
    w_bod = cfg.SCORE_W_BODY
    w_ses = cfg.SCORE_W_SESSION
    ema_norm = cfg.SCORE_EMA_NORM
    z_band_k = cfg.Z_BAND_K

    num, den = 0.0, 0.0

    def _add(weight, value):
        nonlocal num, den
        if not math.isnan(value):
            num += weight * max(0.0, min(1.0, value))
            den += weight

    _add(w_z,   abs(zscore_entry) / z_band_k if not math.isnan(zscore_entry) else math.nan)
    _add(w_vol, (volume_zscore + 2.0) / 4.0  if not math.isnan(volume_zscore) else math.nan)
    _add(w_ema, abs(ema_spread) / ema_norm    if not math.isnan(ema_spread) else math.nan)

    if bar_range_points >= 0.25 and not math.isnan(bar_body_points):
        _add(w_bod, bar_body_points / bar_range_points)

    try:
        hhmm = int(time_of_day_hhmm)
        _add(w_ses, _session_score_from_hhmm(hhmm))
    except (ValueError, TypeError):
        pass

    return num / den if den > 0 else math.nan
