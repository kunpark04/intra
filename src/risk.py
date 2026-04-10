"""
risk.py - Position sizing and stop/target price computation.

Stop methods:
  "fixed" — fixed point distance from config (STOP_FIXED_PTS)
  "atr"   — ATR * multiplier
  "swing" — swing high/low ± buffer ticks
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def tick_round(price: float, tick_size: float) -> float:
    """Round price to the nearest tick."""
    return round(price / tick_size) * tick_size


def compute_stop_price(
    bar_idx: int,
    side: int,
    entry_price: float,
    df: pd.DataFrame,
    cfg,
) -> float:
    """Return the absolute stop-loss price (tick-rounded).

    Parameters
    ----------
    bar_idx     : iloc position in df
    side        : 1 = long, -1 = short
    entry_price : fill price
    df          : indicator DataFrame (needs 'low', 'high', 'atr' per method)
    cfg         : config exposing STOP_METHOD, STOP_FIXED_PTS, TICK_SIZE,
                  ATR_MULTIPLIER, SWING_LOOKBACK, SWING_BUFFER_TICKS
    """
    tick   = cfg.TICK_SIZE
    method = cfg.STOP_METHOD

    if method == "fixed":
        dist = tick_round(cfg.STOP_FIXED_PTS, tick)
        if dist < tick:
            dist = tick
        if side == 1:
            return tick_round(entry_price - dist, tick)
        else:
            return tick_round(entry_price + dist, tick)

    elif method == "atr":
        atr_val = df.iat[bar_idx, df.columns.get_loc("atr")]
        dist = tick_round(atr_val * cfg.ATR_MULTIPLIER, tick)
        if dist < tick:
            dist = tick
        if side == 1:
            return tick_round(entry_price - dist, tick)
        else:
            return tick_round(entry_price + dist, tick)

    elif method == "swing":
        start  = max(0, bar_idx - cfg.SWING_LOOKBACK + 1)
        window = df.iloc[start: bar_idx + 1]
        if side == 1:
            raw = window["low"].min() - cfg.SWING_BUFFER_TICKS * tick
            stop = tick_round(raw, tick)
            if stop >= entry_price:
                stop = tick_round(entry_price - tick, tick)
        else:
            raw = window["high"].max() + cfg.SWING_BUFFER_TICKS * tick
            stop = tick_round(raw, tick)
            if stop <= entry_price:
                stop = tick_round(entry_price + tick, tick)
        return stop

    else:
        raise ValueError(f"Unknown STOP_METHOD: {method!r}")


def compute_tp_price(
    entry_price: float,
    stop_price: float,
    side: int,
    cfg,
) -> float:
    """Return take-profit price at MIN_RR × stop distance (tick-rounded)."""
    tick = cfg.TICK_SIZE
    stop_distance = abs(entry_price - stop_price)
    target_distance = max(stop_distance * cfg.MIN_RR, tick)
    target_distance = tick_round(target_distance, tick)
    if side == 1:
        return tick_round(entry_price + target_distance, tick)
    else:
        return tick_round(entry_price - target_distance, tick)


def compute_contracts(
    equity: float,
    stop_price: float,
    entry_price: float,
    cfg,
) -> int:
    """Number of MNQ contracts sizing risk to RISK_PCT of equity."""
    stop_distance_pts = abs(entry_price - stop_price)
    max_risk = equity * cfg.RISK_PCT
    risk_per_contract = stop_distance_pts * cfg.MNQ_DOLLARS_PER_POINT
    if risk_per_contract <= 0:
        return 0
    return int(max_risk // risk_per_contract)
