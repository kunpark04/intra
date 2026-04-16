"""Position sizing and stop/target price computation for the backtest.

Three stop-loss methods are supported via `cfg.STOP_METHOD`:

- ``"fixed"`` — fixed point distance from `cfg.STOP_FIXED_PTS`.
- ``"atr"``   — `cfg.ATR_MULTIPLIER` × current ATR.
- ``"swing"`` — most recent swing low/high ± `cfg.SWING_BUFFER_TICKS`.

Take-profit distance is `cfg.MIN_RR` × stop distance. Contract count sizes
risk at `cfg.RISK_PCT` × equity using MNQ economics ($2 per point).
All prices are rounded to the instrument tick grid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def tick_round(price: float, tick_size: float) -> float:
    """Round `price` to the nearest multiple of `tick_size`.

    Args:
        price: Raw price value.
        tick_size: Instrument tick size (e.g. 0.25 for MNQ).

    Returns:
        Price snapped to the tick grid.
    """
    return round(price / tick_size) * tick_size


def compute_stop_price(
    bar_idx: int,
    side: int,
    entry_price: float,
    df: pd.DataFrame,
    cfg,
) -> float:
    """Absolute stop-loss price for a pending order (tick-rounded).

    Dispatches on `cfg.STOP_METHOD`. For swing stops, ensures the stop is
    strictly worse than entry by at least one tick so the stop cannot be
    placed on the wrong side of the fill.

    Args:
        bar_idx: iloc position of the signal bar in `df`.
        side: `1` for long, `-1` for short.
        entry_price: Expected fill price.
        df: Indicator DataFrame; needs `low`, `high`, and `atr` columns
            (depending on method).
        cfg: Config exposing `STOP_METHOD`, `STOP_FIXED_PTS`, `TICK_SIZE`,
            `ATR_MULTIPLIER`, `SWING_LOOKBACK`, `SWING_BUFFER_TICKS`.

    Returns:
        Stop price rounded to the tick grid.

    Raises:
        ValueError: If `cfg.STOP_METHOD` is not recognised.
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
    """Take-profit price at `cfg.MIN_RR` × stop distance (tick-rounded).

    Enforces a minimum 1-tick target distance so degenerate (zero-risk)
    stops do not produce a same-price take-profit.

    Args:
        entry_price: Expected fill price.
        stop_price: Pre-computed stop-loss price.
        side: `1` for long, `-1` for short.
        cfg: Config exposing `MIN_RR` and `TICK_SIZE`.

    Returns:
        Take-profit price rounded to the tick grid.
    """
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
    """Floor-rounded MNQ contract count sized to `cfg.RISK_PCT` of equity.

    Risk-per-contract is `stop_distance_pts × MNQ_DOLLARS_PER_POINT`.
    Returns `0` when the stop distance is zero (degenerate order).

    Args:
        equity: Current account equity in USD.
        stop_price: Stop-loss price.
        entry_price: Expected fill price.
        cfg: Config exposing `RISK_PCT` and `MNQ_DOLLARS_PER_POINT`.

    Returns:
        Integer number of contracts to trade (floor-rounded).
    """
    stop_distance_pts = abs(entry_price - stop_price)
    max_risk = equity * cfg.RISK_PCT
    risk_per_contract = stop_distance_pts * cfg.MNQ_DOLLARS_PER_POINT
    if risk_per_contract <= 0:
        return 0
    return int(max_risk // risk_per_contract)
