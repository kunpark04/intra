"""
backtest.py - Bar-by-bar backtest engine for the MNQ 1-minute EMA strategy.

Three-layer dispatch (fastest to slowest):
  backtest_core_cy      — Cython AOT-compiled (preferred; zero JIT overhead)
  _backtest_core        — Numba @njit JIT-compiled (fallback if Cython absent)
  _backtest_core_numpy  — pure-Python/NumPy (always present; last resort)
  run_backtest          — Python wrapper: dispatches to core, post-processes records,
                          builds equity curve, returns results dict.
"""
from __future__ import annotations

import math
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from src.scoring import score_single_trade

# ── Cython AOT extension (preferred) ─────────────────────────────────────────
CYTHON_AVAILABLE = False
try:
    from src.cython_ext.backtest_core import backtest_core_cy  # type: ignore
    CYTHON_AVAILABLE = True
except ImportError:
    backtest_core_cy = None  # type: ignore

# ── Numba JIT fallback ────────────────────────────────────────────────────────
NUMBA_AVAILABLE = False
try:
    from numba import njit  # type: ignore
    NUMBA_AVAILABLE = True
except ImportError:
    # Provide a no-op decorator so the decorated function is just a normal function
    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        # Support both @njit and @njit(cache=True) forms
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


# ── Layer 1a: Numba JIT core ──────────────────────────────────────────────────

@njit(cache=True)
def _backtest_core(
    open_arr,          # float64[:]
    high_arr,          # float64[:]
    low_arr,           # float64[:]
    close_arr,         # float64[:]
    signal_arr,        # int8[:]
    n_bars,            # int
    sl_pts,            # float  — stop distance in points (positive)
    tp_pts,            # float  — tp distance in points (positive)
    same_bar_tp_first, # bool
    exit_on_opposite,  # bool
):
    """Numba JIT bar-by-bar state machine. Returns sliced output arrays."""
    max_trades = n_bars // 2 + 1

    out_side         = np.empty(max_trades, dtype=np.int8)
    out_signal_bar   = np.empty(max_trades, dtype=np.int64)
    out_entry_bar    = np.empty(max_trades, dtype=np.int64)
    out_exit_bar     = np.empty(max_trades, dtype=np.int64)
    out_entry_price  = np.empty(max_trades, dtype=np.float64)
    out_exit_price   = np.empty(max_trades, dtype=np.float64)
    out_sl           = np.empty(max_trades, dtype=np.float64)
    out_tp           = np.empty(max_trades, dtype=np.float64)
    out_exit_reason  = np.empty(max_trades, dtype=np.int8)
    out_mae          = np.empty(max_trades, dtype=np.float64)
    out_mfe          = np.empty(max_trades, dtype=np.float64)
    out_hold_bars    = np.empty(max_trades, dtype=np.int64)
    out_label_tp_first = np.empty(max_trades, dtype=np.int8)

    in_trade    = False
    side        = np.int8(0)
    entry_bar   = -1
    signal_bar  = -1
    entry_price = 0.0
    sl_price    = 0.0
    tp_price    = 0.0
    mae         = 0.0
    mfe         = 0.0
    n_trades    = 0

    for t in range(n_bars):
        if not in_trade:
            sig = signal_arr[t]
            if sig != 0 and t + 1 < n_bars:
                in_trade    = True
                side        = sig
                signal_bar  = t
                entry_bar   = t + 1
                entry_price = open_arr[t + 1]
                sl_price    = entry_price - sl_pts * side
                tp_price    = entry_price + tp_pts * side
                mae         = 0.0
                mfe         = 0.0

        if in_trade and t >= entry_bar:
            # Update MAE / MFE (in points, from entry perspective)
            if side == 1:
                bar_adv = low_arr[t]  - entry_price   # negative = adverse for long
                bar_fav = high_arr[t] - entry_price   # positive = favorable for long
            else:
                bar_adv = entry_price - high_arr[t]   # negative = adverse for short
                bar_fav = entry_price - low_arr[t]    # positive = favorable for short

            if bar_adv < mae:
                mae = bar_adv
            if bar_fav > mfe:
                mfe = bar_fav

            # Check SL / TP hit
            if side == 1:
                sl_hit = low_arr[t]  <= sl_price
                tp_hit = high_arr[t] >= tp_price
            else:
                sl_hit = high_arr[t] >= sl_price
                tp_hit = low_arr[t]  <= tp_price

            exit_reason    = np.int8(0)
            exit_price     = 0.0
            label_tp_first = np.int8(0)

            if sl_hit and tp_hit:
                if same_bar_tp_first:
                    exit_reason    = np.int8(2)
                    exit_price     = tp_price
                    label_tp_first = np.int8(1)
                else:
                    exit_reason    = np.int8(1)
                    exit_price     = sl_price
                    label_tp_first = np.int8(0)
            elif tp_hit:
                exit_reason    = np.int8(2)
                exit_price     = tp_price
                label_tp_first = np.int8(1)
            elif sl_hit:
                exit_reason    = np.int8(1)
                exit_price     = sl_price
                label_tp_first = np.int8(0)
            elif exit_on_opposite and signal_arr[t] == -side and t + 1 < n_bars:
                exit_reason    = np.int8(3)
                exit_price     = open_arr[t + 1]
                label_tp_first = np.int8(0)

            # End of data: close at last close
            if exit_reason == 0 and t == n_bars - 1:
                exit_reason    = np.int8(4)
                exit_price     = close_arr[t]
                label_tp_first = np.int8(0)

            if exit_reason != 0:
                out_side[n_trades]          = side
                out_signal_bar[n_trades]    = signal_bar
                out_entry_bar[n_trades]     = entry_bar
                out_exit_bar[n_trades]      = t
                out_entry_price[n_trades]   = entry_price
                out_exit_price[n_trades]    = exit_price
                out_sl[n_trades]            = sl_price
                out_tp[n_trades]            = tp_price
                out_exit_reason[n_trades]   = exit_reason
                out_mae[n_trades]           = mae
                out_mfe[n_trades]           = mfe
                out_hold_bars[n_trades]     = t - entry_bar + 1
                out_label_tp_first[n_trades] = label_tp_first
                n_trades += 1
                in_trade = False
                side     = np.int8(0)

    return (
        out_side[:n_trades],
        out_signal_bar[:n_trades],
        out_entry_bar[:n_trades],
        out_exit_bar[:n_trades],
        out_entry_price[:n_trades],
        out_exit_price[:n_trades],
        out_sl[:n_trades],
        out_tp[:n_trades],
        out_exit_reason[:n_trades],
        out_mae[:n_trades],
        out_mfe[:n_trades],
        out_hold_bars[:n_trades],
        out_label_tp_first[:n_trades],
    )


# ── Layer 1b: Pure-Python / NumPy fallback (identical logic, no @njit) ───────

def _backtest_core_numpy(
    open_arr,
    high_arr,
    low_arr,
    close_arr,
    signal_arr,
    n_bars,
    sl_pts,
    tp_pts,
    same_bar_tp_first,
    exit_on_opposite,
):
    """Pure-Python/NumPy fallback — same logic as _backtest_core, no Numba."""
    max_trades = n_bars // 2 + 1

    out_side          = np.empty(max_trades, dtype=np.int8)
    out_signal_bar    = np.empty(max_trades, dtype=np.int64)
    out_entry_bar     = np.empty(max_trades, dtype=np.int64)
    out_exit_bar      = np.empty(max_trades, dtype=np.int64)
    out_entry_price   = np.empty(max_trades, dtype=np.float64)
    out_exit_price    = np.empty(max_trades, dtype=np.float64)
    out_sl            = np.empty(max_trades, dtype=np.float64)
    out_tp            = np.empty(max_trades, dtype=np.float64)
    out_exit_reason   = np.empty(max_trades, dtype=np.int8)
    out_mae           = np.empty(max_trades, dtype=np.float64)
    out_mfe           = np.empty(max_trades, dtype=np.float64)
    out_hold_bars     = np.empty(max_trades, dtype=np.int64)
    out_label_tp_first = np.empty(max_trades, dtype=np.int8)

    in_trade    = False
    side        = 0
    entry_bar   = -1
    signal_bar  = -1
    entry_price = 0.0
    sl_price    = 0.0
    tp_price    = 0.0
    mae         = 0.0
    mfe         = 0.0
    n_trades    = 0

    for t in range(n_bars):
        if not in_trade:
            sig = int(signal_arr[t])
            if sig != 0 and t + 1 < n_bars:
                in_trade    = True
                side        = sig
                signal_bar  = t
                entry_bar   = t + 1
                entry_price = float(open_arr[t + 1])
                sl_price    = entry_price - sl_pts * side
                tp_price    = entry_price + tp_pts * side
                mae         = 0.0
                mfe         = 0.0

        if in_trade and t >= entry_bar:
            if side == 1:
                bar_adv = float(low_arr[t])  - entry_price
                bar_fav = float(high_arr[t]) - entry_price
            else:
                bar_adv = entry_price - float(high_arr[t])
                bar_fav = entry_price - float(low_arr[t])

            if bar_adv < mae:
                mae = bar_adv
            if bar_fav > mfe:
                mfe = bar_fav

            if side == 1:
                sl_hit = float(low_arr[t])  <= sl_price
                tp_hit = float(high_arr[t]) >= tp_price
            else:
                sl_hit = float(high_arr[t]) >= sl_price
                tp_hit = float(low_arr[t])  <= tp_price

            exit_reason    = 0
            exit_price     = 0.0
            label_tp_first = 0

            if sl_hit and tp_hit:
                if same_bar_tp_first:
                    exit_reason    = 2
                    exit_price     = tp_price
                    label_tp_first = 1
                else:
                    exit_reason    = 1
                    exit_price     = sl_price
                    label_tp_first = 0
            elif tp_hit:
                exit_reason    = 2
                exit_price     = tp_price
                label_tp_first = 1
            elif sl_hit:
                exit_reason    = 1
                exit_price     = sl_price
                label_tp_first = 0
            elif exit_on_opposite and int(signal_arr[t]) == -side and t + 1 < n_bars:
                exit_reason    = 3
                exit_price     = float(open_arr[t + 1])
                label_tp_first = 0

            if exit_reason == 0 and t == n_bars - 1:
                exit_reason    = 4
                exit_price     = float(close_arr[t])
                label_tp_first = 0

            if exit_reason != 0:
                out_side[n_trades]           = side
                out_signal_bar[n_trades]     = signal_bar
                out_entry_bar[n_trades]      = entry_bar
                out_exit_bar[n_trades]       = t
                out_entry_price[n_trades]    = entry_price
                out_exit_price[n_trades]     = exit_price
                out_sl[n_trades]             = sl_price
                out_tp[n_trades]             = tp_price
                out_exit_reason[n_trades]    = exit_reason
                out_mae[n_trades]            = mae
                out_mfe[n_trades]            = mfe
                out_hold_bars[n_trades]      = t - entry_bar + 1
                out_label_tp_first[n_trades] = label_tp_first
                n_trades += 1
                in_trade = False
                side     = 0

    return (
        out_side[:n_trades],
        out_signal_bar[:n_trades],
        out_entry_bar[:n_trades],
        out_exit_bar[:n_trades],
        out_entry_price[:n_trades],
        out_exit_price[:n_trades],
        out_sl[:n_trades],
        out_tp[:n_trades],
        out_exit_reason[:n_trades],
        out_mae[:n_trades],
        out_mfe[:n_trades],
        out_hold_bars[:n_trades],
        out_label_tp_first[:n_trades],
    )


# ── Layer 2: Python wrapper ───────────────────────────────────────────────────

_EXIT_REASON_MAP = {1: "stop", 2: "take_profit", 3: "opposite_signal", 4: "end_of_data"}


def run_backtest(df: pd.DataFrame, cfg, version: str = "V1") -> dict:
    """Run the bar-by-bar backtest on df (must already have indicators + signals).

    Parameters
    ----------
    df      : DataFrame with columns: open, high, low, close, time, signal,
              ema_fast, ema_slow, ema_spread, zscore (optional), volume,
              volume_zscore (optional), atr (optional), session_break.
    cfg     : config module with all required constants.
    version : iteration label, e.g. "V1".

    Returns
    -------
    dict with keys: trades, equity_curve, n_trades, final_equity, version.
    """
    # ── 1. Extract arrays ────────────────────────────────────────────────────
    open_arr   = df["open"].to_numpy(dtype=np.float64)
    high_arr   = df["high"].to_numpy(dtype=np.float64)
    low_arr    = df["low"].to_numpy(dtype=np.float64)
    close_arr  = df["close"].to_numpy(dtype=np.float64)
    signal_arr = df["signal"].to_numpy(dtype=np.int8)
    n_bars     = len(df)

    # ── 2. Compute SL/TP distances ───────────────────────────────────────────
    sl_pts = float(cfg.STOP_FIXED_PTS)   # positive distance in points
    tp_pts = sl_pts * float(cfg.MIN_RR)  # TP = SL * RR

    same_bar_tp_first = (cfg.SAME_BAR_COLLISION == "tp_first")
    exit_on_opposite  = bool(cfg.EXIT_ON_OPPOSITE_SIGNAL)

    # ── 3. Dispatch: Cython → Numba → NumPy ─────────────────────────────────
    use_cython = CYTHON_AVAILABLE
    use_numba  = (not use_cython) and bool(getattr(cfg, "USE_NUMBA", True)) and NUMBA_AVAILABLE
    if use_cython:
        core_fn = backtest_core_cy
    elif use_numba:
        core_fn = _backtest_core
    else:
        core_fn = _backtest_core_numpy

    (
        raw_side,
        raw_signal_bar,
        raw_entry_bar,
        raw_exit_bar,
        raw_entry_price,
        raw_exit_price,
        raw_sl,
        raw_tp,
        raw_exit_reason,
        raw_mae,
        raw_mfe,
        raw_hold_bars,
        raw_label_tp_first,
    ) = core_fn(
        open_arr, high_arr, low_arr, close_arr,
        signal_arr, n_bars,
        sl_pts, tp_pts,
        same_bar_tp_first, exit_on_opposite,
    )

    n_trades_raw = len(raw_side)

    # ── 4. Resolve column positions for fast .iat access ─────────────────────
    col_pos = {col: df.columns.get_loc(col) for col in df.columns}

    def _get(row_idx: int, col: str, default=float("nan")):
        """Safe column read; returns default if column absent."""
        if col in col_pos:
            return df.iat[row_idx, col_pos[col]]
        return default

    # ── 5. Post-process trades ───────────────────────────────────────────────
    trades: List[Dict[str, Any]] = []
    equity = float(cfg.STARTING_EQUITY)

    # For equity curve: track bar -> equity at close of that bar
    # We'll build equity_curve after we know all trade exit bars.
    # Strategy: maintain a sorted list of (exit_bar, equity_after) events.
    exit_events = []  # list of (exit_bar_idx, equity_after)

    for i in range(n_trades_raw):
        side          = int(raw_side[i])
        entry_price   = float(raw_entry_price[i])
        exit_price    = float(raw_exit_price[i])
        sl_price      = float(raw_sl[i])
        tp_price      = float(raw_tp[i])
        entry_bar_idx = int(raw_entry_bar[i])
        exit_bar_idx  = int(raw_exit_bar[i])
        signal_bar_idx = int(raw_signal_bar[i])
        hold_bars     = int(raw_hold_bars[i])

        stop_distance_pts   = abs(entry_price - sl_price)
        target_distance_pts = abs(tp_price - entry_price)
        rr_planned = (
            target_distance_pts / stop_distance_pts
            if stop_distance_pts > 1e-12 else 0.0
        )

        contracts = (
            int(equity * cfg.RISK_PCT // (stop_distance_pts * cfg.MNQ_DOLLARS_PER_POINT))
            if stop_distance_pts > 0 else 0
        )
        risk_dollars      = stop_distance_pts * contracts * cfg.MNQ_DOLLARS_PER_POINT
        max_risk_allowed  = equity * cfg.RISK_PCT

        gross_pnl_dollars = (
            (exit_price - entry_price) * side * contracts * cfg.MNQ_DOLLARS_PER_POINT
        )
        net_pnl_dollars = gross_pnl_dollars  # commission = 0 for now
        r_multiple = net_pnl_dollars / risk_dollars if risk_dollars > 0 else 0.0

        exit_reason_str = _EXIT_REASON_MAP[int(raw_exit_reason[i])]

        entry_time = df.iat[entry_bar_idx, col_pos["time"]]
        exit_time  = df.iat[exit_bar_idx,  col_pos["time"]]

        # Signal-context features from the SIGNAL bar (no lookahead)
        sig_row_close  = float(_get(signal_bar_idx, "close"))
        sig_row_open   = float(_get(signal_bar_idx, "open"))
        sig_row_high   = float(_get(signal_bar_idx, "high"))
        sig_row_low    = float(_get(signal_bar_idx, "low"))
        sig_row_volume = float(_get(signal_bar_idx, "volume"))
        sig_ema_fast   = float(_get(signal_bar_idx, "ema_fast"))
        sig_ema_slow   = float(_get(signal_bar_idx, "ema_slow"))
        sig_ema_spread = float(_get(signal_bar_idx, "ema_spread"))
        sig_zscore     = float(_get(signal_bar_idx, "zscore", float("nan")))
        sig_vol_z      = float(_get(signal_bar_idx, "volume_zscore", float("nan")))
        sig_atr        = float(_get(signal_bar_idx, "atr", float("nan")))
        sig_session_break = bool(_get(signal_bar_idx, "session_break", False))

        prev_idx = max(0, signal_bar_idx - 1)
        prev_zscore = float(_get(prev_idx, "zscore", float("nan")))

        zscore_delta = (
            sig_zscore - prev_zscore
            if not (math.isnan(sig_zscore) or math.isnan(prev_zscore))
            else float("nan")
        )

        mae_pts = float(raw_mae[i])
        mfe_pts = float(raw_mfe[i])

        equity_before = equity
        equity_after  = equity + net_pnl_dollars
        equity        = equity_after  # carry forward

        exit_events.append((exit_bar_idx, equity_after))

        trade = {
            # Identity + timing
            "trade_id":         i + 1,
            "version":          version,
            "source_iteration": version,
            "symbol":           "MNQ",
            "entry_time":       entry_time,
            "exit_time":        exit_time,
            "entry_date":       entry_time.date(),
            "day_of_week":      entry_time.dayofweek,
            "time_of_day_hhmm": entry_time.strftime("%H%M"),
            "signal_bar_index": signal_bar_idx,
            "entry_bar_index":  entry_bar_idx,
            "exit_bar_index":   exit_bar_idx,
            "session_break_flag": sig_session_break,
            # Direction + execution
            "side":                   "long" if side == 1 else "short",
            "entry_signal_price":     sig_row_close,
            "entry_fill_price":       entry_price,
            "exit_signal_price":      float(sl_price if exit_reason_str == "stop" else tp_price),
            "exit_fill_price":        exit_price,
            "slippage_entry_points":  0.0,
            "slippage_exit_points":   0.0,
            "slippage_total_points":  0.0,
            "spread_at_entry_points": 0.0,
            "exit_reason":            exit_reason_str,
            # Risk + sizing
            "sl_price":                  sl_price,
            "tp_price":                  tp_price,
            "stop_distance_points":      stop_distance_pts,
            "target_distance_points":    target_distance_pts,
            "rr_planned":                rr_planned,
            "contracts":                 contracts,
            "position_notional":         entry_price * contracts * cfg.MNQ_DOLLARS_PER_POINT,
            "risk_dollars":              risk_dollars,
            "max_risk_allowed_dollars":  max_risk_allowed,
            "risk_utilization":          risk_dollars / max_risk_allowed if max_risk_allowed > 0 else 0.0,
            "commission_dollars":        0.0,
            "fees_dollars":              0.0,
            # PnL + path metrics
            "gross_pnl_dollars": gross_pnl_dollars,
            "net_pnl_dollars":   net_pnl_dollars,
            "r_multiple":        r_multiple,
            "hold_bars":         hold_bars,
            "hold_minutes":      hold_bars,
            "mae_points":        mae_pts,
            "mfe_points":        mfe_pts,
            "mae_dollars":       mae_pts * contracts * cfg.MNQ_DOLLARS_PER_POINT,
            "mfe_dollars":       mfe_pts * contracts * cfg.MNQ_DOLLARS_PER_POINT,
            "equity_before":     equity_before,
            "equity_after":      equity_after,
            # Signal-context features (signal bar — no lookahead)
            "zscore_entry":                   sig_zscore,
            "zscore_prev":                    prev_zscore,
            "zscore_delta":                   zscore_delta,
            "z_band_k":                       float(cfg.Z_BAND_K),
            "ema_fast":                       sig_ema_fast,
            "ema_slow":                       sig_ema_slow,
            "ema_spread":                     sig_ema_spread,
            "close_price":                    sig_row_close,
            "open_price":                     sig_row_open,
            "high_price":                     sig_row_high,
            "low_price":                      sig_row_low,
            "bar_range_points":               sig_row_high - sig_row_low,
            "bar_body_points":                abs(sig_row_close - sig_row_open),
            "volume":                         sig_row_volume,
            "volume_zscore":                  sig_vol_z,
            "atr_points":                     sig_atr,
            "distance_to_ema_fast_points":    sig_row_close - sig_ema_fast,
            "distance_to_ema_slow_points":    sig_row_close - sig_ema_slow,
            # Labels + QA
            "label_win":           int(net_pnl_dollars > 0),
            "label_hit_tp_first":  int(raw_label_tp_first[i]),
            "data_quality_flag":   0,
            "rule_violation_flag": int(rr_planned < cfg.MIN_RR - 1e-9),
            # Track A: entry quality metrics
            "mfe_mae_ratio": (
                mfe_pts / abs(mae_pts)
                if not math.isnan(mae_pts) and abs(mae_pts) >= 0.25
                else float("nan")
            ),
            "entry_score": score_single_trade(
                zscore_entry=sig_zscore,
                volume_zscore=sig_vol_z,
                ema_spread=sig_ema_spread,
                bar_body_points=abs(sig_row_close - sig_row_open),
                bar_range_points=sig_row_high - sig_row_low,
                time_of_day_hhmm=entry_time.strftime("%H%M"),
                cfg=cfg,
            ),
        }
        trades.append(trade)

    # ── 6. Build equity curve (one row per bar) ───────────────────────────────
    equity_curve: List[Dict[str, Any]] = []
    if n_trades_raw == 0:
        current_equity = float(cfg.STARTING_EQUITY)
        for t in range(n_bars):
            equity_curve.append({
                "bar_idx": t,
                "time":    df.iat[t, col_pos["time"]],
                "equity":  current_equity,
            })
    else:
        # Build a bar -> equity_after map from exit events
        exit_map: Dict[int, float] = {}
        for exit_bar, eq_after in exit_events:
            # If multiple trades exit on the same bar (shouldn't happen with
            # one-at-a-time), keep the last (highest trade_id, chronological).
            exit_map[exit_bar] = eq_after

        current_equity = float(cfg.STARTING_EQUITY)
        for t in range(n_bars):
            if t in exit_map:
                current_equity = exit_map[t]
            equity_curve.append({
                "bar_idx": t,
                "time":    df.iat[t, col_pos["time"]],
                "equity":  current_equity,
            })

    return {
        "trades":       trades,
        "equity_curve": equity_curve,
        "n_trades":     n_trades_raw,
        "final_equity": equity,
        "version":      version,
    }
