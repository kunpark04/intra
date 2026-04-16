"""Bar-by-bar backtest engine for the MNQ 1-minute strategy.

Three engine tiers are tried in order of speed:

1. **Cython AOT** — `backtest_core_cy` from `src.cython_ext.backtest_core`;
   preferred when the extension is built (no JIT overhead, ~5-10× NumPy).
2. **Numba JIT** — `_backtest_core`; good for development, first-call
   compile cost is amortised with `cache=True`.
3. **NumPy / pure Python** — `_backtest_core_numpy`; always available,
   identical logic to the JIT core, last-resort fallback.

All three cores share the same signature and state machine. The Python
wrapper `run_backtest` extracts arrays from the DataFrame, dispatches to
the best available core, post-processes the output into `LOG_SCHEMA.md`
trade records, builds the bar-by-bar equity curve, and returns a dict
with `trades`, `equity_curve`, `n_trades`, `final_equity`, `version`.
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
        """No-op stand-in for `numba.njit` when Numba is unavailable."""
        def decorator(fn):
            """Pass-through: return `fn` unchanged."""
            return fn
        # Support both @njit and @njit(cache=True) forms
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


# ── Layer 1a: Numba JIT core ──────────────────────────────────────────────────

@njit(cache=True)
def _backtest_core(
    open_arr,            # float64[:]
    high_arr,            # float64[:]
    low_arr,             # float64[:]
    close_arr,           # float64[:]
    signal_arr,          # int8[:]
    n_bars,              # int
    sl_pts,              # float  — stop distance in points (positive)
    tp_pts,              # float  — tp distance in points (positive)
    same_bar_tp_first,   # bool
    exit_on_opposite,    # bool
    use_breakeven_stop,  # bool  — move SL to entry once +1R profit
    max_hold_bars,       # int   — exit at next open after N bars (0 = disabled)
    hour_arr,            # int64[:] — bar hour-of-day (0–23)
    tod_exit_hour,       # int   — force close at this hour (0 = disabled)
):
    """Numba JIT bar-by-bar state machine for one position at a time.

    Exit priority per bar: same-bar SL+TP (configurable), TP only, SL only,
    opposite signal, max hold, time-of-day exit, then end-of-data. Tracks
    MAE/MFE from entry and moves the effective stop to break-even once
    +1R is reached (when enabled). The returned SL in `out_sl` is the
    original pre-breakeven stop so downstream R-sizing stays consistent.

    Args:
        open_arr: float64 bar opens.
        high_arr: float64 bar highs.
        low_arr: float64 bar lows.
        close_arr: float64 bar closes.
        signal_arr: int8 signals (`1` long, `-1` short, `0` flat).
        n_bars: Number of bars.
        sl_pts: Stop distance in positive points.
        tp_pts: Take-profit distance in positive points.
        same_bar_tp_first: True → resolve same-bar SL+TP as TP; False → SL.
        exit_on_opposite: True → close on opposite-side signal.
        use_breakeven_stop: True → move effective SL to entry after +1R.
        max_hold_bars: Force exit after this many bars in position
            (`0` disables).
        hour_arr: int64 hour-of-day per bar (only read when TOD exit is on).
        tod_exit_hour: Force-close hour (0 disables).

    Returns:
        13-tuple of per-trade arrays sliced to the actual trade count
        (side, signal_bar, entry_bar, exit_bar, entry_price, exit_price,
        sl, tp, exit_reason, mae, mfe, hold_bars, label_tp_first).
    """
    max_trades = n_bars // 2 + 1

    out_side           = np.empty(max_trades, dtype=np.int8)
    out_signal_bar     = np.empty(max_trades, dtype=np.int64)
    out_entry_bar      = np.empty(max_trades, dtype=np.int64)
    out_exit_bar       = np.empty(max_trades, dtype=np.int64)
    out_entry_price    = np.empty(max_trades, dtype=np.float64)
    out_exit_price     = np.empty(max_trades, dtype=np.float64)
    out_sl             = np.empty(max_trades, dtype=np.float64)
    out_tp             = np.empty(max_trades, dtype=np.float64)
    out_exit_reason    = np.empty(max_trades, dtype=np.int8)
    out_mae            = np.empty(max_trades, dtype=np.float64)
    out_mfe            = np.empty(max_trades, dtype=np.float64)
    out_hold_bars      = np.empty(max_trades, dtype=np.int64)
    out_label_tp_first = np.empty(max_trades, dtype=np.int8)

    in_trade            = False
    side                = np.int8(0)
    entry_bar           = -1
    signal_bar          = -1
    entry_price         = 0.0
    sl_price            = 0.0
    effective_sl        = 0.0
    tp_price            = 0.0
    mae                 = 0.0
    mfe                 = 0.0
    breakeven_activated = False
    bars_held           = 0
    n_trades            = 0

    for t in range(n_bars):
        if not in_trade:
            sig = signal_arr[t]
            if sig != 0 and t + 1 < n_bars:
                in_trade            = True
                side                = sig
                signal_bar          = t
                entry_bar           = t + 1
                entry_price         = open_arr[t + 1]
                sl_price            = entry_price - sl_pts * side
                effective_sl        = sl_price
                tp_price            = entry_price + tp_pts * side
                mae                 = 0.0
                mfe                 = 0.0
                breakeven_activated = False
                bars_held           = 0

        if in_trade and t >= entry_bar:
            bars_held += 1

            if side == 1:
                bar_adv = low_arr[t]  - entry_price
                bar_fav = high_arr[t] - entry_price
            else:
                bar_adv = entry_price - high_arr[t]
                bar_fav = entry_price - low_arr[t]

            if bar_adv < mae:
                mae = bar_adv
            if bar_fav > mfe:
                mfe = bar_fav

            # Breakeven: once +1R profit, move effective SL to entry
            if use_breakeven_stop and not breakeven_activated:
                if bar_fav >= sl_pts:
                    effective_sl        = entry_price
                    breakeven_activated = True

            # SL / TP hit detection (uses effective_sl)
            if side == 1:
                sl_hit = low_arr[t]  <= effective_sl
                tp_hit = high_arr[t] >= tp_price
            else:
                sl_hit = high_arr[t] >= effective_sl
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
                    exit_price     = effective_sl
            elif tp_hit:
                exit_reason    = np.int8(2)
                exit_price     = tp_price
                label_tp_first = np.int8(1)
            elif sl_hit:
                exit_reason    = np.int8(1)
                exit_price     = effective_sl
            elif exit_on_opposite and signal_arr[t] == -side and t + 1 < n_bars:
                exit_reason    = np.int8(3)
                exit_price     = open_arr[t + 1]
            elif max_hold_bars > 0 and bars_held >= max_hold_bars and t + 1 < n_bars:
                exit_reason    = np.int8(5)
                exit_price     = open_arr[t + 1]
            elif tod_exit_hour > 0 and hour_arr[t] == tod_exit_hour and t + 1 < n_bars:
                exit_reason    = np.int8(5)
                exit_price     = open_arr[t + 1]

            # End of data: close at last close
            if exit_reason == 0 and t == n_bars - 1:
                exit_reason = np.int8(4)
                exit_price  = close_arr[t]

            if exit_reason != 0:
                out_side[n_trades]           = side
                out_signal_bar[n_trades]     = signal_bar
                out_entry_bar[n_trades]      = entry_bar
                out_exit_bar[n_trades]       = t
                out_entry_price[n_trades]    = entry_price
                out_exit_price[n_trades]     = exit_price
                out_sl[n_trades]             = sl_price   # original (pre-breakeven) for sizing
                out_tp[n_trades]             = tp_price
                out_exit_reason[n_trades]    = exit_reason
                out_mae[n_trades]            = mae
                out_mfe[n_trades]            = mfe
                out_hold_bars[n_trades]      = t - entry_bar + 1
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
    use_breakeven_stop,
    max_hold_bars,
    hour_arr,
    tod_exit_hour,
):
    """Pure-Python/NumPy fallback — identical logic to `_backtest_core`.

    Used when neither the Cython extension nor Numba is available. Slow
    compared to the other two cores but requires no compilation. See
    `_backtest_core` for argument and return-value semantics.
    """
    max_trades = n_bars // 2 + 1

    out_side           = np.empty(max_trades, dtype=np.int8)
    out_signal_bar     = np.empty(max_trades, dtype=np.int64)
    out_entry_bar      = np.empty(max_trades, dtype=np.int64)
    out_exit_bar       = np.empty(max_trades, dtype=np.int64)
    out_entry_price    = np.empty(max_trades, dtype=np.float64)
    out_exit_price     = np.empty(max_trades, dtype=np.float64)
    out_sl             = np.empty(max_trades, dtype=np.float64)
    out_tp             = np.empty(max_trades, dtype=np.float64)
    out_exit_reason    = np.empty(max_trades, dtype=np.int8)
    out_mae            = np.empty(max_trades, dtype=np.float64)
    out_mfe            = np.empty(max_trades, dtype=np.float64)
    out_hold_bars      = np.empty(max_trades, dtype=np.int64)
    out_label_tp_first = np.empty(max_trades, dtype=np.int8)

    in_trade            = False
    side                = 0
    entry_bar           = -1
    signal_bar          = -1
    entry_price         = 0.0
    sl_price            = 0.0
    effective_sl        = 0.0
    tp_price            = 0.0
    mae                 = 0.0
    mfe                 = 0.0
    breakeven_activated = False
    bars_held           = 0
    n_trades            = 0

    for t in range(n_bars):
        if not in_trade:
            sig = int(signal_arr[t])
            if sig != 0 and t + 1 < n_bars:
                in_trade            = True
                side                = sig
                signal_bar          = t
                entry_bar           = t + 1
                entry_price         = float(open_arr[t + 1])
                sl_price            = entry_price - sl_pts * side
                effective_sl        = sl_price
                tp_price            = entry_price + tp_pts * side
                mae                 = 0.0
                mfe                 = 0.0
                breakeven_activated = False
                bars_held           = 0

        if in_trade and t >= entry_bar:
            bars_held += 1

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

            # Breakeven: once +1R profit, move effective SL to entry
            if use_breakeven_stop and not breakeven_activated:
                if bar_fav >= sl_pts:
                    effective_sl        = entry_price
                    breakeven_activated = True

            if side == 1:
                sl_hit = float(low_arr[t])  <= effective_sl
                tp_hit = float(high_arr[t]) >= tp_price
            else:
                sl_hit = float(high_arr[t]) >= effective_sl
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
                    exit_price     = effective_sl
            elif tp_hit:
                exit_reason    = 2
                exit_price     = tp_price
                label_tp_first = 1
            elif sl_hit:
                exit_reason    = 1
                exit_price     = effective_sl
            elif exit_on_opposite and int(signal_arr[t]) == -side and t + 1 < n_bars:
                exit_reason    = 3
                exit_price     = float(open_arr[t + 1])
            elif max_hold_bars > 0 and bars_held >= max_hold_bars and t + 1 < n_bars:
                exit_reason    = 5
                exit_price     = float(open_arr[t + 1])
            elif tod_exit_hour > 0 and int(hour_arr[t]) == tod_exit_hour and t + 1 < n_bars:
                exit_reason    = 5
                exit_price     = float(open_arr[t + 1])

            if exit_reason == 0 and t == n_bars - 1:
                exit_reason = 4
                exit_price  = float(close_arr[t])

            if exit_reason != 0:
                out_side[n_trades]           = side
                out_signal_bar[n_trades]     = signal_bar
                out_entry_bar[n_trades]      = entry_bar
                out_exit_bar[n_trades]       = t
                out_entry_price[n_trades]    = entry_price
                out_exit_price[n_trades]     = exit_price
                out_sl[n_trades]             = sl_price   # original (pre-breakeven) for sizing
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

_EXIT_REASON_MAP = {1: "stop", 2: "take_profit", 3: "opposite_signal", 4: "end_of_data", 5: "time_exit"}


def run_backtest(df: pd.DataFrame, cfg, version: str = "V1") -> dict:
    """Run the bar-by-bar backtest and return trades + equity curve.

    Expects `df` to already carry indicators (from `indicators.add_indicators`)
    and signals (from `strategy.generate_signals`). Dispatches to the
    fastest available core (Cython → Numba → NumPy), then post-processes
    each trade into a full `LOG_SCHEMA.md` record with sizing, PnL,
    MAE/MFE, signal-bar features, Track A entry scores, and the
    break-even / rule-violation flags.

    Args:
        df: DataFrame with `open`, `high`, `low`, `close`, `time`, `signal`,
            `ema_fast`, `ema_slow`, `ema_spread`, and optionally `zscore`,
            `volume_zscore`, `atr`, `session_break`, `bar_hour`.
        cfg: Config module exposing `STOP_FIXED_PTS`, `MIN_RR`,
            `SAME_BAR_COLLISION`, `EXIT_ON_OPPOSITE_SIGNAL`,
            `STARTING_EQUITY`, `RISK_PCT`, `MNQ_DOLLARS_PER_POINT`,
            `Z_BAND_K`, and optional `USE_BREAKEVEN_STOP`, `MAX_HOLD_BARS`,
            `TOD_EXIT_HOUR`, `USE_NUMBA`.
        version: Iteration label stamped into each trade record.
            Default ``"V1"``.

    Returns:
        Dict with keys:

        - ``trades`` (list): per-trade dicts in `LOG_SCHEMA.md` form.
        - ``equity_curve`` (list): per-bar `{bar_idx, time, equity}` rows.
        - ``n_trades`` (int): trade count.
        - ``final_equity`` (float): equity after the last trade.
        - ``version`` (str): the passed-in version label.
    """
    # ── 1. Extract arrays ────────────────────────────────────────────────────
    open_arr   = df["open"].to_numpy(dtype=np.float64)
    high_arr   = df["high"].to_numpy(dtype=np.float64)
    low_arr    = df["low"].to_numpy(dtype=np.float64)
    close_arr  = df["close"].to_numpy(dtype=np.float64)
    signal_arr = df["signal"].to_numpy(dtype=np.int8)
    n_bars     = len(df)

    # Hour array for TOD exit (zero array if column absent)
    if "bar_hour" in df.columns:
        hour_arr = df["bar_hour"].to_numpy(dtype=np.int64)
    else:
        hour_arr = np.zeros(n_bars, dtype=np.int64)

    # ── 2. Compute SL/TP distances ───────────────────────────────────────────
    sl_pts = float(cfg.STOP_FIXED_PTS)   # positive distance in points
    tp_pts = sl_pts * float(cfg.MIN_RR)  # TP = SL * RR

    same_bar_tp_first  = (cfg.SAME_BAR_COLLISION == "tp_first")
    exit_on_opposite   = bool(cfg.EXIT_ON_OPPOSITE_SIGNAL)
    use_breakeven_stop = bool(getattr(cfg, "USE_BREAKEVEN_STOP", False))
    max_hold_bars      = int(getattr(cfg, "MAX_HOLD_BARS", 0))
    tod_exit_hour      = int(getattr(cfg, "TOD_EXIT_HOUR", 0))

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
        use_breakeven_stop, max_hold_bars,
        hour_arr, tod_exit_hour,
    )

    n_trades_raw = len(raw_side)

    # ── 4. Resolve column positions for fast .iat access ─────────────────────
    col_pos = {col: df.columns.get_loc(col) for col in df.columns}

    def _get(row_idx: int, col: str, default=float("nan")):
        """Safe positional cell read; returns `default` if the column is absent.

        Args:
            row_idx: iloc row position.
            col: Column name.
            default: Value returned when `col` is not in `df`.

        Returns:
            Cell value or `default`.
        """
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

        fixed_risk = getattr(cfg, "FIXED_RISK_DOLLARS", None)
        max_risk_allowed = (
            float(fixed_risk) if fixed_risk is not None else equity * cfg.RISK_PCT
        )
        contracts = (
            int(max_risk_allowed // (stop_distance_pts * cfg.MNQ_DOLLARS_PER_POINT))
            if stop_distance_pts > 0 else 0
        )
        risk_dollars      = stop_distance_pts * contracts * cfg.MNQ_DOLLARS_PER_POINT

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
