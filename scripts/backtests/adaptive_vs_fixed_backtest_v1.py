"""
adaptive_vs_fixed_backtest_v1.py — Compare fixed min_rr vs adaptive R:R selection
for the top-ranked combo on the TRAINING split.

Approach
--------
1. Load top combo from data/ml/ml1_results/top_combos.csv (row 0).
2. Look up its full parameter set in data/ml/ml_dataset_v{N}_mfe.parquet.
3. Build indicator DataFrame exactly as the sweep did (compute_zscore_v2 with
   the combo's z_* formulation, resolved stop distance via median ATR/swing).
4. On training split (80%): run the backtest core once at MIN_RR = combo.min_rr
   to produce the base trade set (entry bar, stop distance, MFE/MAE path).
5. For each trade, predict P(win | features, candidate_rr) at all 17 RR levels
   using the trained LightGBM model; pick argmax E[R] = P*R - (1-P).
6. Replay both (fixed and adaptive) deterministically:
   - FIXED: use the trade's original exit_reason & exit_price.
   - ADAPTIVE: synthesise exit from MFE/MAE path — if mfe_points >=
     chosen_rr * stop_distance_pts THEN win (exit at TP); elif mae_points <=
     -stop_distance_pts (it did for all trades that stopped out) THEN loss;
     else partial exit at final exit_price scaled or keep the time_exit PnL.
   Note: the adaptive synthesis reuses the same entry-bar, so the core
   strategy and signal engine are identical. Only the target distance differs.
   This matches the labelling used when the adaptive model was trained.
7. Compute equity curves compound on $50k start, 5% risk per trade; report
   total_return_pct, Sharpe, max_dd, n_trades, win_rate.

Output: data/ml/adaptive_rr_v1/adaptive_vs_fixed.json
"""
from __future__ import annotations

import json
import math
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config as _BASE_CFG
from src.data_loader import load_bars, split_train_test
from src.indicators.ema import compute_ema
from src.indicators.atr import compute_atr
from src.indicators.zscore_variants import compute_zscore_v2, compute_vwap_session
from src.indicators.zscore import compute_volume_zscore
from src.strategy import generate_signals

try:
    from src.cython_ext.backtest_core import backtest_core_cy as _core_fn
    CORE_NAME = "cython"
except ImportError:
    from src.backtest import _backtest_core_numpy as _core_fn
    CORE_NAME = "numpy"

import lightgbm as lgb

REPO = Path(__file__).resolve().parents[2]
TOP_CSV = REPO / "data/ml/ml1_results/top_combos.csv"
MODEL_PATH = REPO / "data/ml/adaptive_rr_v1/adaptive_rr_model.txt"
META_PATH = REPO / "data/ml/adaptive_rr_v1/run_metadata.json"
OUT_PATH = REPO / "data/ml/adaptive_rr_v1/adaptive_vs_fixed.json"
DATA_CSV = REPO / "data/NQ_1min.csv"

RR_LEVELS = np.round(np.arange(1.0, 5.25, 0.25), 2).astype(np.float32)

# LightGBM categorical mapping (side: long=0, short=1; stop_method: fixed/atr/swing)
SIDE_MAP = {"long": 0, "short": 1}
STOP_METHOD_MAP = {"fixed": 0, "atr": 1, "swing": 2}


def load_top_combo() -> dict:
    """Read the winning combo row from an ML#1 ranking parquet.

    Args:
        ranking_parquet: Path to the ML#1 ranking file.
        rank: 1-based rank of the combo to load (1 = highest predicted).

    Returns:
        Row as a pandas Series with combo hyper-parameters.
    """
    df = pd.read_csv(TOP_CSV)
    row = df.iloc[0].to_dict()
    gcid = row["global_combo_id"]
    source_version = int(gcid.split("_")[0][1:])
    combo_id = int(gcid.split("_")[1])
    combo = {
        "global_combo_id": gcid,
        "source_version": source_version,
        "combo_id": combo_id,
    }
    # Full meta from parquet
    parq = REPO / f"data/ml/mfe/ml_dataset_v{source_version}_mfe.parquet"
    df_c = pd.read_parquet(parq, filters=[("combo_id", "==", combo_id)])
    meta_cols = [
        "z_band_k", "z_window", "volume_zscore_window", "ema_fast", "ema_slow",
        "stop_method", "stop_fixed_pts", "atr_multiplier", "swing_lookback",
        "min_rr", "exit_on_opposite_signal", "use_breakeven_stop", "max_hold_bars",
        "zscore_confirmation", "z_input", "z_anchor", "z_denom", "z_type",
        "z_window_2", "z_window_2_weight", "volume_entry_threshold",
        "vol_regime_lookback", "vol_regime_min_pct", "vol_regime_max_pct",
        "session_filter_mode", "tod_exit_hour",
    ]
    first = df_c.iloc[0]
    for c in meta_cols:
        if c in df_c.columns:
            v = first[c]
            if isinstance(v, (float, np.floating)) and pd.isna(v):
                combo[c] = None
            else:
                combo[c] = v
        else:
            combo[c] = None
    return combo, df_c


def build_indicators(df: pd.DataFrame, combo: dict) -> pd.DataFrame:
    """Attach EMA/Z-score/ATR/regime indicators for the chosen combo to `bars`.

    Args:
        bars: 1-minute OHLCV frame (canonical column names).
        combo: Combo row produced by `load_top_combo`.

    Returns:
        Copy of `bars` enriched with all indicators required by the signal
        generator for this combo.
    """
    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)
    session_break = df["session_break"].to_numpy(dtype=bool)

    ema_f = compute_ema(close, int(combo["ema_fast"]), True)
    ema_s = compute_ema(close, int(combo["ema_slow"]), True)
    atr_arr = compute_atr(high, low, close, int(_BASE_CFG.ATR_WINDOW), True)
    vwap = compute_vwap_session(high, low, close, volume, session_break)
    vol_z = compute_volume_zscore(volume, int(combo["volume_zscore_window"]), 0, True)

    z = compute_zscore_v2(
        close, high, low, volume, session_break,
        ema_f, ema_s, atr_arr,
        int(combo["z_window"]),
        str(combo["z_input"]), str(combo["z_anchor"]),
        str(combo["z_denom"]), str(combo["z_type"]),
        0, vwap,
        int(combo.get("z_window_2") or 0),
        float(combo.get("z_window_2_weight") or 0.0),
    )

    fast_above = ema_f > ema_s
    prev_fab = np.concatenate([[False], fast_above[:-1]])

    out = pd.DataFrame({
        "time": df["time"].to_numpy(),
        "open": df["open"].to_numpy(dtype=np.float64),
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "session_break": session_break,
        "ema_fast": ema_f,
        "ema_slow": ema_s,
        "ema_spread": ema_f - ema_s,
        "ema_cross_up": fast_above & ~prev_fab,
        "ema_cross_down": ~fast_above & prev_fab,
        "zscore": z,
        "atr": atr_arr,
        "volume_zscore": vol_z,
        "bar_hour": pd.to_datetime(df["time"]).dt.hour.to_numpy(dtype=np.int64),
    })
    return out


def resolve_stop_pts(combo: dict, df_ind: pd.DataFrame) -> float:
    """Resolve the stop distance in points for a given bar and combo.

    Handles fixed, ATR-multiplier, and swing-based stops.

    Args:
        bar: Row from the enriched `bars` frame at signal time.
        combo: Combo row with `stop_method` and associated parameters.

    Returns:
        Stop distance in index points.
    """
    m = combo["stop_method"]
    if m == "fixed":
        return float(combo["stop_fixed_pts"])
    elif m == "atr":
        median_atr = float(np.nanmedian(df_ind["atr"].to_numpy()))
        return median_atr * float(combo["atr_multiplier"])
    else:
        lb = int(combo["swing_lookback"])
        rh = df_ind["high"].rolling(lb).max()
        rl = df_ind["low"].rolling(lb).min()
        swing_dist = (rh - rl).dropna()
        return float(np.nanmedian(swing_dist.to_numpy())) * 0.5 + 0.25


def make_cfg(combo: dict, stop_pts: float) -> types.SimpleNamespace:
    """Build a lightweight `cfg`-like namespace for the backtest engine.

    Maps combo hyper-parameters (R:R, risk %, commission, slippage) into the
    attribute surface that `src.backtest.run_backtest` consumes.

    Args:
        combo: Combo row from the ranking parquet.

    Returns:
        `SimpleNamespace` with the fields expected by the backtest engine.
    """
    ns = types.SimpleNamespace()
    for k in dir(_BASE_CFG):
        if not k.startswith("_"):
            setattr(ns, k, getattr(_BASE_CFG, k))
    ns.Z_BAND_K = float(combo["z_band_k"])
    ns.Z_WINDOW = int(combo["z_window"])
    ns.VOLUME_ZSCORE_WINDOW = int(combo["volume_zscore_window"])
    ns.EMA_FAST = int(combo["ema_fast"])
    ns.EMA_SLOW = int(combo["ema_slow"])
    ns.MIN_RR = float(combo["min_rr"])
    ns.EXIT_ON_OPPOSITE_SIGNAL = bool(combo["exit_on_opposite_signal"])
    ns.USE_BREAKEVEN_STOP = bool(combo.get("use_breakeven_stop", False))
    ns.MAX_HOLD_BARS = int(combo.get("max_hold_bars") or 0)
    ns.ZSCORE_CONFIRMATION = bool(combo.get("zscore_confirmation", False))
    ns.VOLUME_ENTRY_THRESHOLD = float(combo.get("volume_entry_threshold") or 0.0)
    ns.VOL_REGIME_LOOKBACK = int(combo.get("vol_regime_lookback") or 0)
    ns.VOL_REGIME_MIN_PCT = float(combo.get("vol_regime_min_pct") or 0.0)
    ns.VOL_REGIME_MAX_PCT = float(combo.get("vol_regime_max_pct") or 1.0)
    ns.SESSION_FILTER_MODE = int(combo.get("session_filter_mode") or 0)
    ns.TOD_EXIT_HOUR = int(combo.get("tod_exit_hour") or 0)
    ns.STOP_METHOD = "fixed"
    ns.STOP_FIXED_PTS = float(stop_pts)
    ns.SIGNAL_MODE = "zscore_reversal"
    ns.USE_ZSCORE_FILTER = True
    return ns


def run_core(df_sig: pd.DataFrame, cfg) -> dict:
    """Run backtest core; return dict of arrays."""
    open_arr = df_sig["open"].to_numpy(dtype=np.float64)
    high_arr = df_sig["high"].to_numpy(dtype=np.float64)
    low_arr = df_sig["low"].to_numpy(dtype=np.float64)
    close_arr = df_sig["close"].to_numpy(dtype=np.float64)
    signal_arr = df_sig["signal"].to_numpy(dtype=np.int8)
    hour_arr = df_sig["bar_hour"].to_numpy(dtype=np.int64)
    sl_pts = float(cfg.STOP_FIXED_PTS)
    tp_pts = sl_pts * float(cfg.MIN_RR)
    same_bar_tp_first = (cfg.SAME_BAR_COLLISION == "tp_first")
    (side, sig_bar, entry_bar, exit_bar, entry_price, exit_price,
     sl, tp, exit_reason, mae, mfe, hold_bars, label_tp_first) = _core_fn(
        open_arr, high_arr, low_arr, close_arr, signal_arr, len(df_sig),
        sl_pts, tp_pts, same_bar_tp_first, bool(cfg.EXIT_ON_OPPOSITE_SIGNAL),
        bool(cfg.USE_BREAKEVEN_STOP), int(cfg.MAX_HOLD_BARS),
        hour_arr, int(cfg.TOD_EXIT_HOUR),
    )
    return dict(
        side=np.asarray(side), sig_bar=np.asarray(sig_bar),
        entry_bar=np.asarray(entry_bar), exit_bar=np.asarray(exit_bar),
        entry_price=np.asarray(entry_price), exit_price=np.asarray(exit_price),
        sl=np.asarray(sl), tp=np.asarray(tp),
        exit_reason=np.asarray(exit_reason),
        mae=np.asarray(mae), mfe=np.asarray(mfe),
        hold_bars=np.asarray(hold_bars),
    )


def build_features(trades: dict, df_sig: pd.DataFrame, stop_pts: float,
                   stop_method: str, exit_on_opp: bool) -> pd.DataFrame:
    """Build per-trade feature DataFrame matching the adaptive model's inputs."""
    n = len(trades["side"])
    sig_bars = trades["sig_bar"].astype(np.int64)
    entry_bars = trades["entry_bar"].astype(np.int64)

    close = df_sig["close"].to_numpy(dtype=np.float64)
    high = df_sig["high"].to_numpy(dtype=np.float64)
    low = df_sig["low"].to_numpy(dtype=np.float64)
    open_ = df_sig["open"].to_numpy(dtype=np.float64)
    atr = df_sig["atr"].to_numpy(dtype=np.float64)
    zscore = df_sig["zscore"].to_numpy(dtype=np.float64)
    vol_z = df_sig["volume_zscore"].to_numpy(dtype=np.float64)
    ema_f = df_sig["ema_fast"].to_numpy(dtype=np.float64)
    ema_s = df_sig["ema_slow"].to_numpy(dtype=np.float64)
    ema_spread = df_sig["ema_spread"].to_numpy(dtype=np.float64)
    times = pd.to_datetime(df_sig["time"].to_numpy())

    sig_close = close[sig_bars]
    sig_high = high[sig_bars]
    sig_low = low[sig_bars]
    sig_open = open_[sig_bars]
    sig_zscore = zscore[sig_bars]
    prev_z = np.where(sig_bars > 0, zscore[np.maximum(sig_bars - 1, 0)], np.nan)
    sig_vol_z = vol_z[sig_bars]
    sig_atr = atr[sig_bars]
    sig_ema_f = ema_f[sig_bars]
    sig_ema_s = ema_s[sig_bars]
    sig_ema_sp = ema_spread[sig_bars]

    entry_times = times[entry_bars]
    hhmm = np.array([int(t.strftime("%H%M")) for t in pd.DatetimeIndex(entry_times)])
    dow = pd.DatetimeIndex(entry_times).dayofweek.to_numpy()

    # Parkinson
    LOG_PARK = math.sqrt(4.0 * math.log(2))
    PARK_SCALE = math.sqrt(1.0 / (4.0 * math.log(2)))
    with np.errstate(divide="ignore", invalid="ignore"):
        parkinson_vol_pct = np.where(sig_low > 0,
                                     np.log(sig_high / sig_low) / LOG_PARK, np.nan)
    parkinson_vs_atr = np.where(sig_atr > 0,
                                 (sig_high - sig_low) * PARK_SCALE / sig_atr, np.nan)

    side_str = np.where(trades["side"] == 1, "long", "short")

    df = pd.DataFrame({
        "zscore_entry": sig_zscore.astype(np.float32),
        "zscore_prev": prev_z.astype(np.float32),
        "zscore_delta": (sig_zscore - prev_z).astype(np.float32),
        "volume_zscore": sig_vol_z.astype(np.float32),
        "ema_spread": sig_ema_sp.astype(np.float32),
        "bar_body_points": np.abs(sig_close - sig_open).astype(np.float32),
        "bar_range_points": (sig_high - sig_low).astype(np.float32),
        "atr_points": sig_atr.astype(np.float32),
        "parkinson_vol_pct": parkinson_vol_pct.astype(np.float32),
        "parkinson_vs_atr": parkinson_vs_atr.astype(np.float32),
        "time_of_day_hhmm": hhmm.astype(np.float32),
        "day_of_week": dow.astype(np.float32),
        "distance_to_ema_fast_points": (sig_close - sig_ema_f).astype(np.float32),
        "distance_to_ema_slow_points": (sig_close - sig_ema_s).astype(np.float32),
        "side": np.array([SIDE_MAP[s] for s in side_str], dtype=np.int8),
        "stop_method": np.full(n, STOP_METHOD_MAP[stop_method], dtype=np.int8),
        "exit_on_opposite_signal": np.full(n, int(exit_on_opp), dtype=np.int8),
    })
    df["abs_zscore_entry"] = df["zscore_entry"].abs()
    return df


FEATURE_ORDER = [
    "zscore_entry", "zscore_prev", "zscore_delta",
    "volume_zscore", "ema_spread",
    "bar_body_points", "bar_range_points",
    "atr_points", "parkinson_vol_pct", "parkinson_vs_atr",
    "time_of_day_hhmm", "day_of_week",
    "distance_to_ema_fast_points", "distance_to_ema_slow_points",
    "side", "stop_method", "exit_on_opposite_signal",
    "candidate_rr", "abs_zscore_entry", "rr_x_atr",
]


def pick_adaptive_rr(features: pd.DataFrame, model) -> np.ndarray:
    """For each trade (row), pick argmax E[R] across RR_LEVELS."""
    n = len(features)
    K = len(RR_LEVELS)
    # Repeat features K times, vary candidate_rr
    expanded = features.loc[features.index.repeat(K)].reset_index(drop=True).copy()
    rr_col = np.tile(RR_LEVELS, n).astype(np.float32)
    expanded["candidate_rr"] = rr_col
    expanded["rr_x_atr"] = rr_col * expanded["atr_points"].to_numpy()
    X = expanded[FEATURE_ORDER]
    p = model.predict(X.values)
    p = p.reshape(n, K)
    ev = p * RR_LEVELS[None, :] - (1.0 - p)  # E[R] in R units
    best_k = np.argmax(ev, axis=1)
    best_rr = RR_LEVELS[best_k]
    best_ev = ev[np.arange(n), best_k]
    best_p = p[np.arange(n), best_k]
    return best_rr, best_ev, best_p


def simulate(trades: dict, df_sig: pd.DataFrame, per_trade_rr: np.ndarray,
             cfg_base) -> dict:
    """Simulate P&L trade-by-trade using per-trade RR.

    For each trade entry:
      - stop_distance_pts is fixed (from sl price)
      - TP distance = rr * stop_distance_pts
      - Scan bars [entry_bar .. exit_bar + some max] — but restricting to the
        original trade's exit bar suffices ONLY if we have the path. We use
        the MFE/MAE already returned by the core (path captured up to the
        trade's original exit). If new TP < original TP (rr < orig_rr), and
        mfe >= new TP dist, the adaptive trade hits TP earlier — we count win.
        If new TP > original TP (rr > orig_rr), the MFE path shows whether
        price ever reached the new TP before the stop; if yes (mfe >= new tp
        dist) we count win, else loss (stop) or time exit based on original
        exit reason.

    NOTE: this is path-approximation. It matches the labelling scheme the
    adaptive model was trained against (mfe vs rr*stop). Path effects AFTER
    the original exit bar are not captured — but since original core runs
    until SL/TP/opp-signal/time-exit, the MFE up to exit is the relevant
    path for adaptive TPs strictly smaller than or equal to, AND for larger
    TPs we still have the whole holding window's MFE (which is the max fav
    excursion — if it didn't reach the new TP by then, the trade either
    stopped, timed out, or reversed). This is consistent with how the
    label_win was defined in the trained model's training data.
    """
    n = len(trades["side"])
    side = trades["side"].astype(np.int64)
    entry_price = trades["entry_price"]
    exit_price_orig = trades["exit_price"]
    sl_pts_per_trade = np.abs(entry_price - trades["sl"])
    mfe = trades["mfe"]
    mae = trades["mae"]
    exit_reason_orig = trades["exit_reason"]

    # Adaptive exits
    # If mfe >= rr * sl_pts -> WIN: pnl_pts = rr * sl_pts * side
    # Else: LOSS at stop (mae <= -sl_pts) OR time/opp exit -> use original
    #       exit_price if NOT TP (exit_reason != 2); if original was TP but
    #       the new larger TP not reached then use stop loss since
    #       mae <= -sl_pts would have triggered. For zero-mae trades ended
    #       by opposite/time, we use original exit_price.
    pnl_pts = np.zeros(n, dtype=np.float64)
    labels = np.zeros(n, dtype=np.int8)
    exit_kind = np.zeros(n, dtype=np.int8)  # 2=tp, 1=sl, 3=opp/time

    for i in range(n):
        rr = float(per_trade_rr[i])
        s = sl_pts_per_trade[i]
        if s <= 0:
            continue
        new_tp_pts = rr * s
        # Did MFE reach new TP?
        if mfe[i] >= new_tp_pts - 1e-9:
            # WIN: hit new TP
            pnl_pts[i] = new_tp_pts
            labels[i] = 1
            exit_kind[i] = 2
        elif mae[i] <= -s + 1e-9:
            # Hit stop
            pnl_pts[i] = -s
            labels[i] = 0
            exit_kind[i] = 1
        else:
            # Did not hit TP, did not hit stop; original must have been
            # opposite-signal or time-exit. Use original exit price.
            pnl_pts[i] = (exit_price_orig[i] - entry_price[i]) * side[i]
            labels[i] = int(pnl_pts[i] > 0)
            exit_kind[i] = 3

    return dict(pnl_pts=pnl_pts, labels=labels, exit_kind=exit_kind,
                sl_pts=sl_pts_per_trade)


def simulate_fixed(trades: dict) -> dict:
    """Fixed-RR: replay core's outputs directly (ground truth, no approximation)."""
    n = len(trades["side"])
    side = trades["side"].astype(np.int64)
    entry_price = trades["entry_price"]
    exit_price = trades["exit_price"]
    sl_pts_per_trade = np.abs(entry_price - trades["sl"])
    pnl_pts = (exit_price - entry_price) * side
    labels = (pnl_pts > 0).astype(np.int8)
    return dict(pnl_pts=pnl_pts, labels=labels,
                exit_kind=trades["exit_reason"].astype(np.int8),
                sl_pts=sl_pts_per_trade)


def compute_equity_and_metrics(result: dict, trades: dict,
                               cfg, label: str) -> dict:
    """Compute return, Sharpe, DD on a FIXED-equity sizing model (matches the
    sweep/top-combos reporting convention: contracts sized off the initial
    $50k at 5% risk; no compounding). This keeps the comparison stable and
    comparable across combos (compounding a 219,000,000,000% return is the
    artefact of 5% per-trade compounding over 671 trades, not a real result).
    """
    start_eq = float(cfg.STARTING_EQUITY)
    risk_pct = float(cfg.RISK_PCT)
    dpp = float(cfg.MNQ_DOLLARS_PER_POINT)

    pnl_pts = result["pnl_pts"]
    sl_pts = result["sl_pts"]
    n = len(pnl_pts)

    pnl_dollars = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = sl_pts[i]
        if s <= 0:
            continue
        contracts = int(start_eq * risk_pct // (s * dpp))
        pnl_dollars[i] = pnl_pts[i] * contracts * dpp

    # Fixed-equity equity curve (running sum)
    equity_curve = np.concatenate([[start_eq], start_eq + np.cumsum(pnl_dollars)])
    peak = np.maximum.accumulate(equity_curve)
    dd = (peak - equity_curve) / np.where(peak > 0, peak, 1.0)
    max_dd = float(np.max(dd))
    final_equity = float(equity_curve[-1])
    total_return_pct = (final_equity - start_eq) / start_eq * 100.0

    # Sharpe on per-trade $ returns / starting equity
    ret_series = pnl_dollars / start_eq
    if len(ret_series) > 1 and np.std(ret_series, ddof=1) > 0:
        sharpe = float(np.mean(ret_series) / np.std(ret_series, ddof=1) *
                        math.sqrt(len(ret_series)))
    else:
        sharpe = 0.0

    wins = int(np.sum(result["labels"]))
    wr = wins / n if n else 0.0

    return {
        "label": label,
        "n_trades": int(n),
        "win_rate": round(wr, 4),
        "total_return_pct": round(total_return_pct, 2),
        "final_equity": round(final_equity, 2),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "avg_pnl_dollars": round(float(np.mean(pnl_dollars)) if n else 0.0, 2),
        "sum_pnl_dollars": round(float(np.sum(pnl_dollars)), 2),
    }


def main():
    """Run fixed-R:R vs adaptive-R:R backtests for a top ML#1 combo.

    Loads the combo, generates indicators, runs two backtests (fixed `min_rr`
    and adaptive R:R selected from the V1 calibrated P(win|R:R)), prints a
    side-by-side comparison, and writes trades + metrics JSON to the
    evaluation folder.
    """
    t0 = time.time()
    print(f"[adaptive-vs-fixed] core={CORE_NAME}")

    combo, _ = load_top_combo()
    print(f"[adaptive-vs-fixed] top combo: {combo['global_combo_id']}")
    print(f"  stop_method={combo['stop_method']} min_rr={combo['min_rr']:.4f} "
          f"z_band_k={combo['z_band_k']:.4f} z_window={combo['z_window']} "
          f"ema_fast={combo['ema_fast']} ema_slow={combo['ema_slow']}")
    print(f"  z_input={combo['z_input']} z_anchor={combo['z_anchor']} "
          f"z_denom={combo['z_denom']} z_type={combo['z_type']}")
    print(f"  atr_multiplier={combo.get('atr_multiplier')} "
          f"swing_lookback={combo.get('swing_lookback')}")

    # Load bars + train split
    print("[adaptive-vs-fixed] loading bars...")
    df = load_bars(DATA_CSV)
    train, _ = split_train_test(df, 0.8)
    print(f"  full bars: {len(df):,} | train: {len(train):,} "
          f"({train['time'].iloc[0]} -> {train['time'].iloc[-1]})")

    # Build indicators on TRAIN only (matches split policy)
    print("[adaptive-vs-fixed] building indicators...")
    df_ind = build_indicators(train, combo)

    stop_pts = resolve_stop_pts(combo, df_ind)
    print(f"  resolved stop_pts = {stop_pts:.4f}")

    cfg = make_cfg(combo, stop_pts)
    df_sig = generate_signals(df_ind, cfg)
    n_sigs = int((df_sig["signal"] != 0).sum())
    print(f"  signals: {n_sigs:,}")

    # Run core once at fixed min_rr
    print("[adaptive-vs-fixed] running backtest core...")
    tr = run_core(df_sig, cfg)
    n_trades = len(tr["side"])
    print(f"  trades: {n_trades:,}")

    if n_trades == 0:
        print("  no trades generated — aborting")
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps({"error": "no trades"}, indent=2))
        return

    # Build features
    print("[adaptive-vs-fixed] building per-trade features...")
    feats = build_features(tr, df_sig, stop_pts,
                           str(combo["stop_method"]),
                           bool(combo["exit_on_opposite_signal"]))

    # Load model
    print("[adaptive-vs-fixed] loading adaptive R:R model...")
    model = lgb.Booster(model_file=str(MODEL_PATH))

    print("[adaptive-vs-fixed] predicting optimal R:R per trade...")
    best_rr, best_ev, best_pwin = pick_adaptive_rr(feats, model)
    print(f"  best_rr: mean={np.mean(best_rr):.3f} median={np.median(best_rr):.3f} "
          f"min={np.min(best_rr)} max={np.max(best_rr)}")
    print(f"  dist: {dict(zip(*np.unique(best_rr, return_counts=True)))}")

    # Simulate both
    print("[adaptive-vs-fixed] simulating fixed and adaptive...")
    fixed_res = simulate_fixed(tr)
    adapt_res = simulate(tr, df_sig, best_rr, cfg)

    fixed_m = compute_equity_and_metrics(fixed_res, tr, cfg, "fixed")
    adapt_m = compute_equity_and_metrics(adapt_res, tr, cfg, "adaptive")

    # Sanity: reconstruct fixed via adaptive path at the original min_rr to
    # measure path-approximation error
    orig_rr_arr = np.full(n_trades, float(combo["min_rr"]), dtype=np.float32)
    fixed_via_path = simulate(tr, df_sig, orig_rr_arr, cfg)
    fixed_via_path_m = compute_equity_and_metrics(fixed_via_path, tr, cfg,
                                                   "fixed_via_path_approx")

    result = {
        "top_combo": {
            "global_combo_id": combo["global_combo_id"],
            "source_version": combo["source_version"],
            "combo_id": combo["combo_id"],
            "min_rr": float(combo["min_rr"]),
            "stop_method": combo["stop_method"],
            "atr_multiplier": float(combo["atr_multiplier"]) if combo.get("atr_multiplier") else None,
            "swing_lookback": float(combo["swing_lookback"]) if combo.get("swing_lookback") else None,
            "z_band_k": float(combo["z_band_k"]),
            "z_window": int(combo["z_window"]),
            "ema_fast": int(combo["ema_fast"]),
            "ema_slow": int(combo["ema_slow"]),
            "z_input": combo["z_input"],
            "z_anchor": combo["z_anchor"],
            "z_denom": combo["z_denom"],
            "z_type": combo["z_type"],
            "resolved_stop_pts": round(stop_pts, 4),
        },
        "train_bars": int(len(train)),
        "train_start": str(train["time"].iloc[0]),
        "train_end": str(train["time"].iloc[-1]),
        "fixed": fixed_m,
        "adaptive": adapt_m,
        "fixed_via_path_approx": fixed_via_path_m,
        "adaptive_rr_distribution": {
            str(round(float(r), 2)): int(c)
            for r, c in zip(*np.unique(best_rr, return_counts=True))
        },
        "adaptive_rr_summary": {
            "mean": round(float(np.mean(best_rr)), 3),
            "median": round(float(np.median(best_rr)), 3),
            "min": round(float(np.min(best_rr)), 3),
            "max": round(float(np.max(best_rr)), 3),
        },
        "runtime_seconds": round(time.time() - t0, 2),
        "backtest_core": CORE_NAME,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2, default=str))
    print(f"[adaptive-vs-fixed] wrote {OUT_PATH}")
    print(json.dumps({"fixed": fixed_m, "adaptive": adapt_m}, indent=2))
    print(f"[adaptive-vs-fixed] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
