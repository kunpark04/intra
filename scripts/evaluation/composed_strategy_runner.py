"""Helper used by the `evaluation/` notebooks to run composed top-combo strategies.

Loads the 20% chronological test partition, exposes a `run_strategy` entry
point that takes one strategy dict (as emitted by
`scripts/analysis/extract_top_combos_by_freq.py`) and returns a trades
DataFrame + equity curve + headline metrics.

Shared logic (`make_cfg_from_params`, `build_indicators_with_variant`) is
lifted from `scripts/analysis/validate_top_combos_ml1.py` and adapted to
consume the JSON parameter dict shape produced by the extractor.
"""
from __future__ import annotations

import math
import types
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent

import sys
sys.path.insert(0, str(REPO))

from src import config as base_cfg
from src.data_loader import load_bars, split_train_test
from src.indicators.pipeline import add_indicators
from src.indicators.zscore_variants import compute_vwap_session, compute_zscore_v2
from src.strategy import generate_signals
from src.backtest import run_backtest

STARTING_EQUITY = 50_000.0
TEST_BARS_CSV = REPO / "data" / "NQ_1min.csv"


# ── Test partition loader ────────────────────────────────────────────────────

def load_test_bars() -> pd.DataFrame:
    """Load 1-minute bars and return the 20% chronological test partition.

    Returns:
        DataFrame with canonical columns (`time`, `open`, `high`, `low`,
        `close`, `volume`, `session_break`), chronologically sorted, covering
        the test 20% of the full series.
    """
    bars = load_bars(TEST_BARS_CSV)
    _, test = split_train_test(bars)
    return test.reset_index(drop=True)


# ── Param-dict → cfg namespace ───────────────────────────────────────────────

def _is_none_or_nan(x) -> bool:
    if x is None:
        return True
    try:
        return bool(math.isnan(float(x)))
    except (TypeError, ValueError):
        return False


def make_cfg_from_params(params: dict) -> types.SimpleNamespace:
    """Build a `cfg` namespace from a strategy `parameters` dict.

    Starts from `src.config` and overrides the sweep-tunable fields with
    values from `params` (the dict emitted by the top-combo extractor).
    Stop-method is left as specified (`"fixed" | "atr" | "swing"`) so the
    backtest engine resolves the stop at signal time.

    Args:
        params: Entry from `evaluation/top_strategies.json` under the
            `parameters` key.

    Returns:
        `types.SimpleNamespace` ready to pass to `run_backtest`.
    """
    ns = types.SimpleNamespace()
    for k in dir(base_cfg):
        if not k.startswith("_"):
            setattr(ns, k, getattr(base_cfg, k))

    ns.Z_BAND_K                = float(params["z_band_k"])
    ns.Z_WINDOW                = int(params["z_window"])
    ns.VOLUME_ZSCORE_WINDOW    = int(params["volume_zscore_window"])
    ns.EMA_FAST                = int(params["ema_fast"])
    ns.EMA_SLOW                = int(params["ema_slow"])
    ns.MIN_RR                  = float(params["min_rr"])
    ns.EXIT_ON_OPPOSITE_SIGNAL = bool(params.get("exit_on_opposite_signal", False))
    ns.USE_BREAKEVEN_STOP      = bool(params.get("use_breakeven_stop", False))
    ns.MAX_HOLD_BARS           = int(params.get("max_hold_bars", 0) or 0)
    ns.ZSCORE_CONFIRMATION     = bool(params.get("zscore_confirmation", False))

    ns.VOLUME_ENTRY_THRESHOLD = float(params.get("volume_entry_threshold", 0.0) or 0.0)
    ns.VOL_REGIME_LOOKBACK    = int(params.get("vol_regime_lookback", 0) or 0)
    ns.VOL_REGIME_MIN_PCT     = float(params.get("vol_regime_min_pct", 0.0) or 0.0)
    ns.VOL_REGIME_MAX_PCT     = float(params.get("vol_regime_max_pct", 1.0) or 1.0)
    ns.SESSION_FILTER_MODE    = int(params.get("session_filter_mode", 0) or 0)
    ns.TOD_EXIT_HOUR          = int(params.get("tod_exit_hour", 0) or 0)

    stop_method = str(params.get("stop_method", "fixed")).lower()
    ns.STOP_METHOD = stop_method
    if stop_method == "fixed":
        sfp = params.get("stop_fixed_pts")
        ns.STOP_FIXED_PTS = float(sfp) if not _is_none_or_nan(sfp) else base_cfg.STOP_FIXED_PTS
    elif stop_method == "atr":
        mult = params.get("atr_multiplier")
        ns.ATR_MULTIPLIER = float(mult) if not _is_none_or_nan(mult) else base_cfg.ATR_MULTIPLIER
    elif stop_method == "swing":
        lb = params.get("swing_lookback")
        ns.SWING_LOOKBACK = int(lb) if not _is_none_or_nan(lb) else base_cfg.SWING_LOOKBACK

    ns.SIGNAL_MODE = "zscore_reversal"
    return ns


# ── Indicators with z-score variant ──────────────────────────────────────────

def build_indicators_with_variant(bars: pd.DataFrame, cfg, params: dict) -> pd.DataFrame:
    """Attach indicators to `bars`, swapping in the combo's z-score variant.

    Mirrors `validate_top_combos_ml1._build_indicators_with_zscore_variant`:
    builds the default indicator frame, and when the combo declares a
    non-default z-score formulation (`z_input`/`z_anchor`/`z_denom`/
    `z_type`/`z_window_2`), recomputes the `zscore` column via
    `compute_zscore_v2`.

    Args:
        bars: Test-partition bars.
        cfg: Namespace produced by `make_cfg_from_params`.
        params: Strategy `parameters` dict.

    Returns:
        Indicator-enriched DataFrame ready for `generate_signals`.
    """
    df = add_indicators(bars.copy(), cfg)

    z_input  = str(params.get("z_input", "close"))
    z_anchor = str(params.get("z_anchor", "rolling_mean"))
    z_denom  = str(params.get("z_denom", "rolling_std"))
    z_type   = str(params.get("z_type", "parametric"))
    z_w2     = int(params.get("z_window_2", 0) or 0)
    z_w2_wt  = float(params.get("z_window_2_weight", 0.0) or 0.0)

    is_default = (z_input == "close" and z_anchor == "rolling_mean"
                  and z_denom == "rolling_std" and z_type == "parametric"
                  and z_w2 == 0)
    if is_default:
        return df

    close_np = df["close"].to_numpy(np.float64)
    high_np  = df["high"].to_numpy(np.float64)
    low_np   = df["low"].to_numpy(np.float64)
    vol_np   = df["volume"].to_numpy(np.float64)
    sb_np    = df["session_break"].to_numpy()
    ema_f    = df["ema_fast"].to_numpy(np.float64)
    ema_s    = df["ema_slow"].to_numpy(np.float64)
    atr_np   = df["atr"].to_numpy(np.float64)
    vwap_np  = compute_vwap_session(high_np, low_np, close_np, vol_np, sb_np)

    df["zscore"] = compute_zscore_v2(
        close_np, high_np, low_np, vol_np, sb_np,
        ema_f, ema_s, atr_np,
        cfg.Z_WINDOW, z_input, z_anchor, z_denom, z_type, 0,
        vwap_np, z_w2, z_w2_wt,
    )
    return df


# ── Metrics ──────────────────────────────────────────────────────────────────

def _metrics_from_trades(trades: pd.DataFrame, start_equity: float) -> dict:
    """Compute headline metrics from a trades DataFrame."""
    n = len(trades)
    if n == 0:
        return {
            "n_trades": 0, "win_rate": 0.0, "total_pnl_dollars": 0.0,
            "total_return_pct": 0.0, "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0, "max_drawdown_dollars": 0.0,
        }

    pnl = trades["net_pnl_dollars"].to_numpy(np.float64)
    wins = trades["label_win"].to_numpy(np.int8)
    r = trades["r_multiple"].to_numpy(np.float64)

    total_pnl = float(pnl.sum())
    wr = float(wins.mean())
    r_std = float(r.std(ddof=1)) if n > 1 else 0.0
    sharpe = float(r.mean() / r_std) if r_std > 0 else 0.0

    equity = start_equity + np.cumsum(pnl)
    equity_full = np.concatenate([[start_equity], equity])
    peak = np.maximum.accumulate(equity_full)
    dd_dollars = (peak - equity_full).max()
    with np.errstate(divide="ignore", invalid="ignore"):
        dd_pct = np.nan_to_num((peak - equity_full) / peak, nan=0.0).max() * 100

    return {
        "n_trades": int(n),
        "win_rate": round(wr, 4),
        "total_pnl_dollars": round(total_pnl, 2),
        "total_return_pct": round(total_pnl / start_equity * 100, 2),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown_pct": round(float(dd_pct), 2),
        "max_drawdown_dollars": round(float(dd_dollars), 2),
    }


# ── Trade-log shaping ────────────────────────────────────────────────────────

def _shape_trade_log(trades_raw: list[dict]) -> pd.DataFrame:
    """Rename backtest-engine fields to the display column names.

    Adds `date`, `direction`, `entry_px`, `sl_px`, `tp_px`, `dollar_risk`,
    `dollar_reward`, `exit_px`, `exit_reason`, `actual_pnl`, `reason`.
    """
    if not trades_raw:
        return pd.DataFrame()
    df = pd.DataFrame(trades_raw)
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["entry_time"])
    out["direction"] = df["side"]
    out["entry_px"] = df["entry_fill_price"].astype(float)
    out["sl_px"] = df["sl_price"].astype(float)
    out["tp_px"] = df["tp_price"].astype(float)
    out["contracts"] = df["contracts"].astype(float)
    out["dollar_risk"] = df["risk_dollars"].astype(float)
    out["dollar_reward"] = (df["risk_dollars"] * df["rr_planned"]).astype(float)
    out["reason"] = "zscore_reversal"
    out["exit_px"] = df["exit_fill_price"].astype(float)
    out["exit_reason"] = df["exit_reason"].astype(str)
    out["actual_pnl"] = df["net_pnl_dollars"].astype(float)
    return out


# ── Public entry point ───────────────────────────────────────────────────────

def run_strategy(strategy: dict, bars: pd.DataFrame | None = None) -> dict:
    """Run one strategy on the 20% test partition.

    Args:
        strategy: One entry from `top_strategies.json` (`high_freq` or
            `low_freq`), containing `global_combo_id` and `parameters`.
        bars: Optional pre-loaded test-partition bars. If None, reads via
            `load_test_bars`.

    Returns:
        Dict with:
            - `combo_id`: `global_combo_id`.
            - `trades`: shaped trade-log DataFrame (display cols).
            - `equity_curve`: DataFrame of `[time, equity]` after each trade.
            - `metrics`: headline-metrics dict.
    """
    if bars is None:
        bars = load_test_bars()
    params = strategy["parameters"]
    cfg = make_cfg_from_params(params)

    df_ind = build_indicators_with_variant(bars, cfg, params)
    df_sig = generate_signals(df_ind, cfg)
    results = run_backtest(df_sig, cfg,
                           version=f"composed_{strategy['global_combo_id']}")

    raw_trades = results["trades"]
    trades_df = pd.DataFrame(raw_trades)
    display_df = _shape_trade_log(raw_trades)

    if len(raw_trades) > 0:
        eq = STARTING_EQUITY + trades_df["net_pnl_dollars"].cumsum()
        equity_curve = pd.DataFrame({
            "time": pd.to_datetime(trades_df["exit_time"]).values,
            "equity": eq.values,
        })
    else:
        equity_curve = pd.DataFrame(columns=["time", "equity"])

    metrics = _metrics_from_trades(trades_df, STARTING_EQUITY)
    metrics["combo_id"] = strategy["global_combo_id"]
    return {
        "combo_id": strategy["global_combo_id"],
        "trades": display_df,
        "equity_curve": equity_curve,
        "metrics": metrics,
    }
