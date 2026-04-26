"""backfill_combo865.py — forward run of combo-865 from test partition start
through the latest available NQ 1h bar.

This is the "live paper-trading" backfill: a continuity bridge between Probe 2's
historical run (220 trades, 2024-10-22 → 2026-04-08) and the forward paper-trade
window proposed in `tasks/probe5_combo865_paper_trade_preregistration.md`.

Numbers from this script are NOT preregistered and DO NOT count toward Probe 5
§-gate verdicts. The purpose is operational continuity (verify engine repro on
known partition + extend forward through fresh bars) before live ingest begins.

Usage (from repo root):
    python scripts/paper_trade/backfill_combo865.py
    python scripts/paper_trade/backfill_combo865.py --end 2026-04-25
    python scripts/paper_trade/backfill_combo865.py --start 2024-10-22 --out evaluation/probe5_combo865_backfill

Output (default): evaluation/probe5_combo865_backfill/
    metadata.json          — run config + combo-865 params + frozen commit
    trades.csv             — full LOG_SCHEMA-compliant trade list
    trader_log.csv         — minimal 7-column human-readable trades log
    daily_ledger.csv       — one row per calendar day (no-trade days included)
    equity_curve.csv       — bar-by-bar equity at fixed-$500 risk/trade
    monte_carlo.json       — IID bootstrap risk metrics + WR permutation test

Source-of-truth pattern: adapted from
`scripts/backtests/adaptive_vs_fixed_backtest_v1.py` (verified imports).
Sizing pinned to fixed-$500 risk/trade per CLAUDE.md "Reporting & styling
requirements" — no compounding. $5/contract round-trip friction per v11.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src import config as _BASE_CFG
from src.indicators.ema import compute_ema
from src.indicators.atr import compute_atr
from src.indicators.zscore_variants import compute_zscore_v2
from src.indicators.zscore import compute_volume_zscore
from src.strategy import generate_signals
from src.tz_contract import assert_naive_ct
from src.reporting import (
    write_trades_csv,
    write_trader_log,
    write_daily_ledger,
    write_equity_curve,
    run_monte_carlo,
)

try:
    from src.cython_ext.backtest_core import backtest_core_cy as _core_fn
    CORE_NAME = "cython"
except ImportError:
    from src.backtest import _backtest_core_numpy as _core_fn
    CORE_NAME = "numpy"


# ── Combo-865 parameter freeze ────────────────────────────────────────────────
# Lifted verbatim from data/ml/probe2/combo865_1h_test_manifest.json.
# DO NOT MUTATE without amending tasks/probe5_combo865_paper_trade_preregistration.md.
COMBO_865: Dict[str, Any] = {
    "combo_id": 865,
    "z_band_k": 2.3043838978381697,
    "z_window": 41,
    "volume_zscore_window": 47,
    "ema_fast": 6,
    "ema_slow": 48,
    "stop_method": "fixed",
    "stop_fixed_pts": 17.017749351216196,
    "atr_multiplier": None,
    "swing_lookback": None,
    "min_rr": 1.8476511046163293,
    "exit_on_opposite_signal": False,
    "use_breakeven_stop": False,
    "max_hold_bars": 120,
    "zscore_confirmation": False,
    "z_input": "returns",
    "z_anchor": "rolling_mean",
    "z_denom": "parkinson",
    "z_type": "parametric",
    "z_window_2": 47,
    "z_window_2_weight": 0.3308286558845456,
    "volume_entry_threshold": 0.0,
    "vol_regime_lookback": 0,
    "vol_regime_min_pct": 0.0,
    "vol_regime_max_pct": 1.0,
    "session_filter_mode": 0,
    "tod_exit_hour": 0,
    "entry_timing_offset": 1,
    "fill_slippage_ticks": 1,
    "cooldown_after_exit_bars": 3,
}

# ── Project constants (per CLAUDE.md) ─────────────────────────────────────────
BARS_PARQUET = REPO / "data" / "NQ_1h.parquet"
# 1h test partition boundary. floor(0.8 × len(NQ_1h.parquet bars)) lands at
# 2024-10-30 21:00:00 — distinct from `src.config.TRAIN_END_FROZEN`'s
# 2024-10-22 05:07:00 (which is the 1-min split). Probe 2 verdict text quotes
# 2024-10-22 → 2026-04-08 as narrative shorthand; the 1h test partition really
# starts ~8 days later. Reviewer C3 fix.
TEST_PARTITION_START = pd.Timestamp("2024-10-30 21:00:00")
# OOS-scoped sizing override (user directive 2026-04-26): scale paper-account
# economics down from CLAUDE.md project-wide defaults ($50,000 / $500) to a
# small-account configuration. At combo-865's 17.018pt stop the integer-floor
# contract count lands at 1 (50 // 34.04 = 1), so realized per-trade risk is
# $34.04 not $50; rounding up to 2 would put 3.4% of a $2k account at risk
# per trade. Project-wide CLAUDE.md non-negotiables remain $50k / $500 — this
# override is local to the paper-trade backfill.
STARTING_EQUITY = 2_000.0
FIXED_RISK_DOLLARS = 50.0
DOLLARS_PER_POINT = 2.0  # MNQ
TICK_SIZE = 0.25
TRADING_HOURS_PER_BAR = 1  # 1h bars

# Friction includes fill_slippage_ticks adder (param_sweep semantics, lines 895-901):
#   $5 baseline ($3 commission + $2 slippage) + ticks × 2 sides × TICK_SIZE × $/pt
# For combo-865 with fill_slippage_ticks=1 → $5 + 1×2×0.25×2 = $6/contract RT.
COST_PER_CONTRACT_RT = 5.0 + int(COMBO_865.get("fill_slippage_ticks", 0)) * 2 * TICK_SIZE * DOLLARS_PER_POINT

# Engine exit-reason codes are 1-indexed per scripts/param_sweep.py:945.
# Reviewer C1 fix (was a 0-indexed list; every label was scrambled).
_EXIT_REASON_MAP = {1: "stop", 2: "take_profit", 3: "opposite_signal", 4: "end_of_data", 5: "time_exit"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_cfg(combo: Dict[str, Any], stop_pts_resolved: float) -> types.SimpleNamespace:
    """Build a per-combo cfg namespace, mirroring `param_sweep._make_cfg`."""
    cfg = types.SimpleNamespace(
        # Indicators
        EMA_FAST=int(combo["ema_fast"]),
        EMA_SLOW=int(combo["ema_slow"]),
        Z_WINDOW=int(combo["z_window"]),
        Z_BAND_K=float(combo["z_band_k"]),
        ZSCORE_DDOF=getattr(_BASE_CFG, "ZSCORE_DDOF", 0),
        VOLUME_ZSCORE_WINDOW=int(combo["volume_zscore_window"]),
        ATR_WINDOW=getattr(_BASE_CFG, "ATR_WINDOW", 14),
        # Signal
        SIGNAL_MODE="zscore_reversal",
        USE_ZSCORE_FILTER=True,
        ZSCORE_CONFIRMATION=bool(combo["zscore_confirmation"]),
        # Execution
        FILL_MODEL="next_bar_open",
        SAME_BAR_COLLISION="tp_first",
        # Risk
        STARTING_EQUITY=STARTING_EQUITY,
        RISK_PCT=getattr(_BASE_CFG, "RISK_PCT", 0.05),
        FIXED_RISK_DOLLARS=FIXED_RISK_DOLLARS,
        MNQ_DOLLARS_PER_POINT=DOLLARS_PER_POINT,
        TICK_SIZE=TICK_SIZE,
        MIN_RR=float(combo["min_rr"]),
        # Stops
        STOP_METHOD=str(combo["stop_method"]),
        STOP_FIXED_PTS=float(stop_pts_resolved),
        ATR_MULTIPLIER=float(combo["atr_multiplier"]) if combo["atr_multiplier"] else 0.0,
        SWING_LOOKBACK=int(combo["swing_lookback"]) if combo["swing_lookback"] else 0,
        SWING_BUFFER_TICKS=getattr(_BASE_CFG, "SWING_BUFFER_TICKS", 1),
        # Exits
        EXIT_ON_OPPOSITE_SIGNAL=bool(combo["exit_on_opposite_signal"]),
        USE_BREAKEVEN_STOP=bool(combo["use_breakeven_stop"]),
        MAX_HOLD_BARS=int(combo["max_hold_bars"]),
        # Filters
        VOLUME_ENTRY_THRESHOLD=float(combo["volume_entry_threshold"]),
        VOL_REGIME_LOOKBACK=int(combo["vol_regime_lookback"]),
        VOL_REGIME_MIN_PCT=float(combo["vol_regime_min_pct"]),
        VOL_REGIME_MAX_PCT=float(combo["vol_regime_max_pct"]),
        SESSION_FILTER_MODE=int(combo["session_filter_mode"]),
        TOD_EXIT_HOUR=int(combo["tod_exit_hour"]),
        # Numba
        USE_NUMBA=getattr(_BASE_CFG, "USE_NUMBA", True),
        # Monte Carlo
        MC_N_SIMS=getattr(_BASE_CFG, "MC_N_SIMS", 10_000),
        MC_SEED=getattr(_BASE_CFG, "MC_SEED", 42),
        MC_BOOTSTRAP=getattr(_BASE_CFG, "MC_BOOTSTRAP", "iid"),
        MC_RUIN_THRESHOLD=getattr(_BASE_CFG, "MC_RUIN_THRESHOLD", 0.5),
        # Scoring weights (preserved from base; not used in engine, only reporting)
        SCORE_W_ZSCORE=getattr(_BASE_CFG, "SCORE_W_ZSCORE", 0.25),
        SCORE_W_VOLUME=getattr(_BASE_CFG, "SCORE_W_VOLUME", 0.20),
        SCORE_W_EMA=getattr(_BASE_CFG, "SCORE_W_EMA", 0.20),
        SCORE_W_BODY=getattr(_BASE_CFG, "SCORE_W_BODY", 0.20),
        SCORE_W_SESSION=getattr(_BASE_CFG, "SCORE_W_SESSION", 0.15),
        SCORE_EMA_NORM=getattr(_BASE_CFG, "SCORE_EMA_NORM", 5.0),
    )
    return cfg


def build_indicators_for_combo(df: pd.DataFrame, combo: Dict[str, Any], cfg) -> pd.DataFrame:
    """Build indicators using combo-865's specific z-score formulation.

    Mirrors `param_sweep._build_indicator_df` for the v11 zscore_variants path
    rather than the simple `add_indicators` path used in V1/V2 iterations.
    """
    use_numba = bool(cfg.USE_NUMBA)
    ddof = int(cfg.ZSCORE_DDOF)

    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)

    ema_f = compute_ema(close, cfg.EMA_FAST, use_numba)
    ema_s = compute_ema(close, cfg.EMA_SLOW, use_numba)
    atr_arr = compute_atr(high, low, close, cfg.ATR_WINDOW, use_numba)
    vol_z_arr = compute_volume_zscore(volume, cfg.VOLUME_ZSCORE_WINDOW, ddof, use_numba)

    session_break = df["session_break"].to_numpy(dtype=bool) if "session_break" in df.columns \
        else np.zeros(len(df), dtype=bool)

    # v11 zscore variant (input / anchor / denom / type) with optional dual-window blend
    z_w2 = int(combo.get("z_window_2") or 0)
    z_w2_weight = float(combo.get("z_window_2_weight") or 0.0)
    zscore_arr = compute_zscore_v2(
        close=close, high=high, low=low, volume=volume,
        session_break=session_break,
        ema_fast_arr=ema_f, ema_slow_arr=ema_s, atr_arr=atr_arr,
        window=cfg.Z_WINDOW,
        z_input=str(combo["z_input"]),
        z_anchor=str(combo["z_anchor"]),
        z_denom=str(combo["z_denom"]),
        z_type=str(combo["z_type"]),
        ddof=ddof,
        vwap_arr=None,
        z_window_2=z_w2,
        z_window_2_weight=z_w2_weight,
    )

    out = df.copy()
    out["ema_fast"] = ema_f
    out["ema_slow"] = ema_s
    out["ema_spread"] = ema_f - ema_s
    fast_above = ema_f > ema_s
    out["ema_cross_up"] = pd.Series(fast_above, index=out.index) & \
        ~pd.Series(np.concatenate([[False], fast_above[:-1]]), index=out.index)
    out["ema_cross_down"] = ~pd.Series(fast_above, index=out.index) & \
        pd.Series(np.concatenate([[False], fast_above[:-1]]), index=out.index)
    out["zscore"] = zscore_arr
    out["atr"] = atr_arr
    out["volume_zscore"] = vol_z_arr
    return out


def resolve_stop_pts(combo: Dict[str, Any], df_ind: pd.DataFrame) -> float:
    """Resolve stop distance to a single point value, mirroring param_sweep semantics."""
    method = str(combo["stop_method"])
    if method == "fixed":
        return float(combo["stop_fixed_pts"])
    if method == "atr":
        atr_med = float(np.nanmedian(df_ind["atr"].to_numpy()))
        return atr_med * float(combo["atr_multiplier"])
    if method == "swing":
        # Median high-low range over swing_lookback as a proxy
        lb = int(combo["swing_lookback"])
        rng = (df_ind["high"] - df_ind["low"]).rolling(lb, min_periods=1).max()
        return float(np.nanmedian(rng.to_numpy()))
    raise ValueError(f"Unknown stop_method: {method}")


def run_engine(df_sig: pd.DataFrame, cfg) -> Dict[str, np.ndarray]:
    """Invoke the bar-by-bar engine core (Cython preferred, NumPy fallback)."""
    open_arr = df_sig["open"].to_numpy(dtype=np.float64)
    high_arr = df_sig["high"].to_numpy(dtype=np.float64)
    low_arr = df_sig["low"].to_numpy(dtype=np.float64)
    close_arr = df_sig["close"].to_numpy(dtype=np.float64)
    signal_arr = df_sig["signal"].to_numpy(dtype=np.int8)

    # entry_timing_offset shift (mirror param_sweep semantics)
    k = int(COMBO_865.get("entry_timing_offset", 0))
    if k > 0:
        shifted = np.zeros_like(signal_arr)
        shifted[k:] = signal_arr[:-k]
        signal_arr = shifted

    times = pd.to_datetime(df_sig["time"])
    hour_arr = times.dt.hour.to_numpy(dtype=np.int64)

    sl_pts = float(cfg.STOP_FIXED_PTS)
    tp_pts = sl_pts * float(cfg.MIN_RR)

    out = _core_fn(
        open_arr, high_arr, low_arr, close_arr, signal_arr,
        len(df_sig),
        sl_pts, tp_pts,
        cfg.SAME_BAR_COLLISION == "tp_first",
        bool(cfg.EXIT_ON_OPPOSITE_SIGNAL),
        bool(cfg.USE_BREAKEVEN_STOP),
        int(cfg.MAX_HOLD_BARS),
        hour_arr, int(cfg.TOD_EXIT_HOUR),
        int(COMBO_865.get("cooldown_after_exit_bars", 0)),
    )

    keys = [
        "side", "signal_bar", "entry_bar", "exit_bar",
        "entry_price", "exit_price", "sl", "tp",
        "exit_reason", "mae", "mfe", "hold_bars", "label_tp_first",
    ]
    return dict(zip(keys, out))


def trades_to_log_schema(
    eng: Dict[str, np.ndarray],
    df_sig: pd.DataFrame,
    cfg,
    combo: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert engine output arrays to LOG_SCHEMA-compliant per-trade dicts.

    Sizing is fixed-$500 per CLAUDE.md (no compounding). Net PnL is gross PnL
    minus $5/contract round-trip friction (v11 baseline).
    """
    n = len(eng["side"])
    if n == 0:
        return []

    times = pd.to_datetime(df_sig["time"]).to_numpy()
    contracts_per_trade = max(1, int(FIXED_RISK_DOLLARS // (cfg.STOP_FIXED_PTS * DOLLARS_PER_POINT)))

    rows: List[Dict[str, Any]] = []
    cum_net = 0.0
    equity = STARTING_EQUITY
    for i in range(n):
        side = int(eng["side"][i])  # +1 long, -1 short
        eb = int(eng["entry_bar"][i])
        xb = int(eng["exit_bar"][i])
        ep = float(eng["entry_price"][i])
        xp = float(eng["exit_price"][i])
        sl = float(eng["sl"][i])
        tp = float(eng["tp"][i])
        reason = int(eng["exit_reason"][i])
        mae = float(eng["mae"][i])
        mfe = float(eng["mfe"][i])
        hold = int(eng["hold_bars"][i])
        label_tp_first = bool(eng["label_tp_first"][i])

        # PnL in points then dollars
        pnl_pts = (xp - ep) * side
        gross_dollars = pnl_pts * DOLLARS_PER_POINT * contracts_per_trade
        friction_dollars = COST_PER_CONTRACT_RT * contracts_per_trade
        net_dollars = gross_dollars - friction_dollars
        cum_net += net_dollars

        # MFE/MAE ratio (Track A diagnostic). NaN when MAE≈0 to avoid inf
        # leaking into CSV / notebook (reviewer W5).
        mfe_mae_ratio = float(mfe / abs(mae)) if abs(mae) > 1e-12 else float("nan")
        r_multiple = pnl_pts / float(cfg.STOP_FIXED_PTS) if cfg.STOP_FIXED_PTS > 0 else 0.0

        # Engine bar indices must point inside the input bar series — assert
        # rather than silently fall back to None (reviewer W3).
        assert 0 <= eb < len(times), f"entry_bar {eb} out of bounds (n_bars={len(times)})"
        assert 0 <= xb < len(times), f"exit_bar {xb} out of bounds (n_bars={len(times)})"
        entry_ts = pd.Timestamp(times[eb])
        exit_ts = pd.Timestamp(times[xb])
        equity_before = equity
        equity += net_dollars

        rows.append({
            "trade_id": i + 1,
            "combo_id": combo["combo_id"],
            "entry_time": entry_ts.isoformat(),
            "entry_date": entry_ts.date(),
            "exit_time": exit_ts.isoformat(),
            "equity_before": equity_before,
            "equity_after": equity,
            "side": "long" if side == 1 else "short",
            "entry_fill_price": ep,
            "exit_price": xp,
            "sl_price": sl,
            "tp_price": tp,
            "stop_pts": float(cfg.STOP_FIXED_PTS),
            "tp_pts": float(cfg.STOP_FIXED_PTS) * float(cfg.MIN_RR),
            "min_rr_planned": float(cfg.MIN_RR),
            "rr_planned": float(cfg.MIN_RR),
            "z_band_k": float(cfg.Z_BAND_K),
            "contracts": contracts_per_trade,
            "pnl_points": pnl_pts,
            "r_multiple": r_multiple,
            "gross_pnl_dollars": gross_dollars,
            "friction_dollars": friction_dollars,
            "net_pnl_dollars": net_dollars,
            "cumulative_net_pnl_dollars": cum_net,
            "exit_reason": _EXIT_REASON_MAP.get(int(reason), f"code_{int(reason)}"),
            "label_win": bool(net_dollars > 0),
            "label_hit_tp_first": label_tp_first,
            "mae_points": mae,
            "mfe_points": mfe,
            "mfe_mae_ratio": mfe_mae_ratio,
            "hold_bars": hold,
            "entry_bar_idx": eb,
            "exit_bar_idx": xb,
        })
    return rows


def build_equity_curve(df_sig: pd.DataFrame, trades: List[Dict[str, Any]]) -> pd.DataFrame:
    """Bar-by-bar equity curve at fixed-$500 sizing (no compounding)."""
    times = pd.to_datetime(df_sig["time"]).to_numpy()
    eq = np.full(len(df_sig), STARTING_EQUITY, dtype=np.float64)
    for t in trades:
        xb = int(t["exit_bar_idx"])
        if xb < len(eq):
            eq[xb:] += t["net_pnl_dollars"]
    return pd.DataFrame({"time": times, "equity": eq})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", default=str(TEST_PARTITION_START.isoformat()),
                        help="Bar start timestamp (default: 1h test partition start 2024-10-30 21:00:00; not the 1m boundary)")
    parser.add_argument("--end", default=None,
                        help="Bar end timestamp (default: latest bar in NQ_1h.parquet)")
    parser.add_argument("--out", default=str(REPO / "evaluation" / "probe5_combo865_backfill"),
                        help="Output folder (default: evaluation/probe5_combo865_backfill)")
    parser.add_argument("--bars", default=str(BARS_PARQUET),
                        help=f"Bars parquet path (default: {BARS_PARQUET})")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[backfill_combo865] core={CORE_NAME}  start={args.start}  end={args.end or 'latest'}")
    print(f"[backfill_combo865] bars={args.bars}  out={out_dir}")

    # 1. Load + TZ-validate bars
    df = pd.read_parquet(args.bars)
    assert_naive_ct(df, time_col="time")
    print(f"  loaded {len(df):,} bars ({df['time'].iloc[0]} -> {df['time'].iloc[-1]})")

    # 2. Filter to forward window
    start_dt = pd.Timestamp(args.start)
    df = df[df["time"] >= start_dt].reset_index(drop=True)
    if args.end is not None:
        end_dt = pd.Timestamp(args.end)
        df = df[df["time"] <= end_dt].reset_index(drop=True)
    if len(df) == 0:
        print("ERROR: no bars in range.")
        sys.exit(1)
    print(f"  window: {len(df):,} bars ({df['time'].iloc[0]} -> {df['time'].iloc[-1]})")

    # 3. Build cfg + indicators + signals + run engine
    stop_pts = float(COMBO_865["stop_fixed_pts"])  # fixed method
    cfg = make_cfg(COMBO_865, stop_pts)
    print(f"  cfg: stop_pts={stop_pts:.4f}  min_rr={cfg.MIN_RR:.4f}  z_band_k={cfg.Z_BAND_K:.4f}")

    df_ind = build_indicators_for_combo(df, COMBO_865, cfg)
    df_sig = generate_signals(df_ind, cfg)
    n_signals = int((df_sig["signal"] != 0).sum())
    print(f"  signals: {n_signals:,} ({n_signals / max(len(df_sig), 1) * 100:.2f}% of bars)")

    eng = run_engine(df_sig, cfg)
    n_trades = len(eng["side"])
    trades = trades_to_log_schema(eng, df_sig, cfg, COMBO_865)
    print(f"  trades: {n_trades:,}")

    # 4. Equity curve + final metrics
    ec = build_equity_curve(df_sig, trades)
    final_eq = float(ec["equity"].iloc[-1]) if len(ec) else STARTING_EQUITY
    n_wins = sum(1 for t in trades if t["label_win"])
    wr = n_wins / n_trades if n_trades else 0.0
    print(f"  final equity: ${final_eq:,.2f}  win rate: {wr:.1%}")

    # 5. Save artifacts
    write_trades_csv(trades, out_dir / "trades.csv")
    write_trader_log(trades, out_dir / "trader_log.csv")
    write_daily_ledger(trades, df_sig, out_dir / "daily_ledger.csv")
    # write_equity_curve expects a list-of-dicts; convert from DataFrame
    write_equity_curve(ec.to_dict(orient="records"), out_dir / "equity_curve.csv")

    # Monte Carlo (IID bootstrap + WR permutation)
    mc = run_monte_carlo(trades, cfg) if n_trades else {"note": "no trades — MC skipped"}
    (out_dir / "monte_carlo.json").write_text(json.dumps(mc, indent=2, default=str))

    # Metadata
    meta = {
        "version": "probe5_combo865_backfill",
        "source": "scripts/paper_trade/backfill_combo865.py",
        "STARTING_EQUITY": STARTING_EQUITY,
        "FIXED_RISK_DOLLARS": FIXED_RISK_DOLLARS,
        "DOLLARS_PER_POINT": DOLLARS_PER_POINT,
        "COST_PER_CONTRACT_RT": COST_PER_CONTRACT_RT,
        "engine_core": CORE_NAME,
        "bars_parquet": str(args.bars),
        "bars_first": str(df["time"].iloc[0]),
        "bars_last": str(df["time"].iloc[-1]),
        "n_bars": int(len(df)),
        "n_trades": int(n_trades),
        "n_wins": int(n_wins),
        "win_rate": round(wr, 4),
        "final_equity": round(final_eq, 2),
        "total_return_pct": round((final_eq / STARTING_EQUITY - 1) * 100, 2),
        "combo_865_params": COMBO_865,
        "preregistration": "tasks/probe5_combo865_paper_trade_preregistration.md (DRAFT, unsigned)",
        "preregistration_status": "NOT_SIGNED — these numbers do NOT count toward Probe 5 §-gates.",
        "notes": [
            "Backfill is a continuity bridge between Probe 2 (220 trades, 2024-10-22 → 2026-04-08) and forward live ingest.",
            f"Sizing is fixed-${FIXED_RISK_DOLLARS:.0f} per trade on ${STARTING_EQUITY:,.0f} starting equity (no compounding); OOS-scoped override of CLAUDE.md $50,000/$500 defaults per user directive 2026-04-26.",
            "Friction is $6/contract RT (= $5 v11 baseline + $1 fill_slippage_ticks adder for combo-865; see param_sweep.py:895-901).",
            "TZ contract: data/NQ_1h.parquet validated as naive CT via src.tz_contract.assert_naive_ct.",
            "Engine tier: " + CORE_NAME,
        ],
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    print(f"  artifacts -> {out_dir}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
