"""Ablation: does ML#1 v12 UCB ranking add value over random / raw-Sharpe sampling
when fed into ML#2 V4 on the OOS test partition?

Builds three candidate pools of 50 combos each from the v11 post-gate universe
(`data/ml/ml1_results_v12/combo_features_v12.parquet`, n_trades >= 500):

    Pool A — Random 50 (seed=42)
    Pool B — Top-50 by audit_full_net_sharpe (train-partition raw net Sharpe)
    Pool C — Top-50 by v12 UCB (current evaluation/top_strategies_v12_top50.json)

Each pool is run through ML#2 V4 (LightGBM + pooled per-R:R isotonic +
fixed-$500 sizing) on the OOS test partition. Results reported per pool:
  - filtered trade count (combined portfolio)
  - annualized Sharpe p50 + 95% CI (10,000-sim IID bootstrap)
  - max drawdown (p50 / p95 / worst)
  - trades/combo distribution (median, IQR)

Output: evaluation/ablation/ablation_results.json

The three pools share a single underlying backtest + ML#2 filter pass — each
combo is processed once in the union, then pool views subset by combo_id.
This preserves apples-to-apples comparability.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.evaluation._top_perf_common import (  # noqa: E402
    _load_or_build_results_raw,
    _load_or_build_combos_ml2,
    _sim_ml2_portfolio,
    _ml2_net_ev_mask,
    DEFAULT_COST_PER_CONTRACT_RT,
    STARTING_EQUITY,
)
from scripts.evaluation.composed_strategy_runner import load_test_bars  # noqa: E402
from src.reporting import monte_carlo  # noqa: E402

FEATURES_PARQUET = REPO / "data" / "ml" / "ml1_results_v12" / "combo_features_v12.parquet"
TOP50_JSON = REPO / "evaluation" / "top_strategies_v12_top50.json"
OUTPUT_DIR = REPO / "evaluation" / "ablation"
OUTPUT_JSON = OUTPUT_DIR / "ablation_results.json"

MIN_TRADES_GATE = 500
POOL_SIZE = 50
RANDOM_SEED = 42
ML2_VERSION = "v4"
COST_PER_CONTRACT_RT = DEFAULT_COST_PER_CONTRACT_RT  # $5/contract round-trip

PARAM_COLS = [
    "z_band_k", "z_window", "volume_zscore_window",
    "ema_fast", "ema_slow",
    "stop_method", "stop_fixed_pts", "atr_multiplier", "swing_lookback",
    "min_rr", "exit_on_opposite_signal", "use_breakeven_stop",
    "max_hold_bars", "zscore_confirmation",
    "z_input", "z_anchor", "z_denom", "z_type",
    "z_window_2", "z_window_2_weight",
    "volume_entry_threshold",
    "vol_regime_lookback", "vol_regime_min_pct", "vol_regime_max_pct",
    "session_filter_mode", "tod_exit_hour",
]


def _row_to_strategy(row: pd.Series) -> dict:
    """Build a composed_strategy_runner-compatible entry from a features row."""
    params = {}
    for c in PARAM_COLS:
        if c not in row.index:
            continue
        v = row[c]
        if pd.isna(v):
            params[c] = None
        elif hasattr(v, "item"):
            params[c] = v.item()
        else:
            params[c] = v
    sfp = row.get("stop_fixed_pts", None)
    stop_fixed_pts_resolved = (float(sfp) if sfp is not None and not pd.isna(sfp)
                               else 0.0)
    params["stop_fixed_pts_resolved"] = stop_fixed_pts_resolved
    return {
        "global_combo_id": row["global_combo_id"],
        "combo_id": int(row["combo_id"]),
        "parameters": params,
        "stop_fixed_pts_resolved": stop_fixed_pts_resolved,
    }


def _build_pools(df: pd.DataFrame, top50_ids: list[str]) -> dict[str, list[str]]:
    """Return {pool_name: [global_combo_id, ...]} for each of A/B/C."""
    all_ids = df["global_combo_id"].tolist()

    rng = random.Random(RANDOM_SEED)
    pool_a = rng.sample(all_ids, POOL_SIZE)

    # Exclude NaN sharpe (shouldn't happen post-gate but guard).
    df_b = df.dropna(subset=["audit_full_net_sharpe"])
    pool_b = (df_b.sort_values("audit_full_net_sharpe", ascending=False)
              .head(POOL_SIZE)["global_combo_id"].tolist())

    pool_c = list(top50_ids)[:POOL_SIZE]

    return {"A_random": pool_a, "B_raw_sharpe": pool_b, "C_v12_ucb": pool_c}


def _pool_metrics(pool_name: str, pool_ids: list[str],
                  combos_ml2_by_id: dict, bars: pd.DataFrame,
                  years_span: float) -> dict:
    """Subset combos_ml2 to pool_ids, run combined portfolio + MC."""
    # Subset combos_ml2 to pool members (skip any that errored upstream).
    pool_combos = [c for cid in pool_ids if (c := combos_ml2_by_id.get(cid))
                   and not c.get("error") and c.get("n_trades", 0) > 0]
    n_present = len(pool_combos)
    n_missing = len(pool_ids) - n_present
    print(f"[{pool_name}] present={n_present}  missing/empty={n_missing}",
          flush=True)

    # Per-combo trade-count distribution (post ML#2 net-EV filter, pre
    # portfolio-level concurrency drops).
    per_combo_kept = []
    for c in pool_combos:
        mask = _ml2_net_ev_mask(c, COST_PER_CONTRACT_RT)
        per_combo_kept.append(int(mask.sum()))
    tc_arr = np.asarray(per_combo_kept, dtype=int)

    # Combined event-driven portfolio sim.
    port_df = _sim_ml2_portfolio(pool_combos, policy="fixed_dollars_500",
                                 bars=bars,
                                 cost_per_contract_rt=COST_PER_CONTRACT_RT)
    print(f"[{pool_name}] portfolio trades: {len(port_df)}", flush=True)

    if len(port_df) == 0:
        return {
            "pool_name": pool_name,
            "n_combos_requested": len(pool_ids),
            "n_combos_present": n_present,
            "portfolio_trades": 0,
            "note": "empty portfolio",
        }

    pnl = port_df["actual_pnl"].to_numpy(dtype=float)
    risk = port_df["dollar_risk"].to_numpy(dtype=float)

    mc = monte_carlo(pnl, risk_base=risk, policy="fixed_dollars_500",
                     years_span=years_span, n_sims=10_000, seed=42)

    total_pnl = float(pnl.sum())
    equity_path = STARTING_EQUITY + np.cumsum(pnl)
    peak = np.maximum.accumulate(equity_path)
    observed_dd_pct = float(((peak - equity_path) / peak).max() * 100)

    return {
        "pool_name": pool_name,
        "n_combos_requested": len(pool_ids),
        "n_combos_present": n_present,
        "portfolio_trades": int(len(port_df)),
        "total_pnl_dollars": round(total_pnl, 2),
        "observed_max_dd_pct": round(observed_dd_pct, 2),
        "sharpe_p50": mc.get("sharpe_p50"),
        "sharpe_ci_95": mc.get("sharpe_ci_95"),
        "sharpe_pos_prob": mc.get("sharpe_pos_prob"),
        "trades_per_year": mc.get("trades_per_year"),
        "dd_p50_pct": mc.get("dd_p50_pct"),
        "dd_p95_pct": mc.get("dd_p95_pct"),
        "dd_p99_pct": mc.get("dd_p99_pct"),
        "dd_worst_pct": mc.get("dd_worst_pct"),
        "win_rate": mc.get("win_rate"),
        "wr_ci_95": mc.get("wr_ci_95"),
        "risk_of_ruin_prob": mc.get("risk_of_ruin_prob"),
        "trades_per_combo": {
            "n": int(len(tc_arr)),
            "median": int(np.median(tc_arr)) if len(tc_arr) else 0,
            "p25": int(np.percentile(tc_arr, 25)) if len(tc_arr) else 0,
            "p75": int(np.percentile(tc_arr, 75)) if len(tc_arr) else 0,
            "min": int(tc_arr.min()) if len(tc_arr) else 0,
            "max": int(tc_arr.max()) if len(tc_arr) else 0,
            "zero_combos": int((tc_arr == 0).sum()),
        },
    }


def main() -> None:
    print(f"[ablation] loading {FEATURES_PARQUET.relative_to(REPO)}", flush=True)
    df_all = pd.read_parquet(FEATURES_PARQUET)
    df = df_all[df_all["audit_n_trades"] >= MIN_TRADES_GATE].reset_index(drop=True)
    print(f"[ablation] {len(df_all)} combos total, {len(df)} post-gate "
          f"(n_trades >= {MIN_TRADES_GATE})", flush=True)

    top50 = json.loads(TOP50_JSON.read_text())
    top50_ids = [e["global_combo_id"] for e in top50["top"]]
    print(f"[ablation] v12 top-50 source: {TOP50_JSON.name} "
          f"({len(top50_ids)} entries)", flush=True)

    pools = _build_pools(df, top50_ids)
    for name, ids in pools.items():
        print(f"[ablation] pool {name}: {len(ids)} combos "
              f"(first 3: {ids[:3]})", flush=True)

    union_ids = sorted(set().union(*pools.values()))
    print(f"[ablation] union pool: {len(union_ids)} unique combos", flush=True)

    # Build strategy dicts for the union, keyed by global_combo_id. Use
    # drop=False so the id stays accessible as a row value (_row_to_strategy
    # reads row["global_combo_id"]).
    df_by_id = df.set_index("global_combo_id", drop=False)
    union_strats = []
    missing_from_features = []
    for gcid in union_ids:
        if gcid not in df_by_id.index:
            missing_from_features.append(gcid)
            continue
        union_strats.append(_row_to_strategy(df_by_id.loc[gcid]))
    if missing_from_features:
        print(f"[ablation] WARN: {len(missing_from_features)} union ids absent "
              f"from features parquet: {missing_from_features[:5]}...", flush=True)

    # Load OOS bars + years span.
    bars = load_test_bars()
    years_span = ((pd.to_datetime(bars["time"].iloc[-1])
                   - pd.to_datetime(bars["time"].iloc[0])).total_seconds()
                  / (365.25 * 86400))
    print(f"[ablation] OOS bars: {len(bars):,}  "
          f"{bars['time'].iloc[0]} -> {bars['time'].iloc[-1]}  "
          f"years={years_span:.3f}", flush=True)

    # Raw unfiltered per-combo backtest (cached across pool runs).
    _ = _load_or_build_results_raw(union_strats, bars)

    # ML#2 V4 filter pass over the union.
    combo_ids = [s["global_combo_id"] for s in union_strats]
    combos_ml2 = _load_or_build_combos_ml2(combo_ids, bars, version=ML2_VERSION)
    combos_ml2_by_id = {c.get("combo_id"): c for c in combos_ml2}

    # Per-pool metrics.
    results = {}
    for name, ids in pools.items():
        print(f"\n[ablation] === pool {name} ===", flush=True)
        results[name] = _pool_metrics(name, ids, combos_ml2_by_id, bars,
                                      years_span)

    payload = {
        "ablation": "ml1_value_vs_random_vs_raw_sharpe",
        "ml2_version": ML2_VERSION,
        "cost_per_contract_rt": COST_PER_CONTRACT_RT,
        "min_trades_gate": MIN_TRADES_GATE,
        "pool_size": POOL_SIZE,
        "random_seed": RANDOM_SEED,
        "post_gate_universe": int(len(df)),
        "years_span": round(years_span, 3),
        "bars_first": str(bars["time"].iloc[0]),
        "bars_last": str(bars["time"].iloc[-1]),
        "pools": results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\n[ablation] wrote {OUTPUT_JSON.relative_to(REPO)}", flush=True)

    # Console summary table.
    print("\n=== ABLATION SUMMARY ===")
    print(f"{'Pool':<16} {'N':>3} {'Trades':>7} {'Sharpe p50':>11} "
          f"{'95% CI':>22} {'DD worst':>9} {'T/combo (med)':>14}")
    for name, r in results.items():
        if r.get("portfolio_trades", 0) == 0:
            print(f"{name:<16} {r['n_combos_present']:>3} "
                  f"{'0':>7} {'--':>11} {'--':>22} {'--':>9} {'--':>14}")
            continue
        ci = r["sharpe_ci_95"]
        print(f"{name:<16} {r['n_combos_present']:>3} "
              f"{r['portfolio_trades']:>7} {r['sharpe_p50']:>11.3f} "
              f"[{ci[0]:>8.3f},{ci[1]:>8.3f}] "
              f"{r['dd_worst_pct']:>8.1f}% "
              f"{r['trades_per_combo']['median']:>14}")


if __name__ == "__main__":
    main()
