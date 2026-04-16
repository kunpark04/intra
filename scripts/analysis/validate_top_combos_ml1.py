"""
validate_top_combos_ml1.py — Run top ML-recommended combos on the held-out test partition.

Compares training performance (from the sweep) to test performance (unseen data)
to verify that the ML-selected parameters generalize and aren't overfit to
the training period.

This is the critical validation step before live deployment.

Key methodology choices:
- Stop distances use the sweep's pre-resolved fixed value (stop_fixed_pts_resolved
  from the manifest) for apples-to-apples comparison with sweep results.
- Sharpe ratio is computed from R-multiples (not dollar PnLs) so it's independent
  of position sizing method (fixed vs compounding).
- Drawdown is computed from R-multiples as well, treating each trade as +/- R units.

Usage:
    python scripts/analysis/validate_top_combos_ml1.py
    python scripts/analysis/validate_top_combos_ml1.py --n-top 10
    python scripts/analysis/validate_top_combos_ml1.py --partition both
    python scripts/analysis/validate_top_combos_ml1.py --tag high_freq  # saves as validation_high_freq.csv
"""

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import src.config as base_cfg
from src.data_loader import load_bars, split_train_test
from src.indicators import add_indicators
from src.indicators.zscore_variants import compute_zscore_v2, compute_vwap_session
from src.strategy import generate_signals

# ── Constants ────────────────────────────────────────────────────────────────

STARTING_EQUITY = 50_000.0
TOP_COMBOS_PATH = Path("data/ml/ml1_results/top_combos.csv")
OUTPUT_DIR = Path("data/ml/ml1_results/validation")

# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate top ML combos on test partition")
    p.add_argument("--n-top", type=int, default=5,
                   help="Number of top combos to validate (default: 5)")
    p.add_argument("--partition", choices=["test", "train", "both"], default="both",
                   help="Which partition to run on (default: both for comparison)")
    p.add_argument("--top-combos", type=str, default=str(TOP_COMBOS_PATH),
                   help="Path to top_combos.csv from ML optimizer")
    p.add_argument("--tag", type=str, default="",
                   help="Tag for output filenames (e.g. 'high_freq' -> validation_high_freq.csv)")
    return p.parse_args()


# ── Manifest loader ─────────────────────────────────────────────────────────

def load_resolved_stops() -> dict:
    """
    Load stop_fixed_pts_resolved for every combo from all manifests.

    Returns a dict mapping global_combo_id -> resolved stop distance.
    This is the exact stop distance the sweep used, ensuring identical
    entry/exit logic between sweep and validation.
    """
    resolved = {}
    data_dir = Path("data/ml")
    for v in range(2, 11):
        manifest_path = data_dir / "originals" / f"ml_dataset_v{v}_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest:
            gid = f"v{v}_{entry['combo_id']}"
            val = entry.get("stop_fixed_pts_resolved")
            if val is not None:
                resolved[gid] = float(val)
    return resolved


# ── Config builder ───────────────────────────────────────────────────────────

def _make_cfg_from_combo(row: pd.Series, resolved_stop: float) -> types.SimpleNamespace:
    """
    Build a config namespace from a combo row + the manifest's resolved stop.

    Uses stop_fixed_pts_resolved from the manifest directly instead of
    recomputing from ATR/swing medians. This guarantees the same stop
    distance the sweep used, making validation results directly comparable.
    """
    ns = types.SimpleNamespace()

    # Copy all base config values first
    for k in dir(base_cfg):
        if not k.startswith("_"):
            setattr(ns, k, getattr(base_cfg, k))

    # Override with combo-specific parameters
    ns.Z_BAND_K              = float(row["z_band_k"])
    ns.Z_WINDOW              = int(row["z_window"])
    ns.VOLUME_ZSCORE_WINDOW  = int(row["volume_zscore_window"])
    ns.EMA_FAST              = int(row["ema_fast"])
    ns.EMA_SLOW              = int(row["ema_slow"])
    ns.MIN_RR                = float(row["min_rr"])
    ns.EXIT_ON_OPPOSITE_SIGNAL = bool(row["exit_on_opposite_signal"])
    ns.USE_BREAKEVEN_STOP    = bool(row.get("use_breakeven_stop", False))
    ns.MAX_HOLD_BARS         = int(row.get("max_hold_bars", 0))
    ns.ZSCORE_CONFIRMATION   = bool(row.get("zscore_confirmation", False))

    # V5+ filters
    ns.VOLUME_ENTRY_THRESHOLD = float(row.get("volume_entry_threshold", 0.0))
    ns.VOL_REGIME_LOOKBACK   = int(row.get("vol_regime_lookback", 0))
    ns.VOL_REGIME_MIN_PCT    = float(row.get("vol_regime_min_pct", 0.0))
    ns.VOL_REGIME_MAX_PCT    = float(row.get("vol_regime_max_pct", 1.0))
    ns.SESSION_FILTER_MODE   = int(row.get("session_filter_mode", 0))
    ns.TOD_EXIT_HOUR         = int(row.get("tod_exit_hour", 0))

    # Use the sweep's exact resolved stop distance — no recomputation needed.
    # This is the single most important fix: ATR/swing stops are now identical
    # to what the sweep used, regardless of partition.
    ns.STOP_METHOD    = "fixed"
    ns.STOP_FIXED_PTS = resolved_stop

    # Signal mode — all sweep combos use zscore_reversal
    ns.SIGNAL_MODE = "zscore_reversal"

    return ns


# ── Z-score variant indicator builder ────────────────────────────────────────

def _build_indicators_with_zscore_variant(df: pd.DataFrame, cfg, row: pd.Series) -> pd.DataFrame:
    """
    Build the full indicator DataFrame, replacing the default z-score
    with the combo's specific z-score variant formulation if needed.

    The standard add_indicators() always uses the default parametric z-score
    (close / rolling_mean / rolling_std). If the combo uses a different
    formulation (e.g., returns/ema_fast/atr/quantile_rank), we need to
    recompute the zscore column using compute_zscore_v2.
    """
    # Start with standard indicators (EMA, ATR, volume zscore, etc.)
    df_ind = add_indicators(df.copy(), cfg)

    # Check if this combo uses a non-default z-score formulation
    z_input  = str(row.get("z_input", "close"))
    z_anchor = str(row.get("z_anchor", "rolling_mean"))
    z_denom  = str(row.get("z_denom", "rolling_std"))
    z_type   = str(row.get("z_type", "parametric"))
    z_window_2 = int(row.get("z_window_2", 0))
    z_window_2_weight = float(row.get("z_window_2_weight", 0.0))

    is_default = (z_input == "close" and z_anchor == "rolling_mean"
                  and z_denom == "rolling_std" and z_type == "parametric"
                  and z_window_2 == 0)

    if not is_default:
        # Recompute z-score with the variant formulation
        close_np = df_ind["close"].to_numpy(dtype=np.float64)
        high_np  = df_ind["high"].to_numpy(dtype=np.float64)
        low_np   = df_ind["low"].to_numpy(dtype=np.float64)
        volume_np = df_ind["volume"].to_numpy(dtype=np.float64)
        session_brk_np = df_ind["session_break"].to_numpy()

        ema_f = df_ind["ema_fast"].to_numpy(dtype=np.float64)
        ema_s = df_ind["ema_slow"].to_numpy(dtype=np.float64)
        atr_arr = df_ind["atr"].to_numpy(dtype=np.float64)

        # Compute VWAP session array (needed for vwap_session anchor)
        vwap_arr = compute_vwap_session(high_np, low_np, close_np, volume_np, session_brk_np)

        zscore_new = compute_zscore_v2(
            close_np, high_np, low_np, volume_np, session_brk_np,
            ema_f, ema_s, atr_arr,
            cfg.Z_WINDOW, z_input, z_anchor, z_denom, z_type, 0,
            vwap_arr, z_window_2, z_window_2_weight,
        )
        df_ind["zscore"] = zscore_new

    return df_ind


# ── Backtest runner ──────────────────────────────────────────────────────────

def run_single_combo(
    df_partition: pd.DataFrame,
    row: pd.Series,
    partition_name: str,
    resolved_stop: float,
) -> dict:
    """
    Run a full backtest for one combo on a given partition.

    Sharpe ratio is computed from R-multiples (not dollar PnLs) so it's
    independent of position sizing. A 2R win counts the same whether the
    account was $50K or $5M. This makes Sharpe comparable between the
    sweep (fixed sizing) and full backtest (compounding sizing).
    """
    from src.backtest import run_backtest

    cfg = _make_cfg_from_combo(row, resolved_stop)
    combo_id = row["global_combo_id"]

    t0 = time.time()

    # Build indicators (with z-score variant support)
    df_ind = _build_indicators_with_zscore_variant(df_partition, cfg, row)

    # Generate signals
    df_sig = generate_signals(df_ind, cfg)
    n_signals = (df_sig["signal"] != 0).sum()

    # Run full backtest
    results = run_backtest(df_sig, cfg, version=f"ML_{combo_id}_{partition_name}")

    elapsed = time.time() - t0
    n_trades = results["n_trades"]
    trades = results["trades"]

    if n_trades > 0:
        pnls = np.array([t["net_pnl_dollars"] for t in trades])
        wins = np.array([t["label_win"] for t in trades])
        r_mults = np.array([t["r_multiple"] for t in trades])

        total_pnl = pnls.sum()
        win_rate = wins.mean()

        # Sharpe from R-multiples: normalizes by risk so it's independent
        # of position sizing (fixed vs compounding). mean(R) / std(R).
        r_std = np.std(r_mults, ddof=1) if len(r_mults) > 1 else 0.0
        sharpe_r = float(np.mean(r_mults) / r_std) if r_std > 0 else 0.0

        # Max drawdown from R-multiples: cumulative R-sum peak-to-trough.
        # This measures drawdown in risk units, not dollars.
        cum_r = np.cumsum(r_mults)
        cum_r_full = np.concatenate([[0.0], cum_r])
        r_peak = np.maximum.accumulate(cum_r_full)
        # Drawdown in R-units (absolute, not percentage)
        r_drawdowns = r_peak - cum_r_full
        max_dd_r = float(r_drawdowns.max())

        # Also compute dollar-based drawdown for reference
        equity = STARTING_EQUITY + np.cumsum(pnls)
        equity_full = np.concatenate([[STARTING_EQUITY], equity])
        running_peak = np.maximum.accumulate(equity_full)
        with np.errstate(divide="ignore", invalid="ignore"):
            dd_pct = (running_peak - equity_full) / running_peak
        dd_pct = np.nan_to_num(dd_pct, nan=0.0)
        max_dd_pct = float(dd_pct.max() * 100)

        # Profit factor
        win_sum = pnls[pnls > 0].sum()
        loss_sum = abs(pnls[pnls < 0].sum())
        pf = min(float(win_sum / loss_sum), 100.0) if loss_sum > 0 else 100.0
    else:
        total_pnl = 0.0
        win_rate = 0.0
        sharpe_r = 0.0
        max_dd_r = 0.0
        max_dd_pct = 0.0
        pf = 0.0
        r_mults = np.array([])

    return {
        "global_combo_id": combo_id,
        "partition": partition_name,
        "n_trades": n_trades,
        "n_signals": int(n_signals),
        "win_rate": round(float(win_rate), 4),
        "total_return_pct": round(total_pnl / STARTING_EQUITY * 100, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "max_drawdown_R": round(max_dd_r, 2),
        "sharpe_r": round(sharpe_r, 4),
        "profit_factor": round(pf, 2),
        "avg_r_multiple": round(float(np.mean(r_mults)), 4) if n_trades > 0 else 0,
        "runtime_seconds": round(elapsed, 1),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    t_start = time.time()

    print("=" * 70)
    print("Test Partition Validation - Top ML Combos")
    print("=" * 70)

    # Load resolved stops from manifests
    print("[validate] Loading resolved stops from manifests...", flush=True)
    resolved_stops = load_resolved_stops()
    print(f"  Loaded {len(resolved_stops):,} resolved stop values")

    # Load top combos
    top_df = pd.read_csv(args.top_combos)
    top_df = top_df.head(args.n_top)
    print(f"  Validating top {len(top_df)} combos")
    print(f"  Partition: {args.partition}")
    print()

    # Verify all combos have resolved stops
    for _, row in top_df.iterrows():
        gid = row["global_combo_id"]
        if gid not in resolved_stops:
            print(f"  [ERROR] No resolved stop for {gid} — cannot validate.")
            sys.exit(1)
        print(f"  {gid}: stop_resolved={resolved_stops[gid]:.4f} pts")
    print()

    # Load price data
    print("[validate] Loading bars...", flush=True)
    df_raw = load_bars("data/NQ_1min.csv")
    train_df, test_df = split_train_test(df_raw, base_cfg.TRAIN_RATIO)
    print(f"  Train bars: {len(train_df):,}  |  Test bars: {len(test_df):,}")
    print()

    # Determine which partitions to run
    partitions = {}
    if args.partition in ("train", "both"):
        partitions["train"] = train_df
    if args.partition in ("test", "both"):
        partitions["test"] = test_df

    # Run each combo on each partition
    all_results = []
    for idx, (_, row) in enumerate(top_df.iterrows()):
        combo_id = row["global_combo_id"]
        stop = resolved_stops[combo_id]
        print(f"[validate] Combo {idx+1}/{len(top_df)}: {combo_id} (stop={stop:.4f})", flush=True)

        for part_name, part_df in partitions.items():
            print(f"  Running on {part_name} partition...", end=" ", flush=True)
            result = run_single_combo(part_df.copy(), row, part_name, stop)
            all_results.append(result)
            print(f"trades={result['n_trades']}  WR={result['win_rate']:.1%}  "
                  f"return={result['total_return_pct']:.1f}%  "
                  f"DD={result['max_drawdown_pct']:.1f}%  "
                  f"sharpe_R={result['sharpe_r']:.3f}  "
                  f"PF={result['profit_factor']:.2f}  ({result['runtime_seconds']:.1f}s)",
                  flush=True)

        print()

    # Build results DataFrame
    results_df = pd.DataFrame(all_results)

    # ── Print comparison table ──
    if args.partition == "both":
        print("=" * 70)
        print("TRAIN vs TEST COMPARISON")
        print("=" * 70)
        print()

        train_results = results_df[results_df["partition"] == "train"].set_index("global_combo_id")
        test_results = results_df[results_df["partition"] == "test"].set_index("global_combo_id")

        compare_cols = ["n_trades", "win_rate", "total_return_pct",
                        "max_drawdown_pct", "sharpe_r", "profit_factor"]

        for combo_id in train_results.index:
            tr = train_results.loc[combo_id]
            te = test_results.loc[combo_id]
            print(f"  {combo_id}")
            print(f"  {'Metric':<20s}  {'Train':>12s}  {'Test':>12s}  {'Delta':>12s}")
            print(f"  {'-'*20}  {'-'*12}  {'-'*12}  {'-'*12}")

            for col in compare_cols:
                tv = tr[col]
                ev = te[col]
                delta = ev - tv
                delta_str = f"{delta:+.4f}"
                print(f"  {col:<20s}  {tv:>12.4f}  {ev:>12.4f}  {delta_str:>12s}")
            print()

        # Overall summary
        print("=" * 70)
        print("DEGRADATION SUMMARY (test - train averages)")
        print("=" * 70)
        for col in compare_cols:
            train_avg = train_results[col].mean()
            test_avg = test_results[col].mean()
            delta = test_avg - train_avg
            pct_change = (delta / abs(train_avg) * 100) if train_avg != 0 else 0
            print(f"  {col:<20s}  train_avg={train_avg:>10.4f}  test_avg={test_avg:>10.4f}  "
                  f"delta={delta:>+10.4f}  ({pct_change:>+.1f}%)")

    # Save results with optional tag for distinct filenames
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{args.tag}" if args.tag else ""

    results_path = OUTPUT_DIR / f"validation{tag_suffix}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n  Saved: {results_path}")

    json_path = OUTPUT_DIR / f"validation{tag_suffix}.json"
    json_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"  Saved: {json_path}")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
