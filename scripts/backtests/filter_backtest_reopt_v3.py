"""Phase 5A: Re-optimize per-combo filter thresholds using V3 + two-stage calibrator.

Same structure as filter_backtest_per_combo_v3.py but uses the Phase 4f
production calibrator (per-combo isotonic → pooled per-R:R fallback) instead
of the simple per-R:R isotonic.

Hypothesis: the two-stage calibrator produces better-calibrated P(win) per
combo, which may unlock E[R]-threshold improvements that Phase 3's simple
calibrator missed. Phase 3 found null (V3 discrimination gain ≠ Sharpe gain);
this tests whether calibration was the missing link.

Output: data/ml/adaptive_rr_v3/filter_backtest_reopt_v3.json
"""
from __future__ import annotations
import importlib.util, json, sys, traceback
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("fb", REPO / "scripts/backtests/filter_backtest_v2.py")
fb = importlib.util.module_from_spec(spec); spec.loader.exec_module(fb)

spec2 = importlib.util.spec_from_file_location("fbs", REPO / "scripts/backtests/filter_backtest_surrogate_v2.py")
fbs = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(fbs)

spec3 = importlib.util.spec_from_file_location("v3inf", REPO / "scripts/models/inference_v3.py")
v3inf = importlib.util.module_from_spec(spec3); spec3.loader.exec_module(v3inf)

OUT = REPO / "data/ml/adaptive_rr_v3/filter_backtest_reopt_v3.json"
SURROGATE_CSV = REPO / "data/ml/ml1_results/surrogate_top_combos.csv"

THRESHOLDS = [round(x, 2) for x in np.arange(-0.5, 0.51, 0.05)]
MIN_TRADES_FOR_OPTIMUM = 20


def pick_optimum(sweep: list[dict]) -> dict | None:
    """Pick the E[R] threshold that maximises simulated return per combo.

    Sweeps a grid of candidate thresholds and evaluates each via a vectorised
    trade-level simulation using the supplied R:R choice and sizing policy.

    Args:
        rows: Per-trade frame with `e_r`, `r_multiple_at_chosen_rr`, and any
            sizing inputs required by the policy.

    Returns:
        `(threshold, metrics_dict)` for the best-performing threshold.
    """
    eligible = [s for s in sweep if s.get("n_trades", 0) >= MIN_TRADES_FOR_OPTIMUM
                and "sharpe_ratio" in s]
    if not eligible:
        return None
    best = max(eligible, key=lambda s: s["sharpe_ratio"])
    return {
        "threshold": best["threshold"],
        "sharpe_ratio": best["sharpe_ratio"],
        "total_return_pct": best.get("total_return_pct"),
        "max_drawdown_pct": best.get("max_drawdown_pct"),
        "win_rate": best.get("win_rate"),
        "n_trades": best["n_trades"],
        "skip_rate": best.get("skip_rate"),
    }


def backtest_combo_calibrated(gcid: str, combo: dict, booster, two_stage,
                              simple_cals) -> dict:
    """Run backtest for one combo using two-stage calibrated predictions."""
    avf = fb.avf
    rr = float(combo["min_rr"])
    df = avf.load_bars(avf.DATA_CSV)
    train, _ = avf.split_train_test(df, 0.8)
    df_ind = avf.build_indicators(train, combo)
    stop_pts = avf.resolve_stop_pts(combo, df_ind)
    cfg = avf.make_cfg(combo, stop_pts)
    df_sig = avf.generate_signals(df_ind, cfg)
    trades = avf.run_core(df_sig, cfg)
    n = len(trades["side"])
    if n == 0:
        return {"global_combo_id": gcid, "error": "no trades"}

    feats = avf.build_features(trades, df_sig, stop_pts,
                               str(combo["stop_method"]),
                               bool(combo["exit_on_opposite_signal"]))

    # Two-stage calibrated prediction (per-combo isotonic → pooled per-R:R).
    pwin = v3inf.predict_pwin_v3_calibrated_at_rr(
        feats, trades, gcid, stop_pts, rr,
        booster=booster, two_stage=two_stage,
    )
    ev = pwin * rr - (1.0 - pwin)

    # Also compute simple-calibrated for comparison.
    pwin_simple = v3inf.predict_pwin_v3_at_rr(
        feats, trades, gcid, stop_pts, rr,
        booster=booster, calibrators=simple_cals,
    )
    ev_simple = pwin_simple * rr - (1.0 - pwin_simple)

    fixed_res = avf.simulate_fixed(trades)
    fixed_m = avf.compute_equity_and_metrics(fixed_res, trades, cfg, "fixed")

    # Sweep thresholds with two-stage calibrator.
    sweep_calibrated = []
    for thr in THRESHOLDS:
        keep = ev >= thr
        n_kept = int(keep.sum())
        entry = {"threshold": thr, "n_trades": n_kept,
                 "skip_rate": float(1 - keep.mean())}
        if n_kept == 0:
            sweep_calibrated.append(entry); continue
        subset = {k: (np.asarray(v)[keep] if hasattr(v, "__len__") and len(v) == n else v)
                  for k, v in trades.items()}
        fres = avf.simulate_fixed(subset)
        fm = avf.compute_equity_and_metrics(fres, subset, cfg, f"thr_{thr}")
        entry.update({
            "sharpe_ratio": fm.get("sharpe_ratio"),
            "total_return_pct": fm.get("total_return_pct"),
            "max_drawdown_pct": fm.get("max_drawdown_pct"),
            "win_rate": fm.get("win_rate"),
            "avg_pnl_dollars": fm.get("avg_pnl_dollars"),
        })
        sweep_calibrated.append(entry)

    # Sweep thresholds with simple calibrator (for comparison).
    sweep_simple = []
    for thr in THRESHOLDS:
        keep = ev_simple >= thr
        n_kept = int(keep.sum())
        entry = {"threshold": thr, "n_trades": n_kept,
                 "skip_rate": float(1 - keep.mean())}
        if n_kept == 0:
            sweep_simple.append(entry); continue
        subset = {k: (np.asarray(v)[keep] if hasattr(v, "__len__") and len(v) == n else v)
                  for k, v in trades.items()}
        fres = avf.simulate_fixed(subset)
        fm = avf.compute_equity_and_metrics(fres, subset, cfg, f"thr_{thr}")
        entry.update({
            "sharpe_ratio": fm.get("sharpe_ratio"),
            "total_return_pct": fm.get("total_return_pct"),
            "max_drawdown_pct": fm.get("max_drawdown_pct"),
            "win_rate": fm.get("win_rate"),
            "avg_pnl_dollars": fm.get("avg_pnl_dollars"),
        })
        sweep_simple.append(entry)

    return {
        "global_combo_id": gcid,
        "min_rr": rr,
        "n_signals": n,
        "mean_pwin_calibrated": float(pwin.mean()),
        "mean_pwin_simple": float(pwin_simple.mean()),
        "mean_ev_R_calibrated": float(ev.mean()),
        "mean_ev_R_simple": float(ev_simple.mean()),
        "has_per_combo_calibrator": gcid in two_stage["per_combo"],
        "fixed": fixed_m,
        "sweep_calibrated": sweep_calibrated,
        "sweep_simple": sweep_simple,
        "optimum_calibrated": pick_optimum(sweep_calibrated),
        "optimum_simple": pick_optimum(sweep_simple),
    }


def main():
    """Phase 5A: re-optimise per-combo filter thresholds on V3 + two-stage cal.

    Loads the V3 booster + per-combo two-stage calibrator, sweeps thresholds
    per combo, and compares the re-optimised thresholds to the V2/B1 baseline.
    Writes summary JSON including Sharpe deltas and coverage.
    """
    import lightgbm as lgb
    import time

    t0 = time.time()

    high_freq = ["v10_7649", "v10_8617", "v10_9264", "v10_9393", "v6_1676"]
    low_freq = ["v10_9955", "v5_158", "v5_2904", "v7_2114", "v7_215"]
    existing = high_freq + low_freq

    booster = lgb.Booster(model_file=str(v3inf.V3_BOOSTER))
    two_stage = v3inf._load_per_combo_calibrators()
    simple_cals = v3inf._load_calibrators()

    print(f"Loaded two-stage calibrators: "
          f"{len(two_stage['per_combo'])} per-combo, "
          f"{len(two_stage['pooled_per_rr'])} pooled per-R:R")

    all_results = {
        "model": "v3_two_stage_calibrated",
        "thresholds": THRESHOLDS,
        "min_trades_for_optimum": MIN_TRADES_FOR_OPTIMUM,
        "n_per_combo_calibrators": len(two_stage["per_combo"]),
        "combos": [],
    }

    for gcid in existing:
        print(f"\n=== {gcid} ===", flush=True)
        try:
            combo = fb.load_combo_by_id(gcid)
            r = backtest_combo_calibrated(gcid, combo, booster, two_stage, simple_cals)
        except Exception as e:
            r = {"global_combo_id": gcid, "error": str(e),
                 "tb": traceback.format_exc()[:400]}
            print(f"  ERROR: {e}")
        all_results["combos"].append(r)
        if "error" not in r:
            fx = r["fixed"]
            oc = r.get("optimum_calibrated")
            os_ = r.get("optimum_simple")
            has_pc = r.get("has_per_combo_calibrator", False)
            print(f"  per-combo cal: {'YES' if has_pc else 'no (fallback)'}")
            print(f"  fixed: sharpe={fx.get('sharpe_ratio',0):.2f} "
                  f"wr={fx.get('win_rate',0):.3f} n={fx.get('n_trades',0)}")
            if oc:
                print(f"  best(calibrated): thr={oc['threshold']} "
                      f"sharpe={oc['sharpe_ratio']:.2f} "
                      f"wr={oc.get('win_rate',0):.3f} "
                      f"n={oc['n_trades']} skip={oc['skip_rate']:.1%}")
            if os_:
                print(f"  best(simple):     thr={os_['threshold']} "
                      f"sharpe={os_['sharpe_ratio']:.2f} "
                      f"wr={os_.get('win_rate',0):.3f} "
                      f"n={os_['n_trades']} skip={os_['skip_rate']:.1%}")

    # Surrogate combos.
    surrogate = pd.read_csv(SURROGATE_CSV)
    for i, row in surrogate.iterrows():
        gcid = f"surrogate_{i}"
        print(f"\n=== {gcid} ===", flush=True)
        try:
            combo = fbs.row_to_combo(i, row)
            r = backtest_combo_calibrated(gcid, combo, booster, two_stage, simple_cals)
        except Exception as e:
            r = {"global_combo_id": gcid, "error": str(e),
                 "tb": traceback.format_exc()[:400]}
            print(f"  ERROR: {e}")
        all_results["combos"].append(r)
        if "error" not in r:
            fx = r["fixed"]
            oc = r.get("optimum_calibrated")
            os_ = r.get("optimum_simple")
            has_pc = r.get("has_per_combo_calibrator", False)
            print(f"  per-combo cal: {'YES' if has_pc else 'no (fallback)'}")
            print(f"  fixed: sharpe={fx.get('sharpe_ratio',0):.2f} "
                  f"wr={fx.get('win_rate',0):.3f} n={fx.get('n_trades',0)}")
            if oc:
                print(f"  best(calibrated): thr={oc['threshold']} "
                      f"sharpe={oc['sharpe_ratio']:.2f} "
                      f"wr={oc.get('win_rate',0):.3f} "
                      f"n={oc['n_trades']} skip={oc['skip_rate']:.1%}")
            if os_:
                print(f"  best(simple):     thr={os_['threshold']} "
                      f"sharpe={os_['sharpe_ratio']:.2f} "
                      f"wr={os_.get('win_rate',0):.3f} "
                      f"n={os_['n_trades']} skip={os_['skip_rate']:.1%}")

    all_results["runtime_seconds"] = time.time() - t0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSaved: {OUT}")
    print(f"Total runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
