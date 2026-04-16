"""B1: Per-combo optimal filter threshold.

For each combo, sweep absolute E[R] thresholds and pick the one that maximises
Sharpe (with a min-trades guard). Tests the hypothesis that a single global
threshold over-prunes high-freq combos: low-WR / high-freq combos should
prefer a lower threshold, high-WR / low-freq combos a higher one.

Inputs: V2 adaptive R:R model at data/ml/adaptive_rr_v2/adaptive_rr_model.txt
Combos: 10 existing + 50 from surrogate_top_combos.csv.
Output: data/ml/adaptive_rr_v2/filter_backtest_per_combo.json
"""
from __future__ import annotations
import importlib.util, json, sys, traceback
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("fb", REPO / "scripts/filter_backtest_v2.py")
fb = importlib.util.module_from_spec(spec); spec.loader.exec_module(fb)

spec2 = importlib.util.spec_from_file_location("fbs", REPO / "scripts/filter_backtest_surrogate_v2.py")
fbs = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(fbs)

OUT = REPO / "data/ml/adaptive_rr_v2/filter_backtest_per_combo.json"
SURROGATE_CSV = REPO / "data/ml/lgbm_results/surrogate_top_combos.csv"

# Absolute E[R] threshold grid, -0.5 to +0.5 step 0.05 (21 points).
THRESHOLDS = [round(x, 2) for x in np.arange(-0.5, 0.51, 0.05)]
MIN_TRADES_FOR_OPTIMUM = 20


def pick_optimum(sweep: list[dict]) -> dict | None:
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


def backtest_per_combo(gcid: str, combo: dict, model) -> dict:
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
    pwin = fb.predict_pwin_at_rr(feats, model, rr)
    ev = pwin * rr - (1.0 - pwin)

    fixed_res = avf.simulate_fixed(trades)
    fixed_m = avf.compute_equity_and_metrics(fixed_res, trades, cfg, "fixed")

    sweep = []
    for thr in THRESHOLDS:
        keep = ev >= thr
        n_kept = int(keep.sum())
        entry = {"threshold": thr, "n_trades": n_kept,
                 "skip_rate": float(1 - keep.mean())}
        if n_kept == 0:
            sweep.append(entry); continue
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
        sweep.append(entry)

    return {
        "global_combo_id": gcid,
        "min_rr": rr,
        "n_signals": n,
        "mean_pwin": float(pwin.mean()),
        "mean_ev_R": float(ev.mean()),
        "fixed": fixed_m,
        "threshold_sweep": sweep,
        "optimum": pick_optimum(sweep),
    }


def main():
    import lightgbm as lgb
    high_freq = ["v10_7649", "v10_8617", "v10_9264", "v10_9393", "v6_1676"]
    low_freq = ["v10_9955", "v5_158", "v5_2904", "v7_2114", "v7_215"]
    existing = high_freq + low_freq

    model = lgb.Booster(model_file=str(fb.V2_MODEL))
    all_results = {"model": "v2", "thresholds": THRESHOLDS,
                   "min_trades_for_optimum": MIN_TRADES_FOR_OPTIMUM,
                   "combos": []}

    for gcid in existing:
        print(f"\n=== {gcid} ===", flush=True)
        try:
            combo = fb.load_combo_by_id(gcid)
            r = backtest_per_combo(gcid, combo, model)
        except Exception as e:
            r = {"global_combo_id": gcid, "error": str(e),
                 "tb": traceback.format_exc()[:400]}
            print(f"  ERROR: {e}")
        all_results["combos"].append(r)
        if "error" not in r and r.get("optimum"):
            fx, op = r["fixed"], r["optimum"]
            print(f"  fixed: sharpe={fx.get('sharpe_ratio',0):.2f} wr={fx.get('win_rate',0):.3f} n={fx.get('n_trades',0)}")
            print(f"  best: thr={op['threshold']} sharpe={op['sharpe_ratio']:.2f} "
                  f"wr={op.get('win_rate',0):.3f} n={op['n_trades']} skip={op['skip_rate']:.1%}")

    surrogate = pd.read_csv(SURROGATE_CSV)
    for i, row in surrogate.iterrows():
        gcid = f"surrogate_{i}"
        print(f"\n=== {gcid} ===", flush=True)
        try:
            combo = fbs.row_to_combo(i, row)
            r = backtest_per_combo(gcid, combo, model)
        except Exception as e:
            r = {"global_combo_id": gcid, "error": str(e),
                 "tb": traceback.format_exc()[:400]}
            print(f"  ERROR: {e}")
        all_results["combos"].append(r)
        if "error" not in r and r.get("optimum"):
            fx, op = r["fixed"], r["optimum"]
            print(f"  fixed: sharpe={fx.get('sharpe_ratio',0):.2f} wr={fx.get('win_rate',0):.3f} n={fx.get('n_trades',0)}")
            print(f"  best: thr={op['threshold']} sharpe={op['sharpe_ratio']:.2f} "
                  f"wr={op.get('win_rate',0):.3f} n={op['n_trades']} skip={op['skip_rate']:.1%}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
