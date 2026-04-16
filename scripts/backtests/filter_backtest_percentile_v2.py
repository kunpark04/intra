"""B2: Adaptive filter — percentile of E[R] per combo rather than fixed threshold.

For each combo, keep top-X% of trades by E[R]. Tests percentiles [25, 50, 75, 90].
This adapts the filter to each combo's E[R] distribution, so low-freq combos
don't get over-filtered.
"""
import importlib.util, json, sys, traceback
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("fb", REPO / "scripts/backtests/filter_backtest_v2.py")
fb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fb)

spec2 = importlib.util.spec_from_file_location("fbs", REPO / "scripts/backtests/filter_backtest_surrogate_v2.py")
fbs = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(fbs)

OUT = REPO / "data/ml/adaptive_rr_v2/filter_backtest_percentile_v2.json"
SURROGATE_CSV = REPO / "data/ml/ml1_results/surrogate_top_combos.csv"

PERCENTILES = [25, 50, 75, 90]  # keep top-X% by E[R]


def backtest_one_pct(gcid: str, combo: dict, model) -> dict:
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

    results = {"global_combo_id": gcid, "min_rr": rr, "n_signals": n,
               "fixed": fixed_m}

    for pct in PERCENTILES:
        cutoff = np.percentile(ev, 100 - pct)
        keep = ev >= cutoff
        n_kept = int(keep.sum())
        if n_kept == 0:
            results[f"top_{pct}pct"] = {"n_trades": 0, "skipped": "empty"}
            continue
        subset = {k: (np.asarray(v)[keep] if hasattr(v, "__len__") and len(v) == n else v)
                  for k, v in trades.items()}
        fres = avf.simulate_fixed(subset)
        fm = avf.compute_equity_and_metrics(fres, subset, cfg, f"top_{pct}")
        fm["ev_cutoff"] = float(cutoff)
        fm["skip_rate"] = float(1 - keep.mean())
        results[f"top_{pct}pct"] = fm
    return results


def main():
    import lightgbm as lgb
    high_freq = ["v10_7649", "v10_8617", "v10_9264", "v10_9393", "v6_1676"]
    low_freq = ["v10_9955", "v5_158", "v5_2904", "v7_2114", "v7_215"]
    existing = high_freq + low_freq

    model = lgb.Booster(model_file=str(fb.V2_MODEL))
    all_results = {"model": "v2", "percentiles": PERCENTILES, "combos": []}

    # Existing combos
    for gcid in existing:
        print(f"\n=== {gcid} ===")
        try:
            combo = fb.load_combo_by_id(gcid)
            r = backtest_one_pct(gcid, combo, model)
        except Exception as e:
            r = {"global_combo_id": gcid, "error": str(e), "tb": traceback.format_exc()[:400]}
            print(f"  ERROR: {e}")
        all_results["combos"].append(r)
        if "error" not in r:
            fx = r["fixed"]
            print(f"  fixed: ret={fx.get('total_return_pct',0):.1f}% sharpe={fx.get('sharpe_ratio',0):.2f} n={fx.get('n_trades',0)}")
            for pct in PERCENTILES:
                f = r[f"top_{pct}pct"]
                if "skipped" in f: continue
                print(f"  top_{pct}%: ret={f.get('total_return_pct',0):.1f}% sharpe={f.get('sharpe_ratio',0):.2f} skip={f.get('skip_rate',0):.1%}")

    # Surrogate combos
    surrogate = pd.read_csv(SURROGATE_CSV)
    for i, row in surrogate.iterrows():
        gcid = f"surrogate_{i}"
        print(f"\n=== {gcid} ===")
        combo = fbs.row_to_combo(i, row)
        try:
            r = backtest_one_pct(gcid, combo, model)
        except Exception as e:
            r = {"global_combo_id": gcid, "error": str(e), "tb": traceback.format_exc()[:400]}
            print(f"  ERROR: {e}")
        all_results["combos"].append(r)
        if "error" not in r:
            fx = r["fixed"]
            print(f"  fixed sharpe={fx.get('sharpe_ratio',0):.2f} n={fx.get('n_trades',0)}")
            for pct in PERCENTILES:
                f = r[f"top_{pct}pct"]
                if "skipped" in f: continue
                print(f"  top_{pct}%: sharpe={f.get('sharpe_ratio',0):.2f} skip={f.get('skip_rate',0):.1%}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
