"""B4 V3 variant: 50-surrogate-combo filter backtest using V3 booster + Family A."""
import importlib.util, json, sys, traceback
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("fb", REPO / "scripts/filter_backtest.py")
fb = importlib.util.module_from_spec(spec); spec.loader.exec_module(fb)

spec2 = importlib.util.spec_from_file_location("fbs", REPO / "scripts/filter_backtest_surrogate.py")
fbs = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(fbs)

spec3 = importlib.util.spec_from_file_location("v3inf", REPO / "scripts/v3_inference.py")
v3inf = importlib.util.module_from_spec(spec3); spec3.loader.exec_module(v3inf)

SURROGATE_CSV = REPO / "data/ml/lgbm_results/surrogate_top_combos.csv"
OUT = REPO / "data/ml/adaptive_rr_v3/filter_backtest_surrogate_v3.json"


def backtest_one_surrogate_v3(gcid: str, combo: dict, booster, cals) -> dict:
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
    pwin = v3inf.predict_pwin_v3_at_rr(feats, trades, gcid, stop_pts, rr,
                                       booster=booster, calibrators=cals)
    ev = pwin * rr - (1.0 - pwin)

    fixed_res = avf.simulate_fixed(trades)
    fixed_m = avf.compute_equity_and_metrics(fixed_res, trades, cfg, "fixed")

    results = {"global_combo_id": gcid, "min_rr": rr, "n_signals": n,
               "mean_pwin": float(pwin.mean()), "mean_ev_R": float(ev.mean()),
               "fixed": fixed_m}

    for thr in fb.EV_THRESHOLDS:
        keep = ev >= thr
        n_kept = int(keep.sum())
        if n_kept == 0:
            results[f"filter_ev_ge_{thr}"] = {"n_trades": 0, "skipped": "no trades pass filter"}
            continue
        subset = {k: (np.asarray(v)[keep] if hasattr(v, "__len__") and len(v) == n else v)
                  for k, v in trades.items()}
        fres = avf.simulate_fixed(subset)
        fm = avf.compute_equity_and_metrics(fres, subset, cfg, f"filter_{thr}")
        fm["skip_rate"] = float(1 - keep.mean())
        results[f"filter_ev_ge_{thr}"] = fm
    return results


def main():
    import lightgbm as lgb
    surrogate = pd.read_csv(SURROGATE_CSV)
    print(f"Running V3 filter on {len(surrogate)} surrogate combos")
    booster = lgb.Booster(model_file=str(v3inf.V3_BOOSTER))
    cals = v3inf._load_calibrators()
    all_results = {"model": "v3", "combos": []}
    for i, row in surrogate.iterrows():
        gcid = f"surrogate_{i}"
        print(f"\n[{i+1}/{len(surrogate)}] {gcid} min_rr={row['min_rr']:.2f}")
        combo = fbs.row_to_combo(i, row)
        try:
            r = backtest_one_surrogate_v3(gcid, combo, booster, cals)
        except Exception as e:
            r = {"global_combo_id": gcid, "error": str(e), "tb": traceback.format_exc()[:500]}
            print(f"  ERROR: {e}")
        all_results["combos"].append(r)
        if "error" not in r:
            fx = r["fixed"]
            print(f"  fixed: ret={fx.get('total_return_pct',0):.1f}% sharpe={fx.get('sharpe_ratio',0):.2f} n={fx.get('n_trades',0)}")
            for thr in fb.EV_THRESHOLDS:
                f = r[f"filter_ev_ge_{thr}"]
                if "skipped" in f:
                    print(f"  thr={thr}: {f['skipped']}")
                else:
                    print(f"  thr={thr}: ret={f.get('total_return_pct',0):.1f}% sharpe={f.get('sharpe_ratio',0):.2f} skip={f.get('skip_rate',0):.1%}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
