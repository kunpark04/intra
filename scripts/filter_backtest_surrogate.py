"""B4: Filter backtest on all 50 surrogate combos from ML#1.

Surrogate combos are novel parameter samples (no existing sweep runs), so we
build the combo dict directly from the CSV row and run the normal pipeline.
"""
import importlib.util, json, sys, traceback
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("fb", REPO / "scripts/filter_backtest.py")
fb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fb)

SURROGATE_CSV = REPO / "data/ml/lgbm_results/surrogate_top_combos.csv"
OUT = REPO / "data/ml/adaptive_rr_v2/filter_backtest_surrogate.json"

META_COLS = [
    "z_band_k", "z_window", "volume_zscore_window", "ema_fast", "ema_slow",
    "stop_method", "stop_fixed_pts", "atr_multiplier", "swing_lookback",
    "min_rr", "exit_on_opposite_signal", "use_breakeven_stop", "max_hold_bars",
    "zscore_confirmation", "z_input", "z_anchor", "z_denom", "z_type",
    "z_window_2", "z_window_2_weight", "volume_entry_threshold",
    "vol_regime_lookback", "vol_regime_min_pct", "vol_regime_max_pct",
    "session_filter_mode", "tod_exit_hour",
]


def row_to_combo(i: int, row: pd.Series) -> dict:
    combo = {"global_combo_id": f"surrogate_{i}", "source_version": -1, "combo_id": i}
    for c in META_COLS:
        if c in row.index:
            v = row[c]
            combo[c] = None if (isinstance(v, (float, np.floating)) and pd.isna(v)) else v
        else:
            combo[c] = None
    return combo


def backtest_one_surrogate(gcid: str, combo: dict, model) -> dict:
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
    print(f"Running filter on {len(surrogate)} surrogate combos")
    model = lgb.Booster(model_file=str(fb.V2_MODEL))
    all_results = {"model": "v2", "combos": []}
    for i, row in surrogate.iterrows():
        gcid = f"surrogate_{i}"
        print(f"\n[{i+1}/{len(surrogate)}] {gcid} min_rr={row['min_rr']:.2f}")
        combo = row_to_combo(i, row)
        try:
            r = backtest_one_surrogate(gcid, combo, model)
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
