"""Constrained-band adaptive R:R backtest: argmax E[R] restricted to [lo, hi]."""
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Import V3 module by path
spec = importlib.util.spec_from_file_location(
    "avf", REPO / "scripts/adaptive_vs_fixed_backtest_v1.py")
avf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(avf)

OUT = REPO / "data/ml/adaptive_rr_v1/constrained_band_backtest.json"


def pick_constrained(features, model, lo, hi):
    import pandas as pd
    n = len(features)
    K = len(avf.RR_LEVELS)
    expanded = features.loc[features.index.repeat(K)].reset_index(drop=True).copy()
    rr_col = np.tile(avf.RR_LEVELS, n).astype(np.float32)
    expanded["candidate_rr"] = rr_col
    expanded["rr_x_atr"] = rr_col * expanded["atr_points"].to_numpy()
    X = expanded[avf.FEATURE_ORDER]
    p = model.predict(X.values).reshape(n, K)
    ev = p * avf.RR_LEVELS[None, :] - (1.0 - p)
    mask = (avf.RR_LEVELS >= lo) & (avf.RR_LEVELS <= hi)
    ev_m = np.where(mask[None, :], ev, -np.inf)
    best_k = np.argmax(ev_m, axis=1)
    return avf.RR_LEVELS[best_k], ev[np.arange(n), best_k], p[np.arange(n), best_k]


def main():
    import lightgbm as lgb
    combo, _ = avf.load_top_combo()
    print(f"Combo: {combo['global_combo_id']}")
    df = avf.load_bars(avf.DATA_CSV)
    train, _ = avf.split_train_test(df, 0.8)
    df_ind = avf.build_indicators(train, combo)
    stop_pts = avf.resolve_stop_pts(combo, df_ind)
    cfg = avf.make_cfg(combo, stop_pts)
    df_sig = avf.generate_signals(df_ind, cfg)
    trades = avf.run_core(df_sig, cfg)
    features = avf.build_features(trades, df_sig, stop_pts,
                                  str(combo["stop_method"]),
                                  bool(combo["exit_on_opposite_signal"]))
    model = lgb.Booster(model_file=str(avf.MODEL_PATH))

    bands = [(1.5, 2.5), (1.5, 3.0)]
    results = {}
    # Reference: unconstrained V3 metrics
    with open(REPO / "data/ml/adaptive_rr_v1/adaptive_vs_fixed.json") as f:
        ref = json.load(f)
    results["unconstrained"] = ref.get("adaptive", ref)
    results["fixed"] = ref.get("fixed", {})

    for lo, hi in bands:
        rr, ev, p = pick_constrained(features, model, lo, hi)
        sim = avf.simulate(trades, df_sig, rr, cfg)
        metrics = avf.compute_equity_and_metrics(sim, trades, cfg, f"band_{lo}_{hi}")
        metrics["rr_distribution"] = {f"{k:.2f}": int(v) for k, v in
                                      zip(*np.unique(rr, return_counts=True))}
        metrics["mean_pwin"] = float(p.mean())
        metrics["mean_ev_R"] = float(ev.mean())
        results[f"band_{lo}_{hi}"] = metrics
        print(f"Band [{lo},{hi}]: ret={metrics.get('total_return_pct',0):.1f}%  "
              f"sharpe={metrics.get('sharpe',0):.2f}  "
              f"dd={metrics.get('max_dd_pct',0):.1f}%  "
              f"wr={metrics.get('win_rate',0):.3f}  "
              f"n={metrics.get('n_trades',0)}")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
