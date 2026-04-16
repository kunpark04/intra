"""Per-R:R isotonic recalibration of adaptive R:R OOF predictions."""
from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

REPO = Path(__file__).resolve().parents[1]
OOF_PATH = REPO / "data/ml/adaptive_rr_v1/oof_predictions.parquet"
OUT_DIR = REPO / "data/ml/adaptive_rr_v1"
ISO_DIR = OUT_DIR / "isotonic_per_rr"
ISO_DIR.mkdir(parents=True, exist_ok=True)

RR_LEVELS = np.round(np.arange(1.0, 5.25, 0.25), 2)
N_RR = len(RR_LEVELS)


def ece_equal_mass(p, y, n_bins=20):
    order = np.argsort(p)
    p_s, y_s = p[order], y[order]
    idx = np.array_split(np.arange(len(p)), n_bins)
    ece = 0.0
    mce = 0.0
    for ix in idx:
        if len(ix) == 0:
            continue
        gap = abs(p_s[ix].mean() - y_s[ix].mean())
        ece += (len(ix) / len(p)) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def main():
    df = pd.read_parquet(OOF_PATH)
    n = len(df)
    assert n % N_RR == 0, f"rows {n} not divisible by {N_RR}"
    p_raw = df["oof_pwin"].to_numpy()
    y = df["y_true"].to_numpy().astype(np.int8)
    rr = np.tile(RR_LEVELS, n // N_RR).astype(np.float32)

    p_cal = np.empty_like(p_raw)
    per_rr_ece = {}
    per_rr_ece_raw = {}
    for r in RR_LEVELS:
        mask = rr == np.float32(r)
        pr, yr = p_raw[mask], y[mask]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(pr, yr)
        pc = iso.predict(pr)
        p_cal[mask] = pc
        joblib.dump(iso, ISO_DIR / f"iso_rr_{r:.2f}.joblib")
        er, _ = ece_equal_mass(pr, yr)
        ec, _ = ece_equal_mass(pc, yr)
        per_rr_ece_raw[f"{r:.2f}"] = er
        per_rr_ece[f"{r:.2f}"] = ec

    ece_raw, mce_raw = ece_equal_mass(p_raw, y)
    ece_cal, mce_cal = ece_equal_mass(p_cal, y)

    out = pd.DataFrame({
        "oof_pwin_raw": p_raw, "oof_pwin_cal": p_cal,
        "y_true": y, "candidate_rr": rr,
    })
    out.to_parquet(OUT_DIR / "oof_recalibrated.parquet", index=False)

    # Optimal R:R per trade using calibrated P
    n_trades = n // N_RR
    p_cal_2d = p_cal.reshape(n_trades, N_RR)
    ev = p_cal_2d * RR_LEVELS[None, :] - (1.0 - p_cal_2d)
    best_k = np.argmax(ev, axis=1)
    best_rr = RR_LEVELS[best_k]
    best_ev = ev[np.arange(n_trades), best_k]
    best_p = p_cal_2d[np.arange(n_trades), best_k]

    opt = pd.DataFrame({"optimal_rr": best_rr.astype(np.float32),
                        "best_ev": best_ev.astype(np.float32),
                        "best_pwin": best_p.astype(np.float32)})
    opt.to_parquet(OUT_DIR / "optimal_rr_results_cal.parquet", index=False)

    # Comparison to raw
    opt_raw = pd.read_parquet(OUT_DIR / "optimal_rr_results.parquet")
    rr_dist_raw = opt_raw["optimal_rr"].value_counts().sort_index().to_dict()
    rr_dist_cal = pd.Series(best_rr).value_counts().sort_index().to_dict()

    summary = {
        "ece_raw": ece_raw, "ece_cal": ece_cal,
        "mce_raw": mce_raw, "mce_cal": mce_cal,
        "per_rr_ece_raw": per_rr_ece_raw,
        "per_rr_ece_cal": per_rr_ece,
        "mean_ev_raw": float(opt_raw["best_ev"].mean()),
        "mean_ev_cal": float(best_ev.mean()),
        "pct_positive_ev_raw": float((opt_raw["best_ev"] > 0).mean()),
        "pct_positive_ev_cal": float((best_ev > 0).mean()),
        "rr_dist_raw": {f"{k:.2f}": int(v) for k, v in rr_dist_raw.items()},
        "rr_dist_cal": {f"{k:.2f}": int(v) for k, v in rr_dist_cal.items()},
        "n_trades": int(n_trades),
    }
    with open(OUT_DIR / "recalibration_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"ECE: raw={ece_raw:.4f} -> cal={ece_cal:.4f}")
    print(f"MCE: raw={mce_raw:.4f} -> cal={mce_cal:.4f}")
    print(f"Mean E[R]: raw={summary['mean_ev_raw']:.4f} -> cal={summary['mean_ev_cal']:.4f}")
    print(f"% positive E[R]: raw={summary['pct_positive_ev_raw']:.3%} -> cal={summary['pct_positive_ev_cal']:.3%}")
    print(f"R:R distribution (raw): {summary['rr_dist_raw']}")
    print(f"R:R distribution (cal): {summary['rr_dist_cal']}")


if __name__ == "__main__":
    main()
