"""Calibration audit for adaptive R:R OOF predictions.

Produces equal-mass binned reliability table, ECE, MCE, and per-R:R ECE.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/root/intra")
OOF_PATH = ROOT / "data/ml/adaptive_rr_v1/oof_predictions.parquet"
OUT_JSON = ROOT / "data/ml/adaptive_rr_v1/calibration_audit.json"
OUT_PNG = ROOT / "data/ml/adaptive_rr_v1/calibration_by_rr.png"

N_BINS = 20
RR_LEVELS = np.arange(1.0, 5.25, 0.25)  # 17 levels
N_RR = len(RR_LEVELS)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Args:
        k: Number of successes.
        n: Number of trials.
        z: Standard-normal quantile (default 1.96 → 95% CI).

    Returns:
        `(lo, hi)` tuple clipped to `[0, 1]`, or `(nan, nan)` if `n == 0`.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    halfw = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - halfw), min(1.0, centre + halfw))


def equal_mass_bins(pred: np.ndarray, y: np.ndarray, n_bins: int = N_BINS):
    """Equal-mass reliability diagram with per-bin Wilson CIs + ECE/MCE.

    Partitions `pred` into `n_bins` quantile-mass buckets, then reports
    per-bin predicted mean, observed rate, count, Wilson 95% CI, and gap.
    Also returns the expected (ECE) and max (MCE) calibration error.

    Args:
        pred: Predicted probabilities in `[0, 1]`.
        y: Binary outcomes aligned to `pred`.
        n_bins: Number of quantile bins (default `N_BINS`).

    Returns:
        `(rows, ece, mce)` where `rows` is a list of per-bin dicts.
    """
    # Quantile bin edges; unique() to handle ties.
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(pred, qs))
    # Assign bin index; np.digitize treats last edge as open, clip into range.
    idx = np.clip(np.digitize(pred, edges[1:-1], right=False), 0, len(edges) - 2)
    rows = []
    total = len(pred)
    ece = 0.0
    mce = 0.0
    for b in range(len(edges) - 1):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        pm = float(pred[mask].mean())
        om = float(y[mask].mean())
        k = int(y[mask].sum())
        lo, hi = wilson_ci(k, n)
        gap = abs(pm - om)
        ece += (n / total) * gap
        mce = max(mce, gap)
        rows.append({
            "bin": b,
            "edge_lo": float(edges[b]),
            "edge_hi": float(edges[b + 1]),
            "count": n,
            "pred_mean": pm,
            "obs_rate": om,
            "wilson_lo": lo,
            "wilson_hi": hi,
            "gap": gap,
        })
    return rows, ece, mce


def compute_ece(pred: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> float:
    """Equal-mass Expected Calibration Error.

    Args:
        pred: Predicted probabilities.
        y: Binary outcomes aligned to `pred`.
        n_bins: Target number of quantile bins.

    Returns:
        Count-weighted mean absolute gap between predicted and observed
        per bin, or NaN when fewer than `n_bins` samples are provided.
    """
    if len(pred) < n_bins:
        return float("nan")
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(pred, qs))
    idx = np.clip(np.digitize(pred, edges[1:-1], right=False), 0, len(edges) - 2)
    total = len(pred)
    ece = 0.0
    for b in range(len(edges) - 1):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        ece += (n / total) * abs(float(pred[mask].mean()) - float(y[mask].mean()))
    return ece


def main() -> None:
    """Run the V1 calibration audit and write a JSON report.

    Loads OOF predictions from `OOF_PATH`, reconstructs per-R:R indexing,
    computes global equal-mass ECE/MCE, per-R:R ECE, and writes the
    aggregated result to `OUT_PATH`.
    """
    t0 = time.time()
    print(f"[load] {OOF_PATH}")
    df = pd.read_parquet(OOF_PATH, columns=["oof_pwin", "y_true"])
    n = len(df)
    print(f"[load] rows={n:,}  trades={n // N_RR:,}")
    if n % N_RR != 0:
        print(f"[warn] row count {n} not divisible by {N_RR}; per-rr mapping may be off")

    pred = df["oof_pwin"].to_numpy(dtype=np.float64)
    y = df["y_true"].to_numpy(dtype=np.int8)

    # Reconstruct candidate_rr by row-ordering convention.
    rr_idx = np.arange(n) % N_RR
    candidate_rr = RR_LEVELS[rr_idx]

    # Global reliability (equal-mass)
    print(f"[calib] computing global equal-mass reliability ({N_BINS} bins)")
    rows, ece, mce = equal_mass_bins(pred, y, N_BINS)
    print(f"[calib] ECE={ece:.4f}  MCE={mce:.4f}")

    # Per-R:R ECE
    per_rr = []
    for i, rr in enumerate(RR_LEVELS):
        mask = rr_idx == i
        e = compute_ece(pred[mask], y[mask], N_BINS)
        per_rr.append({
            "rr": float(rr),
            "count": int(mask.sum()),
            "pred_mean": float(pred[mask].mean()),
            "obs_rate": float(y[mask].mean()),
            "ece": float(e),
        })
    per_rr_sorted = sorted(per_rr, key=lambda r: -r["ece"])
    print("[calib] worst per-rr ECE:")
    for r in per_rr_sorted[:5]:
        print(f"   rr={r['rr']:.2f}  ECE={r['ece']:.4f}  pred={r['pred_mean']:.3f}  obs={r['obs_rate']:.3f}")

    # Verdict
    if ece < 0.02:
        verdict = "good"
    elif ece < 0.05:
        verdict = "acceptable"
    else:
        verdict = "concerning"

    out = {
        "n_rows": int(n),
        "n_trades": int(n // N_RR),
        "n_bins": N_BINS,
        "binning": "equal_mass_quantile",
        "ece": float(ece),
        "mce": float(mce),
        "verdict": verdict,
        "reliability_table": rows,
        "per_rr": per_rr,
        "runtime_seconds": None,  # filled below
    }

    # Plot: calibration curves grouped low/mid/high R:R plus global
    print(f"[plot] writing {OUT_PNG}")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: global reliability
    ax = axes[0]
    pm_arr = np.array([r["pred_mean"] for r in rows])
    om_arr = np.array([r["obs_rate"] for r in rows])
    lo_arr = np.array([r["wilson_lo"] for r in rows])
    hi_arr = np.array([r["wilson_hi"] for r in rows])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    ax.errorbar(pm_arr, om_arr, yerr=[om_arr - lo_arr, hi_arr - om_arr],
                fmt="o-", ms=4, capsize=2, label=f"OOF (ECE={ece:.3f})")
    ax.set_xlabel("Predicted P(win)")
    ax.set_ylabel("Observed win rate")
    ax.set_title(f"Global reliability (equal-mass, {N_BINS} bins)\nECE={ece:.4f}  MCE={mce:.4f}")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Right: per-rr-group curves
    ax = axes[1]
    groups = {
        "low (1.00-2.00)": (RR_LEVELS >= 1.0) & (RR_LEVELS <= 2.0),
        "mid (2.25-3.50)": (RR_LEVELS >= 2.25) & (RR_LEVELS <= 3.5),
        "high (3.75-5.00)": (RR_LEVELS >= 3.75) & (RR_LEVELS <= 5.0),
    }
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    for gname, gmask in groups.items():
        grr_idx_set = np.where(gmask)[0]
        row_mask = np.isin(rr_idx, grr_idx_set)
        gp = pred[row_mask]
        gy = y[row_mask]
        grows, gece, _ = equal_mass_bins(gp, gy, N_BINS)
        gpm = np.array([r["pred_mean"] for r in grows])
        gom = np.array([r["obs_rate"] for r in grows])
        ax.plot(gpm, gom, "o-", ms=4, label=f"{gname}  ECE={gece:.3f}  (n={row_mask.sum():,})")
    ax.set_xlabel("Predicted P(win)")
    ax.set_ylabel("Observed win rate")
    ax.set_title("Calibration by R:R group")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120)
    plt.close(fig)

    runtime = time.time() - t0
    out["runtime_seconds"] = runtime

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[write] {OUT_JSON}")
    print(f"[done] ECE={ece:.4f}  MCE={mce:.4f}  verdict={verdict}  runtime={runtime:.1f}s")


if __name__ == "__main__":
    main()
