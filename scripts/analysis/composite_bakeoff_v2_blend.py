"""Weighted-blend composite bakeoff — extends composite_bakeoff_v1.

Takes the 9 candidate composites from v1, fits weighted linear blends that
maximise predictive power on held-out combos, and reports whether the
blend actually beats C1 alone.

Methodology (defends against overfitting):
1. Rebuild per-combo early/late aggregates from sweep parquets (same as v1).
2. Z-score each candidate composite and each late truth across eligible combos.
3. Build a meta-target = mean of z-scored late truths (equal weight across
   pnl/sharpe/avg_r/pf/wr).
4. Split combos into train/test halves (random seeded).
5. Fit ridge regression of meta-target ~ z-composites on train half.
6. Measure Spearman rho of the blended composite vs each late truth on test half.
7. Compare head-to-head to C1 (baseline single-composite) on the same test half.

Output: `data/ml/composite_bakeoff_v2_blend.json`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

REPO = Path(__file__).resolve().parent.parent.parent
ORIG_DIR = REPO / "data" / "ml" / "originals"
OUT_PATH = REPO / "data" / "ml" / "composite_bakeoff_v2_blend.json"
SWEEP_VERSIONS = [2, 3, 4, 5, 6, 7, 8, 9, 10]
RISK_DOLLARS = 500.0
EARLY_FRACTION = 0.80
MIN_EARLY = 80
MIN_LATE = 20
RNG_SEED = 42


def aggregate_slice(r_mult: np.ndarray, label_win: np.ndarray) -> dict:
    n = len(r_mult)
    if n == 0:
        return {"n": 0, "wr": np.nan, "avg_r": np.nan, "sharpe": np.nan,
                "pf": np.nan, "total_pnl": 0.0, "max_dd": 0.0}
    pnl = RISK_DOLLARS * r_mult
    avg_r = float(r_mult.mean())
    std_r = float(r_mult.std(ddof=1)) if n > 1 else 0.0
    sharpe = avg_r / std_r if std_r > 0 else 0.0
    wins_pnl = float(pnl[r_mult > 0].sum())
    loss_pnl = float(-pnl[r_mult < 0].sum())
    pf = wins_pnl / loss_pnl if loss_pnl > 0 else np.inf
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    max_dd = float((peak - equity).max())
    return {"n": int(n), "wr": float(label_win.mean()), "avg_r": avg_r,
            "sharpe": sharpe, "pf": pf, "total_pnl": float(pnl.sum()),
            "max_dd": max_dd}


def split_combo(r_mult: np.ndarray, label_win: np.ndarray) -> dict:
    n = len(r_mult)
    s = int(EARLY_FRACTION * n)
    e = aggregate_slice(r_mult[:s], label_win[:s])
    l = aggregate_slice(r_mult[s:], label_win[s:])
    return {"early_n": e["n"], "early_wr": e["wr"], "early_avg_r": e["avg_r"],
            "early_sharpe": e["sharpe"], "early_pf": e["pf"],
            "early_total_pnl": e["total_pnl"], "early_max_dd": e["max_dd"],
            "late_n": l["n"], "late_wr": l["wr"], "late_avg_r": l["avg_r"],
            "late_sharpe": l["sharpe"], "late_pf": l["pf"],
            "late_total_pnl": l["total_pnl"], "late_max_dd": l["max_dd"]}


def process_version(version: int) -> pd.DataFrame:
    path = ORIG_DIR / f"ml_dataset_v{version}.parquet"
    print(f"[v{version}] reading {path.name}")
    tbl = pq.read_table(path, columns=["combo_id", "r_multiple", "label_win"])
    df = tbl.to_pandas()
    del tbl
    r = df["r_multiple"].to_numpy(np.float64)
    w = df["label_win"].to_numpy(np.int8)
    cids = df["combo_id"].to_numpy()
    del df
    change = np.flatnonzero(np.diff(cids)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(r)]])
    rows = []
    for s, e in zip(starts, ends):
        rec = split_combo(r[s:e], w[s:e])
        rec["global_combo_id"] = f"v{version}_{cids[s]}"
        rows.append(rec)
    return pd.DataFrame(rows)


def build_composites(elig: pd.DataFrame) -> list[str]:
    e_n = elig["early_n"].to_numpy(np.float64)
    e_r = elig["early_avg_r"].to_numpy(np.float64)
    e_s = elig["early_sharpe"].to_numpy(np.float64)
    e_w = elig["early_wr"].to_numpy(np.float64)
    e_pf = np.clip(elig["early_pf"].replace([np.inf, -np.inf], np.nan).to_numpy(np.float64), 0, 10)
    e_pnl = elig["early_total_pnl"].to_numpy(np.float64)
    e_dd = elig["early_max_dd"].to_numpy(np.float64)
    elig["C1_fixed_sharpe_x_logn"] = e_s * np.log1p(e_n)
    elig["C2_avg_r_x_sqrtn"] = e_r * np.sqrt(e_n)
    elig["C3_avg_r_x_n"] = e_r * e_n
    elig["C4_pf_clip_x_logn"] = e_pf * np.log1p(e_n)
    elig["C5_fixed_sharpe"] = e_s
    elig["C6_avg_r"] = e_r
    elig["C7_wr_edge_x_sqrtn"] = (e_w - 0.5) * np.sqrt(e_n)
    elig["C8_pnl_minus_dd"] = e_pnl - e_dd
    elig["C9_sortino_proxy"] = e_r / np.where(np.sqrt(e_dd) > 0, np.sqrt(e_dd), np.nan)
    return [c for c in elig.columns if c.startswith("C")]


def zscore(a: np.ndarray) -> np.ndarray:
    mu = np.nanmean(a); sd = np.nanstd(a, ddof=1)
    return (a - mu) / sd if sd > 0 else np.zeros_like(a)


def main() -> None:
    import json
    frames = [process_version(v) for v in SWEEP_VERSIONS]
    agg = pd.concat(frames, ignore_index=True)
    print(f"\nTotal combos: {len(agg):,}")

    elig = agg[(agg["early_n"] >= MIN_EARLY) & (agg["late_n"] >= MIN_LATE)].copy()
    print(f"Eligible: {len(elig):,}")

    composites = build_composites(elig)
    truths = ["late_total_pnl", "late_sharpe", "late_avg_r", "late_pf", "late_wr"]

    for c in composites + truths:
        elig[c] = elig[c].replace([np.inf, -np.inf], np.nan)

    # Drop combos with any NaN in composites/truths
    keep = elig[composites + truths].notna().all(axis=1)
    clean = elig.loc[keep].copy()
    print(f"After dropping NaN rows: {len(clean):,}")

    # Z-score composites + truths
    Z_comp = np.column_stack([zscore(clean[c].to_numpy(np.float64)) for c in composites])
    Z_truth = np.column_stack([zscore(clean[t].to_numpy(np.float64)) for t in truths])
    meta_target = Z_truth.mean(axis=1)  # equal-weight z-scored combo of all truths

    # Train/test split (by combo)
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.permutation(len(clean))
    half = len(idx) // 2
    train, test = idx[:half], idx[half:]

    # Fit ridge: meta_target ~ Z_composites
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
    ridge.fit(Z_comp[train], meta_target[train])
    weights = dict(zip(composites, ridge.coef_.round(4)))
    print(f"\nRidge weights (alpha={ridge.alpha_}):")
    for c, w in sorted(weights.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {c:24s}  {w:+.4f}")

    # Build blended composite on TEST half
    blend_test = Z_comp[test] @ ridge.coef_

    # Spearman of blend vs each truth on test half
    test_df = clean.iloc[test]
    rho_rows = {}
    for i, t in enumerate(truths):
        y = test_df[t].to_numpy(np.float64)
        m = ~np.isnan(y) & ~np.isnan(blend_test)
        rho, _ = spearmanr(blend_test[m], y[m])
        rho_rows[t] = round(float(rho), 4)
    rho_rows["mean_signed"] = round(float(np.mean(list(rho_rows.values()))), 4)

    # Baseline: C1 alone on same test half
    c1_test = Z_comp[test, composites.index("C1_fixed_sharpe_x_logn")]
    c1_rows = {}
    for t in truths:
        y = test_df[t].to_numpy(np.float64)
        m = ~np.isnan(y) & ~np.isnan(c1_test)
        rho, _ = spearmanr(c1_test[m], y[m])
        c1_rows[t] = round(float(rho), 4)
    c1_rows["mean_signed"] = round(float(np.mean(list(c1_rows.values()))), 4)

    # Also C5 (raw fixed_sharpe) — second best single
    c5_test = Z_comp[test, composites.index("C5_fixed_sharpe")]
    c5_rows = {}
    for t in truths:
        y = test_df[t].to_numpy(np.float64)
        m = ~np.isnan(y) & ~np.isnan(c5_test)
        rho, _ = spearmanr(c5_test[m], y[m])
        c5_rows[t] = round(float(rho), 4)
    c5_rows["mean_signed"] = round(float(np.mean(list(c5_rows.values()))), 4)

    print("\n=== Held-out Spearman rho (test half, n={:,}) ===".format(len(test)))
    cmp_df = pd.DataFrame({
        "Ridge-blend (9 composites)": rho_rows,
        "C1 (fixed_sharpe × log1p n)": c1_rows,
        "C5 (raw fixed_sharpe)": c5_rows,
    })
    print(cmp_df.round(3).to_string())

    # Delta vs baselines
    delta_c1 = {t: round(rho_rows[t] - c1_rows[t], 4) for t in rho_rows}
    delta_c5 = {t: round(rho_rows[t] - c5_rows[t], 4) for t in rho_rows}
    print("\nBlend - C1:", delta_c1)
    print("Blend - C5:", delta_c5)

    payload = {
        "method": "9-composite ridge blend; train/test split by combo (50/50); "
                  "meta-target = equal-weight z-scored mean of 5 late truths",
        "n_eligible": int(len(clean)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "ridge_alpha": float(ridge.alpha_),
        "weights": weights,
        "test_rho_blend": rho_rows,
        "test_rho_C1": c1_rows,
        "test_rho_C5": c5_rows,
        "delta_blend_minus_C1": delta_c1,
        "delta_blend_minus_C5": delta_c5,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {OUT_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
