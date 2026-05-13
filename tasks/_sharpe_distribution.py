"""Inspect Sharpe distribution across the full v12 combo universe.

User question: are there any combos with Sharpe > 1?
  - audit_full_net_sharpe: 1-contract net trade-Sharpe over the full train partition
  - target_median_wf_sharpe: median of 5 ordinal walk-forward window Sharpes
  - target_robust_sharpe: median - 0.5*std  (ML#1 v12 training target)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FEATURES = REPO / "data" / "ml" / "ml1_results_v12" / "combo_features_v12.parquet"

THRESHOLDS = [2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.0]
METRICS = [
    "audit_full_net_sharpe",
    "audit_full_gross_sharpe",
    "target_median_wf_sharpe",
    "target_robust_sharpe",
]


def main():
    df = pd.read_parquet(FEATURES)
    print(f"[load] {len(df):,} total combos")
    elig = df[df["audit_n_trades"] >= 500].copy()
    print(f"[gate] {len(elig):,} eligible (audit_n_trades >= 500)")
    print()

    for m in METRICS:
        if m not in elig.columns:
            continue
        s = elig[m].to_numpy()
        print(f"=== {m} ===")
        print(f"  n={len(s):,}  mean={s.mean():.3f}  median={np.median(s):.3f}  "
              f"std={s.std():.3f}")
        print(f"  min={s.min():.3f}  max={s.max():.3f}")
        print(f"  quantiles: p50={np.quantile(s,.50):.3f}  "
              f"p90={np.quantile(s,.90):.3f}  p99={np.quantile(s,.99):.3f}  "
              f"p99.9={np.quantile(s,.999):.3f}")
        print("  threshold counts (combos with Sharpe >= threshold):")
        for t in THRESHOLDS:
            n_above = int((s >= t).sum())
            pct = 100 * n_above / len(s)
            print(f"    >= {t:>5.2f} : {n_above:>6,} combos  ({pct:5.2f}%)")
        print()

    # Top 10 combos by audit_full_net_sharpe (production ranker)
    print("=== top 10 combos by audit_full_net_sharpe (production ranker) ===")
    cols = ["global_combo_id", "audit_n_trades", "audit_full_net_sharpe",
            "audit_full_gross_sharpe", "target_median_wf_sharpe",
            "target_robust_sharpe"]
    top10 = elig.nlargest(10, "audit_full_net_sharpe")[cols].reset_index(drop=True)
    print(top10.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()

    # Top 10 by target_median_wf_sharpe
    print("=== top 10 combos by target_median_wf_sharpe (temporal-robust) ===")
    top10w = elig.nlargest(10, "target_median_wf_sharpe")[cols].reset_index(drop=True)
    print(top10w.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()

    # Also check: any combo with gross_sharpe > 1 but net_sharpe < 1?
    # (friction-knocked-down candidates)
    bigg = elig[elig["audit_full_gross_sharpe"] >= 1.0]
    print(f"=== combos with audit_full_GROSS_sharpe >= 1.0: "
          f"{len(bigg):,} ({100*len(bigg)/len(elig):.2f}%) ===")
    if len(bigg) > 0:
        print(f"  of those, net_sharpe:")
        s = bigg["audit_full_net_sharpe"].to_numpy()
        print(f"    min={s.min():.3f}  median={np.median(s):.3f}  max={s.max():.3f}")
        print(f"    n with net_sharpe >= 1.0: {int((s>=1.0).sum())}")
        print(f"    n with net_sharpe >= 0.5: {int((s>=0.5).sum())}")
        # Show a few examples of biggest gross-to-net drops
        bigg = bigg.copy()
        bigg["friction_drop"] = bigg["audit_full_gross_sharpe"] - bigg["audit_full_net_sharpe"]
        print()
        print("  top 5 gross>=1 combos by net-sharpe:")
        show_cols = ["global_combo_id", "audit_n_trades", "audit_full_gross_sharpe",
                     "audit_full_net_sharpe", "friction_drop"]
        head = bigg.nlargest(5, "audit_full_net_sharpe")[show_cols].reset_index(drop=True)
        print(head.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
