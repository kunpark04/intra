"""Phase 0 — Pool B ranker audit: basket-overlap baseline.

Ranks the eligible v12 combo pool by three combo-agnostic robustness signals
and reports Jaccard overlap + Spearman rank correlation.

Rankers (all already materialized in combo_features_v12.parquet — no learned
parameters involved):
  A = audit_full_net_sharpe         # production ranker
  B = target_median_wf_sharpe       # median across N=5 walk-forward windows
  C = target_robust_sharpe          # median - 0.5 * std (ML#1 v12 train target)

Eligibility: audit_n_trades >= 500 (same gate as production extractor).

Decision gate per task17_ranker_audit_plan.md:
  If Jaccard(A,B) >= 0.80 AND Jaccard(A,C) >= 0.80 -> baskets near-identical,
  skip Phases 1-2, Option A authorized on structural grounds.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FEATURES = REPO / "data" / "ml" / "ml1_results_v12" / "combo_features_v12.parquet"
OUT_JSON = REPO / "tasks" / "task17_phase0_overlap.json"

RANKERS = {
    "A_audit_full_net": "audit_full_net_sharpe",
    "B_median_wf":      "target_median_wf_sharpe",
    "C_robust":         "target_robust_sharpe",
}
MIN_TRADES = 500
TOP_K = 50
JACCARD_GATE = 0.80


def jaccard(s1: set, s2: set) -> float:
    if not s1 and not s2:
        return 1.0
    return len(s1 & s2) / len(s1 | s2)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ar = pd.Series(a).rank().to_numpy().astype(np.float64, copy=True)
    br = pd.Series(b).rank().to_numpy().astype(np.float64, copy=True)
    ar -= ar.mean()
    br -= br.mean()
    denom = np.sqrt((ar * ar).sum() * (br * br).sum())
    return float((ar * br).sum() / denom) if denom > 0 else 0.0


def main():
    if not FEATURES.exists():
        raise SystemExit(f"[ERR] features parquet not found: {FEATURES}")
    df = pd.read_parquet(FEATURES)
    print(f"[load] {FEATURES.relative_to(REPO)}  rows={len(df):,}", flush=True)

    elig = df[df["audit_n_trades"] >= MIN_TRADES].copy()
    print(f"[gate] audit_n_trades >= {MIN_TRADES}: "
          f"eligible={len(elig):,}  rejected={len(df) - len(elig):,}", flush=True)

    # Per-ranker top-K baskets (global_combo_id sets) + ranking vectors
    baskets: dict[str, set] = {}
    rankvecs: dict[str, np.ndarray] = {}
    stats: dict[str, dict] = {}
    for label, col in RANKERS.items():
        if col not in elig.columns:
            raise SystemExit(f"[ERR] missing column {col}")
        sorted_df = elig.sort_values(col, ascending=False).reset_index(drop=True)
        top = sorted_df.head(TOP_K)
        baskets[label] = set(top["global_combo_id"].tolist())
        rankvecs[label] = elig[col].to_numpy()
        stats[label] = {
            "col": col,
            "top_k_range": [float(top[col].min()), float(top[col].max())],
            "top_k_median_production_rank_sharpe": float(top["audit_full_net_sharpe"].median()),
            "top_k_median_audit_n_trades": float(top["audit_n_trades"].median()),
            "top_k_median_target_median_wf": float(top["target_median_wf_sharpe"].median()),
        }

    print()
    print("=== per-basket stats (top-50 by each ranker) ===")
    for label, s in stats.items():
        print(f"  {label} ({s['col']}):")
        print(f"    {s['col']} range in top-50 : "
              f"[{s['top_k_range'][0]:.3f} .. {s['top_k_range'][1]:.3f}]")
        print(f"    median audit_full_net_sharpe: {s['top_k_median_production_rank_sharpe']:.3f}")
        print(f"    median audit_n_trades       : {s['top_k_median_audit_n_trades']:.0f}")
        print(f"    median target_median_wf     : {s['top_k_median_target_median_wf']:.3f}")
    print()

    # Jaccard overlap matrix
    labels = list(baskets.keys())
    print("=== Jaccard overlap matrix (top-50 baskets) ===")
    header = "               " + "".join(f"{l:>14}" for l in labels)
    print(header)
    jaccard_matrix: dict[str, dict[str, float]] = {}
    for r in labels:
        jaccard_matrix[r] = {}
        row = f"  {r:<13}"
        for c in labels:
            j = jaccard(baskets[r], baskets[c])
            jaccard_matrix[r][c] = j
            row += f"{j:>14.3f}"
        print(row)
    print()

    # Spearman rank correlations (full eligible pool)
    print("=== Spearman rank correlation on full eligible pool ===")
    spearman_matrix: dict[str, dict[str, float]] = {}
    for r in labels:
        spearman_matrix[r] = {}
        row = f"  {r:<13}"
        for c in labels:
            s = spearman(rankvecs[r], rankvecs[c])
            spearman_matrix[r][c] = s
            row += f"{s:>14.3f}"
        print(row)
    print()

    # Decision gate
    ab = jaccard_matrix["A_audit_full_net"]["B_median_wf"]
    ac = jaccard_matrix["A_audit_full_net"]["C_robust"]
    gate_passed = (ab >= JACCARD_GATE) and (ac >= JACCARD_GATE)
    print("=== decision gate (task17_ranker_audit_plan.md) ===")
    print(f"  Jaccard(A,B) = {ab:.3f}  (gate >= {JACCARD_GATE})")
    print(f"  Jaccard(A,C) = {ac:.3f}  (gate >= {JACCARD_GATE})")
    print(f"  gate {'PASSED' if gate_passed else 'FAILED'} "
          f"-> {'skip Phases 1-2, Option A authorized' if gate_passed else 'proceed to Phase 1 (alternate basket s3-s6)'}")
    print()

    # Which combos are in A but not in B (these are the "selection-only" combos)
    a_only = sorted(baskets["A_audit_full_net"] - baskets["B_median_wf"])
    b_only = sorted(baskets["B_median_wf"] - baskets["A_audit_full_net"])
    print(f"=== symmetric difference A_audit_full_net vs B_median_wf ===")
    print(f"  in A only ({len(a_only)}): {a_only[:10]}{'...' if len(a_only)>10 else ''}")
    print(f"  in B only ({len(b_only)}): {b_only[:10]}{'...' if len(b_only)>10 else ''}")
    print()

    # Persist
    out = {
        "features_path": str(FEATURES.relative_to(REPO)),
        "min_trades": MIN_TRADES,
        "top_k": TOP_K,
        "jaccard_gate": JACCARD_GATE,
        "pool_sizes": {
            "total_combos": int(len(df)),
            "eligible_combos": int(len(elig)),
        },
        "per_basket_stats": stats,
        "jaccard_matrix": jaccard_matrix,
        "spearman_matrix": spearman_matrix,
        "gate": {
            "jaccard_A_B": ab,
            "jaccard_A_C": ac,
            "passed": bool(gate_passed),
            "next_action": ("skip Phases 1-2, authorize Option A"
                            if gate_passed else "proceed to Phase 1"),
        },
        "baskets": {
            label: sorted(list(ids)) for label, ids in baskets.items()
        },
        "symmetric_difference_A_vs_B": {
            "a_only": a_only,
            "b_only": b_only,
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, default=float))
    print(f"[persist] {OUT_JSON.relative_to(REPO)}")


if __name__ == "__main__":
    main()
