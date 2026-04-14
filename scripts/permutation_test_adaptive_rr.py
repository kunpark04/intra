"""
permutation_test_adaptive_rr.py — Verify adaptive R:R model's OOF AUC is not leakage.

Permutes candidate_rr (and the derived rr_x_atr) WITHIN each base trade's
17-row block. Labels are preserved so they reflect the ORIGINAL rr alignment,
meaning the features can no longer recover the label logic. If AUC collapses
to ~0.5, the model's signal was genuinely derived from candidate_rr and its
interactions; otherwise something else (e.g. a leak via atr_points or feature
correlations) is informative.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import adaptive_rr_model as arr  # type: ignore

import sys
# Ensure local import works regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))


def permute_rr_within_blocks(df: pd.DataFrame, n_rr: int,
                             seed: int = 1337) -> pd.DataFrame:
    """Permute candidate_rr within each contiguous 17-row block.

    Row ordering from expand_rr_levels is:
      trade0_rr0..trade0_rr16, trade1_rr0..trade1_rr16, ...

    For each block we draw a random permutation of length 17 and apply it to
    candidate_rr and rr_x_atr (which is rr * atr_points). The label
    `would_win` stays put — so in the permuted dataset, candidate_rr within
    a block no longer matches the row's label.
    """
    assert len(df) % n_rr == 0, "Row count must be divisible by n_rr"
    n_blocks = len(df) // n_rr
    rng = np.random.default_rng(seed)

    # Draw per-block permutations: shape (n_blocks, n_rr)
    perms = np.argsort(rng.random((n_blocks, n_rr)), axis=1).astype(np.int32)
    # Convert to flat indices: for block b, global index b*n_rr + perms[b]
    block_starts = (np.arange(n_blocks, dtype=np.int64) * n_rr)[:, None]
    flat_idx = (block_starts + perms).ravel()

    rr = df["candidate_rr"].to_numpy()
    df["candidate_rr"] = rr[flat_idx]

    # Recompute rr_x_atr with the permuted rr — atr_points is per-trade so
    # within a block it's constant; shuffling rr * atr_points is equivalent
    # to recomputing, but we do it explicitly in case of any drift.
    if "rr_x_atr" in df.columns and "atr_points" in df.columns:
        df["rr_x_atr"] = (df["candidate_rr"].to_numpy()
                          * df["atr_points"].to_numpy()).astype(np.float32)

    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--versions", type=int, nargs="+",
                   default=list(range(2, 11)))
    p.add_argument("--max-rows", type=int, default=2_000_000,
                   help="Cap expanded rows (default 2M for fast permutation test)")
    p.add_argument("--min-trades-per-combo", type=int, default=30)
    p.add_argument("--n-folds", type=int, default=5)
    args = p.parse_args()

    t_start = time.time()
    n_rr = len(arr.RR_LEVELS)
    target_base = args.max_rows // n_rr

    print(f"[permtest] Loading sweeps v{args.versions}, target base trades={target_base:,}")
    df = arr.load_mfe_parquets(args.versions, target_base_trades=target_base)
    df = arr.filter_combos(df, args.min_trades_per_combo)

    print("\n[permtest] Expanding R:R levels (produces correct labels first)...")
    expanded = arr.expand_rr_levels(df, arr.RR_LEVELS, args.max_rows)
    del df

    print(f"\n[permtest] Permuting candidate_rr within each {n_rr}-row block...")
    expanded = permute_rr_within_blocks(expanded, n_rr=n_rr, seed=1337)

    # Sanity check: within-block correlation of permuted rr with row position
    # should be near zero (was perfect 1.0 before permutation).
    pos_in_block = np.tile(np.arange(n_rr), len(expanded) // n_rr)
    corr = np.corrcoef(pos_in_block, expanded["candidate_rr"].to_numpy())[0, 1]
    print(f"  corr(row_pos_in_block, candidate_rr) = {corr:+.4f} (was +1.0 before)")

    # Per-rr-value label rate should now be uniform (the label no longer
    # depends on the permuted rr value — it depends on the true rr for the
    # row's position, which is shuffled).
    wr_by_permuted_rr = (expanded.groupby("candidate_rr", observed=True)
                         ["would_win"].mean())
    print("  Win rate by (permuted) candidate_rr — should be ~constant:")
    for rr_v, wr in wr_by_permuted_rr.items():
        print(f"    R:R {float(rr_v):.2f}: {wr:.1%}")

    print("\n[permtest] Training with permuted features...")
    result = arr.train_model(expanded, n_folds=args.n_folds)
    permuted_metrics = result["overall"]

    # Load original metrics
    orig_path = Path("data/ml/adaptive_rr/run_metadata.json")
    with open(orig_path) as f:
        orig = json.load(f)
    original = orig["overall_metrics"]

    passed = 0.48 <= permuted_metrics["auc"] <= 0.52

    out = {
        "original_auc": original["auc"],
        "permuted_auc": permuted_metrics["auc"],
        "original_brier": original["brier"],
        "permuted_brier": permuted_metrics["brier"],
        "original_log_loss": original["log_loss"],
        "permuted_log_loss": permuted_metrics["log_loss"],
        "pass": bool(passed),
        "n_rows_tested": int(len(expanded)),
        "n_folds": args.n_folds,
        "runtime_seconds": time.time() - t_start,
        "fold_metrics_permuted": result["fold_metrics"],
    }

    out_path = Path("data/ml/adaptive_rr/permutation_test.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)

    print("\n" + "=" * 60)
    print("PERMUTATION TEST RESULTS")
    print("=" * 60)
    print(f"  Original  AUC: {original['auc']:.4f}  Brier: {original['brier']:.4f}")
    print(f"  Permuted  AUC: {permuted_metrics['auc']:.4f}  Brier: {permuted_metrics['brier']:.4f}")
    print(f"  Pass (AUC in [0.48, 0.52]): {passed}")
    print(f"  Total runtime: {out['runtime_seconds']:.1f} s")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
