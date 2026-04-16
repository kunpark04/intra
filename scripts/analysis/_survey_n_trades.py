"""One-shot distribution survey for n_trades across all sweep combos.

Prints summary stats, percentiles, and a log-spaced histogram to stdout.
Used to choose principled high/low frequency thresholds for the
top-combo extractor. Not part of the production pipeline.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PARQUET = Path("/root/intra/data/ml/lgbm_results_v2filtered/combo_features.parquet")


def main() -> None:
    """Print n_trades summary stats, key percentiles, log-spaced histogram."""
    df = pd.read_parquet(PARQUET)
    n = df["n_trades"]
    print(f"N combos: {len(n):,}")
    print(f"min: {n.min()}, max: {n.max()}")
    print(f"mean: {n.mean():.1f}, median: {n.median():.1f}, std: {n.std():.1f}")
    print()
    print("Percentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{p:02d}: {int(np.percentile(n, p)):>7,}")
    print()
    print("Log-spaced histogram:")
    edges = [0, 50, 100, 200, 400, 800, 1600, 3200,
             6400, 12800, 25600, 51200, 10_000_000]
    hist = pd.cut(n, edges, right=False).value_counts().sort_index()
    for k, v in hist.items():
        pct = 100 * v / len(n)
        print(f"  [{int(k.left):>7,}-{int(k.right):>8,}):"
              f" {v:>6,} combos ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
