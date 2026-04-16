"""Build per-combo r_multiple statistics across sweep parquets v2–v10.

For each combo, streams the sweep parquet row group by row group and
accumulates a single-pass mean / variance using Welford's algorithm
(memory constant per combo rather than O(n_trades)).

Output: data/ml/full_combo_r_stats.parquet with columns
    global_combo_id, n_trades, mean_r, std_r, fixed_sharpe

`fixed_sharpe = mean_r / std_r` — sizing-invariant (no dollars, no
compounding). Downstream ML#1 retrain uses this as the C1 target input
(C1 = fixed_sharpe * log1p(n_trades)).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent.parent
ORIG_DIR = REPO / "data" / "ml" / "originals"
OUT_PATH = REPO / "data" / "ml" / "full_combo_r_stats.parquet"
SWEEP_VERSIONS = [2, 3, 4, 5, 6, 7, 8, 9, 10]


class WelfordAcc:
    __slots__ = ("n", "mean", "m2")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update_batch(self, x: np.ndarray) -> None:
        # Vectorised Welford merge of existing running stats with a new batch.
        k = x.size
        if k == 0:
            return
        batch_mean = float(x.mean())
        batch_m2 = float(((x - batch_mean) ** 2).sum())
        if self.n == 0:
            self.n = k
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        delta = batch_mean - self.mean
        total = self.n + k
        self.mean = self.mean + delta * k / total
        self.m2 = self.m2 + batch_m2 + (delta ** 2) * self.n * k / total
        self.n = total

    def finalize(self) -> tuple[int, float, float]:
        if self.n == 0:
            return 0, 0.0, 0.0
        if self.n == 1:
            return self.n, self.mean, 0.0
        var = self.m2 / (self.n - 1)
        return self.n, self.mean, float(np.sqrt(var))


def process_version(version: int, accs: dict[str, WelfordAcc]) -> None:
    path = ORIG_DIR / f"ml_dataset_v{version}.parquet"
    if not path.exists():
        print(f"[v{version}] MISSING: {path}")
        return
    pf = pq.ParquetFile(str(path))
    print(f"[v{version}] {pf.metadata.num_rows:,} rows, "
          f"{pf.metadata.num_row_groups} row groups")
    for rg in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["combo_id", "r_multiple"])
        df = tbl.to_pandas()
        del tbl
        # Accumulate per-combo via groupby (vectorised within the row group)
        for cid, grp in df.groupby("combo_id", sort=False):
            gid = f"v{version}_{int(cid)}"
            acc = accs.get(gid)
            if acc is None:
                acc = WelfordAcc()
                accs[gid] = acc
            acc.update_batch(grp["r_multiple"].to_numpy(np.float64))
        del df
        if rg % 10 == 0:
            print(f"  rg {rg+1}/{pf.metadata.num_row_groups}  "
                  f"cumulative combos: {len(accs):,}")


def main() -> None:
    accs: dict[str, WelfordAcc] = {}
    for v in SWEEP_VERSIONS:
        process_version(v, accs)

    print(f"\nTotal combos tracked: {len(accs):,}")
    rows = []
    for gid, acc in accs.items():
        n, mean, std = acc.finalize()
        fsharpe = (mean / std) if std > 0 else 0.0
        rows.append({
            "global_combo_id": gid,
            "n_trades_full": n,
            "mean_r_full": mean,
            "std_r_full": std,
            "fixed_sharpe_full": fsharpe,
        })
    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH.relative_to(REPO)}  "
          f"({len(out):,} rows, {OUT_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
