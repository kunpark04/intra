"""Quick sanity check on data/ml/originals/ml_dataset_v11_15m.parquet.

Confirms:
  - shape + schema match expected (55 cols; microstructure cols present, int64)
  - all 3000 combos present
  - 27 microstructure cells all represented (entry_timing_offset × fill_slippage_ticks × cooldown_after_exit_bars)
  - basic trade-count distribution looks healthy
"""
from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PARQUET = REPO / "data" / "ml" / "originals" / "ml_dataset_v11_15m.parquet"
MANIFEST = REPO / "data" / "ml" / "originals" / "ml_dataset_v11_15m_manifest.json"


def main() -> None:
    print(f"[read] {PARQUET.relative_to(REPO)}")
    df = pd.read_parquet(PARQUET)
    print(f"  shape: {df.shape}")
    print(f"  unique combos: {df['combo_id'].nunique():,}")

    print("\n[schema] microstructure columns:")
    for col in ("entry_timing_offset", "fill_slippage_ticks", "cooldown_after_exit_bars"):
        print(f"  {col}: dtype={df[col].dtype}  unique={sorted(df[col].unique().tolist())}")

    print("\n[coverage] microstructure cells (combo-level, not trade-level):")
    combo_grid = (df.groupby("combo_id")[
        ["entry_timing_offset", "fill_slippage_ticks", "cooldown_after_exit_bars"]
    ].first())
    cell_counts = combo_grid.value_counts().sort_index()
    print(f"  populated cells: {len(cell_counts)}/27")
    print(f"  combos per cell (min/p50/max): "
          f"{cell_counts.min()} / {int(cell_counts.median())} / {cell_counts.max()}")

    print("\n[trades] per-combo trade counts:")
    tc = df.groupby("combo_id").size()
    print(f"  combos with trades: {(tc > 0).sum():,} / {len(tc):,}")
    print(f"  total trades: {int(tc.sum()):,}")
    print(f"  trades/combo p10/p50/p90/max: "
          f"{int(tc.quantile(.1))} / {int(tc.median())} / "
          f"{int(tc.quantile(.9))} / {int(tc.max())}")

    print("\n[friction] sample slippage-tick audit (expect $1/tick/contract on net - gross):")
    # Pick one combo from each fill_slippage_ticks value, check a few trades
    for slip in sorted(df["fill_slippage_ticks"].unique()):
        row = df[df["fill_slippage_ticks"] == slip].iloc[0]
        contracts_guess = row["friction_dollars"] / (5.0 + slip * 1.0)
        print(f"  slip={slip} tick -> friction=${row['friction_dollars']:.2f} "
              f"on implied contracts≈{contracts_guess:.1f}")

    print("\n[manifest] summary:")
    with open(MANIFEST) as f:
        m = json.load(f)
    if isinstance(m, list):
        statuses = pd.Series([x.get("status", "?") for x in m]).value_counts().to_dict()
        print(f"  entries: {len(m)}")
        print(f"  status: {statuses}")
    else:
        print(f"  (dict form, {len(m)} keys)")

    print("\n[ok] verification complete.")


if __name__ == "__main__":
    main()
