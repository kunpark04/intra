"""C4: Probe 1 gross-ceiling readout on 15m + 1h parquets.

Per preregistration §3/§4:
  - MIN_TRADES_GATE = 50 (bar-count-adjusted floor for fast timeframes)
  - N_1.3(tf) := count of combos with full-train gross Sharpe >= 1.3
                  AND trade count >= MIN_TRADES_GATE
  - Branch A (sunset): N_1.3(15m) < 10 AND N_1.3(1h) < 10
  - Branch B (K-fold): N_1.3(15m) >= 10 OR N_1.3(1h) >= 10

Sharpe formula (mirrors scripts/analysis/build_combo_features_ml1_v12.py:254-260):
  gross_pnl_dollars column already = 1-contract gross PnL per trade.
  sharpe = mean(gross) / std(gross, ddof=1) * sqrt(n / YEARS_SPAN_TRAIN)

YEARS_SPAN_TRAIN = 5.8056 is the 1min-baseline value and applies equally to
15m/1h because all three sweeps use the first 80% of calendar time (resampled
bars share the window). Cross-check: 1h parquet train = 33,520 bars,
33,520 / (23 * 252) = 5.78 yr ≈ 5.8056 ✓.

Reference: 1min v12 ceiling was N_1.3 = 1 (combo v11_23634 @ Sharpe 1.108).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
YEARS_SPAN_TRAIN = 5.8056
MIN_TRADES_GATE = 50
SHARPE_THRESHOLD = 1.3

PARQUETS = {
    "15m": REPO / "data" / "ml" / "originals" / "ml_dataset_v11_15m.parquet",
    "1h":  REPO / "data" / "ml" / "originals" / "ml_dataset_v11_1h.parquet",
}


def compute_combo_sharpes(df: pd.DataFrame) -> pd.DataFrame:
    """Per-combo gross-Sharpe table (annualised, train-partition years)."""
    rows = []
    for combo_id, g in df.groupby("combo_id", sort=False):
        gross = g["gross_pnl_dollars"].to_numpy()
        n = len(gross)
        if n < 2:
            sharpe = 0.0
        else:
            std = float(np.std(gross, ddof=1))
            if std <= 0:
                sharpe = 0.0
            else:
                mean = float(np.mean(gross))
                tpy = n / YEARS_SPAN_TRAIN
                sharpe = mean / std * np.sqrt(tpy)

        # also capture a little metadata for downstream eyeballing
        rows.append({
            "combo_id": int(combo_id),
            "n_trades": n,
            "gross_sharpe": float(sharpe),
            "mean_gross": float(gross.mean()),
            "std_gross": float(np.std(gross, ddof=1)) if n >= 2 else 0.0,
            "win_rate": float((g["label_win"] == 1).mean()),
            "entry_timing_offset": int(g["entry_timing_offset"].iloc[0]),
            "fill_slippage_ticks": int(g["fill_slippage_ticks"].iloc[0]),
            "cooldown_after_exit_bars": int(g["cooldown_after_exit_bars"].iloc[0]),
        })
    return pd.DataFrame(rows).sort_values("gross_sharpe", ascending=False).reset_index(drop=True)


def report(tf: str, path: Path) -> tuple[int, pd.DataFrame]:
    print(f"\n{'='*68}\n  {tf.upper()}   ({path.name})\n{'='*68}")
    df = pd.read_parquet(path)
    print(f"shape={df.shape}   unique_combos={df['combo_id'].nunique():,}")

    combo = compute_combo_sharpes(df)
    print(f"combos analysed: {len(combo):,}")
    gated = combo[combo["n_trades"] >= MIN_TRADES_GATE].copy()
    print(f"combos passing MIN_TRADES_GATE>={MIN_TRADES_GATE}: {len(gated):,}")

    # Distribution snapshot on gated pool
    s = gated["gross_sharpe"]
    print("\ngated gross-Sharpe distribution:")
    qs = [0.50, 0.75, 0.90, 0.95, 0.99, 1.00]
    qvals = s.quantile(qs).to_dict()
    for q, v in qvals.items():
        print(f"  p{int(q*100):02d}: {v:+.3f}")
    print(f"  max : {s.max():+.3f}")

    # Threshold counts on gated pool
    print("\nthreshold counts (gated):")
    for thr in (0.5, 1.0, 1.3, 1.5, 2.0):
        k = int((s >= thr).sum())
        print(f"  N(gross Sharpe >= {thr:.1f}): {k}")

    n_1_3 = int((s >= SHARPE_THRESHOLD).sum())
    print(f"\nN_1.3({tf}) = {n_1_3}")

    # Top-10 combos so we can inspect the ceiling
    top = gated.head(10)[["combo_id", "n_trades", "gross_sharpe", "win_rate",
                          "entry_timing_offset", "fill_slippage_ticks",
                          "cooldown_after_exit_bars"]]
    print(f"\ntop 10 gated combos ({tf}):")
    print(top.to_string(index=False))

    return n_1_3, gated


def main() -> None:
    print(f"YEARS_SPAN_TRAIN = {YEARS_SPAN_TRAIN}")
    print(f"MIN_TRADES_GATE  = {MIN_TRADES_GATE}")
    print(f"SHARPE_THRESHOLD = {SHARPE_THRESHOLD}")

    results: dict[str, tuple[int, pd.DataFrame]] = {}
    for tf, path in PARQUETS.items():
        results[tf] = report(tf, path)

    # Branch decision
    n15 = results["15m"][0]
    n1h = results["1h"][0]
    print(f"\n{'='*68}\n  BRANCH DECISION (preregistration §4)\n{'='*68}")
    print(f"N_1.3(15m) = {n15}")
    print(f"N_1.3(1h)  = {n1h}")
    print(f"1min baseline (reference) N_1.3(1min) = 1 (v12 combo_id 23634 @ 1.108)")

    both_below = (n15 < 10) and (n1h < 10)
    either_ok  = (n15 >= 10) or (n1h >= 10)
    if both_below:
        print("\n>>> BRANCH A — family-level sunset.")
        print("    Both timeframes fail N_1.3 >= 10. Strategy family")
        print("    (z-score + EMA mean-reversion on NQ) does not produce a")
        print("    viable gross ceiling within the sampled microstructure grid.")
        print("    Next step: C6 verdict doc (Branch A).")
    elif either_ok:
        print("\n>>> BRANCH B — combo-agnostic K-fold audit.")
        print("    At least one timeframe reaches N_1.3 >= 10 combos.")
        print("    Next step: C5 GroupKFold(5) on combo_id with v3_no_memory")
        print("    feature set, ship gate = cross-fold Sharpe p50 >= 1.0,")
        print("    ruin <= 20%, >= 10 combos in basket in >= 4/5 folds.")
    else:
        print("\n[warn] decision logic inconsistency — investigate.")

    # Persist snapshot to disk for audit trail
    out_dir = REPO / "tasks"
    for tf, (_, gated) in results.items():
        out_path = out_dir / f"probe1_{tf}_gross_sharpe.csv"
        gated.to_csv(out_path, index=False)
        print(f"[saved] {out_path.relative_to(REPO)}  ({len(gated):,} rows)")


if __name__ == "__main__":
    main()
