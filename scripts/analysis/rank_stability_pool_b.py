"""Pool B rank-stability gauntlet: Jaccard overlap of raw-Sharpe top-50 under date-shifted train cutoffs."""
from __future__ import annotations

import gc
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Paths & constants (hard-coded per spec)
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
V10_MFE = REPO / "data" / "ml" / "mfe" / "ml_dataset_v10_mfe.parquet"
NQ_CSV = REPO / "data" / "NQ_1min.csv"
OUTPUT = REPO / "data" / "ml" / "ml1_results_v12" / "rank_stability_pool_b.json"

MIN_TRADES_GATE = 500
TOP_K = 50
COST_PER_CONTRACT_RT = 5.0  # MNQ $5/contract round-trip friction
DOLLARS_PER_POINT = 2.0     # MNQ: $2 per full index point per contract
YEARS_SPAN_TRAIN_BASELINE = 5.8056  # 80/20 split from NQ_1min.csv
SHIFT_DAYS = [-180, -90, 0, 90, 180]  # 0 = baseline (canonical 80/20)

# Only these columns are pulled from the parquet — keeps peak RAM ~500MB.
READ_COLS = ["combo_id", "r_multiple", "stop_distance_pts"]


# ---------------------------------------------------------------------------
# NQ-CSV geometry
# ---------------------------------------------------------------------------
def load_nq_geometry() -> dict:
    """Load NQ timestamps and return train-cutoff geometry + bars/day estimate."""
    print(f"[rank-stability] loading {NQ_CSV.relative_to(REPO)} timestamps...",
          flush=True)
    df = pd.read_csv(NQ_CSV, usecols=["Time"], parse_dates=["Time"])
    df = df.dropna(subset=["Time"]).reset_index(drop=True)
    total_bars = len(df)
    t_first = df["Time"].iloc[0]
    t_last = df["Time"].iloc[-1]
    span_days = (t_last - t_first).total_seconds() / 86400.0
    bars_per_day = total_bars / span_days

    train_end_baseline = int(math.floor(0.8 * total_bars))
    t_train_end_baseline = df["Time"].iloc[train_end_baseline - 1]

    print(f"[rank-stability] total_bars={total_bars:,} "
          f"span_days={span_days:.2f} bars_per_day={bars_per_day:.2f}",
          flush=True)
    print(f"[rank-stability] baseline train_cutoff_bar={train_end_baseline:,} "
          f"(time {t_train_end_baseline})", flush=True)
    sys.stdout.flush()

    return {
        "times": df["Time"].to_numpy(),
        "total_bars": total_bars,
        "bars_per_day": float(bars_per_day),
        "train_end_baseline": train_end_baseline,
        "t_first": t_first,
        "t_last": t_last,
    }


def arm_geometry(geo: dict, shift_days: int) -> dict:
    """For a given shift, return the arm's train_cutoff_bar and timestamp span."""
    total_bars = geo["total_bars"]
    shift_bars = int(round(shift_days * geo["bars_per_day"]))
    raw_cutoff = geo["train_end_baseline"] + shift_bars
    clipped = max(1, min(total_bars, raw_cutoff))
    was_clipped = clipped != raw_cutoff
    times = geo["times"]
    t_end = times[clipped - 1]
    years_span = (pd.Timestamp(t_end) - pd.Timestamp(times[0])).total_seconds() \
                 / (86400.0 * 365.25)
    return {
        "shift_days": int(shift_days),
        "shift_bars": int(shift_bars),
        "train_cutoff_bar": int(clipped),
        "was_clipped": was_clipped,
        "years_span": float(years_span),
        "train_frac": clipped / total_bars,
    }


# ---------------------------------------------------------------------------
# v10 MFE streaming aggregator
# ---------------------------------------------------------------------------
def accumulate_trades(src: Path) -> dict[int, np.ndarray]:
    """Stream v10 MFE parquet; return per-combo 1-contract gross-PnL array in entry order.

    Per the v12 builder's documented invariant, trades within a combo are appended
    in entry-bar order by the sweep runner, so np.concatenate of per-row-group
    slices preserves within-combo chronological order. v10 MFE has no
    entry_bar_idx column, so ordinal position is our only time proxy.
    """
    pf = pq.ParquetFile(str(src))
    available = set(pf.schema_arrow.names)
    missing = [c for c in READ_COLS if c not in available]
    if missing:
        raise RuntimeError(f"v10 MFE parquet missing required cols: {missing}")

    acc_pnl: dict[int, list[np.ndarray]] = defaultdict(list)

    t0 = time.time()
    n_rows = 0
    n_rg = pf.metadata.num_row_groups
    for rg in range(n_rg):
        chunk = pf.read_row_group(rg, columns=READ_COLS).to_pandas()
        cid_arr = chunk["combo_id"].astype(np.int64).to_numpy()
        rmul = chunk["r_multiple"].to_numpy(dtype=np.float32)
        stop = chunk["stop_distance_pts"].to_numpy(dtype=np.float32)
        # 1-contract gross PnL (matches v12 builder line 229)
        gross_pnl = rmul * stop * np.float32(DOLLARS_PER_POINT)

        order = np.argsort(cid_arr, kind="stable")
        cid_sorted = cid_arr[order]
        uniq, starts = np.unique(cid_sorted, return_index=True)
        ends = np.append(starts[1:], len(cid_sorted))
        gross_pnl_ord = gross_pnl[order]
        for cid, st, en in zip(uniq, starts, ends):
            acc_pnl[int(cid)].append(gross_pnl_ord[st:en])

        n_rows += len(chunk)
        if (rg + 1) % 20 == 0 or rg == n_rg - 1:
            elapsed = time.time() - t0
            rate = n_rows / max(elapsed, 1e-9)
            print(f"  rg {rg+1}/{n_rg} | {n_rows:,} rows | "
                  f"{rate:,.0f} rows/s | combos={len(acc_pnl)}", flush=True)
        del chunk, gross_pnl, gross_pnl_ord

    combos: dict[int, np.ndarray] = {}
    for cid in list(acc_pnl.keys()):
        combos[cid] = np.concatenate(acc_pnl.pop(cid))
    gc.collect()
    print(f"[rank-stability] accumulated {len(combos):,} combos "
          f"({n_rows:,} trades, {time.time()-t0:.1f}s)", flush=True)
    return combos


# ---------------------------------------------------------------------------
# Per-arm Sharpe ranking
# ---------------------------------------------------------------------------
def arm_top50(combos: dict[int, np.ndarray], train_frac: float,
              years_span: float) -> tuple[list[str], int]:
    """Compute top-50 global_combo_ids for a given arm's train fraction.

    Approach: for each combo, keep trades[0 : int(train_frac * n_combo_trades)],
    then compute annualized 1-contract net Sharpe over the arm's years_span.
    Gate: arm-local n_trades >= MIN_TRADES_GATE.

    NOTE on the proportional-slicing proxy: v10 MFE has no entry_bar_idx, so
    we cannot filter trades by bar. Ordinal slicing inside a combo is the
    documented surrogate (see v12 builder docstring lines 25–31). Within-combo
    entry order is preserved by np.concatenate of row-group chunks.
    """
    rows: list[tuple[int, float]] = []
    n_gated = 0
    for cid, gross_pnl in combos.items():
        n_full = len(gross_pnl)
        cut = int(round(train_frac * n_full))
        if cut < MIN_TRADES_GATE:
            n_gated += 1
            continue
        arm_pnl = gross_pnl[:cut]
        n_arm = cut
        net = arm_pnl - np.float32(COST_PER_CONTRACT_RT)
        std = float(np.std(net, ddof=1))
        if std <= 0.0 or not np.isfinite(std) or n_arm < 2:
            n_gated += 1
            continue
        trades_per_year = n_arm / years_span
        sharpe = float(np.mean(net) / std * math.sqrt(trades_per_year))
        rows.append((cid, sharpe))

    rows.sort(key=lambda r: r[1], reverse=True)
    top = rows[:TOP_K]
    top_ids = [f"v10_{cid}" for cid, _ in top]
    eligible_count = len(rows)
    return top_ids, eligible_count


def jaccard(a: list[str], b: list[str]) -> float:
    """Set Jaccard overlap."""
    sa, sb = set(a), set(b)
    u = sa | sb
    if not u:
        return 0.0
    return len(sa & sb) / len(u)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    t_start = time.time()
    geo = load_nq_geometry()
    arms_meta = {d: arm_geometry(geo, d) for d in SHIFT_DAYS}
    for d, meta in arms_meta.items():
        flag = " (CLIPPED)" if meta["was_clipped"] else ""
        print(f"[rank-stability] shift={d:+d}d -> cutoff_bar={meta['train_cutoff_bar']:,} "
              f"years_span={meta['years_span']:.3f} frac={meta['train_frac']:.4f}"
              f"{flag}", flush=True)

    combos = accumulate_trades(V10_MFE)

    # Compute top-50 per arm.
    arm_results: dict[str, dict] = {}
    baseline_ids: list[str] | None = None
    for d in SHIFT_DAYS:
        meta = arms_meta[d]
        t0 = time.time()
        top_ids, eligible_count = arm_top50(combos, meta["train_frac"],
                                            meta["years_span"])
        print(f"[rank-stability] shift={d:+d}d: "
              f"eligible={eligible_count} top50[0..4]={top_ids[:5]} "
              f"({time.time()-t0:.1f}s)", flush=True)
        if d == 0:
            baseline_ids = top_ids
            arm_results["baseline"] = {
                "top50": top_ids,
                "n_gated_combos": len(combos) - eligible_count,
                "n_eligible_combos": eligible_count,
                "train_cutoff_bar": meta["train_cutoff_bar"],
                "years_span": meta["years_span"],
            }
        else:
            sign = "minus" if d < 0 else "plus"
            key = f"shift_{sign}_{abs(d)}d"
            arm_results[key] = {
                "top50": top_ids,
                "n_gated_combos": len(combos) - eligible_count,
                "n_eligible_combos": eligible_count,
                "train_cutoff_bar": meta["train_cutoff_bar"],
                "years_span": meta["years_span"],
                "shift_days": d,
                "was_clipped": meta["was_clipped"],
            }

    if baseline_ids is None:
        raise RuntimeError("shift=0 baseline arm missing — cannot compute Jaccard")

    # Jaccard vs baseline for non-zero shifts.
    jaccard_values: list[float] = []
    for d in SHIFT_DAYS:
        if d == 0:
            continue
        sign = "minus" if d < 0 else "plus"
        key = f"shift_{sign}_{abs(d)}d"
        j = jaccard(baseline_ids, arm_results[key]["top50"])
        arm_results[key]["jaccard"] = j
        jaccard_values.append(j)
        print(f"[rank-stability] Jaccard baseline vs shift={d:+d}d: {j:.4f}",
              flush=True)

    j_mean = float(np.mean(jaccard_values)) if jaccard_values else 0.0
    j_min = float(np.min(jaccard_values)) if jaccard_values else 0.0

    payload = {
        "baseline_top50": baseline_ids,
        "train_cutoff_bar_baseline": geo["train_end_baseline"],
        "arms": arm_results,
        "jaccard_mean": j_mean,
        "jaccard_min": j_min,
        "jaccard_values": jaccard_values,
        "shifts_days": [d for d in SHIFT_DAYS if d != 0],
        "bars_per_day": geo["bars_per_day"],
        "total_bars": geo["total_bars"],
        "min_trades_gate": MIN_TRADES_GATE,
        "top_k": TOP_K,
        "cost_per_contract_rt": COST_PER_CONTRACT_RT,
        "years_span_train_baseline": YEARS_SPAN_TRAIN_BASELINE,
        "source_parquet": str(V10_MFE.relative_to(REPO)),
        "method_note": (
            "v10 MFE parquet has no entry_bar_idx column. Date-shifted train "
            "cutoffs are approximated by proportional ordinal slicing within "
            "each combo's trade stream (entry order is preserved by the sweep "
            "runner; see scripts/analysis/build_combo_features_ml1_v12.py "
            "docstring lines 25-31). 1-contract gross PnL is reconstructed "
            "as r_multiple * stop_distance_pts * 2.0 (MNQ $2/pt)."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, default=float))
    print(f"[rank-stability] wrote {OUTPUT.relative_to(REPO)} "
          f"({OUTPUT.stat().st_size/1e3:.1f} KB)", flush=True)
    print(f"[rank-stability] Jaccard mean={j_mean:.4f} min={j_min:.4f} "
          f"values={[f'{v:.3f}' for v in jaccard_values]}", flush=True)
    print(f"[rank-stability] Done: total runtime {time.time()-t_start:.1f}s",
          flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
