"""1m-only stratified recount — streams both CT and ET parquets via
pyarrow.iter_batches to avoid OOM. Previous version used pd.read_parquet()
and hit systemd MemoryMax=6G on the 6.19 GB CT parquet (per rule 5 of
memory/reference_remote_job_workflow.md — never pd.read_parquet() whole
files).

The CT parquet (data/ml/originals/ml_dataset_v11.parquet) was appended to
across multiple v11 1m sweep generations; unique_combos ≈ 24,618 even though
the Probe 1 signed 1m sweep was only 16,384 combos. The ET re-sweep
(data/ml/probe1_audit/ml_dataset_v11_1m_et.parquet) is a fresh 16,384-combo
run with combo_ids 0..16383 under seed=0.

To get an apples-to-apples comparison aligned with Probe 1's signed contract,
the CT recount is restricted to combo_ids 0..16383 (same ID range as ET).
The full-CT count across all 24,618 combos is also reported for context.

Per-combo accumulators use running sums (n, Σx, Σx²) so Sharpe is one
calculation per combo at the end, regardless of parquet size.
"""
from __future__ import annotations

import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
AUDIT_DIR = REPO / "data" / "ml" / "probe1_audit"
OUT_PATH = AUDIT_DIR / "stratified_recount_1m.json"

YEARS_SPAN_TRAIN = 5.8056
MIN_TRADES_GATE  = 50
SHARPE_THRESHOLD = 1.3

ET_COMBO_MAX = 16384  # ET re-sweep had combo_ids 0..16383

CT_1M_PATH = REPO / "data" / "ml" / "originals" / "ml_dataset_v11.parquet"
ET_1M_PATH = AUDIT_DIR / "ml_dataset_v11_1m_et.parquet"

BATCH_SIZE = 500_000


def stream_accumulate(path: Path, max_combo_id: int | None = None) -> dict:
    """Stream parquet via iter_batches; accumulate per-combo (n, Σx, Σx²).

    Returns dict[combo_id -> {'n', 'sum', 'sum_sq', 'mode'}].

    If max_combo_id is given, combos with combo_id >= max_combo_id are
    accumulated separately under a second dict for reporting purposes but
    not included in the primary return.
    """
    agg = defaultdict(lambda: {"n": 0, "sum": 0.0, "sum_sq": 0.0, "mode": None})
    agg_out = defaultdict(lambda: {"n": 0, "sum": 0.0, "sum_sq": 0.0, "mode": None})
    pf = pq.ParquetFile(str(path))
    row_groups = pf.num_row_groups
    total_rows = pf.metadata.num_rows
    print(f"[stream] {path.name}: {total_rows:,} rows across "
          f"{row_groups} row groups; batch_size={BATCH_SIZE:,}", flush=True)

    n_rows_seen = 0
    for batch in pf.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["combo_id", "gross_pnl_dollars", "session_filter_mode"],
    ):
        combo_id_arr = batch.column("combo_id").to_numpy()
        gross_arr    = batch.column("gross_pnl_dollars").to_numpy().astype(np.float64)
        mode_arr     = batch.column("session_filter_mode").to_numpy()

        # Unique combo_ids in this batch — use np.unique for O(n) grouping
        # rather than pandas groupby to keep memory tight.
        unique_ids, idx_first = np.unique(combo_id_arr, return_index=True)

        # For each unique combo_id in this batch, mask and aggregate
        for cid, first_idx in zip(unique_ids, idx_first):
            cid_int = int(cid)
            mask = combo_id_arr == cid
            g = gross_arr[mask]
            target = agg_out if (max_combo_id is not None and cid_int >= max_combo_id) else agg
            a = target[cid_int]
            a["n"] += int(g.size)
            a["sum"] += float(g.sum())
            a["sum_sq"] += float((g * g).sum())
            if a["mode"] is None:
                a["mode"] = int(mode_arr[first_idx])

        n_rows_seen += len(combo_id_arr)
        if n_rows_seen % (BATCH_SIZE * 5) == 0 or n_rows_seen == total_rows:
            print(f"[stream]   {n_rows_seen:,} / {total_rows:,} rows "
                  f"({100*n_rows_seen/total_rows:.0f}%)  "
                  f"n_combos_seen={len(agg)+len(agg_out):,}", flush=True)

    return dict(agg), dict(agg_out)


def _compute_sharpes(agg: dict) -> list:
    rows = []
    for cid, a in agg.items():
        n = a["n"]
        mean = a["sum"] / n if n > 0 else 0.0
        if n >= 2:
            var = (a["sum_sq"] - n * mean * mean) / (n - 1)
            if var > 0 and np.isfinite(var):
                sharpe = mean / np.sqrt(var) * np.sqrt(n / YEARS_SPAN_TRAIN)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0
        rows.append({
            "combo_id":            int(cid),
            "n_trades":            int(n),
            "gross_sharpe":        float(sharpe),
            "mean_gross":          float(mean),
            "session_filter_mode": int(a["mode"]) if a["mode"] is not None else -1,
        })
    return sorted(rows, key=lambda r: -r["gross_sharpe"])


def _stratified(combos: list, label: str) -> dict:
    """Compute N_1.3 + stratified breakdown by session_filter_mode."""
    gated = [r for r in combos if r["n_trades"] >= MIN_TRADES_GATE]
    passing = [r for r in gated if r["gross_sharpe"] >= SHARPE_THRESHOLD]
    near_miss = [r for r in gated if 1.1 <= r["gross_sharpe"] < SHARPE_THRESHOLD]

    result = {
        "label":                label,
        "n_combos":             len(combos),
        "n_combos_with_trades": sum(1 for r in combos if r["n_trades"] > 0),
        "n_combos_gated":       len(gated),
        "n_pass_overall":       len(passing),
        "max_gross_sharpe":     combos[0]["gross_sharpe"] if combos else 0.0,
        "near_miss_1_1_to_1_3": len(near_miss),
    }

    by_mode = {}
    for mode in sorted({r["session_filter_mode"] for r in combos}):
        pool = [r for r in combos if r["session_filter_mode"] == mode]
        p_gated = [r for r in pool if r["n_trades"] >= MIN_TRADES_GATE]
        p_pass  = [r for r in p_gated if r["gross_sharpe"] >= SHARPE_THRESHOLD]
        p_nm    = [r for r in p_gated if 1.1 <= r["gross_sharpe"] < SHARPE_THRESHOLD]
        by_mode[f"mode_{mode}"] = {
            "n_combos":             len(pool),
            "n_combos_gated":       len(p_gated),
            "n_pass":               len(p_pass),
            "near_miss_count":      len(p_nm),
            "pass_combo_ids":       sorted(r["combo_id"] for r in p_pass)[:30],
        }
    result["by_session_filter_mode"] = by_mode

    # Top-10 for human inspection
    result["top_10_combos"] = [
        {k: v for k, v in r.items() if k in ("combo_id", "n_trades", "gross_sharpe", "session_filter_mode")}
        for r in combos[:10]
    ]
    return result


def main() -> None:
    out = {"task": "probe1_stratified_recount_1m", "timeframe": "1min"}

    # ── CT: stream, subset by combo_id < 16384 for apples-to-apples ─────────
    if CT_1M_PATH.exists():
        print(f"\n=== CT — {CT_1M_PATH.name} (stream; subset to combo_id < {ET_COMBO_MAX}) ===", flush=True)
        ct_in, ct_out = stream_accumulate(CT_1M_PATH, max_combo_id=ET_COMBO_MAX)
        ct_primary_combos = _compute_sharpes(ct_in)
        ct_all_combos = _compute_sharpes({**ct_in, **ct_out})
        out["ct_subset_to_16384"] = _stratified(ct_primary_combos, "CT_subset_combo_id_<_16384")
        out["ct_full_parquet"]    = _stratified(ct_all_combos, "CT_full_parquet_24618_combos")
        print(f"[recount_1m] CT subset N_1.3 = {out['ct_subset_to_16384']['n_pass_overall']}  "
              f"(of {out['ct_subset_to_16384']['n_combos_gated']} gated, "
              f"max Sharpe {out['ct_subset_to_16384']['max_gross_sharpe']:.3f})", flush=True)
        print(f"[recount_1m] CT full   N_1.3 = {out['ct_full_parquet']['n_pass_overall']}  "
              f"(of {out['ct_full_parquet']['n_combos_gated']} gated, "
              f"max Sharpe {out['ct_full_parquet']['max_gross_sharpe']:.3f})", flush=True)
    else:
        print(f"[recount_1m] CT parquet missing: {CT_1M_PATH}", flush=True)

    # ── ET: stream (no subsetting needed) ──────────────────────────────────
    if ET_1M_PATH.exists():
        print(f"\n=== ET — {ET_1M_PATH.name} (stream) ===", flush=True)
        et_in, _ = stream_accumulate(ET_1M_PATH, max_combo_id=None)
        et_combos = _compute_sharpes(et_in)
        out["et"] = _stratified(et_combos, "ET_16384")
        print(f"[recount_1m] ET N_1.3 = {out['et']['n_pass_overall']}  "
              f"(of {out['et']['n_combos_gated']} gated, "
              f"max Sharpe {out['et']['max_gross_sharpe']:.3f})", flush=True)
    else:
        print(f"[recount_1m] ET parquet missing: {ET_1M_PATH}", flush=True)

    # ── CT/ET comparison ───────────────────────────────────────────────────
    if "ct_subset_to_16384" in out and "et" in out:
        ct_n = out["ct_subset_to_16384"]["n_pass_overall"]
        et_n = out["et"]["n_pass_overall"]
        out["comparison_ct_vs_et_1m"] = {
            "ct_n_pass":         ct_n,
            "et_n_pass":         et_n,
            "delta":             et_n - ct_n,
            "gate_n_1_3":        10,
            "ct_crosses_gate":   ct_n >= 10,
            "et_crosses_gate":   et_n >= 10,
            "family_falsification_survives_1m": (ct_n < 10 and et_n < 10),
        }
        print("\n=== 1m CT vs ET comparison ===", flush=True)
        cmp = out["comparison_ct_vs_et_1m"]
        print(f"  CT subset pass={cmp['ct_n_pass']}  "
              f"ET pass={cmp['et_n_pass']}  "
              f"delta={cmp['delta']:+d}  gate=10  "
              f"survives_1m={cmp['family_falsification_survives_1m']}", flush=True)
        print("\n  CT stratified by session_filter_mode:", flush=True)
        for k, v in out["ct_subset_to_16384"]["by_session_filter_mode"].items():
            print(f"    {k}: gated={v['n_combos_gated']:>6}  pass={v['n_pass']:>3}  "
                  f"nm={v['near_miss_count']:>3}  ids={v['pass_combo_ids']}", flush=True)
        print("\n  ET stratified by session_filter_mode:", flush=True)
        for k, v in out["et"]["by_session_filter_mode"].items():
            print(f"    {k}: gated={v['n_combos_gated']:>6}  pass={v['n_pass']:>3}  "
                  f"nm={v['near_miss_count']:>3}  ids={v['pass_combo_ids']}", flush=True)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[recount_1m] wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
