"""Probe 1 recount under the SIGNED v12 Sharpe formula (1-contract per-trade).

Replaces the dollar-PnL-sized recount in _probe1_stratified_recount.py and
_probe1_stratified_recount_1m.py, which the stats-ml-logic-reviewer flagged
as UNSOUND (measurement mismatch with signed verdict; see
tasks/_agent_bus/probe1_1m_audit_2026-04-23/stats-ml-logic-reviewer.md).

Signed formula (mirrors scripts/analysis/build_combo_features_ml1_v12.py:229):
    per_trade_pnl = r_multiple * stop_distance_pts * DOLLARS_PER_POINT
    sharpe = mean(per_trade_pnl) / std(per_trade_pnl, ddof=1)
             * sqrt(n / YEARS_SPAN_TRAIN)

This is "1-contract net" Sharpe (r_multiple is computed net of friction at
param_sweep.py:1038). It ISOLATES signal quality from contract-count sizing
policy, which is what Probe 1 §3 was intended to gate on.

Runs ALL three timeframes, CT and ET, and emits a single comparison JSON
at data/ml/probe1_audit/v12_formula_recount.json.

For 1m CT, uses the pre-computed v12 combo-feature parquet at
data/ml/ml1_results_v12/combo_features_v12.parquet (since streaming the
6.2 GB raw parquet from local is not feasible; the v12 parquet has the
same audit_full_gross_sharpe per combo).
"""
from __future__ import annotations

import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data" / "ml" / "probe1_audit"
OUT_PATH = OUT_DIR / "v12_formula_recount.json"

YEARS_SPAN_TRAIN = 5.8056
MIN_TRADES_GATE = 50
SHARPE_THRESHOLD = 1.3
DOLLARS_PER_POINT = 2.0

# 15m/1h parquets are small — pandas read is fine
PARQUETS_SMALL = {
    "15m_CT": REPO / "data" / "ml" / "originals" / "ml_dataset_v11_15m.parquet",
    "15m_ET": REPO / "data" / "ml" / "probe1_audit" / "ml_dataset_v11_15m_et.parquet",
    "1h_CT":  REPO / "data" / "ml" / "originals" / "ml_dataset_v11_1h.parquet",
    "1h_ET":  REPO / "data" / "ml" / "probe1_audit" / "ml_dataset_v11_1h_et.parquet",
}

# 1m ET parquet is 3.6 GB — stream. 1m CT is 6.2 GB on remote only; use v12 pre-computed.
ET_1M_PATH = REPO / "data" / "ml" / "probe1_audit" / "ml_dataset_v11_1m_et.parquet"
V12_COMBO_FEATURES = REPO / "data" / "ml" / "ml1_results_v12" / "combo_features_v12.parquet"

BATCH_SIZE = 500_000


def compute_signed_sharpes_from_df(df: pd.DataFrame) -> list[dict]:
    """Compute 1-contract-net Sharpe per combo using rmul * stop * $/pt."""
    # Compute 1-contract net PnL per trade
    pt = (df["r_multiple"] * df["stop_distance_pts"] * DOLLARS_PER_POINT).to_numpy(dtype=np.float64)
    # Attach back — needed for groupby
    work = df[["combo_id", "session_filter_mode"]].copy()
    work["_pt"] = pt
    rows = []
    for cid, g in work.groupby("combo_id", sort=False):
        arr = g["_pt"].to_numpy()
        n = arr.size
        if n < 2:
            sharpe = 0.0
            mean = float(arr.mean()) if n else 0.0
        else:
            mean = float(arr.mean())
            std = float(arr.std(ddof=1))
            sharpe = (mean / std * np.sqrt(n / YEARS_SPAN_TRAIN)) if (std > 0 and np.isfinite(std)) else 0.0
        rows.append({
            "combo_id":            int(cid),
            "n_trades":            int(n),
            "sharpe_signed":       float(sharpe),
            "mean_per_trade":      float(mean),
            "session_filter_mode": int(g["session_filter_mode"].iloc[0]),
        })
    return sorted(rows, key=lambda r: -r["sharpe_signed"])


def stream_signed_sharpes_from_parquet(path: Path) -> list[dict]:
    """Stream large parquet with per-combo running sums (n, Σx, Σx²) of 1-contract-net PnL."""
    agg = defaultdict(lambda: {"n": 0, "s": 0.0, "ssq": 0.0, "mode": None})
    pf = pq.ParquetFile(str(path))
    total = pf.metadata.num_rows
    seen = 0
    for batch in pf.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["combo_id", "r_multiple", "stop_distance_pts", "session_filter_mode"],
    ):
        cid_arr = batch.column("combo_id").to_numpy()
        rmul    = batch.column("r_multiple").to_numpy().astype(np.float64)
        stop    = batch.column("stop_distance_pts").to_numpy().astype(np.float64)
        mode    = batch.column("session_filter_mode").to_numpy()
        pt      = rmul * stop * DOLLARS_PER_POINT
        for cid in np.unique(cid_arr):
            m = cid_arr == cid
            arr = pt[m]
            a = agg[int(cid)]
            a["n"]   += int(arr.size)
            a["s"]   += float(arr.sum())
            a["ssq"] += float((arr * arr).sum())
            if a["mode"] is None:
                a["mode"] = int(mode[np.where(m)[0][0]])
        seen += len(cid_arr)
    rows = []
    for cid, a in agg.items():
        n = a["n"]
        if n < 1:
            continue
        mean = a["s"] / n
        if n >= 2:
            var = (a["ssq"] - n * mean * mean) / (n - 1)
            sharpe = (mean / np.sqrt(var) * np.sqrt(n / YEARS_SPAN_TRAIN)) if (var > 0 and np.isfinite(var)) else 0.0
        else:
            sharpe = 0.0
        rows.append({
            "combo_id":            int(cid),
            "n_trades":            int(n),
            "sharpe_signed":       float(sharpe),
            "mean_per_trade":      float(mean),
            "session_filter_mode": int(a["mode"]) if a["mode"] is not None else -1,
        })
    return sorted(rows, key=lambda r: -r["sharpe_signed"])


def stratified(combos: list, label: str) -> dict:
    gated = [c for c in combos if c["n_trades"] >= MIN_TRADES_GATE]
    passing = [c for c in gated if c["sharpe_signed"] >= SHARPE_THRESHOLD]
    near = [c for c in gated if 1.1 <= c["sharpe_signed"] < SHARPE_THRESHOLD]
    result = {
        "label":                label,
        "formula":              "1-contract-net: rmul * stop * $/pt",
        "n_combos":             len(combos),
        "n_combos_gated":       len(gated),
        "n_pass_overall":       len(passing),
        "max_sharpe_gated":     gated[0]["sharpe_signed"] if gated else 0.0,
        "near_miss_1_1_to_1_3": len(near),
    }
    by_mode = {}
    for mode in sorted({c["session_filter_mode"] for c in combos}):
        pool = [c for c in combos if c["session_filter_mode"] == mode]
        g = [c for c in pool if c["n_trades"] >= MIN_TRADES_GATE]
        p = [c for c in g if c["sharpe_signed"] >= SHARPE_THRESHOLD]
        by_mode[f"mode_{mode}"] = {
            "n_combos_gated":       len(g),
            "n_pass":               len(p),
            "pass_combo_ids":       sorted(c["combo_id"] for c in p)[:30],
        }
    result["by_session_filter_mode"] = by_mode
    result["top_10_passing"] = [
        {k: v for k, v in c.items() if k in ("combo_id","n_trades","sharpe_signed","session_filter_mode")}
        for c in passing[:10]
    ]
    return result


def main() -> None:
    out = {
        "task":     "probe1_recount_v12_formula",
        "authority": "stats-ml-logic-reviewer UNSOUND verdict 2026-04-23 on dollar-PnL-sized recount",
        "by_timeframe": {},
    }

    # 15m + 1h (small parquets, pandas OK)
    for label, path in PARQUETS_SMALL.items():
        if not path.exists():
            print(f"[v12] MISSING: {path}", flush=True)
            continue
        print(f"\n=== {label} — {path.name} ===", flush=True)
        df = pd.read_parquet(
            path,
            columns=["combo_id", "r_multiple", "stop_distance_pts", "session_filter_mode"],
        )
        print(f"  rows={len(df):,}  unique_combos={df['combo_id'].nunique():,}", flush=True)
        combos = compute_signed_sharpes_from_df(df)
        r = stratified(combos, label)
        out["by_timeframe"][label] = r
        print(f"  N_1.3 = {r['n_pass_overall']}  (gated {r['n_combos_gated']}, "
              f"max Sharpe {r['max_sharpe_gated']:.4f}, near-miss {r['near_miss_1_1_to_1_3']})",
              flush=True)
        for k, v in r["by_session_filter_mode"].items():
            print(f"    {k}: gated={v['n_combos_gated']:>5}  pass={v['n_pass']:>3}  "
                  f"ids={v['pass_combo_ids']}", flush=True)

    # 1m ET (stream)
    if ET_1M_PATH.exists():
        print(f"\n=== 1m_ET — {ET_1M_PATH.name} (stream) ===", flush=True)
        combos = stream_signed_sharpes_from_parquet(ET_1M_PATH)
        r = stratified(combos, "1m_ET")
        out["by_timeframe"]["1m_ET"] = r
        print(f"  N_1.3 = {r['n_pass_overall']}  (gated {r['n_combos_gated']}, "
              f"max Sharpe {r['max_sharpe_gated']:.4f})", flush=True)
        for k, v in r["by_session_filter_mode"].items():
            print(f"    {k}: gated={v['n_combos_gated']:>5}  pass={v['n_pass']:>3}  "
                  f"ids={v['pass_combo_ids']}", flush=True)
    else:
        print(f"[v12] 1m ET parquet MISSING: {ET_1M_PATH}", flush=True)

    # 1m CT — use v12 pre-computed combo features (already has audit_full_gross_sharpe)
    if V12_COMBO_FEATURES.exists():
        print(f"\n=== 1m_CT — via v12 combo_features ({V12_COMBO_FEATURES.name}) ===", flush=True)
        df = pd.read_parquet(V12_COMBO_FEATURES)
        # v12 schema: audit_full_gross_sharpe is the 1-contract-net Sharpe (misnamed).
        sc = "audit_full_gross_sharpe"
        if sc not in df.columns:
            print(f"  [v12] column {sc} missing; have: {[c for c in df.columns if 'sharpe' in c]}", flush=True)
        else:
            # Filter for 1m: v12 may have all TFs mixed. Check the schema.
            nt = "audit_full_n_trades" if "audit_full_n_trades" in df.columns else "n_trades"
            print(f"  rows={len(df):,}  max sharpe={df[sc].max():.4f}", flush=True)
            gated = df[df[nt] >= MIN_TRADES_GATE] if nt in df.columns else df
            pass_count = int((gated[sc] >= SHARPE_THRESHOLD).sum())
            near_count = int(((gated[sc] >= 1.1) & (gated[sc] < SHARPE_THRESHOLD)).sum())
            out["by_timeframe"]["1m_CT_v12"] = {
                "label":            "1m_CT_via_v12_combo_features",
                "n_combos":         int(len(df)),
                "n_combos_gated":   int(len(gated)),
                "n_pass_overall":   pass_count,
                "max_sharpe_gated": float(gated[sc].max()) if len(gated) else 0.0,
                "near_miss_1_1_to_1_3": near_count,
                "note":             "1m_CT via v12 pre-computed — 1-contract-net Sharpe (column misnamed 'audit_full_gross_sharpe')",
            }
            print(f"  N_1.3 = {pass_count}  (gated {len(gated)}, max Sharpe {gated[sc].max():.4f}, near-miss {near_count})",
                  flush=True)

    # Summary table — the headline comparison across all 6 (CT,ET) × (1m,15m,1h) cells
    print("\n" + "=" * 78, flush=True)
    print(" SUMMARY — N_1.3 under SIGNED v12 formula (gate = 10)", flush=True)
    print("=" * 78, flush=True)
    summary = []
    for key in ("1m_CT_v12", "1m_ET", "15m_CT", "15m_ET", "1h_CT", "1h_ET"):
        r = out["by_timeframe"].get(key)
        if not r:
            continue
        summary.append({
            "tf": key,
            "n_pass": r["n_pass_overall"],
            "max_sharpe": r.get("max_sharpe_gated", 0.0),
            "gated": r["n_combos_gated"],
            "crosses_gate": r["n_pass_overall"] >= 10,
        })
        print(f"  {key:>10}: N_1.3 = {r['n_pass_overall']:>4}  "
              f"max={r.get('max_sharpe_gated',0.0):>7.3f}  "
              f"gated={r['n_combos_gated']:>5}  "
              f"crosses_gate={r['n_pass_overall'] >= 10}", flush=True)
    out["summary"] = summary

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[v12] wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
