"""ET-sweep comparison recount — run after the ET re-sweep completes.

Stratified N_1.3 counts on the ET-labeled re-sweep parquets, mirroring the
CT-sweep recount produced by _probe1_stratified_recount.py. Emits a side-by-
side comparison JSON so the council can read whether the Probe 1 family-level
falsification survives the TZ fix.

Authority: chairman verdict of Probe 3 COUNCIL_RECONVENE. See
tasks/tz_bug_provenance_log_2026-04-23.md for provenance.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
AUDIT_DIR = REPO / "data" / "ml" / "probe1_audit"
OUT_PATH  = AUDIT_DIR / "stratified_recount_et.json"
CT_JSON   = AUDIT_DIR / "stratified_recount.json"

YEARS_SPAN_TRAIN = 5.8056
MIN_TRADES_GATE  = 50
SHARPE_THRESHOLD = 1.3

ET_PARQUETS = {
    "15m": AUDIT_DIR / "ml_dataset_v11_15m_et.parquet",
    "1h":  AUDIT_DIR / "ml_dataset_v11_1h_et.parquet",
}


def compute_combo_sharpes(df: pd.DataFrame) -> pd.DataFrame:
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
                sharpe = float(np.mean(gross)) / std * float(np.sqrt(n / YEARS_SPAN_TRAIN))
        rows.append({
            "combo_id":            int(combo_id),
            "n_trades":            int(n),
            "gross_sharpe":        float(sharpe),
            "session_filter_mode": int(g["session_filter_mode"].iloc[0]),
            "tod_exit_hour":       int(g["tod_exit_hour"].iloc[0])
                                    if "tod_exit_hour" in g.columns else 0,
        })
    return pd.DataFrame(rows).sort_values("gross_sharpe", ascending=False).reset_index(drop=True)


def stratified_counts(combo_df: pd.DataFrame) -> dict:
    gated = combo_df[combo_df["n_trades"] >= MIN_TRADES_GATE]
    passing = gated[gated["gross_sharpe"] >= SHARPE_THRESHOLD]
    r = {
        "n_combos_with_trades": int(len(combo_df)),
        "n_combos_gated":       int(len(gated)),
        "n_pass_overall":       int(len(passing)),
    }
    by_mode = {}
    for mode in sorted(combo_df["session_filter_mode"].unique()):
        m = int(mode)
        pool = combo_df[combo_df["session_filter_mode"] == m]
        pool_gated = pool[pool["n_trades"] >= MIN_TRADES_GATE]
        pool_pass = pool_gated[pool_gated["gross_sharpe"] >= SHARPE_THRESHOLD]
        by_mode[f"mode_{m}"] = {
            "n_combos_with_trades": int(len(pool)),
            "n_combos_gated":       int(len(pool_gated)),
            "n_pass":               int(len(pool_pass)),
            "pass_combo_ids":       sorted(int(x) for x in pool_pass["combo_id"].tolist()),
        }
    r["by_session_filter_mode"] = by_mode
    r["pass_combos_full"] = [
        {
            "combo_id":            int(x["combo_id"]),
            "gross_sharpe":        round(float(x["gross_sharpe"]), 4),
            "n_trades":            int(x["n_trades"]),
            "session_filter_mode": int(x["session_filter_mode"]),
            "tod_exit_hour":       int(x["tod_exit_hour"]),
        }
        for _, x in passing.iterrows()
    ]
    return r


def main() -> None:
    out = {"task": "probe1_stratified_recount_et", "by_timeframe": {}}
    for tf, path in ET_PARQUETS.items():
        if not path.exists():
            print(f"[recount_et] SKIP {tf}: parquet missing at {path}", flush=True)
            continue
        print(f"\n=== {tf.upper()} — {path.name} ===", flush=True)
        df = pd.read_parquet(path)
        print(f"rows={len(df):,}  unique_combos={df['combo_id'].nunique():,}", flush=True)
        combo = compute_combo_sharpes(df)
        r = stratified_counts(combo)
        out["by_timeframe"][tf] = r
        print(f"ET N_1.3 = {r['n_pass_overall']}  (gated={r['n_combos_gated']})", flush=True)
        for k, v in r["by_session_filter_mode"].items():
            print(f"  {k}: gated={v['n_combos_gated']:>4}  pass={v['n_pass']:>3}  "
                  f"ids={v['pass_combo_ids']}", flush=True)

    # Side-by-side comparison if CT recount exists
    if CT_JSON.exists():
        ct = json.loads(CT_JSON.read_text())
        comparison = {}
        for tf in ("15m", "1h"):
            ct_r = ct.get("by_timeframe", {}).get(tf)
            et_r = out["by_timeframe"].get(tf)
            if ct_r and et_r:
                comparison[tf] = {
                    "ct_n_pass":         ct_r["n_pass_overall"],
                    "et_n_pass":         et_r["n_pass_overall"],
                    "delta":             et_r["n_pass_overall"] - ct_r["n_pass_overall"],
                    "gate_n_1_3":        10,
                    "ct_crosses_gate":   ct_r["n_pass_overall"] >= 10,
                    "et_crosses_gate":   et_r["n_pass_overall"] >= 10,
                    "family_falsification_survives": (
                        ct_r["n_pass_overall"] < 10 and et_r["n_pass_overall"] < 10
                    ),
                }
        out["comparison_ct_vs_et"] = comparison
        print("\n=== CT vs ET comparison ===", flush=True)
        for tf, c in comparison.items():
            print(f"  {tf}: CT pass={c['ct_n_pass']}  ET pass={c['et_n_pass']}  "
                  f"delta={c['delta']:+d}  gate=10  "
                  f"survives_falsification={c['family_falsification_survives']}", flush=True)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[recount_et] wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
