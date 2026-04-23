"""1m-only stratified recount — runs on sweep-runner-1 where both the 6.2 GB
CT v11 1m sweep and the new ET 1m sweep live.

Loads three-column subsets (combo_id, gross_pnl_dollars, session_filter_mode)
to avoid the rule-5 OOM trap (full parquet reads peak >10 GB). Emits a
side-by-side CT vs ET comparison at
data/ml/probe1_audit/stratified_recount_1m.json.

N_1.3(1m) under CT is expected to be ~0-1 per the Probe 1 verdict reference.
ET outcome is unknown — if 1m N_1.3 crosses gate=10, that's a second
timeframe that flips Branch A.
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
OUT_PATH = AUDIT_DIR / "stratified_recount_1m.json"

YEARS_SPAN_TRAIN = 5.8056
MIN_TRADES_GATE  = 50
SHARPE_THRESHOLD = 1.3

CT_1M_PATH = REPO / "data" / "ml" / "originals" / "ml_dataset_v11.parquet"
ET_1M_PATH = AUDIT_DIR / "ml_dataset_v11_1m_et.parquet"


def compute_combo_sharpes(df: pd.DataFrame) -> pd.DataFrame:
    """Per-combo Sharpe + session_filter_mode. Expects 3 columns loaded."""
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
        })
    return pd.DataFrame(rows).sort_values("gross_sharpe", ascending=False).reset_index(drop=True)


def stratified(combo_df: pd.DataFrame, label: str) -> dict:
    gated = combo_df[combo_df["n_trades"] >= MIN_TRADES_GATE]
    passing = gated[gated["gross_sharpe"] >= SHARPE_THRESHOLD]
    near_miss = gated[(gated["gross_sharpe"] >= 1.1) & (gated["gross_sharpe"] < 1.3)]
    r = {
        "label":                label,
        "n_combos_with_trades": int(len(combo_df)),
        "n_combos_gated":       int(len(gated)),
        "n_pass_overall":       int(len(passing)),
        "max_gross_sharpe":     float(combo_df["gross_sharpe"].max()) if len(combo_df) else 0.0,
        "near_miss_1_1_to_1_3": int(len(near_miss)),
    }
    by_mode = {}
    for mode in sorted(combo_df["session_filter_mode"].unique()):
        m = int(mode)
        pool = combo_df[combo_df["session_filter_mode"] == m]
        pool_gated = pool[pool["n_trades"] >= MIN_TRADES_GATE]
        pool_pass = pool_gated[pool_gated["gross_sharpe"] >= SHARPE_THRESHOLD]
        pool_nm = pool_gated[(pool_gated["gross_sharpe"] >= 1.1) & (pool_gated["gross_sharpe"] < 1.3)]
        by_mode[f"mode_{m}"] = {
            "n_combos_with_trades": int(len(pool)),
            "n_combos_gated":       int(len(pool_gated)),
            "n_pass":               int(len(pool_pass)),
            "near_miss_count":      int(len(pool_nm)),
            "pass_combo_ids":       sorted(int(x) for x in pool_pass["combo_id"].tolist())[:20],
        }
    r["by_session_filter_mode"] = by_mode
    return r


def load_and_count(path: Path, label: str) -> dict:
    if not path.exists():
        print(f"[recount_1m] {label} parquet missing: {path}", flush=True)
        return None
    print(f"[recount_1m] loading {label}: {path}", flush=True)
    print(f"[recount_1m]   file size: {path.stat().st_size / (1024**3):.2f} GB", flush=True)
    # Load only needed columns — rule 5 of reference_remote_job_workflow.md.
    df = pd.read_parquet(
        path,
        columns=["combo_id", "gross_pnl_dollars", "session_filter_mode"],
    )
    print(f"[recount_1m]   rows={len(df):,}  unique_combos={df['combo_id'].nunique():,}", flush=True)
    combo = compute_combo_sharpes(df)
    del df
    result = stratified(combo, label)
    print(f"[recount_1m] {label}: N_1.3 = {result['n_pass_overall']}  "
          f"(max Sharpe {result['max_gross_sharpe']:.3f}, "
          f"gated {result['n_combos_gated']}, "
          f"near-miss {result['near_miss_1_1_to_1_3']})", flush=True)
    for k, v in result["by_session_filter_mode"].items():
        print(f"  {k}: gated={v['n_combos_gated']:>6}  pass={v['n_pass']:>3}  "
              f"nm={v['near_miss_count']:>3}  ids={v['pass_combo_ids']}", flush=True)
    return result


def main() -> None:
    out = {"task": "probe1_stratified_recount_1m", "timeframe": "1min"}
    ct = load_and_count(CT_1M_PATH, "CT")
    et = load_and_count(ET_1M_PATH, "ET")
    if ct is not None:
        out["ct"] = ct
    if et is not None:
        out["et"] = et

    if ct is not None and et is not None:
        delta = et["n_pass_overall"] - ct["n_pass_overall"]
        comparison = {
            "ct_n_pass":         ct["n_pass_overall"],
            "et_n_pass":         et["n_pass_overall"],
            "delta":             delta,
            "gate_n_1_3":        10,
            "ct_crosses_gate":   ct["n_pass_overall"] >= 10,
            "et_crosses_gate":   et["n_pass_overall"] >= 10,
            "family_falsification_survives_1m": (
                ct["n_pass_overall"] < 10 and et["n_pass_overall"] < 10
            ),
        }
        out["comparison"] = comparison
        print("\n=== 1m CT vs ET comparison ===", flush=True)
        print(f"  CT pass={comparison['ct_n_pass']}  "
              f"ET pass={comparison['et_n_pass']}  "
              f"delta={comparison['delta']:+d}  gate=10  "
              f"survives_falsification_1m={comparison['family_falsification_survives_1m']}",
              flush=True)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[recount_1m] wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
