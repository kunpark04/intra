"""Scope D — post-hoc SES_2 sub-window split on combos {865, 1298, 664}.

Authority: `tasks/scope_d_brief.md` (stage = characterization; council chairman's
one-thing-to-do-first from `tasks/council-report-2026-04-23-b1-scope.html`).

Splits the Amendment-1 SES_2 complement-of-RTH bucket into two mechanically
distinct sub-windows and reports per-combo metrics across a 3-bucket partition
of the 1h test partition:

    SES_1  RTH                   et_min >= 570  AND et_min <  960    (09:30 - 16:00 ET)
    SES_2a GLOBEX overnight      et_min >= 1080 OR  et_min <  570    (18:00 - 09:30 ET)
    SES_2b post-RTH + halt       et_min >= 960  AND et_min < 1080    (16:00 - 18:00 ET)

Reuses the tz_convert("America/New_York") pattern from `_probe4_readout.py`
and `_probe3_1h_ritual.py` so EDT/EST transitions are handled transparently
across the test window.

No gates. No pass/fail. No branch routing. This is a characterization
footnote — zero engine compute (pure re-bucketing of existing per-trade
parquets), wall-clock < 1 s.

Inputs:
  data/ml/probe2/combo865_1h_test.parquet     (220 trades, full schema)
  data/ml/probe4/combo1298_SES_0_trades.parquet (123 trades, {net_pnl, entry_bar_idx})
  data/ml/probe4/combo664_SES_0_trades.parquet  (780 trades, {net_pnl, entry_bar_idx})
  data/NQ_1h.parquet

Output:
  data/ml/scope_d/readout.json

Sharpe scaling preserves `YEARS_SPAN_TEST = 1.4799` inherited from
`_probe4_readout.py` so cross-probe metric comparability holds
(see memory/feedback_years_span_cross_tf.md on the ~1.3% conservative
bias — retained for comparability, not silently fixed).
"""
from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import split_train_test  # noqa: E402

OUT_DIR      = REPO_ROOT / "data" / "ml" / "scope_d"
OUT_PATH     = OUT_DIR / "readout.json"
BARS_PARQUET = REPO_ROOT / "data" / "NQ_1h.parquet"

# Per-combo input: (parquet_path, pnl_col). Probe 2's combo865 parquet has full
# schema (one row per trade with all features); Probe 4's trade parquets carry
# only the two columns the readout needs. Schema difference is handled by the
# `pnl_col` lookup below.
COMBO_INPUTS: dict[int, tuple[Path, str]] = {
    865:  (REPO_ROOT / "data" / "ml" / "probe2" / "combo865_1h_test.parquet",   "net_pnl_dollars"),
    1298: (REPO_ROOT / "data" / "ml" / "probe4" / "combo1298_SES_0_trades.parquet", "net_pnl_dollars"),
    664:  (REPO_ROOT / "data" / "ml" / "probe4" / "combo664_SES_0_trades.parquet",  "net_pnl_dollars"),
}

SES_LABELS = {
    "SES_0":  "all trades (full 24h baseline)",
    "SES_1":  "RTH 09:30-16:00 ET",
    "SES_2a": "GLOBEX overnight 18:00-09:30 ET (tradable)",
    "SES_2b": "post-RTH + CME settlement halt 16:00-18:00 ET (partially untradable)",
}

# Inherited from _probe4_readout.py — preserves cross-probe Sharpe comparability.
YEARS_SPAN_TEST = 1.4799


# ── Session masks ───────────────────────────────────────────────────────────
# Note: masks are mutually exclusive and cover [0, 1440) exactly.
def _ses_1_rth_mask(et_min: np.ndarray) -> np.ndarray:
    return (et_min >= 570) & (et_min < 960)


def _ses_2a_overnight_mask(et_min: np.ndarray) -> np.ndarray:
    return (et_min >= 1080) | (et_min < 570)


def _ses_2b_post_rth_halt_mask(et_min: np.ndarray) -> np.ndarray:
    return (et_min >= 960) & (et_min < 1080)


# ── Metrics (Sharpe formula identical to _probe4_readout.py._compute_metrics) ─
def _compute_metrics(pnl: np.ndarray) -> dict:
    n = int(pnl.size)
    if n == 0:
        return {
            "n_trades":             0,
            "net_sharpe":           0.0,
            "net_dollars_per_year": 0.0,
            "mean_pnl_dollars":     0.0,
            "std_pnl_dollars":      0.0,
            "total_net_dollars":    0.0,
        }
    mean_pnl = float(pnl.mean())
    total    = float(pnl.sum())
    if n < 2:
        std_pnl    = 0.0
        net_sharpe = 0.0
    else:
        std_pnl = float(pnl.std(ddof=1))
        if std_pnl <= 0 or not math.isfinite(std_pnl):
            net_sharpe = 0.0
        else:
            net_sharpe = (mean_pnl / std_pnl) * math.sqrt(n / YEARS_SPAN_TEST)
    net_dollars_per_year = mean_pnl * n / YEARS_SPAN_TEST
    return {
        "n_trades":             n,
        "net_sharpe":           float(net_sharpe),
        "net_dollars_per_year": float(net_dollars_per_year),
        "mean_pnl_dollars":     mean_pnl,
        "std_pnl_dollars":      std_pnl,
        "total_net_dollars":    total,
    }


def _load_test_bars_with_et_minutes() -> pd.DataFrame:
    bars = pd.read_parquet(BARS_PARQUET)
    _, test_part = split_train_test(bars, 0.8)
    test_part = test_part.reset_index(drop=True)
    ts = pd.to_datetime(test_part["time"])
    if getattr(ts.dt, "tz", None) is None:
        ts_utc = ts.dt.tz_localize("UTC")
    else:
        ts_utc = ts.dt.tz_convert("UTC")
    ts_et = ts_utc.dt.tz_convert("America/New_York")
    et_min = ts_et.dt.hour * 60 + ts_et.dt.minute
    test_part = test_part.copy()
    test_part["_et_minutes"] = et_min.values.astype(np.int32)
    return test_part


def _partition_combo(pnl: np.ndarray, et_trade: np.ndarray) -> dict:
    rth_mask     = _ses_1_rth_mask(et_trade)
    overnight    = _ses_2a_overnight_mask(et_trade)
    post_rth_halt = _ses_2b_post_rth_halt_mask(et_trade)

    # Sanity: masks must be disjoint and cover all trades.
    assert (rth_mask & overnight).sum() == 0
    assert (rth_mask & post_rth_halt).sum() == 0
    assert (overnight & post_rth_halt).sum() == 0
    covered = rth_mask.sum() + overnight.sum() + post_rth_halt.sum()
    assert covered == len(pnl), f"mask coverage {covered} != {len(pnl)}"

    return {
        "SES_0":  _compute_metrics(pnl),
        "SES_1":  _compute_metrics(pnl[rth_mask]),
        "SES_2a": _compute_metrics(pnl[overnight]),
        "SES_2b": _compute_metrics(pnl[post_rth_halt]),
    }


def _share_of(total: float, part: float) -> float | None:
    """Fraction of `total` attributable to `part`. Returns None when total is
    near zero to avoid misleading ratios on combos with near-flat aggregate $."""
    if abs(total) < 1e-9:
        return None
    return part / total


def _dominance_label(share_2a: float | None, share_2b: float | None) -> str:
    """Human-readable regime label per `scope_d_brief.md` Outcome
    interpretations §1-4. Thresholds are descriptive, not gates."""
    if share_2a is None or share_2b is None:
        return "indeterminate (near-zero total $)"
    if share_2a >= 0.80 and share_2b <= 0.25:
        return "SES_2a dominates (pure overnight)"
    if share_2b >= 0.50:
        return "SES_2b dominates (post-RTH/halt artifact)"
    if share_2a >= 0.25 and share_2b >= 0.25:
        return "mixed (both sub-windows contribute)"
    return "weak / no clear pattern"


def main() -> None:
    print("[scope_d] loading 1h test-partition bar timeline + ET-minute index")
    bars = _load_test_bars_with_et_minutes()
    et_min_arr = bars["_et_minutes"].to_numpy()
    n_bars = len(et_min_arr)
    print(f"[scope_d] test partition bars: {n_bars}  "
          f"(ET min range: {et_min_arr.min()} .. {et_min_arr.max()})")

    per_combo = {}
    for cid, (parquet_path, pnl_col) in COMBO_INPUTS.items():
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"combo {cid} input missing: {parquet_path}  "
                f"(scope_d_brief.md §Method lists required artifacts)"
            )
        df = pd.read_parquet(parquet_path, columns=[pnl_col, "entry_bar_idx"])
        idx = df["entry_bar_idx"].to_numpy()
        if idx.max() >= n_bars:
            raise RuntimeError(
                f"combo {cid}: entry_bar_idx max {idx.max()} >= n_bars {n_bars}  "
                f"— bar timeline / engine partition mismatch"
            )
        pnl      = df[pnl_col].to_numpy().astype(float)
        et_trade = et_min_arr[idx]
        metrics  = _partition_combo(pnl, et_trade)

        # SES_2a/SES_2b share-of-SES_0 summary (per brief §Outcome interpretations).
        share_2a = _share_of(metrics["SES_0"]["total_net_dollars"],
                              metrics["SES_2a"]["total_net_dollars"])
        share_2b = _share_of(metrics["SES_0"]["total_net_dollars"],
                              metrics["SES_2b"]["total_net_dollars"])

        # Trade-count shares (useful when $ shares are ambiguous).
        n_share_2a = metrics["SES_2a"]["n_trades"] / metrics["SES_0"]["n_trades"]
        n_share_2b = metrics["SES_2b"]["n_trades"] / metrics["SES_0"]["n_trades"]

        per_combo[str(cid)] = {
            "source_parquet":    str(parquet_path.relative_to(REPO_ROOT)),
            "metrics":           metrics,
            "ses_2a_dollar_share": share_2a,
            "ses_2b_dollar_share": share_2b,
            "ses_2a_count_share":  n_share_2a,
            "ses_2b_count_share":  n_share_2b,
            "regime_label":        _dominance_label(share_2a, share_2b),
        }

    # Cross-combo agreement summary.
    regimes = {c: d["regime_label"] for c, d in per_combo.items()}
    regimes_agree = len(set(regimes.values())) == 1

    out = {
        "task":                "scope_d_ses2_subwindow_split",
        "stage":               "characterization (not gate, not preregistered confirmation)",
        "authority":           "tasks/council-report-2026-04-23-b1-scope.html (chairman's one-thing-to-do-first)",
        "brief":               "tasks/scope_d_brief.md",
        "predecessors":        [
            "tasks/probe2_verdict.md (combo865)",
            "tasks/probe3_verdict.md (combo865 4-gate PASS)",
            "tasks/probe4_verdict.md (combo1298, combo664 SESSION_CONFOUND)",
        ],
        "timeframe":           "1h",
        "partition":           "test (2024-10-22 -> 2026-04-08)",
        "combos":              list(COMBO_INPUTS.keys()),
        "session_labels":      SES_LABELS,
        "session_partition_method": (
            "post-hoc ET-minute mask on entry_bar_idx via "
            "tz_convert('America/New_York') on data/NQ_1h.parquet test_part"
        ),
        "years_span_test":     YEARS_SPAN_TEST,
        "per_combo":           per_combo,
        "cross_combo_regime_agreement": regimes_agree,
        "cross_combo_regimes": regimes,
        "caveats": [
            "Partition reuse: 1h test partition has been read by Probes 2, 3, 4, and this footnote. "
            "Numbers are less trustworthy as independent confirmation with each re-read.",
            "No holding-time axis in Probe 4 trade parquets; Outsider's holding-time/opportunity-cost "
            "concern from council is not addressed here.",
            "SES_2b includes the 17:00-18:00 CME settlement halt — if SES_2b is material, a finer "
            "16:00-17:00 vs 17:00-18:00 split is an immediate follow-up.",
            "Per-bucket metrics with n_trades < 30 are directional-only, not statistically resolved.",
            "No gate, no pass/fail, no branch auto-fires. Human reading of the table decides the next step.",
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"[scope_d] wrote {OUT_PATH}")
    print()
    print("─── Per-combo 3-bucket decomposition ─" + "─" * 45)
    for cid_str, d in per_combo.items():
        m = d["metrics"]
        print(f"\ncombo {cid_str}  ({d['source_parquet']})")
        print(f"  regime_label:       {d['regime_label']}")
        hdr = f"  {'bucket':<7} {'n':>4}  {'sharpe':>8}  {'$/yr':>12}  {'mean $':>9}  {'total $':>12}"
        print(hdr)
        for b in ("SES_0", "SES_1", "SES_2a", "SES_2b"):
            bm = m[b]
            print(f"  {b:<7} {bm['n_trades']:>4}  {bm['net_sharpe']:>8.3f}  "
                  f"{bm['net_dollars_per_year']:>12,.0f}  "
                  f"{bm['mean_pnl_dollars']:>9,.0f}  "
                  f"{bm['total_net_dollars']:>12,.0f}")
        if d["ses_2a_dollar_share"] is not None:
            print(f"  SES_2a $ share of SES_0: {d['ses_2a_dollar_share']:+.1%}  "
                  f"(count share {d['ses_2a_count_share']:.1%})")
        if d["ses_2b_dollar_share"] is not None:
            print(f"  SES_2b $ share of SES_0: {d['ses_2b_dollar_share']:+.1%}  "
                  f"(count share {d['ses_2b_count_share']:.1%})")
    print()
    print(f"[scope_d] cross_combo_regime_agreement: {regimes_agree}")
    print(f"[scope_d] regimes: {regimes}")


if __name__ == "__main__":
    main()
