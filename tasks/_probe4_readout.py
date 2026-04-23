"""Probe 4 aggregate readout — §4.3 Welch-t + §4.4 post-hoc ET session
decomposition + §5 ordered branch routing.

Reads the 2 per-combo SES_0 JSONs + 2 SES_0 per-trade parquets produced by
`tasks/_probe4_run_combo.py`, partitions each combo's SES_0 trades into SES_1
(RTH 09:30-16:00 ET) / SES_2 (GLOBEX overnight) post-hoc using
`tz_convert("America/New_York")` on the 1h bar timeline, computes the Welch-t
primary gate (§4.3), then applies §5 branch routing mechanically (ordered
first-match).

Inputs:
  data/ml/probe4/combo{1298,664}_SES_0.json
  data/ml/probe4/combo{1298,664}_SES_0_trades.parquet
  data/NQ_1h.parquet  (pre-built 1h bar cache)

Output:
  data/ml/probe4/readout.json  — machine-readable verdict

Session definitions (per prereg §4.4, mirror Probe 3 §4.4):
  SES_0 — all sessions (baseline, from the engine run as-is)
  SES_1 — RTH only:              ET minute >= 570  AND  ET minute <  960
  SES_2 — overnight / GLOBEX:    ET minute >= 1080 OR   ET minute <  570

The 1h bar timeline is loaded via the same convention used by `param_sweep.py`
and `_probe3_1h_ritual.py`: `pd.read_parquet("data/NQ_1h.parquet")` then
`split_train_test(df, 0.8)` → test_part reset_index. `entry_bar_idx` from the
engine trade parquet indexes directly into that test_part. tz_convert to
America/New_York handles EDT/EST transitions transparently (fixed UTC-offset
arithmetic would silently miscategorize ~50% of a multi-year window).

§4.3 Welch-t formula (one-tailed, H1: 1298 > 664):
    t = (mean_pnl_1298 - mean_pnl_664)
        / sqrt((var_1298 / n_1298) + (var_664 / n_664))
  Gate threshold: t >= 2.0.
  Insufficient-n defense: if either SES_0 run has n_trades < 50, emit
  "welch_t": null AND "welch_gate_pass": false. Never emit
  welch_gate_pass: null — downstream §5 comparisons treat null as ambiguous.

§5 branch routing is ordered (first match wins):
  1. gate_1298_abs_pass == False AND welch_gate_pass == False          -> INCONCLUSIVE
  2. gate_1298_abs_pass == True  AND SES_2 abs_pass AND NOT SES_1 abs_pass
     AND (SES_2.net_sharpe - SES_1.net_sharpe) > 1.0                   -> SESSION_CONFOUND
  3. gate_1298_abs_pass == True  AND gate_664_abs_pass == False
     AND welch_gate_pass == True                                        -> PROPERTY_VALIDATED
  4. gate_1298_abs_pass == True  AND gate_664_abs_pass == True         -> COUNCIL_RECONVENE
  5. Catch-all (narrow-miss per §5.2, or any unforeseen combination)   -> COUNCIL_RECONVENE

Row 2 fires BEFORE row 3 — an 1298 that passes absolute and Welch-t but
concentrates in SES_2 routes to SESSION_CONFOUND, not PROPERTY_VALIDATED.

References:
- tasks/probe4_preregistration.md §4.3, §4.4, §5, §5.1, §5.2, §8.2, §8.3
- Precedent: tasks/_probe3_1h_ritual.py:124-140, 180-193, 229-271
  (canonical post-hoc ET-minute session filter pattern)

Runs anywhere (no engine compute). Wall-clock < 1 s.
"""
from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# UTF-8 stdout (Windows cp1252 chokes on any non-ASCII symbol).
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import split_train_test  # noqa: E402

IN_DIR    = REPO_ROOT / "data" / "ml" / "probe4"
OUT_PATH  = IN_DIR / "readout.json"
BARS_PARQUET = REPO_ROOT / "data" / "NQ_1h.parquet"

COMBOS      = (1298, 664)
SES_LABELS  = {0: "SES_0_all", 1: "SES_1_RTH", 2: "SES_2_GLOBEX"}

WELCH_T_GATE         = 2.0
WELCH_MIN_N          = 50   # per-combo; must hold for BOTH to compute t
SESSION_CONFOUND_DLT = 1.0  # (SES_2 Sharpe - SES_1 Sharpe) > this

# Probe 4 gate constants — must match _probe4_run_combo.py exactly so SES_1 /
# SES_2 partitions use identical thresholds as the SES_0 aggregate.
YEARS_SPAN_TEST = 1.4799
SHARPE_GATE     = 1.3
NTRADES_GATE    = 50
DOLLARS_GATE    = 5_000.0


def _load_run(cid: int) -> dict:
    p = IN_DIR / f"combo{cid}_SES_0.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Per-run readout missing: {p}. "
            f"Run `tasks/_probe4_run_combo.py --combo-id {cid}` first."
        )
    return json.loads(p.read_text())


def _load_ses0_trades(cid: int) -> pd.DataFrame:
    p = IN_DIR / f"combo{cid}_SES_0_trades.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"SES_0 per-trade parquet missing: {p}. "
            f"Welch-t primary gate (§4.3) + §4.4 session decomposition "
            f"both depend on it."
        )
    return pd.read_parquet(p)


def _load_test_bars_with_et_minutes() -> pd.DataFrame:
    """Load the 1h test partition and attach an ET minute-of-day column.

    Mirrors `_probe3_1h_ritual.py::_load_test_bars_with_et_minutes`. The
    returned DataFrame's row index aligns with the engine's entry_bar_idx.
    tz_convert handles EDT/EST transitions across the multi-year window.
    """
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


def _ses_rth_mask(et_min: np.ndarray) -> np.ndarray:
    """SES_1 RTH: 09:30 - 16:00 ET."""
    return (et_min >= 570) & (et_min < 960)


def _ses_overnight_mask(et_min: np.ndarray) -> np.ndarray:
    """SES_2 GLOBEX: everything outside RTH (complement of SES_1)."""
    return ~_ses_rth_mask(et_min)


def _compute_metrics(pnl: np.ndarray) -> dict:
    """Same formula as _probe4_run_combo._aggregate_metrics — duplicated here
    so the readout can compute SES_1 / SES_2 partition metrics without a
    round-trip through the engine."""
    n = int(pnl.size)
    if n < 2:
        return {
            "n_trades":             n,
            "net_sharpe":           0.0,
            "net_dollars_per_year": 0.0,
            "mean_pnl_dollars":     0.0,
            "std_pnl_dollars":      0.0,
            "sharpe_pass":          False,
            "n_trades_pass":        n >= NTRADES_GATE,
            "dollars_pass":         False,
            "abs_pass":             False,
        }
    mean_pnl = float(pnl.mean())
    std_pnl  = float(pnl.std(ddof=1))
    if std_pnl <= 0 or not math.isfinite(std_pnl):
        net_sharpe = 0.0
    else:
        net_sharpe = (mean_pnl / std_pnl) * math.sqrt(n / YEARS_SPAN_TEST)
    net_dollars_per_year = mean_pnl * n / YEARS_SPAN_TEST
    sharpe_pass   = net_sharpe           >= SHARPE_GATE
    n_trades_pass = n                    >= NTRADES_GATE
    dollars_pass  = net_dollars_per_year >= DOLLARS_GATE
    return {
        "n_trades":             n,
        "net_sharpe":           float(net_sharpe),
        "net_dollars_per_year": float(net_dollars_per_year),
        "mean_pnl_dollars":     mean_pnl,
        "std_pnl_dollars":      std_pnl,
        "sharpe_pass":          bool(sharpe_pass),
        "n_trades_pass":        bool(n_trades_pass),
        "dollars_pass":         bool(dollars_pass),
        "abs_pass":             bool(sharpe_pass and n_trades_pass and dollars_pass),
    }


def _partition_session_metrics(
    trades: pd.DataFrame,
    et_min_arr: np.ndarray,
) -> dict:
    """Decompose SES_0 trades into SES_1 / SES_2 using entry_bar_idx -> ET minute."""
    if len(trades) == 0:
        empty = _compute_metrics(np.array([]))
        return {"SES_1": empty, "SES_2": empty}
    idx = trades["entry_bar_idx"].to_numpy()
    if idx.max() >= len(et_min_arr):
        raise RuntimeError(
            f"entry_bar_idx max {idx.max()} >= len(test_part) {len(et_min_arr)} "
            f"— bar timeline / engine partition mismatch"
        )
    et_trade = et_min_arr[idx]
    pnl      = trades["net_pnl_dollars"].to_numpy()
    rth_mask = _ses_rth_mask(et_trade)
    return {
        "SES_1": _compute_metrics(pnl[rth_mask]),
        "SES_2": _compute_metrics(pnl[~rth_mask]),
    }


def _welch_t(pnl_a: np.ndarray, pnl_b: np.ndarray) -> dict:
    """Welch's t on 1298 vs 664 per-trade PnL.

    Insufficient-n defense per §4.3: if either n < 50, return t=None and
    gate_pass=False (never None). Also guards against zero variance.
    """
    n_a, n_b = int(pnl_a.size), int(pnl_b.size)
    base = {
        "n_1298":        n_a,
        "n_664":         n_b,
        "mean_1298":     float(pnl_a.mean()) if n_a >= 1 else None,
        "mean_664":      float(pnl_b.mean()) if n_b >= 1 else None,
        "var_1298":      float(pnl_a.var(ddof=1)) if n_a >= 2 else None,
        "var_664":       float(pnl_b.var(ddof=1)) if n_b >= 2 else None,
        "threshold":     WELCH_T_GATE,
        "min_n_per_arm": WELCH_MIN_N,
    }
    if n_a < WELCH_MIN_N or n_b < WELCH_MIN_N:
        return {
            **base,
            "welch_t":           None,
            "welch_gate_pass":   False,
            "insufficient_n":    True,
            "insufficient_reason": (
                f"n_1298={n_a} or n_664={n_b} < {WELCH_MIN_N}; "
                f"§4.3 defaults welch_gate_pass to False."
            ),
        }
    var_a, var_b = base["var_1298"], base["var_664"]
    denom = math.sqrt((var_a / n_a) + (var_b / n_b))
    if denom <= 0 or not math.isfinite(denom):
        return {
            **base,
            "welch_t":           None,
            "welch_gate_pass":   False,
            "insufficient_n":    False,
            "insufficient_reason": "zero pooled SE (degenerate variance)",
        }
    t = (base["mean_1298"] - base["mean_664"]) / denom
    return {
        **base,
        "welch_t":         float(t),
        "welch_gate_pass": bool(t >= WELCH_T_GATE),
        "insufficient_n":  False,
    }


def _route_branch(
    gate_1298_abs_pass: bool,
    gate_664_abs_pass:  bool,
    welch_gate_pass:    bool,
    ses_1_1298_metrics: dict,
    ses_2_1298_metrics: dict,
) -> tuple[str, int, str]:
    """Apply §5 routing table in order. First match wins. Returns
    (branch, matched_row, rationale)."""
    ses1_pass  = bool(ses_1_1298_metrics["abs_pass"])
    ses2_pass  = bool(ses_2_1298_metrics["abs_pass"])
    ses1_sharp = float(ses_1_1298_metrics["net_sharpe"])
    ses2_sharp = float(ses_2_1298_metrics["net_sharpe"])
    sharp_dlt  = ses2_sharp - ses1_sharp

    # Row 1 — INCONCLUSIVE
    if (not gate_1298_abs_pass) and (not welch_gate_pass):
        return (
            "INCONCLUSIVE",
            1,
            "1298 failed absolute gate AND Welch-t < 2.0 (§5 row 1).",
        )

    # Row 2 — SESSION_CONFOUND (must fire BEFORE row 3 per §5 ordering)
    if gate_1298_abs_pass and ses2_pass and (not ses1_pass) and sharp_dlt > SESSION_CONFOUND_DLT:
        return (
            "SESSION_CONFOUND",
            2,
            f"1298 passed absolute; SES_2 abs_pass={ses2_pass}, SES_1 abs_pass={ses1_pass}, "
            f"(SES_2 Sharpe - SES_1 Sharpe) = {sharp_dlt:.3f} > {SESSION_CONFOUND_DLT} (§5 row 2).",
        )

    # Row 3 — PROPERTY_VALIDATED
    if gate_1298_abs_pass and (not gate_664_abs_pass) and welch_gate_pass:
        return (
            "PROPERTY_VALIDATED",
            3,
            "1298 PASS absolute, 664 FAIL absolute, Welch-t >= 2.0 (§5 row 3).",
        )

    # Row 4 — COUNCIL_RECONVENE (both-pass adjudication)
    if gate_1298_abs_pass and gate_664_abs_pass:
        return (
            "COUNCIL_RECONVENE",
            4,
            "Both 1298 and 664 passed absolute gate (§5 row 4; both-pass adjudication).",
        )

    # Row 5 — catch-all (§5.2 narrow-miss or any unforeseen anomaly)
    return (
        "COUNCIL_RECONVENE",
        5,
        "Caught by §5 row 5 (anomaly / narrow-miss per §5.2). "
        "Verdict document MUST flag this as such.",
    )


def main() -> None:
    # ── Load per-run SES_0 JSONs + trade parquets ──────────────────────────
    ses_0_1298 = _load_run(1298)
    ses_0_664  = _load_run(664)
    trades_1298 = _load_ses0_trades(1298)
    trades_664  = _load_ses0_trades(664)

    # ── §4.1 / §4.2 — SES_0 absolute gates ──────────────────────────────────
    gate_1298_abs_pass = bool(ses_0_1298["metrics"]["abs_pass"])
    gate_664_abs_pass  = bool(ses_0_664["metrics"]["abs_pass"])

    # ── §4.4 post-hoc ET-minute session decomposition ──────────────────────
    print("[readout] loading 1h bar cache + building ET minute index")
    bars = _load_test_bars_with_et_minutes()
    et_min_arr = bars["_et_minutes"].to_numpy()
    sessions_1298 = _partition_session_metrics(trades_1298, et_min_arr)
    sessions_664  = _partition_session_metrics(trades_664,  et_min_arr)

    # ── §4.3 — Welch-t primary gate (on full SES_0 per-trade series) ───────
    pnl_1298 = trades_1298["net_pnl_dollars"].to_numpy()
    pnl_664  = trades_664["net_pnl_dollars"].to_numpy()
    welch    = _welch_t(pnl_1298, pnl_664)
    welch_gate_pass = bool(welch["welch_gate_pass"])

    # ── §5 branch routing ────────────────────────────────────────────────────
    branch, matched_row, rationale = _route_branch(
        gate_1298_abs_pass=gate_1298_abs_pass,
        gate_664_abs_pass=gate_664_abs_pass,
        welch_gate_pass=welch_gate_pass,
        ses_1_1298_metrics=sessions_1298["SES_1"],
        ses_2_1298_metrics=sessions_1298["SES_2"],
    )

    # §5.2 narrow-miss flagging: 1298 abs FAIL + welch PASS caught by row 5
    narrow_miss = bool(
        matched_row == 5
        and (not gate_1298_abs_pass)
        and welch_gate_pass
    )

    # Session decomposition summary (non-gate-bound per §4.4)
    session_decomp = {
        "1298": {
            "SES_0": ses_0_1298["metrics"],
            "SES_1": sessions_1298["SES_1"],
            "SES_2": sessions_1298["SES_2"],
        },
        "664": {
            "SES_0": ses_0_664["metrics"],
            "SES_1": sessions_664["SES_1"],
            "SES_2": sessions_664["SES_2"],
        },
    }

    out = {
        "probe":                    "probe4_b2_second_combo_carveout",
        "preregistration_signed_commit": "432fb3d",
        "timeframe":                "1h",
        "partition":                "test (2024-10-22 -> 2026-04-08)",
        "combos_under_test":        list(COMBOS),
        "session_modes":            {str(k): v for k, v in SES_LABELS.items()},
        "session_partition_method": (
            "post-hoc ET-minute mask on SES_0 entry_bar_idx via "
            "tz_convert('America/New_York') on data/NQ_1h.parquet test_part "
            "(engine session_filter_mode=0 for both runs)"
        ),
        "gates": {
            "§4.1_abs_1298": {
                "gate_pass": gate_1298_abs_pass,
                "metrics":   ses_0_1298["metrics"],
            },
            "§4.2_abs_664": {
                "gate_pass": gate_664_abs_pass,
                "metrics":   ses_0_664["metrics"],
            },
            "§4.3_welch_t": welch,
        },
        "branch":                   branch,
        "matched_row":              matched_row,
        "rationale":                rationale,
        "narrow_miss_flag":         narrow_miss,
        "narrow_miss_note": (
            "§5.2 narrow-miss: 1298 failed absolute gate but Welch-t >= 2.0; "
            "verdict document must flag this explicitly and NOT post-hoc promote "
            "to PROPERTY_VALIDATED."
            if narrow_miss else None
        ),
        "session_decomposition":    session_decomp,
        "per_run_sources": {
            f"combo{cid}_SES_0": str((IN_DIR / f"combo{cid}_SES_0.json").relative_to(REPO_ROOT))
            for cid in COMBOS
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"[readout] wrote {OUT_PATH}")
    print(f"[readout] gate_1298_abs_pass = {gate_1298_abs_pass}  "
          f"(sharpe={ses_0_1298['metrics']['net_sharpe']:.3f}, "
          f"n={ses_0_1298['metrics']['n_trades']}, "
          f"$/yr={ses_0_1298['metrics']['net_dollars_per_year']:.0f})")
    print(f"[readout] gate_664_abs_pass  = {gate_664_abs_pass}  "
          f"(sharpe={ses_0_664['metrics']['net_sharpe']:.3f}, "
          f"n={ses_0_664['metrics']['n_trades']}, "
          f"$/yr={ses_0_664['metrics']['net_dollars_per_year']:.0f})")
    for cid in COMBOS:
        sd = session_decomp[str(cid)]
        print(f"[readout] combo{cid} SES_1 RTH:      "
              f"sharpe={sd['SES_1']['net_sharpe']:.3f}  n={sd['SES_1']['n_trades']}  "
              f"$/yr={sd['SES_1']['net_dollars_per_year']:.0f}  abs_pass={sd['SES_1']['abs_pass']}")
        print(f"[readout] combo{cid} SES_2 GLOBEX:   "
              f"sharpe={sd['SES_2']['net_sharpe']:.3f}  n={sd['SES_2']['n_trades']}  "
              f"$/yr={sd['SES_2']['net_dollars_per_year']:.0f}  abs_pass={sd['SES_2']['abs_pass']}")
    wt = welch["welch_t"]
    wt_str = f"{wt:.3f}" if isinstance(wt, (int, float)) else "null"
    print(f"[readout] welch_t            = {wt_str}  "
          f"welch_gate_pass={welch_gate_pass}  "
          f"(n_1298={welch['n_1298']}, n_664={welch['n_664']})")
    print(f"[readout] branch             = {branch}  (row {matched_row})")
    print(f"[readout] rationale          = {rationale}")
    if narrow_miss:
        print(f"[readout] NARROW-MISS FLAG (§5.2): verdict must call this out.")


if __name__ == "__main__":
    main()
