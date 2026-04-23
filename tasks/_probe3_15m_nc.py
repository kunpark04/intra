"""Probe 3 gate §4.3 — 15m negative-control session/exit ritual.

Runs combo 865 × 4 EX variants on the 15m test partition (4 engine runs);
post-hoc applies 4 ET-session filters to split each EX's trades into 4
cells, yielding 4 × 4 = 16 cells. Each cell is evaluated against the
Probe 2 §4 three-gate set:

  net_sharpe ≥ 1.3  AND  n_trades ≥ 50  AND  net_$/yr ≥ 5000

Gate §4.3 PASSES iff ≤ 2 of 16 cells clear the three-gate set.
(Negative control: under the friction-regime interpretation of Probe 2,
15m's per-trade gross (~$355) cannot clear per-contract friction (~$438)
regardless of session/exit tuning. Multiple 15m cells clearing would
contradict the mechanical story and threaten the 1h PASS interpretation.)

Post-hoc design notes:
- SES filter applies to ENTRIES ONLY (§2.3 final paragraph: "Session filter
  applies only to entries. Open positions managed by native exit rules
  regardless of session."). Implemented as entry-time filter on the
  resulting trade parquet.
- EX_2 (TOD exit @ 15:00 ET) faces a spec↔engine units mismatch: the
  preregistration says "Override `tod_exit_hour` to 15" but the engine
  field is UTC hour-of-day, not ET. Resolved here as tod_exit_hour=19 UTC
  (= 15:00 EDT exactly; 14:00 EST = 1h early). Test window is ~53% EDT /
  47% EST; the seasonal mismatch is disclosed in the readout JSON.

References:
- tasks/probe3_preregistration.md §2.3, §2.4, §2.5, §3.2.3, §4.3
- tasks/probe3_multiplicity_memo.md §4 (power calibration)

Runs on sweep-runner-1 after remote `git pull`. Expected wall-clock ~10 min.
"""
from __future__ import annotations

import io
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# UTF-8 stdout (Windows cp1252 chokes on any non-ASCII symbol).
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import split_train_test  # noqa: E402

# ── Timeframe parameters (15m-specific) ──────────────────────────────────────
TF            = "15min"
BARS_PER_HOUR = 4
EX1_MAXHOLD   = 60 * BARS_PER_HOUR   # 60 hours × 4 bars/hour = 240 bars
EX2_TOD_UTC   = 19                    # UTC approximation of 15:00 ET (EDT-exact)

OUT_DIR       = REPO_ROOT / "data" / "ml" / "probe3" / "nc_15m"
BARS_PARQUET  = REPO_ROOT / "data" / f"NQ_{TF}.parquet"
READOUT_OUT   = REPO_ROOT / "data" / "ml" / "probe3" / "15m_nc.json"

# ── Gate thresholds (Probe 2 §4 three-gate set) ──────────────────────────────
YEARS_SPAN_TEST    = 1.4799
SHARPE_GATE        = 1.3
NTRADES_GATE       = 50
DOLLARS_GATE       = 5_000.0
NC_COMPOSITE_MAX_N = 2    # gate PASSES iff ≤ 2 of 16 cells clear three-gate set
GRID_N             = 16

# ── Base combo 865 (engine-grade values from Probe 2 1h manifest) ────────────
BASE_COMBO_865: dict = {
    "combo_id":                 865,
    "z_band_k":                 2.3043838978381697,
    "z_window":                 41,
    "volume_zscore_window":     47,
    "ema_fast":                 6,
    "ema_slow":                 48,
    "stop_method":              "fixed",
    "stop_fixed_pts":           17.017749351216196,
    "atr_multiplier":           None,
    "swing_lookback":           None,
    "min_rr":                   1.8476511046163293,
    "exit_on_opposite_signal":  False,
    "use_breakeven_stop":       False,
    "max_hold_bars":            120,
    "zscore_confirmation":      False,
    "z_input":                  "returns",
    "z_anchor":                 "rolling_mean",
    "z_denom":                  "parkinson",
    "z_type":                   "parametric",
    "z_window_2":               47,
    "z_window_2_weight":        0.3308286558845456,
    "volume_entry_threshold":   0.0,
    "vol_regime_lookback":      0,
    "vol_regime_min_pct":       0.0,
    "vol_regime_max_pct":       1.0,
    "session_filter_mode":      0,
    "tod_exit_hour":            0,
    "entry_timing_offset":      1,
    "fill_slippage_ticks":      1,
    "cooldown_after_exit_bars": 3,
}

# ── 4 EX variants (exit-rule overrides; session_filter_mode stays 0) ─────────
EX_VARIANTS: list[dict] = [
    {   # EX_0: native (combo 865's 1h-signed-off exit rules, transposed to 15m)
        "label":              "EX_0_native",
        "max_hold_bars":      120,   # 865 native value (preregistration §2.1)
        "use_breakeven_stop": False,
        "tod_exit_hour":      0,
    },
    {   # EX_1: 60-hour hold cap (= 240 bars at 15m)
        "label":              "EX_1_maxhold_60h",
        "max_hold_bars":      EX1_MAXHOLD,
        "use_breakeven_stop": False,
        "tod_exit_hour":      0,
    },
    {   # EX_2: TOD exit at ~15:00 ET (UTC approximation; see header disclosure)
        "label":              "EX_2_TOD_1500ET",
        "max_hold_bars":      120,
        "use_breakeven_stop": False,
        "tod_exit_hour":      EX2_TOD_UTC,
    },
    {   # EX_3: breakeven stop after 1R profit
        "label":              "EX_3_breakeven_after_1R",
        "max_hold_bars":      120,
        "use_breakeven_stop": True,
        "tod_exit_hour":      0,
    },
]

# ── 4 SES variants (ET windows; applied post-hoc as entry-time filter) ───────
def _ses_rth_only(minutes_since_midnight_et: np.ndarray) -> np.ndarray:
    # 09:30 ET (570) – 16:00 ET (960)
    return (minutes_since_midnight_et >= 570) & (minutes_since_midnight_et < 960)

def _ses_overnight(minutes_since_midnight_et: np.ndarray) -> np.ndarray:
    # 18:00 ET (1080) – 09:30 ET (570) NEXT day → wraparound
    return (minutes_since_midnight_et >= 1080) | (minutes_since_midnight_et < 570)

def _ses_rth_excl_lunch(minutes_since_midnight_et: np.ndarray) -> np.ndarray:
    # RTH-only minus 12:00 ET (720) – 14:00 ET (840)
    rth = _ses_rth_only(minutes_since_midnight_et)
    lunch = (minutes_since_midnight_et >= 720) & (minutes_since_midnight_et < 840)
    return rth & ~lunch

SES_VARIANTS: list[dict] = [
    {"label": "SES_0_all",             "filter": None},
    {"label": "SES_1_RTH_only",        "filter": _ses_rth_only},
    {"label": "SES_2_overnight_only",  "filter": _ses_overnight},
    {"label": "SES_3_RTH_excl_lunch",  "filter": _ses_rth_excl_lunch},
]


def _make_ex_combo(ex: dict) -> dict:
    """Build engine combo dict from EX override, combo_id = 10_100+ (disjoint)."""
    c = dict(BASE_COMBO_865)
    c["combo_id"]           = 10_100 + EX_VARIANTS.index(ex)
    c["max_hold_bars"]      = ex["max_hold_bars"]
    c["use_breakeven_stop"] = ex["use_breakeven_stop"]
    c["tod_exit_hour"]      = ex["tod_exit_hour"]
    return c


def _write_combos_and_run() -> dict[str, Path]:
    """Write combos.json + launch 4 engine runs, one per EX variant.

    Returns mapping {ex_label -> parquet path}.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parquets: dict[str, Path] = {}
    for ex in EX_VARIANTS:
        combo = _make_ex_combo(ex)
        combos_json = OUT_DIR / f"{ex['label']}_combos.json"
        parquet_out = OUT_DIR / f"{ex['label']}.parquet"
        combos_json.write_text(json.dumps([combo], indent=2, default=str))
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "param_sweep.py"),
            "--explicit-combos-json", str(combos_json),
            "--timeframe",            TF,
            "--eval-partition",       "test",
            "--output",               str(parquet_out),
            "--workers",              "1",
        ]
        print(f"[15m_nc] running EX variant {ex['label']}...")
        print(f"[15m_nc] $ {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"param_sweep.py for {ex['label']} returned {proc.returncode}"
            )
        parquets[ex["label"]] = parquet_out
    return parquets


def _load_test_bars_with_et_minutes() -> pd.DataFrame:
    """Load 15m bars test partition + attach et_minutes_since_midnight."""
    bars = pd.read_parquet(BARS_PARQUET)
    _, test_part = split_train_test(bars, 0.8)
    test_part = test_part.reset_index(drop=True)
    # Source CSV timestamps are CENTRAL TIME (Barchart export) — see
    # scripts/data_pipeline/update_bars_yfinance.py:37. The prior "UTC-naive"
    # comment here was wrong and propagated a TZ bug across Probe 3/4/Scope D.
    # Localize CT -> ET so 15m session cell boundaries are wall-clock correct.
    ts = pd.to_datetime(test_part["time"])
    if getattr(ts.dt, "tz", None) is None:
        ts_ct = ts.dt.tz_localize("America/Chicago", ambiguous="infer",
                                  nonexistent="shift_forward")
    else:
        ts_ct = ts.dt.tz_convert("America/Chicago")
    ts_et = ts_ct.dt.tz_convert("America/New_York")
    # Minutes since midnight (ET) — enables single-array range checks.
    et_min = ts_et.dt.hour * 60 + ts_et.dt.minute
    test_part = test_part.copy()
    test_part["_et_minutes"] = et_min.values.astype(np.int32)
    return test_part


def _compute_cell_gate(pnl: np.ndarray) -> dict:
    """Apply the Probe 2 §4 three-gate set to one cell's trade PnL."""
    n = int(pnl.size)
    if n < 2:
        return {
            "n_trades":             n,
            "net_sharpe":           0.0,
            "net_dollars_per_year": 0.0,
            "sharpe_pass":          False,
            "n_trades_pass":        n >= NTRADES_GATE,
            "dollars_pass":         False,
            "cell_pass":            False,
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
        "sharpe_pass":          bool(sharpe_pass),
        "n_trades_pass":        bool(n_trades_pass),
        "dollars_pass":         bool(dollars_pass),
        "cell_pass":            bool(sharpe_pass and n_trades_pass and dollars_pass),
    }


def _apply_cells(parquets: dict[str, Path], bars: pd.DataFrame) -> list[dict]:
    cells: list[dict] = []
    for ex in EX_VARIANTS:
        parquet_path = parquets[ex["label"]]
        if not parquet_path.exists():
            print(f"[15m_nc] WARN: {parquet_path} missing; treating as 0 trades.")
            for ses in SES_VARIANTS:
                cells.append({
                    "ex_label":  ex["label"],
                    "ses_label": ses["label"],
                    **_compute_cell_gate(np.array([])),
                })
            continue
        trades = pd.read_parquet(parquet_path)
        if len(trades) == 0:
            print(f"[15m_nc] {ex['label']}: zero trades from engine")
            for ses in SES_VARIANTS:
                cells.append({
                    "ex_label":  ex["label"],
                    "ses_label": ses["label"],
                    **_compute_cell_gate(np.array([])),
                })
            continue
        idx = trades["entry_bar_idx"].to_numpy()
        if idx.max() >= len(bars):
            raise RuntimeError(
                f"{ex['label']}: entry_bar_idx max {idx.max()} >= "
                f"len(test_part) {len(bars)}"
            )
        et_min = bars["_et_minutes"].to_numpy()[idx]
        pnl    = trades["net_pnl_dollars"].to_numpy()
        for ses in SES_VARIANTS:
            if ses["filter"] is None:
                mask = np.ones(len(pnl), dtype=bool)
            else:
                mask = ses["filter"](et_min)
            cell = _compute_cell_gate(pnl[mask])
            cells.append({
                "ex_label":  ex["label"],
                "ses_label": ses["label"],
                **cell,
            })
    return cells


def main() -> None:
    print(f"[15m_nc] launching {len(EX_VARIANTS)} engine runs on {TF} test")
    parquets = _write_combos_and_run()
    print(f"[15m_nc] loading {TF} bar cache + building ET minute index")
    bars = _load_test_bars_with_et_minutes()
    print(f"[15m_nc] applying 4 SES filters × 4 EX runs = {GRID_N} cells")
    cells = _apply_cells(parquets, bars)
    n_pass    = sum(1 for r in cells if r["cell_pass"])
    gate_pass = n_pass <= NC_COMPOSITE_MAX_N
    readout = {
        "gate_id":         "§4.3_15m_negative_control",
        "timeframe":       TF,
        "grid_n":          GRID_N,
        "n_pass":          int(n_pass),
        "threshold_max":   NC_COMPOSITE_MAX_N,
        "years_span_test": YEARS_SPAN_TEST,
        "ex2_tod_utc_hour": EX2_TOD_UTC,
        "ex2_approximation_note": (
            "Preregistration §2.4 EX_2 specifies 'tod_exit_hour=15' with "
            "intent '15:00 ET'; engine's tod_exit_hour is UTC hour-of-day. "
            f"Applied tod_exit_hour={EX2_TOD_UTC} UTC = 15:00 EDT exactly / "
            "14:00 EST (1h early). Test window ~53% EDT / 47% EST."
        ),
        "ex1_maxhold_bars": EX1_MAXHOLD,
        "thresholds": {
            "net_sharpe":           SHARPE_GATE,
            "n_trades":             NTRADES_GATE,
            "net_dollars_per_year": DOLLARS_GATE,
            "composite_max_n":      NC_COMPOSITE_MAX_N,
        },
        "cells":           cells,
        "gate_pass":       bool(gate_pass),
    }
    READOUT_OUT.parent.mkdir(parents=True, exist_ok=True)
    READOUT_OUT.write_text(json.dumps(readout, indent=2, default=str))
    print(f"[15m_nc] wrote {READOUT_OUT}")
    print(f"[15m_nc] n_pass = {n_pass} / {GRID_N}  "
          f"(threshold_max = {NC_COMPOSITE_MAX_N})")
    print(f"[15m_nc] GATE §4.3 -> "
          f"{'PASS (friction story holds)' if gate_pass else 'FAIL (rescue)'}")


if __name__ == "__main__":
    main()
