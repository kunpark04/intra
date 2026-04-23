"""Probe 4 — run one combo on 1h test partition (SES_0 all-sessions only).

Invoked twice by `tasks/_run_probe4_remote.py`:
  --combo-id 1298
  --combo-id  664

Engine runs at session_filter_mode=0 (all sessions). Per-session (SES_1 RTH /
SES_2 GLOBEX) decomposition is performed post-hoc in `_probe4_readout.py` using
`tz_convert("America/New_York")` on the 1h bar timeline — the engine's built-in
`session_filter_mode` branches filter by raw UTC hour-of-day (bar_hour is
tz-naive at scripts/param_sweep.py:1567), which cannot reproduce the
preregistration §4.4 ET-minute definitions (09:30-16:00 ET). See
`_probe3_1h_ritual.py:124-140, 180-193, 229-271` for the canonical pattern.

For each invocation:
  1. Load the frozen parameter dict for `combo_id`.
  2. Assign a disjoint engine-side `combo_id` (ENGINE_ID_BASE + CID) for
     parquet idempotency.
  3. Invoke `scripts/param_sweep.py --explicit-combos-json ... --timeframe 1h
     --eval-partition test`.
  4. Aggregate the resulting trade parquet into SES_0 metrics JSON (net_sharpe,
     n_trades, net_dollars_per_year, sub-gate booleans).
  5. Emit a per-trade parquet (`{tag}_trades.parquet`) with net_pnl_dollars and
     entry_bar_idx — consumed by the readout for both the §4.3 Welch-t primary
     gate AND the §4.4 post-hoc ET-minute session decomposition.

Absolute sub-gates (§4.1 / §4.2 of the preregistration):
  net_sharpe >= 1.3  AND  n_trades >= 50  AND  net_$/yr >= 5_000

Friction model: $5/contract RT (baked into v11 sweep engine PnL).
Sizing: fixed $500 risk per trade.

References:
- tasks/probe4_preregistration.md §2.1, §2.2, §4.1, §4.2, §4.4, §8.2, §8.3
- tasks/_probe4_param_dicts.json (cached dicts, single source of truth)
- Precedent: tasks/_probe3_1h_ritual.py (post-hoc ET-minute session filter)

Runs on sweep-runner-1 after remote `git pull`. Wall-clock per invocation ~2 min.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# UTF-8 stdout (Windows cp1252 chokes on any non-ASCII symbol).
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Frozen partition / gate constants
TF                = "1h"
# YEARS_SPAN_TEST inherited unchanged from Probe 3 for cross-probe continuity.
# True 1h partition span is ~533 days = 1.4593 y; 1.4799 biases Sharpe ~1.3%
# conservative. Per `feedback_years_span_cross_tf.md`: document, do not "fix".
YEARS_SPAN_TEST   = 1.4799
SHARPE_GATE       = 1.3
NTRADES_GATE      = 50
DOLLARS_GATE      = 5_000.0

OUT_DIR           = REPO_ROOT / "data" / "ml" / "probe4"
PARAM_DICTS_JSON  = REPO_ROOT / "tasks" / "_probe4_param_dicts.json"
V11_1H_PARQUET    = REPO_ROOT / "data" / "ml" / "originals" / "ml_dataset_v11_1h.parquet"

VALID_COMBO_IDS   = {1298, 664}

# Engine-side combo_ids must be disjoint from the v11 sampler namespace and
# from Probe 3's 10_000-band (`10_000 + i` for param_nbhd, `10_200 + i` for
# 1h_ritual). 20_000 + CID keeps per-run parquets idempotent across re-launches.
ENGINE_ID_BASE    = 20_000


def _load_param_dict(combo_id: int) -> dict:
    """Resolve the frozen parameter dict for `combo_id`.

    Prefers the cached JSON (single source of truth; committed alongside the
    preregistration). Falls back to parquet lookup if the JSON is missing.
    """
    if PARAM_DICTS_JSON.exists():
        dicts = json.loads(PARAM_DICTS_JSON.read_text())
        key = str(combo_id)
        if key not in dicts:
            raise KeyError(
                f"combo_id {combo_id} missing from {PARAM_DICTS_JSON}. "
                f"Expected keys: {sorted(dicts.keys())}."
            )
        return dict(dicts[key])

    print(f"[probe4_run] {PARAM_DICTS_JSON} missing; falling back to parquet.")
    df = pd.read_parquet(V11_1H_PARQUET)
    sub = df[df["combo_id"] == combo_id]
    if len(sub) == 0:
        raise RuntimeError(
            f"combo_id {combo_id} not found in {V11_1H_PARQUET}."
        )
    param_cols = [
        "z_band_k", "z_window", "z_input", "z_anchor", "z_denom", "z_type",
        "z_window_2", "z_window_2_weight", "volume_zscore_window", "ema_fast",
        "ema_slow", "stop_method", "stop_fixed_pts", "atr_multiplier",
        "swing_lookback", "min_rr", "max_hold_bars", "exit_on_opposite_signal",
        "use_breakeven_stop", "zscore_confirmation", "volume_entry_threshold",
        "vol_regime_lookback", "vol_regime_min_pct", "vol_regime_max_pct",
        "session_filter_mode", "tod_exit_hour", "entry_timing_offset",
        "fill_slippage_ticks", "cooldown_after_exit_bars",
    ]
    row = sub.iloc[0]
    out: dict = {}
    for col in param_cols:
        val = row[col]
        if isinstance(val, float) and math.isnan(val):
            out[col] = None
        elif hasattr(val, "item"):
            out[col] = val.item()
        else:
            out[col] = val
    return out


def _build_engine_combo(combo_id: int, base: dict) -> dict:
    """Force session_filter_mode=0 (all-sessions) + assign disjoint engine id.

    Engine-side UTC-hour filters (session_filter_mode ∈ {1,2,3}) cannot
    reproduce the preregistration §4.4 ET-minute session labels; leaving it
    at 0 here forces the readout to partition post-hoc via tz_convert.
    """
    c = dict(base)
    c["session_filter_mode"] = 0
    c["combo_id"]            = ENGINE_ID_BASE + combo_id
    return c


def _run_sweep(combos_json: Path, parquet_out: Path) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "param_sweep.py"),
        "--explicit-combos-json", str(combos_json),
        "--timeframe",            TF,
        "--eval-partition",       "test",
        "--output",               str(parquet_out),
        "--workers",              "1",
    ]
    print(f"[probe4_run] $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"param_sweep.py returned {proc.returncode}")


def _aggregate_metrics(pnl: np.ndarray) -> dict:
    """Compute net_sharpe, n_trades, net_$/yr + sub-gate booleans from PnL series."""
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


def _extract_trades_slim(trades: pd.DataFrame) -> pd.DataFrame:
    """Per-trade slice for readout input: net_pnl_dollars + entry_bar_idx.

    entry_bar_idx is partition-relative (same convention as Probe 3 parquets);
    the readout resolves it to an ET minute via tz_convert on the 1h bar
    timeline.
    """
    keep = ["net_pnl_dollars", "entry_bar_idx"]
    if "exit_bar_idx" in trades.columns:
        keep.append("exit_bar_idx")
    if "entry_time" in trades.columns:
        keep.append("entry_time")
    if "exit_time" in trades.columns:
        keep.append("exit_time")
    return trades[keep].copy()


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe 4 — one combo SES_0 run.")
    ap.add_argument("--combo-id", type=int, required=True,
                    help=f"combo_id in {sorted(VALID_COMBO_IDS)}")
    args = ap.parse_args()

    if args.combo_id not in VALID_COMBO_IDS:
        raise ValueError(f"--combo-id {args.combo_id} not in {VALID_COMBO_IDS}")

    cid = args.combo_id
    tag = f"combo{cid}_SES_0"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combos_json  = OUT_DIR / f"{tag}_combos.json"
    parquet_out  = OUT_DIR / f"{tag}.parquet"
    json_out     = OUT_DIR / f"{tag}.json"
    trades_out   = OUT_DIR / f"{tag}_trades.parquet"

    print(f"[probe4_run] combo_id={cid}  session_filter_mode=0 (SES_0 baseline)  tag={tag}")

    base  = _load_param_dict(cid)
    combo = _build_engine_combo(cid, base)
    combos_json.write_text(json.dumps([combo], indent=2, default=str))
    print(f"[probe4_run] wrote engine combo -> {combos_json}  "
          f"(engine_combo_id={combo['combo_id']})")

    _run_sweep(combos_json, parquet_out)

    if not parquet_out.exists():
        raise RuntimeError(f"engine did not produce {parquet_out}")
    trades = pd.read_parquet(parquet_out)
    print(f"[probe4_run] engine produced {len(trades)} trades")

    if len(trades) == 0:
        pnl = np.array([])
        gross_dollars    = 0.0
        friction_dollars = 0.0
    else:
        pnl = trades["net_pnl_dollars"].to_numpy()
        gross_dollars    = (
            float(trades["gross_pnl_dollars"].sum())
            if "gross_pnl_dollars" in trades.columns else float("nan")
        )
        friction_dollars = (
            float(trades["friction_dollars"].sum())
            if "friction_dollars" in trades.columns else float("nan")
        )

    metrics = _aggregate_metrics(pnl)

    readout = {
        "probe":              "probe4",
        "combo_id":           cid,
        "session_filter_mode": 0,
        "engine_combo_id":    combo["combo_id"],
        "timeframe":          TF,
        "years_span_test":    YEARS_SPAN_TEST,
        "thresholds": {
            "net_sharpe":           SHARPE_GATE,
            "n_trades":             NTRADES_GATE,
            "net_dollars_per_year": DOLLARS_GATE,
        },
        "gross_dollars":      gross_dollars,
        "friction_dollars":   friction_dollars,
        "metrics":            metrics,
        "source_parquet":     str(parquet_out.relative_to(REPO_ROOT)),
    }
    json_out.write_text(json.dumps(readout, indent=2, default=str))
    print(f"[probe4_run] wrote {json_out}")

    if len(trades) > 0:
        slim = _extract_trades_slim(trades)
        slim.to_parquet(trades_out, index=False)
        print(f"[probe4_run] wrote {trades_out}  (n={len(slim)} trades)")
    else:
        # Empty-trades shim so the readout can distinguish "engine ran, zero
        # trades" from "invocation skipped". Readout's insufficient-n branch
        # handles n_a==0 cleanly.
        pd.DataFrame({"net_pnl_dollars": pd.Series(dtype="float64"),
                      "entry_bar_idx":   pd.Series(dtype="int64")}).to_parquet(
            trades_out, index=False
        )
        print(f"[probe4_run] wrote empty {trades_out} (engine returned 0 trades)")

    print(f"[probe4_run] metrics: sharpe={metrics['net_sharpe']:.3f}  "
          f"n={metrics['n_trades']}  $/yr={metrics['net_dollars_per_year']:.0f}  "
          f"abs_pass={metrics['abs_pass']}")


if __name__ == "__main__":
    main()
