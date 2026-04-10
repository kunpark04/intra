"""
rerun_and_benchmark.py — Rerun V1 and V2 with $50k starting equity and
measure Cython vs Numba vs NumPy core-loop performance on real training data.

Usage (from repo root):
    python scripts/rerun_and_benchmark.py
"""
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import src.config as cfg
from src.data_loader import load_bars, split_train_test
from src.indicators import add_indicators
from src.strategy import generate_signals
from src.backtest import (
    run_backtest,
    _backtest_core_numpy,
    CYTHON_AVAILABLE,
    NUMBA_AVAILABLE,
)
from src.reporting import save_iteration

# ── V1 and V2 parameter sets (equity already set to 50k in config) ─────────────
VERSIONS = {
    "V1": {"MIN_RR": 2.0, "STOP_FIXED_PTS": 30.0},
    "V2": {"MIN_RR": 3.0, "STOP_FIXED_PTS": 20.0},
}

SEP = "-" * 60


def patch_cfg(overrides: dict) -> None:
    """Apply key=value overrides onto the config module in-place."""
    for k, v in overrides.items():
        setattr(cfg, k, v)


def make_core_inputs(train):
    """Extract the raw arrays needed by all three core functions."""
    open_arr   = train["open"].to_numpy(dtype=np.float64)
    high_arr   = train["high"].to_numpy(dtype=np.float64)
    low_arr    = train["low"].to_numpy(dtype=np.float64)
    close_arr  = train["close"].to_numpy(dtype=np.float64)
    signal_arr = train["signal"].to_numpy(dtype=np.int8)
    n_bars     = len(train)
    sl_pts     = float(cfg.STOP_FIXED_PTS)
    tp_pts     = sl_pts * float(cfg.MIN_RR)
    same_bar   = cfg.SAME_BAR_COLLISION == "tp_first"
    exit_opp   = bool(cfg.EXIT_ON_OPPOSITE_SIGNAL)
    return (open_arr, high_arr, low_arr, close_arr,
            signal_arr, n_bars, sl_pts, tp_pts, same_bar, exit_opp)


def time_fn(fn, args, n_runs: int = 3) -> float:
    """Return mean wall-clock seconds over n_runs calls."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def main():
    print(SEP)
    print("  Starting equity : $50,000")
    print(SEP)

    # ── 1. Load data once ─────────────────────────────────────────────────────
    print("\n[1/4] Loading bars and computing indicators/signals...")
    t0 = time.perf_counter()
    df = load_bars("data/NQ_1min.csv")
    train, _ = split_train_test(df, cfg.TRAIN_RATIO)
    train = add_indicators(train, cfg)
    train = generate_signals(train, cfg)
    print(f"      {len(train):,} training bars ready  ({time.perf_counter()-t0:.1f}s)")

    # ── 2. Benchmark core engines ─────────────────────────────────────────────
    print(f"\n[2/4] Benchmarking core engines (3 runs each on {len(train):,} bars)...")

    core_args = make_core_inputs(train)
    N_RUNS = 3
    results_bench = {}

    # Cython
    if CYTHON_AVAILABLE:
        from src.cython_ext.backtest_core import backtest_core_cy
        t_cy = time_fn(backtest_core_cy, core_args, N_RUNS)
        results_bench["Cython (AOT)"] = t_cy
        print(f"      Cython AOT    : {t_cy:.3f}s avg")
    else:
        print("      Cython        : NOT AVAILABLE")

    # Numba (warm up first to exclude JIT compilation from timing)
    if NUMBA_AVAILABLE:
        from src.backtest import _backtest_core
        print("      Numba         : warming up JIT...", end=" ", flush=True)
        _backtest_core(*core_args)                       # JIT compile pass
        print("done")
        t_nb = time_fn(_backtest_core, core_args, N_RUNS)
        results_bench["Numba (JIT)"] = t_nb
        print(f"      Numba JIT     : {t_nb:.3f}s avg")
    else:
        print("      Numba         : NOT AVAILABLE")

    # NumPy fallback
    print("      NumPy fallback: running...", end=" ", flush=True)
    t_np = time_fn(_backtest_core_numpy, core_args, N_RUNS)
    results_bench["NumPy (fallback)"] = t_np
    print(f"{t_np:.3f}s avg")

    # Speedup summary
    print(f"\n{SEP}")
    print("  ENGINE SPEEDUP SUMMARY")
    print(SEP)
    baseline = results_bench.get("NumPy (fallback)", 1.0)
    for name, t in results_bench.items():
        speedup = baseline / t
        print(f"  {name:<22} {t:.3f}s   {speedup:6.1f}x vs NumPy")
    if "Cython (AOT)" in results_bench and "Numba (JIT)" in results_bench:
        cy_vs_nb = results_bench["Numba (JIT)"] / results_bench["Cython (AOT)"]
        print(f"\n  Cython vs Numba: {cy_vs_nb:.2f}x faster (post-warmup)")
    print(SEP)

    # ── 3. Run V1 and V2 backtests ────────────────────────────────────────────
    print("\n[3/4] Running V1 and V2 backtests with $50k equity...\n")

    for version, overrides in VERSIONS.items():
        patch_cfg(overrides)
        print(f"  [{version}] MIN_RR={cfg.MIN_RR}  SL={cfg.STOP_FIXED_PTS}pts  "
              f"Equity=${cfg.STARTING_EQUITY:,.0f}")

        t0 = time.perf_counter()
        # Regenerate signals in case they depend on config (they don't currently,
        # but be explicit for correctness)
        result = run_backtest(train, cfg, version=version)
        elapsed = time.perf_counter() - t0

        n   = result["n_trades"]
        eq  = result["final_equity"]
        wins = sum(1 for t in result["trades"] if t["label_win"])
        wr  = wins / n * 100 if n else 0.0
        ret = (eq / cfg.STARTING_EQUITY - 1) * 100

        print(f"         Trades: {n:,}  Win rate: {wr:.1f}%  "
              f"Final equity: ${eq:,.2f}  Return: {ret:+.2f}%  ({elapsed:.1f}s)")

        save_iteration(version, result, train, cfg)
        print()

    # ── 4. Regenerate analysis notebooks ─────────────────────────────────────
    print("[4/4] Regenerating and executing analysis notebooks...")
    import importlib
    import subprocess
    subprocess.run([sys.executable, "scripts/gen_analysis_notebook.py"], check=True)
    subprocess.run([sys.executable, "scripts/exec_analysis.py"], check=True)

    print(f"\n{SEP}")
    print("  All done.")
    print(SEP)


if __name__ == "__main__":
    main()
