"""
run_backtest.py — CLI entry point for running a training backtest iteration.

Usage (from repo root):
    python scripts/run_backtest.py --version V1
    python scripts/run_backtest.py --version V2 --data data/NQ_1min.csv
"""
import argparse
import sys
import time
from pathlib import Path

# Ensure repo root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import src.config as cfg
from src.data_loader import load_bars, split_train_test
from src.indicators import add_indicators
from src.strategy import generate_signals
from src.backtest import run_backtest
from src.reporting import save_iteration


def main():
    """CLI entry point: run one iteration version end-to-end (training partition)."""
    parser = argparse.ArgumentParser(description="Run MNQ 1-min EMA crossover backtest (training partition only).")
    parser.add_argument("--version", default="V1", help="Iteration version label, e.g. V1, V2 (default: V1)")
    parser.add_argument("--data", default="data/NQ_1min.csv", help="Path to NQ 1-min CSV (default: data/NQ_1min.csv)")
    args = parser.parse_args()

    t0 = time.time()

    print(f"[run_backtest] version={args.version}  data={args.data}")

    # 1. Load data
    print("Loading bars...")
    df = load_bars(args.data)
    train, test = split_train_test(df, cfg.TRAIN_RATIO)
    print(f"  Total bars: {len(df):,}  |  Train: {len(train):,}  |  Test: {len(test):,}")

    # 2. Indicators
    print("Computing indicators...")
    train = add_indicators(train, cfg)

    # 3. Signals
    print("Generating signals...")
    train = generate_signals(train, cfg)
    n_signals = (train["signal"] != 0).sum()
    print(f"  Signals: {n_signals:,} ({n_signals/len(train)*100:.2f}% of bars)")

    # 4. Backtest
    print("Running backtest (training partition only)...")
    results = run_backtest(train, cfg, version=args.version)
    n_trades = results["n_trades"]
    final_eq = results["final_equity"]
    wins = sum(1 for t in results["trades"] if t["label_win"])
    wr = wins / n_trades * 100 if n_trades else 0.0
    print(f"  Trades: {n_trades:,}  |  Wins: {wins:,}  |  Win rate: {wr:.1f}%  |  Final equity: ${final_eq:,.2f}")

    # 5. Save artifacts
    print(f"Saving artifacts to iterations/{args.version}/...")
    out_dir = save_iteration(args.version, results, train, cfg)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s  ->  {out_dir}")


if __name__ == "__main__":
    main()
