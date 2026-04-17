"""Pull executed evaluation notebooks — artifacts are published via git by
sweep-runner-1's run_eval_nbs.sh wrapper. This script just runs `git pull`."""
from __future__ import annotations
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def main() -> None:
    print(f"git -C {REPO} pull --ff-only origin master")
    subprocess.run(
        ["git", "-C", str(REPO), "pull", "--ff-only", "origin", "master"],
        check=True,
    )
    for rel in ("evaluation/top_performance.ipynb", "evaluation/top_trade_log.xlsx"):
        p = REPO / rel
        print(f"  {rel}: {p.stat().st_size:,} B")


if __name__ == "__main__":
    main()
