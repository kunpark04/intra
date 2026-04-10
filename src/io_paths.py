from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
ITERATIONS_DIR = REPO_ROOT / "iterations"
EVALUATION_DIR = REPO_ROOT / "evaluation"

def iteration_dir(version: str) -> Path:
    return ITERATIONS_DIR / version

def evaluation_dir() -> Path:
    return EVALUATION_DIR
