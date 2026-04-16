"""Centralised filesystem path constants for repo-relative artifacts.

Importers get the same `REPO_ROOT`, `DATA_DIR`, `ITERATIONS_DIR`, and
`EVALUATION_DIR` regardless of the caller's current working directory,
which keeps notebook and script artifacts from landing in the wrong place.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
ITERATIONS_DIR = REPO_ROOT / "iterations"
EVALUATION_DIR = REPO_ROOT / "evaluation"

def iteration_dir(version: str) -> Path:
    """Return the artifacts folder for a given iteration version.

    Args:
        version: Iteration label like `"V1"`, `"V2"`, `"V3"`.

    Returns:
        Path to `iterations/<version>/` (not created by this function).
    """
    return ITERATIONS_DIR / version

def evaluation_dir() -> Path:
    """Return the single hold-out evaluation artifact folder.

    Returns:
        Path to `evaluation/` — the one-shot final hold-out run location.
    """
    return EVALUATION_DIR
