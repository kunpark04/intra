"""Run adaptive_vs_fixed with the held-out model against v10_9955."""
import importlib.util, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location(
    "avf", REPO / "scripts/adaptive_vs_fixed_backtest.py")
avf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(avf)

# Override model + output paths
avf.MODEL_PATH = REPO / "data/ml/adaptive_rr_heldout/adaptive_rr_model.txt"
avf.OUT_PATH = REPO / "data/ml/adaptive_rr_heldout/adaptive_vs_fixed_heldout.json"

avf.main()
