"""B9: V2 adaptive R:R with monotonic constraint on candidate_rr and rr_x_atr.

Forces P(win) to be monotonically decreasing in candidate_rr (higher R:R → lower
win probability). Runs same training pipeline as V2 but with constraint added.
"""
import importlib.util, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location(
    "v2", REPO / "scripts/adaptive_rr_model_v2.py")
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)

# Override output dir
v2.OUTPUT_DIR = v2.DATA_DIR / "adaptive_rr_monotonic_v2"
v2.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Build monotone constraints list aligned with ALL_FEATURES order.
# Only candidate_rr and rr_x_atr get -1 (decreasing); others = 0 (unconstrained).
constraints = []
for f in v2.ALL_FEATURES:
    if f in ("candidate_rr", "rr_x_atr"):
        constraints.append(-1)
    else:
        constraints.append(0)

v2.LGB_PARAMS = {**v2.LGB_PARAMS,
                 "monotone_constraints": constraints,
                 "monotone_constraints_method": "advanced"}

print(f"B9: monotone_constraints = {constraints}")
print(f"B9: features = {v2.ALL_FEATURES}")
print(f"B9: output dir = {v2.OUTPUT_DIR}")

v2.main()
