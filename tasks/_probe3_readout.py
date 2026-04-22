"""Probe 3 aggregate readout — §5 branch routing over 4 gate JSONs.

Reads:
  - data/ml/probe3/regime_halves.json   (gate §4.1)
  - data/ml/probe3/param_nbhd.json      (gate §4.2)
  - data/ml/probe3/15m_nc.json          (gate §4.3)
  - data/ml/probe3/1h_ritual.json       (gate §4.4)

Counts F = number of FAILs across the four gates and routes:
  F = 0  → PAPER_TRADE        (memo §6 posterior ≈ 0.91)
  F = 1  → COUNCIL_RECONVENE  (memo §6 gate-specific posterior lookup)
  F ≥ 2  → SUNSET_OPTION_Z    (memo §6 posterior < 0.06)

Writes data/ml/probe3/readout.json with machine-readable verdict record.

Mirrors tasks/_probe2_readout.py structure. Runs on sweep-runner-1
(or locally — reads JSON only; no engine compute). Wall-clock < 1 s.

References:
- tasks/probe3_preregistration.md §3.3, §5
- tasks/probe3_multiplicity_memo.md §6 (Bayes-factor table)
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

# UTF-8 stdout (Windows cp1252 chokes on any non-ASCII symbol).
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent

GATE_FILES: dict[str, Path] = {
    "§4.1_regime_halves":          REPO_ROOT / "data" / "ml" / "probe3" / "regime_halves.json",
    "§4.2_param_nbhd":             REPO_ROOT / "data" / "ml" / "probe3" / "param_nbhd.json",
    "§4.3_15m_negative_control":   REPO_ROOT / "data" / "ml" / "probe3" / "15m_nc.json",
    "§4.4_1h_session_exit_ritual": REPO_ROOT / "data" / "ml" / "probe3" / "1h_ritual.json",
}
READOUT_OUT = REPO_ROOT / "data" / "ml" / "probe3" / "readout.json"

# ── Posterior lookup per memo §6 table (prior = 0.167) ───────────────────────
# When F = 1, posterior depends on which single gate failed; table from
# memo §6.4. When F ≥ 2, posterior < 0.06 (sunset triggered).
POSTERIOR_F0                = 0.91
POSTERIOR_F1_REGIME_HALVES  = 0.055   # 3/4 PASS, regime halves FAIL
POSTERIOR_F1_PARAM_NBHD     = 0.038   # 3/4 PASS, param nbhd FAIL
POSTERIOR_F1_15M_NC         = 0.05    # approx — memo §6.4 lumps ≤ sunset range
POSTERIOR_F1_SESSION_EXIT   = 0.10    # approx — memo §7 "possible reduced-deployment"
POSTERIOR_F2                = 0.01
POSTERIOR_F3                = 0.002
POSTERIOR_F4                = 0.0005  # prior * ~tiny BF

POSTERIOR_F1_BY_FAIL_GATE: dict[str, float] = {
    "§4.1_regime_halves":          POSTERIOR_F1_REGIME_HALVES,
    "§4.2_param_nbhd":             POSTERIOR_F1_PARAM_NBHD,
    "§4.3_15m_negative_control":   POSTERIOR_F1_15M_NC,
    "§4.4_1h_session_exit_ritual": POSTERIOR_F1_SESSION_EXIT,
}


def _load_gate(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Gate readout missing: {path}. Run the corresponding _probe3_*.py "
            f"before invoking this readout."
        )
    return json.loads(path.read_text())


def _posterior_for(F: int, failing_gates: list[str]) -> float:
    if F == 0:
        return POSTERIOR_F0
    if F == 1:
        return POSTERIOR_F1_BY_FAIL_GATE.get(failing_gates[0], POSTERIOR_F2)
    if F == 2:
        return POSTERIOR_F2
    if F == 3:
        return POSTERIOR_F3
    return POSTERIOR_F4


def _branch_for(F: int) -> str:
    if F == 0:
        return "PAPER_TRADE"
    if F == 1:
        return "COUNCIL_RECONVENE"
    return "SUNSET_OPTION_Z"


def main() -> None:
    gates: dict[str, dict] = {}
    per_gate_summary: list[dict] = []
    for gate_id, path in GATE_FILES.items():
        readout = _load_gate(path)
        gates[gate_id] = readout
        per_gate_summary.append({
            "gate_id":    gate_id,
            "gate_pass":  bool(readout.get("gate_pass")),
            "source_file": str(path.relative_to(REPO_ROOT)),
        })

    failing_gates = [g["gate_id"] for g in per_gate_summary if not g["gate_pass"]]
    F             = len(failing_gates)
    branch        = _branch_for(F)
    posterior     = _posterior_for(F, failing_gates)

    out = {
        "probe":      "probe3_combo865_session_exit",
        "preregistration_signed_commit": "8636167",
        "base_commit": "f8447af",
        "gates_summary": per_gate_summary,
        "failing_gates": failing_gates,
        "F":           F,
        "branch":      branch,
        "posterior":   posterior,
        "posterior_prior": 0.167,
        "posterior_source": (
            "tasks/probe3_multiplicity_memo.md §6.4 Bayes-factor table"
        ),
        "gates":      gates,
    }
    READOUT_OUT.parent.mkdir(parents=True, exist_ok=True)
    READOUT_OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"[readout] wrote {READOUT_OUT}")
    print(f"[readout] F = {F}  branch = {branch}  posterior = {posterior}")
    if failing_gates:
        print(f"[readout] failing: {failing_gates}")
    else:
        print(f"[readout] all 4 gates PASS")


if __name__ == "__main__":
    main()
