"""Quick check that the combo-agnostic V3 refit artifacts exist on
sweep-runner-1 before launching the ship-blocker audit.

Two modes (SSH mirrors the V4 template; local is an extension for use
after SFTP'ing the booster back to the workstation):

  python tasks/_check_v3_no_gcid_artifact.py
      - SSH to sweep-runner-1, ls the remote v3_no_gcid artifact dir,
        cat head of metrics_v3.json, assert booster + calibrator files
        exist. Non-zero exit if missing.

  python tasks/_check_v3_no_gcid_artifact.py --local PATH_TO_BOOSTER_TXT
      - Load a locally-available booster_v3.txt via LightGBM, assert
        `global_combo_id` is NOT in `booster.feature_name()` (the
        *structural* anti-leak invariant for the combo-agnostic refit),
        and that the feature count is in a sensible range (>10, <100).
        Also reads metrics_v3.json next to the booster if present and
        prints OOF AUC (sanity bar: >0.80 — V3 production sits near 0.85
        and V4 no-gcid came in at 0.8463).

The local-mode assertion set mirrors the Phase 2.2 checklist in
`tasks/plan_v3_audit_and_ranker_null.md`:
  - booster loads via `lgb.Booster(model_file=...)`
  - `feature_name()` does not contain `"global_combo_id"`
  - AUC within a sanity band (explicit threshold: OOF AUC > 0.80)
  - ECE per-RR-bin < 1e-4 when present in metrics_v3.json

Feature-count thresholds are deliberately soft (>10, <100) rather than
an exact match to V3: the shipped V3 feature list and the combo-agnostic
list differ by exactly one element (`global_combo_id`), but we assert
the anti-leak invariant explicitly rather than relying on a count match.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import paramiko

HOST = "195.88.25.157"
USER = "root"
PASS = "J@maicanP0wer123"
REMOTE = "/root/intra/data/ml/adaptive_rr_v3_no_gcid"


def ssh_mode() -> int:
    """Mirror the V4 template: ls remote dir + presence checks."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=15)

    _, so, _ = ssh.exec_command(f"ls -la {REMOTE} 2>&1")
    listing = so.read().decode()
    print(f"=== ls -la {REMOTE} ===")
    print(listing)

    _, so, _ = ssh.exec_command(
        f"test -f {REMOTE}/booster_v3.txt && echo HAVE_BOOSTER || echo NO_BOOSTER"
    )
    booster_status = so.read().decode().strip()
    print(f"booster: {booster_status}")

    _, so, _ = ssh.exec_command(
        f"test -f {REMOTE}/isotonic_calibrators_v3.json && echo HAVE_CAL || echo NO_CAL"
    )
    cal_status = so.read().decode().strip()
    print(f"calibrators: {cal_status}")

    _, so, _ = ssh.exec_command(f"cat {REMOTE}/metrics_v3.json 2>/dev/null | head -40")
    metrics = so.read().decode()
    if metrics:
        print(f"=== metrics_v3.json (head) ===")
        print(metrics)

    ssh.close()

    return 0 if booster_status == "HAVE_BOOSTER" and cal_status == "HAVE_CAL" else 2


def local_mode(booster_path: Path) -> int:
    """Load booster via LightGBM + enforce structural anti-leak invariants."""
    import lightgbm as lgb

    if not booster_path.is_file():
        print(f"[fail] booster not found: {booster_path}")
        return 2

    print(f"[check] loading booster: {booster_path}")
    booster = lgb.Booster(model_file=str(booster_path))

    feature_names = booster.feature_name()
    print(f"[check] feature count: {len(feature_names)}")
    print(f"[check] features: {feature_names}")

    failures: list[str] = []

    # Anti-leak invariant: global_combo_id must be absent.
    if "global_combo_id" in feature_names:
        failures.append("global_combo_id present in booster.feature_name() — "
                        "combo-agnostic refit is contaminated")
    else:
        print("[ok] global_combo_id NOT in feature_name()")

    # Sanity band on feature count (V3 should be ~20-30 features; leave
    # room for Family A + any minor drift). Exact match to V3 is not
    # asserted because the anti-leak invariant above is load-bearing.
    if not (10 < len(feature_names) < 100):
        failures.append(f"feature count {len(feature_names)} outside sanity band (10, 100)")
    else:
        print(f"[ok] feature count {len(feature_names)} in sanity band (10, 100)")

    # Metrics sidecar (optional but expected).
    metrics_path = booster_path.parent / "metrics_v3.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text())
        print(f"[check] metrics_v3.json keys: {sorted(metrics.keys())}")

        auc = metrics.get("oof_auc") or metrics.get("auc") or metrics.get("AUC")
        if auc is not None:
            print(f"[check] OOF AUC: {auc}")
            if auc <= 0.80:
                failures.append(f"OOF AUC {auc} <= 0.80 sanity bar "
                                "(V3 production ~0.85, V4 no-gcid 0.8463)")
            else:
                print("[ok] OOF AUC > 0.80")
        else:
            print("[warn] AUC key not found in metrics_v3.json — skipping AUC check")

        # ECE per-RR-bin check — mirror V4's threshold.
        ece_per_rr = metrics.get("ece_per_rr") or metrics.get("ece_cal_per_rr")
        if isinstance(ece_per_rr, dict):
            max_ece = max(float(v) for v in ece_per_rr.values())
            print(f"[check] max per-RR-bin ECE: {max_ece}")
            if max_ece >= 1e-4:
                failures.append(f"max per-RR-bin ECE {max_ece} >= 1e-4 "
                                "(V4 no-gcid came in at 8.14e-7)")
            else:
                print("[ok] max per-RR-bin ECE < 1e-4")
        else:
            print("[warn] ece_per_rr key not found — skipping ECE check")
    else:
        print(f"[warn] {metrics_path} not found — skipping AUC + ECE checks")

    # Calibrators sidecar.
    cal_path = booster_path.parent / "isotonic_calibrators_v3.json"
    if cal_path.is_file():
        print(f"[ok] found {cal_path}")
    else:
        failures.append(f"missing isotonic calibrators at {cal_path}")

    if failures:
        print("\n[FAIL] check failed:")
        for f in failures:
            print(f"  - {f}")
        return 2
    print("\n[PASS] combo-agnostic V3 artifact checks all cleared")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local", type=Path, default=None,
                   help="Path to a locally-available booster_v3.txt "
                        "(after SFTP). Runs structural anti-leak checks.")
    args = p.parse_args()

    if args.local is not None:
        return local_mode(args.local)
    return ssh_mode()


if __name__ == "__main__":
    sys.exit(main())
