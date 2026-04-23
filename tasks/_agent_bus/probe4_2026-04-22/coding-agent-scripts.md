---
from: coding-agent
run_id: probe4_2026-04-22
pass: 3
timestamp: 2026-04-23T00:00:00Z
for: [code-logic-reviewer, user]
task_summary: Authored the three Probe 4 execution scripts per signed preregistration at commit 432fb3d.
files_authored:
  - tasks/_probe4_run_combo.py
  - tasks/_probe4_readout.py
  - tasks/_run_probe4_remote.py
context_loaded:
  - tasks/probe4_preregistration.md
  - tasks/_probe4_param_dicts.json
  - tasks/_probe3_1h_ritual.py
  - tasks/_probe3_readout.py
  - tasks/_run_probe3_remote.py
  - tasks/_probe3_regime_halves.py
  - tasks/_probe3_param_nbhd.py
  - scripts/param_sweep.py (greps on session_filter_mode + _COMBO_META_KEYS)
  - tasks/_agent_bus/probe4_2026-04-22/ (existing bus artifacts)
  - ~/.claude/projects/<slug>/memory/reference_remote_job_workflow.md
libraries_consulted: []
ready_for_review: true
---

## What I changed

- Authored `tasks/_probe4_run_combo.py` (~210 LOC) — single (combo_id, session_filter_mode) backtest runner per §4.4 session override mechanism.
- Authored `tasks/_probe4_readout.py` (~220 LOC) — aggregates 6 per-run JSONs + 2 SES_0 per-trade parquets, computes §4.3 Welch-t, applies §5 ordered branch routing.
- Authored `tasks/_run_probe4_remote.py` (~185 LOC) — paramiko orchestrator: git-pull on sweep-runner-1, upload scripts, launch 6-run suite under `systemd-run --scope -p MemoryMax=9G` in screen `probe4`, poll at 60 s for up to 30 min, SFTP-pull artifacts.

LOC totals: ~615 across the three scripts.

## Key design decisions (anywhere I deviated from Probe 3 precedent)

1. **Runner split by (combo, session)** — 6 invocations of one runner script instead of Probe 3's per-gate scripts. Each invocation writes one JSON (aggregate metrics) + one engine parquet; SES_0 additionally writes a lean per-trade parquet for Welch-t. Per preregistration §8.2 "single-run-with-tagging was considered and rejected".

2. **Engine `combo_id` encoding**: `20_000 + combo_id*10 + ses_mode` so each of the 6 runs produces an idempotent engine parquet keyed off a unique engine-side id, disjoint from the v11 sampler namespace (Probe 3 used `10_000 + i` in `_param_nbhd` and `10_200 + i` in `_1h_ritual`; I continued the pattern in the 20_xxx band).

3. **Parameter materialization — JSON-first, parquet fallback**: `_probe4_run_combo.py::_load_param_dict` prefers `tasks/_probe4_param_dicts.json` (cached, committed next to the preregistration) and only falls back to `data/ml/originals/ml_dataset_v11_1h.parquet` if the JSON is missing. Probe 3 hard-coded the base combo 865 dict inline — Probe 4's JSON-first approach avoids duplicating the 29-column dict across three scripts and keeps the preregistration's §2.1/§2.2 tables as the single source of truth.

4. **Welch-t insufficient-n defense per §4.3** — strict: if either combo has `n_trades < 50` on SES_0, emit `"welch_t": null` AND `"welch_gate_pass": false` (never `null`). The routing table in §5 never compares the `null` t-value directly; it only reads `welch_gate_pass`. Zero pooled SE degenerate case is guarded symmetrically.

5. **§5 ordered routing implemented as an explicit if-cascade** (not a list-of-tuples) for readability. Row 2 (SESSION_CONFOUND) fires before row 3 (PROPERTY_VALIDATED), matching the preregistration's ordering disclosure. Branch string values are exactly `{PROPERTY_VALIDATED, SESSION_CONFOUND, INCONCLUSIVE, COUNCIL_RECONVENE}` — no `PROPERTY_FALSIFIED` (removed in prereg pass-1).

6. **§5.2 narrow-miss flag**: readout emits a separate `narrow_miss_flag` boolean when row 5 fires with `1298 abs FAIL` AND `welch_gate_pass == True`. This surfaces the case in the JSON for the verdict document without allowing any post-hoc promotion to PROPERTY_VALIDATED.

7. **Orchestrator memory cap = 9G, no CPU quota**: aligned with `run_eval_notebooks_remote.py`'s 9G convention. Per preregistration §8.4 the 6 runs are ~12 min total — well under the 3h Kamatera burst threshold — so `CPUQuota=280%` (Probe 3 used it for longer gates) is omitted. Documented inline in the script header.

8. **Remote git-sync pattern**: mirrors Probe 3 exactly (`git stash -u` → `git fetch origin master` → `git reset --hard origin/master`). Per `feedback_remote_git_sync.md`: never SFTP-patch `src/`. The param-dicts JSON lives under `tasks/` and is picked up by the same git pull.

9. **Cython .so wipe**: kept (same reason as Probe 3 — Numba fallback ensures parity with the signed test partition baseline used in Probes 2/3).

10. **Poll cadence = 60 s** (not the 10-min `feedback_poll_interval.md` default): matches Probe 3's in-script polling. The 10-min default applies to `/loop` tool polls from the Windows host, not to the in-script screen polling inside a single paramiko orchestrator run.

## Open questions for the reviewer

- **Verification that `session_filter_mode` override propagates correctly on the sweep engine**: I confirmed by grep that `session_filter_mode` is in `_COMBO_META_KEYS` (line 121) and is consumed at `param_sweep.py:886` via `ns.SESSION_FILTER_MODE = int(combo.get("session_filter_mode", 0))`. Probe 3's §4.4 ritual used the same `tod_exit_hour`/`use_breakeven_stop`/`max_hold_bars` override mechanism successfully. **Worth a reviewer spot-check** that the override branch actually engages the RTH/GLOBEX filter logic at engine runtime (not just stored in the combo dict). This is the primary place where a latent bug would silently give identical SES_0/SES_1/SES_2 outputs.

- **YEARS_SPAN_TEST = 1.4799 reused unchanged** from Probe 3. Preregistration §3 locks the partition to the same 2024-10-22 → 2026-04-08 window, so this should be fine, but flagging for the reviewer since the same constant appears in three separate Probe 4 compute sites (metric aggregation inside `_probe4_run_combo.py`) and silent drift would corrupt all 6 metric tuples.

- **Empty trades-parquet emitted on SES_0 with zero trades**: I write an empty parquet with `net_pnl_dollars`/`entry_bar_idx` columns rather than skipping, so the readout can distinguish "engine ran, zero trades" from "invocation never launched". Preregistration doesn't explicitly speak to this edge case — reviewer should confirm this doesn't surprise the `_welch_t()` insufficient-n branch (it doesn't; `n_a == 0 < 50` trips the guard cleanly).

## Ready-for-review

Yes. All three scripts compile-visible and pattern-match Probe 3 precedents. No engine invocation performed. No remote calls. No commits. Awaiting `code-logic-reviewer` pass before user authorizes remote launch.
