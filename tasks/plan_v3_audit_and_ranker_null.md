# Plan: V3 Combo-Agnostic Audit + Ranker-Label Overlap Null

**Date**: 2026-04-21
**Authority**: LLM Council verdict 2026-04-20 13:49 CDT + V4 FAIL 2026-04-21 00:34 UTC
**Supersedes**: tasks #10–11 of `C:\Users\kunpa\.claude\plans\sleepy-roaming-kettle.md` (reclassified — not dropped)
**Parent findings**: `tasks/v4_no_gcid_audit_verdict.md`, memory `project_v4_combo_id_leak_confirmed.md`

## Context

The 2026-04-21 V4 ship-blocker audit failed catastrophically: Sharpe p50 collapsed from the shipped +2.13 (with `global_combo_id`) to −0.42 under combo-agnostic refit, and ruin probability jumped 0.02% → 56.37%. This confirmed `global_combo_id` was the *source* of V4's measured alpha, not a covariate of it.

Two items carry forward from the LLM Council 6-step framework and one is now promoted to urgent:

1. **V3 combo-agnostic audit** — V3 is declared production in `CLAUDE.md`, and it carries the same `global_combo_id` LightGBM categorical as V4. Every V3-based claim is suspect until cleared.
2. **Council step 6 (ranker-label overlap null)** — still-live test that asks whether `audit_full_net_sharpe` (the Pool B ranker) leaks between the partition used to pick combos and the partition used to train ML#2. Independent of ML#2 outcome.
3. **Council step 5 (random-K null)** — MOOT under current config (precondition: step 4 Sharpe CI overlaps shipped; didn't happen). Revivable against any *clean* filter that re-opens a ship question.

---

## Phase 0 — Documentation Discovery (COMPLETE)

**Completed**: 2026-04-20 via Explore subagent. Infrastructure mapping of the V4 combo-agnostic work at commit `3e40223` (with upstream `d66af88`, `4c8bd98`).

### Copy template → target map

| Concern | V4 template | V3 target | Notes |
|---|---|---|---|
| Training CLI flag | `scripts/models/adaptive_rr_model_v4.py::--no-combo-id` argparse block | same pattern in `scripts/models/adaptive_rr_model_v3.py` | re-grep exact line numbers before editing |
| Feature rebinding | V4 L69/72/74 (dropped `global_combo_id`) | V3 L64/67/69 per `project_v4_combo_id_leak_confirmed.md` | verify at exec time |
| Inference module | `scripts/models/inference_v4_no_gcid.py` | new `scripts/models/inference_v3_no_gcid.py` sibling of `inference_v3.py` | keeps V3's pooled per-R:R isotonic calibrator |
| Eval dispatch | `scripts/evaluation/_top_perf_common.py::_load_eval_module` has `v4_no_gcid` branch | add `v3_no_gcid` branch | same file |
| Notebook builder | `scripts/evaluation/_build_v2_notebooks.py::_build_net_variant` emits `v4_no_gcid` | add `v3_no_gcid` variant | net-only per 2026-04-20 policy |
| Remote runner | `scripts/runners/run_eval_notebooks_remote.py` task list contains `v12_top50_raw_sharpe_v4_no_gcid` | add `v12_top50_raw_sharpe_v3_no_gcid` | same screen+systemd-run wrapper |
| Model artifact path | `data/ml/adaptive_rr_v4_no_gcid/booster_v4.txt` (13 MB, AUC 0.8463) | `data/ml/adaptive_rr_v3_no_gcid/booster_v3.txt` | mirrored directory layout |
| Calibrators | V4 has none (no isotonic in V4 production) | V3 requires pooled per-R:R isotonic under `data/ml/adaptive_rr_v3_no_gcid/isotonic/` | **V3-specific extension** — this step does not exist in V4 template |

### Anti-patterns to avoid

- **Do not copy V4's training script wholesale** — V3's feature set differs; preserve V3-specific features and the pooled per-R:R isotonic calibrator (see `CLAUDE.md` Phase 5D).
- **Do not leave `global_combo_id` aliases** — check derived features (one-hot encodings, string hashes) that could reintroduce per-combo memorization.
- **Do not emit gross notebooks** — use `_build_net_variant` only (2026-04-20 policy in `CLAUDE.md`).
- **Do not call Kelly sizing** — the two-stage per-combo calibrator was deprecated in Phase 5D. Stick to `fixed_dollars_500`.

---

## Phase 1 — V3 Combo-Agnostic Refit Infrastructure

**Goal**: Mechanical copy of V4 combo-agnostic scaffolding into V3, without changing V3's training objective or feature set beyond removing `global_combo_id`.

### 1.1 Training CLI flag

**What**: Add `--no-combo-id` argparse flag + runtime feature-list guard to `scripts/models/adaptive_rr_model_v3.py`, mirroring the V4 pattern from commit `d66af88`.

**Documentation reference**: Read `scripts/models/adaptive_rr_model_v4.py` and locate:
- the `--no-combo-id` argparse block (new in `d66af88`),
- the feature-list assembly (L69/72/74 in V4),
- the LightGBM `categorical_feature` argument (must drop `global_combo_id` when flag set).

Copy the same three edits into V3 at its analogous locations (L64/67/69). Re-grep before editing — the line numbers are from the memory snapshot and may drift.

### 1.2 Inference module

**What**: Create `scripts/models/inference_v3_no_gcid.py` as a sibling to `inference_v3.py`.

**Documentation reference**: Read `scripts/models/inference_v4_no_gcid.py` in full. Copy the module shape (sys.path insertion, feature-list rebinding, predict function signature). Name the exported function `predict_pwin_v3_no_gcid()`; internal booster should load from `data/ml/adaptive_rr_v3_no_gcid/booster_v3.txt`.

**Critical**: Keep V3's pooled per-R:R isotonic calibrator wired in — V4 had no isotonic, V3 does. Load from `data/ml/adaptive_rr_v3_no_gcid/isotonic/pooled_rr_{10,15,20,25,30}.pkl`.

### Verification for Phase 1

- [ ] `grep -n "no-combo-id" scripts/models/adaptive_rr_model_v3.py` shows the flag guard
- [ ] `python scripts/models/adaptive_rr_model_v3.py --no-combo-id --dry-run` (or `--help`) prints a feature list lacking `global_combo_id`
- [ ] `scripts/models/inference_v3_no_gcid.py` imports cleanly and `predict_pwin_v3_no_gcid` exists
- [ ] `grep -n "global_combo_id" scripts/models/inference_v3_no_gcid.py` returns no matches

### Anti-pattern guards

- Do not assume exact line numbers; re-grep `global_combo_id` in `adaptive_rr_model_v3.py` and verify each occurrence is handled by the flag guard.
- Do not change V3's feature set beyond removing `global_combo_id` — this is a surgical refit, not a retrain from scratch.

---

## Phase 2 — V3 Remote Refit + Artifact Verification

**Goal**: Produce a refit `adaptive_rr_v3_no_gcid/` artifact tree on sweep-runner-1, mirroring the V4 refit that produced AUC 0.8463.

### 2.1 Remote launcher

**What**: Create `scripts/runners/run_adaptive_rr_v3_refit_remote.py` as a copy of `scripts/runners/run_adaptive_rr_v4_refit_remote.py` with the model version swap.

**Documentation reference**: Read `scripts/runners/run_adaptive_rr_v4_refit_remote.py` in full. Preserve the screen + systemd-run wrapper, 9G `MemoryMax`, 280% `CPUQuota`. Change only:
- script name passed to remote (`adaptive_rr_model_v3.py` not `_v4.py`),
- output directory (`data/ml/adaptive_rr_v3_no_gcid/`),
- screen session name.

### 2.2 Artifact verification

**What**: Create `tasks/_check_v3_no_gcid_artifact.py` (copy + swap version from `tasks/_check_v4_no_gcid_artifact.py`).

Must assert:
- booster file exists and loads via `lgb.Booster(model_file=...)`,
- `booster.feature_name()` does not contain `"global_combo_id"`,
- AUC on held-out bar is within 0.05 of V3 production baseline (`data/ml/adaptive_rr_v3/`),
- ECE < 1e-4 per R:R bin (matches V4's 8.14e-7 scale; V3 should come in similar).

### 2.3 Calibrator export

**What**: Run `scripts/calibration/export_calibrators_v3.py` against the new booster, writing pooled per-R:R isotonic pickles under `data/ml/adaptive_rr_v3_no_gcid/isotonic/`.

### Verification for Phase 2

- [ ] Remote refit completes with AUC within 0.05 of V3 production
- [ ] `tasks/_check_v3_no_gcid_artifact.py` exits 0
- [ ] `feature_name()` list lacks `global_combo_id`
- [ ] All 5 isotonic pickles exist at expected R:R values

### Anti-pattern guards

- Do not SFTP-patch source into remote — use the git-sync pattern (`feedback_remote_git_sync.md`): push local, then `git pull` on sweep-runner-1.
- Do not exceed the Kamatera CPU cap (`feedback_kamatera_cpu_cap.md`): 280% CPUQuota, 3 LightGBM threads.

---

## Phase 3 — V3 Ship-Blocker Audit

**Goal**: Run the same 12-notebook s6_net audit that ran on V4, now on the refit V3, and decide PASS/FAIL against a pre-registered criterion.

### 3.1 Notebook variant builder

**What**: Extend `scripts/evaluation/_build_v2_notebooks.py` to emit a `v3_no_gcid` variant via `_build_net_variant` (net-only).

**Documentation reference**: Read the existing `_build_net_variant` call for `v4_no_gcid` in `_build_v2_notebooks.py`. Mirror it for `v3_no_gcid` — same Pool B (`evaluation/top_strategies_v12_raw_sharpe_top50.json`), same MC kernel, same n_sims=2000 pinning on s3_net.

### 3.2 Dispatch wiring

**What**: Add `v3_no_gcid` branch to `scripts/evaluation/_top_perf_common.py::_load_eval_module`.

**Documentation reference**: Read the existing `v4_no_gcid` branch. The new branch should import from `inference_v3_no_gcid` and use the v3_no_gcid isotonic calibrators.

### 3.3 Remote runner wiring

**What**: Add `v12_top50_raw_sharpe_v3_no_gcid` to the task dispatch in `scripts/runners/run_eval_notebooks_remote.py`.

### 3.4 Remote audit execution

**What**: `python scripts/runners/run_eval_notebooks_remote.py v12_top50_raw_sharpe_v3_no_gcid` — 12 notebooks under `evaluation/v12_topk_top50_raw_sharpe_{,net_}v3_no_gcid/`.

### 3.5 Verdict extraction

**What**: From `evaluation/v12_topk_top50_raw_sharpe_net_v3_no_gcid/s6_mc_combined_ml2_net.ipynb`, extract the MC row: `sharpe_p50`, `sharpe_ci_95`, `risk_of_ruin_prob`, `dd_worst_pct`, `n_trades`, `win_rate`, `sharpe_pos_prob`, `trades_per_year`.

**Decision criterion** (pre-registered, mirrors V4 audit):

| Outcome | Rule |
|---|---|
| **PASS** | Sharpe CI overlaps V3 production-with-ID baseline **AND** ruin < 10% **AND** `sharpe_pos_prob > 90%` |
| **FAIL** | Any of: Sharpe CI disjoint, ruin > 30%, `sharpe_pos_prob < 70%` |
| **AMBIGUOUS** | Any other configuration — escalate to council |

### 3.6 Verdict documentation

**What**: Write `tasks/v3_no_gcid_audit_verdict.md` mirroring `tasks/v4_no_gcid_audit_verdict.md` structure. Cross-link from `tasks/ship_decision.md`.

### Verification for Phase 3

- [ ] All 12 notebooks executed without errors
- [ ] s6_net.ipynb contains a populated MC row
- [ ] `tasks/v3_no_gcid_audit_verdict.md` exists with head-to-head table
- [ ] Decision recorded (PASS/FAIL/AMBIGUOUS)
- [ ] `tasks/ship_decision.md` updated with V3 result row

### Anti-pattern guards

- Do not skip the n_sims=2000 pinning on s3_net — this is the OOM guard (`CLAUDE.md` remote eval section).
- Do not run gross notebooks — net-only.
- Do not mark PASS on ruin < 10% alone — all three criteria must hold.

---

## Phase 4 — Ranker-Label Overlap Null (Council Step 6)

**Goal**: Test whether `audit_full_net_sharpe` (the Pool B ranker) depends on combo-level memorization between the selection partition and the ML#2 training partition. **Independent of Phases 1–3 — can run in parallel.**

### 4.1 Partition labeling (REMOTE)

**What**: For each of the 13,814 post-gate combos in `data/ml/ml1_results_v12/combo_features_v12.parquet`, label whether any of its trade rows appear in the V3 (or V4) training parquet `data/ml/mfe/ml_dataset_v11_mfe.parquet` post-gate.

**Where**: executes on sweep-runner-1 — the 102M-row V11 MFE parquet lives there and must NOT be pulled down to local (shipping GB of data to compute a 13k-row label output is wasteful and violates the remote-execution contract).

**Implementation**: `scripts/analysis/build_combo_overlap_labels.py` runs on the remote via the standard paramiko + screen + systemd-run wrapper. Read `scripts/analysis/build_combo_features_ml1.py` locally to trace how the feature parquet was built, then implement the overlap check with the same join semantics. After remote execution, SFTP only `data/ml/ranker_null/combo_overlap_labels.parquet` (small — one row per combo, ~13k rows) back to local.

Output: `data/ml/ranker_null/combo_overlap_labels.parquet` with columns `(combo_id, trades_in_training, overlap_pct)`.

### 4.2 Held-out Pool B construction (REMOTE)

**What**: `scripts/analysis/extract_top_combos_heldout_v12.py` — same raw-Sharpe ranking as `extract_top_combos_by_raw_sharpe_v12.py`, but restricted to combos with `trades_in_training == 0` (or `overlap_pct < threshold`).

**Where**: remote. Reads `combo_features_v12.parquet` + the new `combo_overlap_labels.parquet` both in place on sweep-runner-1. After remote execution, SFTP the small JSON back.

Output: `evaluation/top_strategies_v12_raw_sharpe_top50_heldout.json`.

**Decision point**: If fewer than 50 combos survive the held-out filter, reduce top-K to whatever the pool supports (≥20 combos) and note in the verdict doc. If <20 survive, the null is infeasible and we need a different design (e.g., combo-level K-fold).

### 4.3 Notebook variant + remote audit

**What**: Add a `heldout_ranker` variant to `_build_v2_notebooks.py` that reads the held-out top-K JSON, using the *current* ML#2 filter (fixed V3-with-ID as diagnostic, or V3 combo-agnostic if Phase 3 passed and artifacts exist).

Remote-execute the 12-notebook audit.

### 4.4 Decision criterion

| Outcome | Rule |
|---|---|
| **PASS (ranker clean)** | Held-out Pool B Sharpe CI overlaps visible Pool B Sharpe CI |
| **FAIL (ranker leaks)** | Sharpe deltas exceed CI width, or held-out Pool B collapses to ruin-regime (>30% ruin) |

### 4.5 Verdict documentation

**What**: Write `tasks/ranker_label_overlap_null_verdict.md` with visible-vs-heldout Sharpe/ruin table.

### Verification for Phase 4

- [ ] `combo_overlap_labels.parquet` exists and sums overlap_pct sensibly (most combos should have high overlap given the training set's breadth)
- [ ] Held-out Pool B JSON exists with ≥20 combos
- [ ] 12 notebooks executed
- [ ] Verdict doc with decision

### Anti-pattern guards

- Do not use a string match on trade IDs without verifying schema compatibility between ml_dataset_v11_mfe and combo_features_v12 — confirm the join key before labeling.
- Do not reuse `v4_no_gcid` artifacts here — this null tests the *ranker*, not the *filter*; the filter should stay constant across visible and held-out runs.

---

## Phase 5 — Decision Fork

Once Phase 3 and Phase 4 verdicts exist, the next move is prescribed by the following matrix:

| Phase 3 (V3) | Phase 4 (Ranker) | Action |
|---|---|---|
| **PASS** | **PASS** | Revive council step 5 (random-K null vs refit V3). If step 5 passes, V3 is ship-ready on Pool B. Propose re-ship via new council review. |
| **PASS** | **FAIL** | V3 is clean but Pool B's ranker leaks. Redesign pool selection with held-out combo partitioning before any ship claim. V3 filter stays, ranker gets rebuilt. |
| **FAIL** | **PASS** | ML#2 architectural redesign needed — both V3 and V4 memorize via `global_combo_id`, so the ML#2 feature set itself must be reworked. Ranker is clean, so a new filter on existing Pool B could still ship eventually. |
| **FAIL** | **FAIL** | Deep redesign across both layers. Pause all ship claims, re-scope Phase B, new council review. |

---

## Final Verification (cross-cutting)

- [ ] All memory entries current: `project_council_plan_stage.md`, `project_v4_combo_id_leak_confirmed.md`, `project_ml1_topk_concentration.md`, MEMORY.md index
- [ ] `tasks/v3_no_gcid_audit_verdict.md` exists and is cross-linked from `tasks/ship_decision.md`
- [ ] `tasks/ranker_label_overlap_null_verdict.md` exists
- [ ] `grep -r "global_combo_id" scripts/models/` shows occurrences only in the *with-ID* sources (`adaptive_rr_model_v3.py`, `adaptive_rr_model_v4.py` outside their `--no-combo-id` guards) — no `_no_gcid` variant or inference module contains it
- [ ] Paper-trading remains halted until Phase 5 decision resolves
- [ ] Phase 5 decision recorded in `tasks/ship_decision.md` with next-steps ladder

## Execution order & parallelism

```
Phase 1 → Phase 2 → Phase 3      (sequential)
Phase 4                           (parallel with Phase 1–3)
Phase 5                           (after Phase 3 AND Phase 4 complete)
```

Phase 4 requires no artifacts from Phases 1–3 (it uses the current filter as a diagnostic), so it can run concurrently on a separate remote screen session.
