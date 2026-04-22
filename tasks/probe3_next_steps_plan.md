# Probe 3 — next-steps execution plan

**Baseline commit**: `4e5a114` — Probe 3 pre-registration drafted (unsigned)
**Authoring date**: 2026-04-21 evening CDT
**Scope**: from current state (unsigned draft + pre-sign review findings) through
final verdict and branch action.
**Execution policy**: all Python/data/compute work runs on sweep-runner-1.
Local-only actions: `Write` / `Edit` file tool calls and `git` commands.
See `memory/feedback_remote_execution_only.md`.

---

## Current state (entry conditions)

- `tasks/probe3_preregistration.md` drafted at 455 lines, committed at `4e5a114`,
  pushed to origin.
- Pre-sign reviewing agent returned **FIX WARNINGS THEN SIGN** verdict:
  - 0 CRITICAL findings.
  - 4 WARN findings (W1–W4), all in-text clarity refinements (≤10 lines of
    edits, no restructuring).
  - 3 INFO items (advisory, not blocking).
- A-level (transcription / arithmetic), B-level (structural), C-level (memo
  consistency), D-level (reference integrity) all PASSED.
- No §9 signature populated yet.

---

## Phase B2a — Address WARN items, save review, re-commit

**Scope**: local-only. `Edit` on preregistration + `Write` on review report +
`git add/commit/push`. No compute.

**Steps**:

1. **`Edit tasks/probe3_preregistration.md`** — four small in-text additions:
   - **W1** (§4.1 regime halves gate): alongside the "71 % power at
     SR = 2.5 (conservative)" line, also cite "~83 % power at observed
     SR = 2.89" for full disclosure, with a footnote pointing at
     `probe3_multiplicity_memo.md` §4 power table.
   - **W2** (§4.2 parameter neighborhood gate): after "≥ 14 / 27 combos pass
     …", add a parenthetical of the form
     "(denominator = 27 axis combos; ≥ 14 / 27 under an H0 per-combo pass
     rate ≤ 0.25 yields likelihood ratio > 850× vs H0 — derivation in
     `probe3_multiplicity_memo.md` §3)".
   - **W3** (§4.3 15m negative-control gate): after "≤ 2 / 16 cells pass",
     explicitly name the per-cell H0 pass rate used to calibrate the
     threshold (5 %, → P(≥ 3 / 16 pass | H0) ≈ 0.021). Point at the
     closed-form binomial calc in the memo.
   - **W4** (§5.2 methodology binding clause): upgrade the Rule 2 reference
     from a "per `feedback_council_methodology.md`" citation to an
     explicit inline restatement: "For each diagnostic gate, this
     preregistration commits to P(PASS | H1) and P(PASS | H0) pre-data
     (memo §3–§4). Any future relaxation of these gates requires a new
     preregistration cycle and fresh council, not a mid-flight edit."
2. **`Write tasks/probe3_preregistration_review.md`** — save the reviewing
   agent's full findings inline (CRITICAL / WARN / INFO breakdown, A–D
   check matrix, verdict).
3. **`git add tasks/probe3_preregistration.md tasks/probe3_preregistration_review.md`**.
4. **`git commit -m "Probe 3 preregistration: address pre-sign review W1–W4"`**
   — this creates a new commit hash that supersedes `4e5a114` as the signing
   referent.
5. **`git push origin master`**.

**Verification**:

- `grep -n "83 %" tasks/probe3_preregistration.md` returns §4.1 hit.
- `grep -n "> 850" tasks/probe3_preregistration.md` returns §4.2 hit.
- `grep -n "5 %" tasks/probe3_preregistration.md` returns §4.3 hit.
- `grep -n "new preregistration cycle" tasks/probe3_preregistration.md`
  returns §5.2 hit.
- `git log -1 --oneline` shows the new commit with message containing W1–W4.

**Anti-patterns to avoid**:

- Do **not** change any gate thresholds. W1–W4 are disclosure / traceability
  refinements, not methodology changes. Gates must stay bit-identical to
  `4e5a114`.
- Do **not** touch §9 — the signature block stays blank until Phase B3.
- Do **not** amend `4e5a114`. Create a fresh commit.

**End-state**: new commit hash `<B2a-hash>` on origin/master; signature block
still blank; ready for user review.

---

## Phase B3 — Sign + freeze

**Scope**: local-only. `Edit` on §9 of preregistration + `git commit + push`.

**Entry condition**: explicit user approval of the form
"Sign Probe 3 at commit `<B2a-hash>`". Do **not** proceed without this message.

**Steps**:

1. **`Edit tasks/probe3_preregistration.md` §9**:
   - Signing date/time: current local time (stamp in UTC too).
   - Signing commit: `<B2a-hash>` (the post-WARN-fix hash the user approves).
   - Signer identifier: as recorded for Probes 1 & 2.
2. **`git add tasks/probe3_preregistration.md`**.
3. **`git commit -m "Sign Probe 3 pre-registration at <B2a-hash>"`**.
4. **`git push origin master`**.

**Verification**:

- `git log -1 --oneline` shows signing commit.
- The signing commit itself must **only** populate §9. No other sections
  change. Use `git diff <B2a-hash> HEAD -- tasks/probe3_preregistration.md`
  to confirm the diff is the §9 block only.

**End-state**: Probe 3 hypothesis frozen. Any future deviation from §4 gates
or §5 branch map requires a new preregistration cycle and fresh council.

---

## Phase C — Remote probe execution (4 gates)

**Scope**: sweep-runner-1 via paramiko. All Python runs remote. Local-only
actions are: drafting scripts (`Write`), git commit + push, polling (`ssh`
read-only inspection), SFTP pull of artifacts.

### C1 — Draft 4 probe scripts + 1 launcher (local)

Draft in `tasks/` and push so sweep-runner-1 can `git pull`.

- **`tasks/_probe3_regime_halves.py`** — gate §4.1
  - Reads the 1h test-partition trades for combo 865 (Probe 2 cache).
  - Splits on calendar midpoint 2025-07-15.
  - Computes net Sharpe on each half, applies PASS rule: both halves ≥ 1.3
    AND each half n_trades ≥ 50.
  - Emits `data/ml/probe3/regime_halves.json` with {half_1_sharpe,
    half_1_n, half_2_sharpe, half_2_n, pass}.
- **`tasks/_probe3_param_nbhd.py`** — gate §4.2
  - Sweeps 3 × 3 × 3 = 27 combos across (z_band_k, stop_fixed_pts, min_rr)
    at ±5 % around combo 865's §2.1 values (exact values already
    enumerated in preregistration §2.2).
  - Runs each on 1h test partition using existing param-sweep engine.
  - Applies per-combo §4 three-gate set (net Sharpe ≥ 1.3 AND n_trades
    ≥ 50 AND net $/yr ≥ 5 000).
  - Emits `data/ml/probe3/param_nbhd.json` with
    {combo_id, pass_flag, n_trades, net_sharpe, net_dollars_per_yr} × 27
    plus scalar `n_pass` and PASS rule `n_pass ≥ 14`.
- **`tasks/_probe3_15m_nc.py`** — gate §4.3 (negative control)
  - 4 sessions × 4 exits = 16 combos on the 15m test partition.
  - Combo 865 params held fixed; only session filter and exit rule vary
    per §2.3 and §2.4.
  - Applies per-cell §4 three-gate set.
  - Emits `data/ml/probe3/15m_nc.json` with {cell_id, session, exit,
    pass_flag, n_trades, net_sharpe, net_dollars_per_yr} × 16 plus
    scalar `n_pass` and PASS rule `n_pass ≤ 2`.
  - **Note**: PASS on the negative control = FAIL for the probe.
    If more than 2 of 16 cells clear the gates on 15m, the hypothesis
    that 15m's Probe 2 failure was friction-driven (not systematic) is
    weakened — i.e. session / exit manipulation can rescue 15m, suggesting
    combo 865's "edge" is fragile.
- **`tasks/_probe3_1h_ritual.py`** — gate §4.4
  - Same 16-cell grid, 1h test partition.
  - Applies per-cell §4 three-gate set.
  - Emits `data/ml/probe3/1h_ritual.json` with `n_pass`, PASS rule
    `n_pass ≥ 8`.
- **`tasks/_run_probe3_remote.py`** — launcher
  - `paramiko` upload no-ops (scripts arrive via `git pull`).
  - On sweep-runner-1:
    - `systemd-run --scope -p MemoryMax=9G -p CPUQuota=280% …`
    - `screen -dmS probe3_<gate> python tasks/_probe3_<gate>.py`
    - One screen per gate; 4 screens running in parallel if RAM headroom
      allows, else sequential.
  - Expected RAM: 15m NC is the fattest (larger trade row count); sequence
    NC → ritual → param_nbhd → regime halves if OOM risk appears.

### C2 — Commit + push scripts

```
git add tasks/_probe3_*.py tasks/_run_probe3_remote.py
git commit -m "Probe 3 C1: four gate probes + remote launcher"
git push origin master
```

### C3 — Launch remote

```
ssh sweep-runner-1 "cd ~/intra && git pull origin master"
python tasks/_run_probe3_remote.py   # paramiko; starts 4 screens
```

### C4 — Poll

Per `memory/feedback_poll_interval.md`: `delaySeconds=600` on ML probes.
Expected runtime per gate on sweep-runner-1 (based on Probe 2 timing):
regime halves ~ 2 min, param_nbhd ~ 15 min, 15m NC ~ 20 min, 1h ritual
~ 15 min. Total wall-clock ≤ 60 min parallel, ≤ 90 min sequential.

### C5 — SFTP pull

When all four JSON artifacts land on remote:

```
paramiko SFTP pull data/ml/probe3/*.json
git add data/ml/probe3/*.json
git commit -m "Probe 3 C5: four gate artifacts pulled"
git push origin master
```

**Verification**:

- `ls data/ml/probe3/` shows four JSON files.
- Each JSON loads and contains at minimum {pass_flag, n_trades, …} per
  gate's spec above.
- `git log -1` shows the artifact-pull commit.

**Anti-patterns**:

- Do **not** inspect any gate result before all four artifacts are pulled.
  Partial-result inspection breaks the §6 "nine irrevocable commitments"
  (commit 7: no early-stop, no sequential inspection).
- Do **not** re-run any gate because a result looked "off" — §6 commits
  bind this.

---

## Phase D — Readout + verdict

**Scope**: readout script runs remote (Python). Verdict doc is local Write.

### D1 — Draft + push readout

- **`tasks/_probe3_readout.py`**:
  - Load all four JSON artifacts.
  - Count FAIL gates → `F ∈ {0, 1, 2, 3, 4}` per §5.
  - Compute posterior per §6 memo: F = 0 → 0.91; F = 1 → ~0.5; F ≥ 2
    → < 0.06.
  - Emit `data/ml/probe3/readout.json` with {F, posterior, branch, per-gate
    pass_flags}.
- Push, pull on remote, run on remote, SFTP back.

### D2 — Write verdict document

**`tasks/probe3_verdict.md`** — structure mirrors `probe2_verdict.md`:

- Pre-registered commit hash cited (signing commit from Phase B3).
- Per-gate numeric table (n_trades, net Sharpe, pass / fail).
- Scalar F and posterior.
- Branch declaration per §5.
- If F = 0: authorize paper-trade setup (Phase E1).
- If F = 1: re-convene council (Phase E2).
- If F ≥ 2: sunset carve-out (Phase E3).

### D3 — Commit + push

```
git add tasks/probe3_verdict.md data/ml/probe3/readout.json
git commit -m "Probe 3 verdict: F=<N>, branch <paper_trade|council|sunset>"
git push origin master
```

**Verification**:

- The verdict commit must land **after** the artifacts commit from C5.
  Never commit a verdict before the artifacts it reads.
- `F` in `readout.json` must match `F` in `probe3_verdict.md`.

---

## Phase E — Branch action (conditional on F)

### E1 — If F = 0 (paper-trade authorized)

- **`Write tasks/combo865_1h_paper_trade_plan.md`**:
  - Instrument: MNQ on 1h bars, combo 865 parameter dict from
    preregistration §2.1.
  - Sizing: fixed $500 / trade (project convention).
  - Duration: 20 trading days minimum OR 30 filled trades, whichever
    comes first.
  - Live / paper broker adapter spec (TBD — outside non-negotiable
    "backtest first" scope; may need a new preregistration for the
    adapter contract).
  - Pre-paper-trade checklist: data feed stability, order-routing
    dry-run, stop / target precision at 0.25-point tick.
- Spawn a fresh LLM Council *before* paper-trade scope is signed
  — Probe 2 §5.6 binding equivalent applies here by analogy.
- Commit + push.

### E2 — If F = 1 (council re-convene)

- **`Write tasks/probe3_post_verdict_council_framing.md`**:
  - Summary of which single gate failed and which three passed.
  - Framing question: "Given F = 1 on gate X, should the carve-out
    advance to a narrower probe, to paper-trade with a tightened
    sizing cap, or to sunset?"
  - Council is bound by preregistration §5 — *the council does not
    override the F = 1 branch, it scopes the next-step probe*.
- Fire the council (5 advisors + peer review + chairman) per
  `memory/feedback_council_methodology.md` rules (both rules
  prepended to framing).
- Commit council artifacts + any follow-up preregistration.

### E3 — If F ≥ 2 (sunset)

- **`Edit CLAUDE.md`**: revoke the "Combo-865 carve-out" block; mark
  the full Z-score family terminal (bar-timeframe axis, per Probe 1
  §7.6, now reinforced).
- **`Edit STRATEGY.md`**: remove any combo 865 / adaptive-RR production
  language; replace with sunset banner.
- **`Write tasks/combo865_sunset_verdict.md`**:
  - Chain of authority: Probe 1 family-level sunset → Probe 2 carve-out
    PASS on 1h → Probe 3 FAIL (F ≥ 2) → full retirement.
  - Post-mortem summary: why the 1h Probe 2 result was a small-sample
    artifact / favorable test window / noise (per the specific failing
    gates — regime half failure implies test-window bias; param nbhd
    failure implies p-hacking; 15m NC pass implies non-structural
    effect; 1h ritual failure implies session/exit fragility).
- Fire a fresh LLM Council on project-level next step (family swap,
  instrument swap, project sunset). Must be spawned *after* the
  sunset verdict is committed — the council's input is the verdict,
  not the decision.
- Commit + push.

---

## Cross-phase invariants

1. **Every phase ends with `git push origin master`.** Local commits
   that aren't pushed violate the user directive "after each task,
   commit and push". This preserves remote audit trail and enables
   sweep-runner-1 to `git pull` the next batch of scripts.
2. **No gate threshold moves post-signing.** W1–W4 fixes happen
   *before* B3 signing for exactly this reason. Any observed need to
   shift thresholds after data = new preregistration cycle.
3. **Remote execution for all compute.** Every phase from C1 launcher
   onward uses sweep-runner-1. Local = drafting + committing only.
4. **Council framing prepends both Rule 1 and Rule 2.** When E1 or E2
   fires a council, the framing block must explicitly instruct
   advisors to identify Stage 1 vs Stage 2 and to compute
   P(PASS | H1) / P(PASS | H0) per gate.
5. **No post-hoc gate edits.** §6 of preregistration lists 9
   irrevocable commitments — commit 3 (no gate edits), commit 5 (no
   timeframe swaps), commit 7 (no early-stop), commit 9 (no
   multiplicity-correction relaxation) are load-bearing here.

---

## File-reference appendix

| File | Role | Status |
|---|---|---|
| `tasks/probe3_preregistration.md` | Hypothesis freeze | Drafted at `4e5a114`; WARN fixes pending |
| `tasks/probe3_multiplicity_memo.md` | Statistical calibration | Committed `6dfee2d` |
| `tasks/probe3_preregistration_review.md` | Pre-sign agent report | To be written in B2a |
| `tasks/probe2_preregistration.md` | Structural template | Signed `a49f370` |
| `tasks/probe2_verdict.md` | Verdict template | `f573f45` |
| `memory/feedback_council_methodology.md` | Rules 1 & 2 | Saved 2026-04-21 |
| `memory/feedback_remote_execution_only.md` | Execution policy | Saved 2026-04-21 |
| `memory/feedback_poll_interval.md` | `delaySeconds=600` | Pre-existing |
| `memory/reference_remote_job_workflow.md` | Paramiko + screen + systemd-run pattern | Pre-existing |
| `memory/reference_sweep_runner_ssh.md` | SSH credentials | Pre-existing |

---

## Entry-point for next turn

If the user says **"fire B2a"** or **"execute in recommended order"**:
begin Phase B2a by applying W1–W4 edits to `tasks/probe3_preregistration.md`,
writing the reviewing agent's report to
`tasks/probe3_preregistration_review.md`, committing, pushing. Then stop and
present the new commit hash to the user for Phase B3 approval.

If the user says **"sign"** directly: treat as an implicit authorization to
run B2a → B3 in one motion; still emit the post-B2a hash *before* signing so
the user sees what they're signing against.
