# Probe 3 Pre-Registration — Combo-865 Session/Exit + Robustness Probe on 1h

**Date**: 2026-04-21 UTC
**Authority**: Probe 3 scope LLM Council verdict,
`tasks/council-report-2026-04-21-probe3-scope.html`
**Calibration memo**: `tasks/probe3_multiplicity_memo.md`
**Template**: Mirrors `tasks/probe2_preregistration.md`.
**Process discipline**: Thresholds committed before the experiment runs,
applied mechanically after. Ties go to the stricter (sunset) side.

---

## Status

**DRAFT — UNSIGNED** as of 2026-04-21 UTC.

No part of Probe 3 execution may begin until the signer countersigns §9
at a specific git commit hash. The §4 gate rules are binding from the
moment of signing regardless of which branch the observed metrics fall
into.

---

## 1. Purpose

Test whether combo 865's held-out 1h edge (Probe 2: net Sharpe **2.89**
on 220 trades, +$124,896/yr) is:

- (a) a **regime-persistent real edge** for this parameter realization at
  1h on NQ, or
- (b) an artifact of **test-window favorability** + **parameter-space
  knife-edge** + **tolerant §4 gates** that happens to be reproducible
  only at one specific session/exit configuration.

The probe answers one compound four-binary question. Four **independent
diagnostic gates** (defined in §4) apply to combo 865 on the same
held-out 20% test partition used in Probe 2:

1. Regime halves (does the edge survive a temporal split within the
   test window?)
2. Parameter neighborhood (is combo 865 a needle or a ridge in parameter
   space?)
3. 15m negative control (does session/exit tuning resurrect 15m, which
   Probe 2 declared dead? If so, the friction-regime interpretation of
   Probe 2 is wrong and the 1h PASS may itself be sweep-findable.)
4. Session/exit ritual on 1h (is the 1h edge robust across reasonable
   session/exit permutations, not brittle to the specific config in
   combo 865's native params?)

**Branch map** (full spec in §5):
- 4 / 4 PASS → **paper-trade authorization** for combo 865 @ 1h on MNQ.
- ≥ 2 / 4 FAIL → **Option Z (project sunset)** of the Z-score
  mean-reversion family carve-out for combo 865.
- Exactly 1 / 4 FAIL → **council re-convene** (ambiguous branch).

Probe 3 does NOT reopen the bar-timeframe axis (Probe 1 §7.6 remains
terminal); does NOT test additional combos from the v11 sweep (Probe 1
§7.6 family-level falsification remains intact); does NOT vary combo
865's non-session/exit parameters beyond the ±5% neighborhood in §4.2.
Scope is locked to combo 865 on 1h (with the 15m NC being symmetric
structural evidence, not a new family-level probe).

---

## 2. Combo + Neighborhood + Session/Exit Grid Under Test

### 2.1 Center combo (frozen, identical to Probe 2 §2)

`combo_id = 865` from the v11 sampler (`--range-mode v11 --seed 0`).
Full parameter dict as in `tasks/probe2_preregistration.md` §2. Reproduced
here with explicit call-outs to the parameters swept by §4:

| Parameter | Value | Probe 3 gate |
|---|---|---|
| `z_band_k` | 2.304384 | §4.2 swept at ×0.95 / ×1.00 / ×1.05 |
| `z_window` | 41 | frozen |
| `z_input` | `returns` | frozen |
| `z_anchor` | `rolling_mean` | frozen |
| `z_denom` | `parkinson` | frozen |
| `z_type` | `parametric` | frozen |
| `z_window_2` | 47 | frozen |
| `z_window_2_weight` | 0.330829 | frozen |
| `volume_zscore_window` | 47 | frozen |
| `ema_fast` | 6 | frozen |
| `ema_slow` | 48 | frozen |
| `stop_method` | `fixed` | frozen |
| `stop_fixed_pts` | 17.017749 | §4.2 swept at ×0.95 / ×1.00 / ×1.05 |
| `atr_multiplier` | NaN (unused) | frozen |
| `swing_lookback` | NaN (unused) | frozen |
| `min_rr` | 1.847651 | §4.2 swept at ×0.95 / ×1.00 / ×1.05 |
| `max_hold_bars` | 120 | §4.3/§4.4 grid override (exit rule) |
| `exit_on_opposite_signal` | False | §4.3/§4.4 grid override |
| `use_breakeven_stop` | False | §4.3/§4.4 grid override |
| `zscore_confirmation` | False | frozen |
| `volume_entry_threshold` | 0.0 | frozen |
| `vol_regime_lookback` | 0 | frozen |
| `vol_regime_min_pct` | 0.0 | frozen |
| `vol_regime_max_pct` | 1.0 | frozen |
| `session_filter_mode` | 0 | §4.3/§4.4 grid override |
| `tod_exit_hour` | 0 | §4.3/§4.4 grid override |
| `entry_timing_offset` | 1 | frozen |
| `fill_slippage_ticks` | 1 | frozen |
| `cooldown_after_exit_bars` | 3 | frozen |

### 2.2 Parameter neighborhood (for §4.2)

3 axes × 3 levels = **27 combos** (one level per axis is the 865 center;
discarded as duplicate → 26 neighbors + 1 center; all 27 evaluated for
gate purposes).

| Axis | ×0.95 | ×1.00 (865 center) | ×1.05 |
|---|---:|---:|---:|
| `z_band_k` | 2.189165 | 2.304384 | 2.419603 |
| `stop_fixed_pts` | 16.166862 | 17.017749 | 17.868636 |
| `min_rr` | 1.755268 | 1.847651 | 1.940034 |

All 28+ other parameters frozen at the §2.1 values. Timeframe fixed at
**1h**. Partition fixed at **test**.

### 2.3 Session filter grid (for §4.3 and §4.4)

| Label | Definition (US Eastern Time) |
|---|---|
| `SES_0_all` | No filter. Any 1h bar eligible (identical to 865 native). |
| `SES_1_RTH_only` | Entries allowed only at 1h bars whose start is within **09:30 – 16:00 ET**. |
| `SES_2_overnight_only` | Entries allowed only at 1h bars whose start is within **18:00 – 09:30 ET** (overnight + Asian/European). |
| `SES_3_RTH_excl_lunch` | `SES_1_RTH_only` minus the **12:00 – 14:00 ET** lunch window. |

Session filter applies only to **entries**. Open positions managed by
native exit rules regardless of session.

### 2.4 Exit rule grid (for §4.3 and §4.4)

| Label | Definition |
|---|---|
| `EX_0_native` | Combo 865 native exit (fixed-stop 17.018 pts, TP from `min_rr`=1.848, `max_hold_bars`=120). |
| `EX_1_maxhold_60h` | Override `max_hold_bars` to 60 (1h × 60 = 60 hours hold cap). |
| `EX_2_TOD_1500ET` | Override `tod_exit_hour` to 15 (force all open positions flat at 15:00 ET). |
| `EX_3_breakeven_after_1R` | Override `use_breakeven_stop` to `True` (move stop to entry once price reaches entry ± 1R). |

### 2.5 Full (session × exit) grid

4 sessions × 4 exits = **16 cells**. Applied once at 1h (§4.4) and once
at 15m (§4.3) — 32 cell evaluations total.

---

## 3. Protocol

### 3.1 Partitions (identical to Probe 2)

- **Data source**: `data/NQ_1min.csv` (2,573,239 valid 1-min bars).
- **Split**: chronological 80/20 at 1-minute level. Test set: bars
  2,058,591 – 2,573,238 (2024-10-22 05:06 → 2026-04-08 20:20,
  **`YEARS_SPAN_TEST = 1.4799` — frozen at signing**).
- **Bar resampling**: `src/indicators/bar_resample.py`, left-anchored /
  partial-window-dropped semantics (identical to Probe 1 / Probe 2).

### 3.2 Execution per gate

All execution is **remote on sweep-runner-1** per
`memory/feedback_remote_execution_only.md`. Local work is limited to
script authoring (Write), git commits, and reading returned JSON / verdict
writeups.

#### 3.2.1 §4.1 — Regime halves

**No new compute**. Re-use `data/ml/probe2/combo865_1h_test.parquet`
(signed commit `a49f370`). Write `tasks/_probe3_regime_halves.py` which:

- Loads the Probe 2 1h test parquet.
- Splits trades by `entry_time` at **2025-07-15 00:00 UTC** (midpoint of
  test window).
- For each half, computes `n_trades`, `net_sharpe(annualized)`,
  `net_dollars_per_year` using half-year-adjusted `YEARS_SPAN_HALF`
  (computed from actual bar counts in each half, NOT fixed at 0.74 —
  exact values reported at execution; do NOT retune split date
  post-hoc if the count is unequal).
- Applies the §4.1 composite gate (both halves pass their sub-gates).
- Writes `data/ml/probe3/regime_halves.json`.

Runs on sweep-runner-1 after remote `git pull`. Expected wall-clock <
30 seconds.

#### 3.2.2 §4.2 — Parameter neighborhood

Write `tasks/_probe3_param_nbhd.py` which:

- Generates the 27 combos from §2.2.
- Runs each through the param_sweep backtest engine (v11 sweep
  friction + microstructure semantics) on 1h **test** partition bars.
- Writes `data/ml/probe3/param_nbhd/combo_{i}.parquet` for each combo
  and a `param_nbhd_manifest.json` summary.

Local author + commit the script. Upload + screen + systemd-run per
remote job workflow. Wall-clock ≈ 5 min.

#### 3.2.3 §4.3 — 15m negative control

Write `tasks/_probe3_15m_nc.py` which:

- Generates combo 865 × §2.3 × §2.4 (16 cells).
- Runs each on the **15m test** partition.
- Writes `data/ml/probe3/nc_15m/{session}_{exit}.parquet` + manifest.

Wall-clock ≈ 10 min.

#### 3.2.4 §4.4 — 1h session/exit ritual

Write `tasks/_probe3_1h_ritual.py` which:

- Generates combo 865 × §2.3 × §2.4 (16 cells).
- Runs each on the **1h test** partition.
- Writes `data/ml/probe3/ritual_1h/{session}_{exit}.parquet` + manifest.

Wall-clock ≈ 10 min.

### 3.3 Aggregate readout

Write `tasks/_probe3_readout.py` (mirrors `_probe2_readout.py`) which:

- Reads the four outputs from §3.2.1 – §3.2.4.
- Applies §4 gates mechanically per §4's specification.
- Applies §5 branch rule (count of FAILs across 4 gates).
- Writes `data/ml/probe3/readout.json` — machine-readable verdict record
  with the same structure as Probe 2's readout.

Runs on sweep-runner-1. Script itself is plain Python + numpy +
pandas — no heavy compute.

### 3.4 Metrics (per §4 sub-probe)

Consistent with Probe 2 §3:
- `net_sharpe = mean(net_pnl_dollars) / std(net_pnl_dollars, ddof=1)
  * sqrt(n_trades / YEARS_SPAN)`
- `net_dollars_per_year = mean(net_pnl_dollars) * n_trades / YEARS_SPAN`
- `net_pnl_dollars` already net of `$5/contract RT` via v11 sweep
  engine. Gross fields logged for audit.

---

## 4. Gates (binding, mechanically applied)

Each of the four gates produces a **binary PASS / FAIL**. All four are
evaluated independently. §5 aggregates the four binary outcomes.

### 4.1 Regime halves gate

**Specification**: Split combo 865's 1h test-partition trades at
**2025-07-15 00:00 UTC**. For each half `h ∈ {H1, H2}`, compute:

1. `net_sharpe(h, annualized) ≥ 1.3`
2. `n_trades(h) ≥ 25`
3. `net_dollars_per_year(h, annualized) ≥ 5,000`

**Gate composite**: **both halves** pass **all three** sub-gates.

**Rationale / power** (from calibration memo §4): at true SR = 2.5, both
halves pass ≈ **71%**; at null, both pass ≈ **1.8%**. Likelihood ratio
≈ 40× at this gate alone.

### 4.2 Parameter neighborhood gate

**Specification**: Run all 27 combos from §2.2 on the 1h test partition.
For each of the 27, apply the Probe 2 §4 three-gate set
(`net_sharpe ≥ 1.3 AND n_trades ≥ 50 AND net $/yr ≥ 5,000`).

**Gate composite**: **≥ 14 of 27** combos pass all three.

**Rationale / power**: under H1 (ridge), ±5% jitter preserves most
passes (expected ≈ 22 – 25 of 27). Under H0 (needle / favorable window),
≤ 7 of 27 expected. Base-rate check: uniform pass rate across the
13,814 v11 sweep combos is ~0.03% at this three-gate level; 14 / 27 is
~1,700× base rate.

### 4.3 15m negative control gate

**Specification**: Run combo 865 + (§2.3 × §2.4) = 16 cells on the 15m
test partition. For each, apply Probe 2 §4 three-gate set.

**Gate composite**: **≤ 2 of 16** 15m cells clear §4 three-gate set.

**Rationale**: under the friction-regime interpretation of Probe 2,
15m gross per trade (~$355) cannot clear the per-contract friction
(~$438) regardless of session/exit structure — session/exit tuning
should NOT rescue 15m. If ≥ 3 of 16 cells clear §4 on 15m, that
contradicts the friction mechanical story and implies the 1h Probe 2
PASS may itself be findable by session/exit tuning, not a genuine
single-TF edge. Expected under H0 (with modest session/exit search):
< 2 passes. Expected under H1 (friction story): 0 passes.

### 4.4 1h session/exit ritual gate

**Specification**: Run combo 865 + (§2.3 × §2.4) = 16 cells on the 1h
test partition. For each, apply Probe 2 §4 three-gate set.

**Gate composite**: **≥ 8 of 16** 1h cells clear §4 three-gate set.

**Rationale**: under H1 (real edge), combo 865's 1h edge should persist
across most reasonable session/exit permutations (not be brittle to
the one native session/exit config). Under H0 (favorable-window), edge
decays under session/exit perturbation and < 8 cells pass. Baseline
cell (SES_0_all × EX_0_native) is expected to pass (already did in
Probe 2) — the gate is about the rest.

### 4.5 Tie-breaking (all gates)

- "Approximately meets" is **not** meets. `net_sharpe = 1.29`,
  `n_trades = 24`, `net $/yr = 4999`, `13 / 27`, `3 / 16`,
  `7 / 16` → FAIL on the respective gate.
- Gates are evaluated on the parquet outputs exactly as produced by the
  v11 sweep engine. No post-hoc cost adjustments, no metric
  re-derivations.
- If a gate's input data is corrupt (e.g., fewer than expected bars, a
  run crash), that is an **out-of-band event** triggering compute
  investigation — NOT an auto-FAIL. The probe restarts after root-cause.

---

## 5. Branches

Let `F` = count of gates among {4.1, 4.2, 4.3, 4.4} that FAIL.

| `F` | Branch | Action |
|---:|---|---|
| 0 | **PAPER-TRADE** | All 4 gates PASS. Draft paper-trade setup (broker selection, capital sizing, monitoring dashboard). Posterior per memo §6 ≈ **0.91**. |
| 1 | **COUNCIL RE-CONVENE** | Ambiguous. Spawn a fresh LLM Council scoped to the single failed gate's implications. Council output determines whether to partial-deploy, re-probe, or sunset. |
| ≥ 2 | **SUNSET (Option Z)** | Write `tasks/project_sunset_verdict.md`; retire combo 865 @ 1h. Update `CLAUDE.md` combo-865 carve-out bullet with sunset banner. Posterior per memo §6 ≈ **<0.06**. |

Branch decision is **mechanical** once §4 gates are resolved. No
post-hoc "the 1-of-4 fail was really a 0.5-of-4 fail" reinterpretation.

### 5.1 Paper-trade trigger details (PT branch)

If `F = 0`, paper-trade setup doc must cover:

- Broker: named choice (AMP / NinjaTrader / Tradovate / other) with
  rationale.
- Instrument: MNQ (per `CLAUDE.md` non-negotiable constraints).
- Position sizing: flat `$500 risk/trade` (fixed-$500 convention
  per `CLAUDE.md` Sharpe-basis block).
- Starting paper-trade equity: $50,000 (matches project stated capital).
- Kill-switch rules: daily loss cap, weekly loss cap, monthly loss cap
  (specific $ values pre-registered in the setup doc).
- Monitoring cadence: at minimum, weekly review of paper-trade log vs
  expected distribution.
- Go-live gate: after paper-trade setup draft, reviewing agent + user
  approval before capital commits.

### 5.2 Council re-convene details (ambiguous branch)

If `F = 1`, the council's framing must include:

- Which gate failed and by how much (point estimate and σ).
- Whether the other 3 gates' passes were marginal or comfortable.
- Calibration memo's `P(PASS|H1)` and `P(PASS|H0)` for the failed gate.
- Explicit reference to `memory/feedback_council_methodology.md` Rule 1
  (Stage 1 vs Stage 2) and Rule 2 (`P(PASS|H1)/P(PASS|H0)` per gate).

### 5.3 Sunset details (Z branch)

If `F ≥ 2`, `tasks/project_sunset_verdict.md` must include:

- Observed metrics per gate with PASS/FAIL determination.
- Bayesian posterior update using memo §6 BFs (prior 0.167 → final).
- Authority chain: Probe 1 → Probe 2 (this carve-out) → Probe 3
  (this sunset).
- CLAUDE.md update: replace combo-865 carve-out bullet with sunset
  banner; link to this verdict.
- `lessons.md` entry: the 3000-combo-sweep coincidence rate, observed.

---

## 6. Irrevocable Commitments

1. **Results read mechanically.** No post-hoc reinterpretation of
   thresholds once the readout JSON is produced.
2. **Combo 865 parameters frozen.** §2.1 is the complete specification;
   no substitution or "neighboring" combos outside the ±5% neighborhood
   in §2.2.
3. **No post-hoc timeframe switching.** 1h is the test-passing TF per
   Probe 2 §4; 1h is the gate TF here. 15m is a negative control only.
   1m, 30m, 2h, etc. remain inadmissible per Probe 1 §7.6.
4. **No post-hoc gate relaxation.** §4's four gate specifications are
   binding. A `13/27` or `7/16` or `n_trades(H1) = 24` is a FAIL.
5. **No post-hoc split-date retuning.** §4.1's split at 2025-07-15
   is frozen. If trade counts end up unequal (e.g., 95 / 125), that is
   the observed partition; no rebalancing.
6. **No post-hoc session/exit grid augmentation.** §2.3 and §2.4 are
   the complete grids. No "what if we also tried the overnight-only +
   maxhold-24h" probe after the fact.
7. **Scope lock stands.** NQ/MNQ only (`memory/feedback_nq_mnq_scope_only.md`).
8. **Paper-trade still requires reviewer + user approval.** A Probe 3
   `F = 0` PASS does NOT skip the paper-trade setup review.
9. **Council re-convene must follow `memory/feedback_council_methodology.md`.**
   Rule 1 (Stage 1 vs Stage 2 multiplicity naming) and Rule 2
   (`P(PASS|H1)/P(PASS|H0)` per gate) must prepend the framing.

---

## 7. Compute Budget

| Step | Wall-clock | Notes |
|---|---|---|
| Local: write 4 probe scripts + readout | 60 min | One-time authoring |
| Local: git commit + push | 5 min | Signing commit; §9 hash |
| Remote: git pull + env sync | 2 min | Via `feedback_remote_git_sync.md` |
| §4.1 regime halves (re-use probe2 parquet) | < 1 min | No new sweep |
| §4.2 param neighborhood (27 combos × 1h test) | ~5 min | Backtest engine only |
| §4.3 15m NC (16 cells × 15m test) | ~10 min | |
| §4.4 1h ritual (16 cells × 1h test) | ~10 min | |
| SFTP pull parquets + manifests | 3 min | |
| Remote readout script | < 1 min | |
| Local: verdict document | 60 min | `tasks/probe3_verdict.md` |
| **Total** | **~2.5 hours** | ~35 min compute; rest is authoring |

**Abort threshold**: if any individual remote sweep exceeds **45
minutes** wall-clock, abort and investigate. Abort is an out-of-band
event (§4.5) separate from the gate-reading discipline.

---

## 8. References

- Probe 2 verdict (authority to probe at all): `tasks/probe2_verdict.md`
- Probe 2 preregistration (structural template): `tasks/probe2_preregistration.md`
- Probe 3 council HTML: `tasks/council-report-2026-04-21-probe3-scope.html`
- Probe 3 council transcript: `tasks/council-transcript-2026-04-21-probe3-scope.md`
- Probe 3 calibration memo (power + prior): `tasks/probe3_multiplicity_memo.md`
- Probe 1 family-level falsification (precedent + §7.6 terminal):
  `tasks/probe1_verdict.md`, `tasks/probe1_preregistration.md`
- Council methodology feedback (framing rules):
  `memory/feedback_council_methodology.md`
- Remote execution scope: `memory/feedback_remote_execution_only.md`
- Remote job workflow: `memory/reference_remote_job_workflow.md`
- NQ/MNQ scope lock: `memory/feedback_nq_mnq_scope_only.md`
- CPU/memory envelope: `memory/feedback_kamatera_cpu_cap.md`
- Bar resampling: `src/indicators/bar_resample.py`
- Sweep engine: `scripts/param_sweep.py`

---

## 9. Signature

By signing below, the signer commits to this document's decision rule
without post-hoc reinterpretation. §4 thresholds, §5 branch routing,
and §6 irrevocable commitments are all mechanically applied to the
observed metrics.

- **Signed**: _(awaiting signer; countersign via explicit "Sign
  Probe 3 at commit `<hash>`" reply; signing is a separate commit from
  this drafting commit)_
- **Date/time of signature**: _(populated at signing)_
- **Commit hash at time of signature**: _(populated at signing; see
  `git log` immediately after the signing commit lands)_
