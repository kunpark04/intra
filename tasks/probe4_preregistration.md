# Probe 4 Pre-Registration — Cross-TF-Coherence Property Test (Path B · B2 second-combo carve-out)

**Date**: 2026-04-22 UTC
**Authority**: LLM Council verdict 2026-04-22, `tasks/council-report-2026-04-22-b2-scope.html`
**Predecessor**: `tasks/probe3_verdict.md` (Probe 3 F=0, branch=PAPER_TRADE — NOT executed; user selected Path B instead of Path A)
**Semantic label**: Path B sub-option B2 (second-combo carve-out within the Z-score mean-reversion family)
**Template**: Mirrors `tasks/probe3_preregistration.md`.
**Process discipline**: Thresholds committed before the experiment runs, applied mechanically after. Ties go to the stricter (more-cautious) side — borderline outcomes route to INCONCLUSIVE or COUNCIL_RECONVENE rather than to PROPERTY_VALIDATED.

---

## Status

**UNSIGNED DRAFT** as of 2026-04-22 UTC.

Awaits: (1) pre-sign review bus round by `code-logic-reviewer` + `stats-ml-logic-reviewer`, (2) user explicit authorization, (3) signing commit (a separate git commit recording this document at the freeze-point hash). No §4 gate rules, §5 branch routing, or §6 commitments are binding until signing.

---

## 1. Purpose

Test whether the **cross-TF-coherence property** that admitted combo-865 to Probe 2 (top-10 on both 15m and 1h in the 3000-combo Probe 1 sweep) identifies **genuine edge** or is a **Stage-1 selection artifact reverse-engineered from 865's observed success**.

This is a **property test**, not a combo generalization test. Probe 2 asked "does combo-865 hold out of sample?" Probe 4 asks "does the selection procedure that produced combo-865 reliably identify held-out edge, or does it over-produce false positives?"

The experimental design follows the First Principles council reframe (2026-04-22 council, unanimous reviewer pick): **combo-664 is the treatment, not a negative control.** Combo-664 shares identical microstructure with combo-865 but fails the Probe 1 gross-Sharpe pre-gate (1.200 vs ≥ 1.3). Under H0 (property is overfit), 664 should pass the held-out absolute gate at the same rate as combo-1298 (rank-1 gross Sharpe 2.272). Under H1 (property identifies edge), 664 should systematically underperform 1298.

The probe also answers a **session-structure confound** question that Probe 3 surfaced post-hoc but did not test prospectively: combo-865's 1h edge concentrates overnight (per-trade net ~3.45× SES_2 overnight vs SES_1 RTH). If 1298 and 664 share that session structure, the signal under test is session-structural rather than combo-identity-structural, and Path B's B1 (session-structure sweep) is logically prior to further B2-style carve-outs. §4.4 decomposes every gate by session to surface this prospectively.

**Branch map** (full spec in §5):
- **PROPERTY_VALIDATED** → 1298 PASS absolute, 664 FAIL absolute, Welch-t ≥ 2.0. Authorizes B1 (signal family / session-structure probe) as a generalization follow-on.
- **SESSION_CONFOUND** → 1298 PASS absolute, but edge concentrates in SES_2 only (mirroring 865), per the SES_2 vs SES_1 inequality in §5 row 2. B1 session-structure probe is logically prior; no further combo-level B2 work until B1 resolves.
- **INCONCLUSIVE** → 1298 FAILs absolute AND Welch-t < 2.0. Property did not survive one admitted test; B2 retires; B1 remains independently open.
- **COUNCIL_RECONVENE** → Both 1298 and 664 PASS absolute (both-pass adjudication: weak property validation vs property also present in 664), or any unforeseen anomaly (e.g. 1298 abs FAIL but Welch-t ≥ 2.0). Fresh council adjudicates.

Probe 4 does **NOT** reopen the bar-timeframe axis (Probe 1 §7.6 remains terminal); does **NOT** test 1149 (dropped per council: 101 trades cannot survive a regime-halves sub-gate; distant parameters = separate hypothesis); does **NOT** modify combo-865 evidence (Probe 3 verdict stands).

---

## 2. Combos Under Test

### 2.1 Combo-1298 (frozen, Probe 1 rank-1 of 3000 on 1h gross Sharpe)

Source: v11 sampler (`--range-mode v11 --seed 0`), combo_id = 1298. Parameter dict extracted from `data/ml/originals/ml_dataset_v11_1h.parquet` (498 training-partition trades; all 29 parameter columns verified constant across trades via `tasks/_extract_probe4_params.py`, output cached at `tasks/_probe4_param_dicts.json`).

| Parameter | Value | Probe 4 status |
|---|---|---|
| `z_band_k` | 2.268105 | frozen |
| `z_window` | 14 | frozen |
| `z_input` | `returns` | frozen |
| `z_anchor` | `rolling_mean` | frozen |
| `z_denom` | `parkinson` | frozen |
| `z_type` | `parametric` | frozen |
| `z_window_2` | 0 | frozen |
| `z_window_2_weight` | 0.0 | frozen |
| `volume_zscore_window` | 12 | frozen |
| `ema_fast` | 4 | frozen |
| `ema_slow` | 15 | frozen |
| `stop_method` | `fixed` | frozen |
| `stop_fixed_pts` | 15.199019 | frozen |
| `atr_multiplier` | NaN (unused) | frozen |
| `swing_lookback` | NaN (unused) | frozen |
| `min_rr` | 2.786175 | frozen |
| `max_hold_bars` | 480 | frozen |
| `exit_on_opposite_signal` | True | frozen |
| `use_breakeven_stop` | False | frozen |
| `zscore_confirmation` | True | frozen |
| `volume_entry_threshold` | 0.0 | frozen |
| `vol_regime_lookback` | 0 | frozen |
| `vol_regime_min_pct` | 0.0 | frozen |
| `vol_regime_max_pct` | 1.0 | frozen |
| `session_filter_mode` | 0 | §4.4 grid override (SES_0 baseline / SES_1 RTH / SES_2 GLOBEX) |
| `tod_exit_hour` | 0 | frozen |
| `entry_timing_offset` | 0 | frozen |
| `fill_slippage_ticks` | 2 | frozen |
| `cooldown_after_exit_bars` | 3 | frozen |

**Probe 1 readout properties** (continuity of narrative; not derived from the dict):
- 1h gross Sharpe 2.272 (rank 1 of 3000 on that timeframe)
- 1h trade count 498
- 15m gross Sharpe 1.167 (rank 14 — top-20 on both TFs)
- Microstructure delta from combo-865 (865 dict per `tasks/probe3_preregistration.md` §2.1): `entry_timing_offset` Δ=−1 (1298=0 vs 865=1), `fill_slippage_ticks` Δ=+1 (1298=2 vs 865=1), `cooldown_after_exit_bars` Δ=0 (both 3) — confirmed against the extracted dict above.

### 2.2 Combo-664 (frozen, Probe 1 rank-6 on 1h but FAILS Probe 1 pre-gate)

Source: v11 sampler (`--range-mode v11 --seed 0`), combo_id = 664. Parameter dict extracted from `data/ml/originals/ml_dataset_v11_1h.parquet` (2,658 training-partition trades; all 29 parameter columns verified constant across trades via `tasks/_extract_probe4_params.py`).

| Parameter | Value | Probe 4 status |
|---|---|---|
| `z_band_k` | 1.780825 | frozen |
| `z_window` | 7 | frozen |
| `z_input` | `returns` | frozen |
| `z_anchor` | `rolling_mean` | frozen |
| `z_denom` | `n/a` | frozen |
| `z_type` | `quantile_rank` | frozen |
| `z_window_2` | 30 | frozen |
| `z_window_2_weight` | 0.277201 | frozen |
| `volume_zscore_window` | 29 | frozen |
| `ema_fast` | 8 | frozen |
| `ema_slow` | 50 | frozen |
| `stop_method` | `fixed` | frozen |
| `stop_fixed_pts` | 38.867811 | frozen |
| `atr_multiplier` | NaN (unused) | frozen |
| `swing_lookback` | NaN (unused) | frozen |
| `min_rr` | 1.183628 | frozen |
| `max_hold_bars` | 15 | frozen |
| `exit_on_opposite_signal` | False | frozen |
| `use_breakeven_stop` | False | frozen |
| `zscore_confirmation` | True | frozen |
| `volume_entry_threshold` | 0.0 | frozen |
| `vol_regime_lookback` | 0 | frozen |
| `vol_regime_min_pct` | 0.0 | frozen |
| `vol_regime_max_pct` | 1.0 | frozen |
| `session_filter_mode` | 0 | §4.4 grid override (SES_0 baseline / SES_1 RTH / SES_2 GLOBEX) |
| `tod_exit_hour` | 0 | frozen |
| `entry_timing_offset` | 1 | frozen |
| `fill_slippage_ticks` | 1 | frozen |
| `cooldown_after_exit_bars` | 3 | frozen |

**Probe 1 readout properties** (continuity of narrative; not derived from the dict):
- 1h gross Sharpe 1.200 (rank 6 on 1h — fails Probe 1 gate of ≥ 1.3)
- 1h trade count 2,658
- 15m gross Sharpe 1.182 (rank 13)
- Microstructure: identical to combo-865 on all three axes (`entry_timing_offset`=1, `fill_slippage_ticks`=1, `cooldown_after_exit_bars`=3) — confirmed against the extracted dict above. This identity is the core experimental rationale: 664 isolates "non-microstructure parameter realization" as the variable distinguishing it from 865.

### 2.3 Combo-1149 — EXCLUDED

Per council verdict: 101 trades at 1h cannot survive a Probe-3-style regime-halves sub-gate (each half would have ~50 trades, below stability threshold); distant cooldown parameter (Δ=+7 from 865) tests a separate hypothesis (distant-param replication rather than the coherence property). Inclusion would contaminate the property test. Not part of Probe 4 scope.

---

## 3. Data partition

**Identical to Probe 2 and Probe 3**: 1h bars on the chronological test partition `2024-10-22 00:00 UTC → 2026-04-08 00:00 UTC` (the 20% held-out tail of the full NQ 1-min bar series after the 80/20 split). Sourced from `data/NQ_1min.csv` resampled to 1h.

No 15m component in Probe 4. The 15m negative control structure from Probe 3 §4.3 already established that 15m is friction-binding; running it again on 1298/664 does not add information.

---

## 4. Gate structure

Three gates plus a non-gate-bound session decomposition readout. Gate passes computed mechanically from `data/ml/probe4/readout.json` after the remote run completes. No manual interpretation during application.

### 4.1 Absolute gate on combo-1298 (secondary)

Combo-1298 is run on the test partition with its frozen parameter dict. Three sub-gates must all hold:

- `net_sharpe` ≥ **1.3**
- `n_trades` ≥ **50**
- `net_dollars_per_year` ≥ **$5,000**

Friction: $5/contract RT (same as Probes 1/2/3). Sizing: fixed $500 risk per trade. All three sub-gates together define `gate_1298_abs_pass`.

### 4.2 Absolute gate on combo-664 (secondary)

Same three sub-gates applied to combo-664's frozen dict on the same test partition. Together define `gate_664_abs_pass`.

### 4.3 Per-trade Welch t-test (Sharpe-differential signal) — PRIMARY

Compute Welch's t-statistic on the per-trade net PnL distributions of combo-1298 vs combo-664 on the 1h test partition:

t = (mean_pnl_1298 − mean_pnl_664) / sqrt((var_1298 / n_1298) + (var_664 / n_664))

**Gate threshold**: `t ≥ 2.0` (one-tailed, H1: 1298 > 664). This corresponds to a preregistered α ≈ 0.025 under the large-sample Welch approximation — tighter than the original 1.3 Sharpe-differential threshold (which stats-ml-logic-reviewer's C1 audit showed was calibrated to P(PASS | H0) ≈ 13.3% given σ_diff ≈ 1.17 at the 1.48-year window, 2.3× looser than the absolute gate's 5.7%).

Both combos must have `n_trades ≥ 50` on the test partition for this gate to be computable. If either has fewer, the primary gate defaults to FAIL.

Readout JSON semantics: when either combo has n_trades < 50, the readout emits `"welch_t": null` and `"welch_gate_pass": false`. The branch-routing comparison in §5 treats `welch_gate_pass == true` as the explicit positive; `false` (from either insufficient-n or t < 2.0) as the explicit negative; `null` is never compared directly. This prevents silent row-5 dumping from JSON type ambiguity.

Why Welch (not pooled-t): per-trade PnL variances for 1298 and 664 are not expected to be equal (different trade frequencies, different sizing-footprint distributions). Welch makes no equal-variance assumption.

Why per-trade (not annualized Sharpe differential): annualized Sharpe SE is dominated by window duration (σ(SR_ann) ≈ sqrt(1/years_span) ≈ 0.82 at 1.48 years, almost independent of n). Per-trade Welch-t exploits the full n_1 + n_2 − 2 degrees of freedom, which is the real statistical leverage available here.

### 4.4 Session decomposition (mandated readout; non-gate-bound)

Every metric computed in §4.1/§4.2/§4.3 is additionally decomposed by session filter:

- **SES_0** — all sessions (baseline)
- **SES_1** — RTH only (09:30–16:00 ET)
- **SES_2** — overnight / GLOBEX (excluding RTH)

(Session definitions mirror Probe 3 §4.4. SES_3 RTH-minus-lunch is not reported — Probe 3 established it is consistently thinnest and not decision-relevant.)

Session decomposition is **not gate-bound**. Its purpose is to surface the session-confound branch route in §5 prospectively. If 1298 passes only in SES_2 (mirroring 865's overnight concentration), §5 routes to SESSION_CONFOUND regardless of whether the aggregate absolute and relative gates pass.

---

## 5. Branch routing (mechanical)

Applied in order. First match wins. Routing computed by `tasks/_probe4_readout.py` after all gate JSONs land.

| # | Condition | Branch |
|---|---|---|
| 1 | `gate_1298_abs_pass` == FALSE AND Welch-t < 2.0 | **INCONCLUSIVE** |
| 2 | `gate_1298_abs_pass` == TRUE AND SES_2 absolute PASS AND SES_1 absolute FAIL AND (SES_2 net Sharpe − SES_1 net Sharpe) > 1.0 | **SESSION_CONFOUND** |
| 3 | `gate_1298_abs_pass` == TRUE AND `gate_664_abs_pass` == FALSE AND Welch-t ≥ 2.0 | **PROPERTY_VALIDATED** |
| 4 | `gate_1298_abs_pass` == TRUE AND `gate_664_abs_pass` == TRUE | **COUNCIL_RECONVENE** (both-pass adjudication: is this weak property validation, or is the property also present in 664?) |
| 5 | None of the above (including `gate_1298_abs_pass` == FALSE with Welch-t ≥ 2.0 anomaly, or any other unforeseen combination) | **COUNCIL_RECONVENE** |

Ordering is load-bearing. In particular, SESSION_CONFOUND (row 2) fires **before** PROPERTY_VALIDATED (row 3), so a 1298 that passes absolute and beats 664 on Welch-t but does so only in SES_2 routes to SESSION_CONFOUND rather than validation.

### 5.1 Downstream bindings per branch

- **PROPERTY_VALIDATED** binding: next step is LLM Council on B1 (signal family / session-structure probe) scope. No paper-trade authorization for 1298 itself. This is a family-level evidence result, not a combo-level deployment result.
- **SESSION_CONFOUND** binding: B1 preregistration becomes the next authoring step, specifically a session-structure sweep across multiple combos (not a single-combo carve-out). B2 parks.
- **INCONCLUSIVE** binding: B2 retires; Path B continues via B1 only. No retroactive change to Probe 3 verdict.
- **COUNCIL_RECONVENE** binding: fresh LLM Council adjudicates. Two distinct paths land here: (a) both combos pass absolute (row 4) — council weighs whether the result is weak property validation or evidence the property is also present in 664; (b) the anomaly path (row 5) where 1298 fails absolute yet Welch-t ≥ 2.0, or any other unforeseen combination. No automatic memory update or downstream binding fires until the council writes a verdict.

**Ordering interaction disclosure**: if combo-1298 and combo-664 both PASS absolute AND combo-1298's edge concentrates in SES_2 only (SES_2 abs PASS, SES_1 abs FAIL, (SES_2 Sharpe − SES_1 Sharpe) > 1.0), the case routes to SESSION_CONFOUND (row 2) rather than to COUNCIL_RECONVENE both-pass adjudication (row 4), because row 2 fires first under ordered-match. This is intentional — a session-confounded signal is diagnostically more informative than a both-pass adjudication, and the binding session-structure interpretation should not be suppressed by concurrent 664 absolute pass.

### 5.2 Narrow-miss disclosure

If `gate_1298_abs_pass == FALSE` because `net_sharpe_1298` falls in [1.1, 1.3) AND Welch-t ≥ 2.0, the case routes to **COUNCIL_RECONVENE via §5 row 5** (row 1 does not fire because Welch-t ≥ 2.0 violates row 1's `Welch-t < 2.0` conjunction). The verdict document must explicitly flag this as a **narrow-miss anomaly**: 1298 demonstrated Sharpe-differential edge over 664 but missed the absolute floor by a small margin. Council adjudication (not mechanical INCONCLUSIVE retirement) is the correct routing because the anomaly warrants interpretation rather than automatic dismissal. Do not post-hoc promote to PROPERTY_VALIDATED.

---

## 6. Irrevocable commitments

1. **No mid-flight gate edits.** §4 gate thresholds and §5 branch routing apply as written at the signing commit hash. No relaxation, no tightening, no "the observed t was 1.95 which is close enough to 2.0" re-interpretation.
2. **No early-stop inspection.** The full 2-combo × 3-session-decomposition suite runs to completion on sweep-runner-1 before any readout is read. `_probe4_readout.py` writes `readout.json` atomically after all gate JSONs land.
3. **Results read mechanically.** Branch routing looked up from §5 table. No subjective re-interpretation of borderline metrics.
4. **No post-hoc methodology shift.** The session decomposition in §4.4 is a readout requirement; its results cannot be promoted to a primary gate after the data lands.
5. **Scope lock.** Probe 4 tests combos 1298 and 664 on 1h only. No spillover to 1149, other combos, 15m, other timeframes, other instruments.
6. **1298 is not deployed under any branch.** PROPERTY_VALIDATED authorizes B1 as a follow-on; it does NOT authorize 1298 paper trading, live trading, or any form of capital-adjacent deployment. Combo-865 paper-trade remains on hold per user's Path B selection.
7. **Stage-0 multiplicity is acknowledged, not Bonferroni-inflated.** The Council verdict explicitly declined Bonferroni α/3 inflation of the absolute gate. The Welch-t ≥ 2.0 primary gate (§4.3) is how this probe pays for the Stage-0 selection cost. Any post-hoc proposal to tighten the absolute gate after the data lands is out of scope per §6.1.

---

## 7. Methodology disclosures (transparency for future auditors)

### 7.1 No post-hoc reinterpretation
Same rule as Probes 1/2/3. Observations outside the §5 table route to COUNCIL_RECONVENE, not to creative re-application of gates.

### 7.2 Rule 2 per-gate and per-branch probability table (binding)

Populated from stats-ml-logic-reviewer pre-sign audit (2026-04-22). Derived assuming σ(SR_ann) ≈ 0.82 per combo at the 1.48-year test window (Jobson-Korkie SE formula), σ_diff ≈ 1.17 under independence, per-trade PnL variance from Probe 2 baseline (std ~$3,538/trade at 1h).

**Per-gate probabilities**:

| Gate | P(PASS \| H1a strong) | P(PASS \| H1b moderate) | P(PASS \| H1c shared-weak) | P(PASS \| H0) | LR H1a/H0 |
|---|---:|---:|---:|---:|---:|
| §4.1 1298 abs ≥ 1.3 | 0.974 | 0.730 | 0.500 | 0.057 | 17.1× |
| §4.2 664 abs ≥ 1.3 | 0.057 | 0.057 | 0.165 | 0.057 | 1.0× |
| §4.3 Welch-t ≥ 2.0 | ~0.95 (at α=0.025) | ~0.75 | ~0.30 | 0.025 | ~38× |

(§4.3 row updated from reviewer's original Sharpe-differential calculation to reflect the Welch-t ≥ 2.0 primary gate per Edit 1. Under H0, Welch-t has exact P(PASS) = α = 0.025. Under H1, approximate power at the expected Δ from Probe 2/3.)

**Branch-route probabilities**:

| Branch | P(route \| H1a strong) | P(route \| H1b moderate) | P(route \| H1c shared-weak) | P(route \| H0) |
|---|---:|---:|---:|---:|
| PROPERTY_VALIDATED | ~0.88 | ~0.55 | ~0.10 | ~0.0014 |
| SESSION_CONFOUND | ~0.10 (if 1298 shares 865's overnight pattern) | lower | lower | ~0.01 |
| INCONCLUSIVE | ~0.025 | ~0.22 | ~0.50 | ~0.94 |
| COUNCIL_RECONVENE | ~0.00 | ~0.20 (both pass + small Welch) | ~0.35 | ~0.05 |

PROPERTY_VALIDATED LR ≈ **629× (H1a vs H0)** under the Welch-t ≥ 2.0 primary gate — substantially stronger discrimination than the 1.3 Sharpe-differential the draft originally proposed (263×). Design is well-powered on validation; PROPERTY_FALSIFIED has been removed because it could not achieve comparable discrimination at the sample sizes available.

### 7.4 Trade count discrepancy
Combo-1298 has 498 training-partition trades at 1h vs combo-664's 2,658. This asymmetry is a known feature of the v11 sampler, not a confound — it reflects different entry frequencies from the parameter dicts. Both easily clear the 50-trade floor on the held-out partition. Sample-size variance on per-combo Sharpe will differ and is expected. The Welch-t primary gate exploits the n asymmetry through the degrees-of-freedom denominator (Satterthwaite approximation), which is a consequence of the Welch formula rather than a designed-in property of this probe.

### 7.5 No family-level claim attached to PROPERTY_VALIDATED
Per Probe 1 §7.6, the family-level bar-timeframe falsification is terminal. Even if Probe 4 routes to PROPERTY_VALIDATED, this does not re-open the family-level question. It only says: "the property that singled out 865 is not pure noise." Two 1h survivors is still not N_1.3 ≥ 10. For family-level combo-counting purposes, combo-1298's Δ=1 microstructure neighborhood from combo-865 means these two combos are closer to **one observation than two** in the same parameter basin. Probe 4 testing 1298 does NOT produce a second independent basin-survivor for Probe 1 N_1.3 counting purposes under any branch routing.

### 7.6 Partition-reuse caveat

The 2024-10-22 → 2026-04-08 1h test partition has now been consumed by Probe 2 (combo-865 PASS), Probe 3 (combo-865 4-gate PASS), and is being consumed again by Probe 4 (combos 1298, 664). Combos 1298 and 664 have never been run against this partition — so no strict data leakage exists — but the partition as a whole has been "looked at" enough that if the window is mildly favorable to Z-score mean-reversion as a family, both absolute gates (§4.1, §4.2) inherit that favorability. The primary Welch-t gate (§4.3) is **differential** between 1298 and 664, which cancels common-factor window favorability; the absolute gates do not. Verdict interpretation must respect this: PROPERTY_VALIDATED should lean on the Welch-t result as primary evidence, not on 1298's absolute Sharpe in isolation.

---

## 8. Execution plan

### 8.1 Locality
Remote on sweep-runner-1 per `feedback_remote_execution_only.md`. Local authoring (this document, gate scripts, readout script, bus artifacts) is admissible; compute is not.

### 8.2 Scripts to author (coding-agent task, post-signing)
- `tasks/_probe4_run_combo.py` — runs one combo's frozen parameter dict against the 1h test partition with session decomposition; writes per-session JSON. Execution model: one backtest run per (combo, session filter) pair, using the existing `session_filter_mode` override (Probe-3-style, per §4.4 of `tasks/probe3_preregistration.md`). This yields 6 backtest runs total (2 combos × 3 session filters {SES_0, SES_1, SES_2}). The Welch-t §4.3 gate uses the SES_0 (all-sessions) trade series from each combo; the per-session decomposition in §4.4 uses the SES_1 and SES_2 runs for the row-2 SESSION_CONFOUND operationalization. Single-run-with-tagging was considered and rejected because it would require engine changes outside the pre-sweep-gate-validated code surface.
- `tasks/_probe4_readout.py` — aggregates per-combo JSONs into `data/ml/probe4/readout.json`, applies §5 branch routing, writes final verdict row.
- `tasks/_run_probe4_remote.py` — paramiko orchestrator per `reference_remote_job_workflow.md`. Uploads scripts, git-pulls on remote (per `feedback_remote_git_sync.md`), launches under `systemd-run --scope -p MemoryMax=9G`, polls via screen/tail/free.

### 8.3 Artifacts produced
- `data/ml/probe4/readout.json` — machine-readable branch routing
- `data/ml/probe4/combo1298_{SES_0,SES_1,SES_2}.parquet` — per-session trade tables
- `data/ml/probe4/combo664_{SES_0,SES_1,SES_2}.parquet`
- `data/ml/probe4/combo1298_gate.json` + `combo664_gate.json`
- `data/ml/probe4/combo1298_SES_0_trades.parquet` — per-trade net_pnl_dollars series for Welch-t computation (§4.3 primary gate input)
- `data/ml/probe4/combo664_SES_0_trades.parquet` — same, for combo-664
- `tasks/probe4_verdict.md` (written after remote run; not part of this prereg)

### 8.4 Compute envelope
Two frozen combos × 3 session filters × 1 timeframe × 1 partition ≈ 6 backtest runs. Estimated wall-clock on sweep-runner-1: < 10 minutes (combo-865 Probe 2 took ~2 minutes per session on 1h).

---

## 9. Pre-sign review workflow

Mirrors Probe 3's review flow, adapted to the agent-bus infrastructure shipped 2026-04-22:

1. **Unsigned draft** (this document). Committed locally to `tasks/probe4_preregistration.md` at a WIP hash not flagged as signing.
2. **Bus run-id established** at `tasks/_agent_bus/probe4_2026-04-22/`.
3. **Parameter manifest population**: COMPLETED 2026-04-23 (this draft). Queried `data/ml/originals/ml_dataset_v11_1h.parquet` for combo_id ∈ {1298, 664} via `tasks/_extract_probe4_params.py`; verified all 29 parameter columns constant across each combo's training-partition trades; cached at `tasks/_probe4_param_dicts.json`; populated §2.1/§2.2 tables in this document. (Note: prior draft referenced `data/ml/mfe/ml_dataset_v11_mfe.parquet`; v11 emits MFE inline per CLAUDE.md, so the canonical 1h parquet is `data/ml/originals/ml_dataset_v11_1h.parquet` — corrected here.)
4. **`code-logic-reviewer` audit** via agent bus: validates gate script authoring plan (§8.2) against engine semantics, edge cases, and off-by-one risks. Writes findings to `tasks/_agent_bus/probe4_2026-04-22/code-logic-reviewer.md`.
5. **`stats-ml-logic-reviewer` audit** via agent bus: validates §4 gate math, §5 branch routing coverage, §7.2 Rule 2 probability table, Stage-0 multiplicity treatment. Writes findings to `tasks/_agent_bus/probe4_2026-04-22/stats-ml-logic-reviewer.md`.
6. **CRITICAL findings**: must be resolved (either fix or explicit council override) before signing. WARN findings: document inline or defer with rationale.
7. **User explicit authorization to sign.** No signing without.
8. **Signing commit**: a separate commit tagged with this document at its freeze-point hash, functionally analogous to Probe 3 commits `f8447af` / `8636167`.

---

## 10. References

### Authority
- Council verdict: `tasks/council-report-2026-04-22-b2-scope.html` + `tasks/council-transcript-2026-04-22-b2-scope.md`

### Predecessors (evidence chain)
- `tasks/probe1_verdict.md` (family-level sunset; Branch A)
- `tasks/probe2_verdict.md` (combo-865 @ 1h PASS on held-out)
- `tasks/probe3_verdict.md` (combo-865 4-gate robustness PASS; F=0)
- `tasks/probe3_multiplicity_memo.md` (multiplicity framework authority)

### Memory
- `project_probe3_combo865_pass.md` — posterior [0.65, 0.85], §4.4 at floor
- `project_probe2_combo865_1h_pass.md`
- `project_probe1_branch_a_verdict.md`
- `feedback_council_methodology.md` — Rule 1 multiplicity + Rule 2 probability framing
- `feedback_report_corrected_truth_only.md` — no correction-surfacing in downstream reports
- `feedback_nq_mnq_scope_only.md` — scope fence
- `feedback_remote_execution_only.md` — compute locality
- `project_post_probe3_fork_and_resume.md` — Path A/B/C context

### Code conventions
- `reference_remote_job_workflow.md` — paramiko → screen + systemd-run → SFTP artifacts
- `feedback_lgbm_threads.md` — not applicable (no LightGBM in Probe 4)
- `feedback_kamatera_cpu_cap.md` — applicable if any step runs > 1 hour; Probe 4 does not
