# LLM Council Transcript — Probe 3 Scope (Session-Structure Sweep Axes)

- **Date**: 2026-04-21 UTC
- **Trigger**: `tasks/probe2_verdict.md` §Next-Actions — "Convene LLM Council on Probe 3 scope" (preregistration §5.6 binding).
- **Scope-lock**: NQ/MNQ only (`memory/feedback_nq_mnq_scope_only.md`); 1h timeframe only per Probe 2 §4 PASS branch; §7.6 of Probe 1 remains terminal for bar-TF axis.
- **Sizing convention**: flat $500 risk per trade (user directive 2026-04-21 UTC — locks project-standard convention for Probe 3 and supersedes the v11 sweep engine's 5%-of-equity sizing for evaluation).
- **Compute**: 5 advisors in parallel + 5 anonymized peer reviewers in parallel + 1 chairman synthesis.

---

## 1. Framed Question (given to all five advisors)

Probe 2 PASSED on 1h (combo-865 isolation, held-out test partition 2024-10-22 → 2026-04-08):

| Timeframe | n_trades | net Sharpe | net $/yr | §4 verdict |
|---|---:|---:|---:|:---:|
| 15m | 780 | −0.55 | −$43,968 | FAIL |
| 1h | 220 | **+2.89** | **+$124,896** | **PASS** |

Preregistration §4 PASS-branch authorizes Probe 3 (Option Y session-structure sweep) on
the test-passing timeframe (1h only). §5.6 of Probe 2 preregistration requires a fresh
council before Probe 3 signing — session-structure probes have their own axis decisions.

**Council task**: recommend Probe 3 scope decisions along the following 6 axes, and flag
any axis the council wants to add or remove:

1. **Session-filter set**: which buckets? {all / RTH / RTH-lunch-exclude / overnight-only}?
   Broader (7+ buckets)? Or narrower (≤3)?
2. **Parameter holdover rule**: freeze all 28 non-session params from combo 865, or sweep
   a local neighborhood (±δ) around 865's parameter dict?
3. **Timeframe lock**: 1h only (per §4) or is there a principled exception?
4. **Trade-count floor**: Probe 2 produced only 220 trades on 1h over 1.48 years;
   session filters will halve or quarter that. What's the preregistered `n_trades_min`?
5. **Regime-window stress test**: should Probe 3 preregister a temporal sub-window split
   (e.g. halves of the test window) as an additional gate, to pressure-test whether the
   2024-10-22 → 2026-04-08 window was regime-favorable to combo 865?
6. **Sizing / friction model**: confirm flat $500/trade sizing and $5/contract RT friction
   as locked, or revisit?

**Key constraints to respect**:
- Probe 1 §7.6 terminal on bar-timeframe axis still binds — 30min / 2h / 5min not admissible.
- Probe 2 PASS is **one combo × one TF × one test window** — narrow evidence base.
- Scope lock stands: NQ/MNQ only, no cross-instrument.
- Test-partition net Sharpe (2.89) **exceeds** training-partition gross Sharpe (2.098) —
  an unusual inversion flagged in `tasks/probe2_verdict.md` §Relationship-to-Probe-1 as
  requiring Probe-3 pressure-testing for (a) regime favorability and (b) small-sample variance.
- The project's ML#2 stacks V3 and V4 BOTH shipped on clean OOS evaluation and subsequently
  failed combo-agnostic audits — Probe 3 design must consider this precedent.

---

## 2. Advisor Responses (independent, 150–300 words each)

### The Contrarian — Narrow probe with preregistered sunset trigger

Combo 865 at 1h is a post-hoc survivor of a 13,814-combo sweep. Test-partition net Sharpe
2.89 on 220 trades sounds impressive, but the *null* distribution of the maximum Sharpe
from 13,814 independent draws has a large upper tail — and 220 trades is a small sample
where Sharpe estimator noise alone produces ±0.5 swings. Before any session probe runs,
the honest Probe 3 design should preregister a **multiplicity-corrected Sharpe floor**,
not the naive 1.3 gate. At α=0.05 with Bonferroni over ~9,000 effective tests (post-trade-
count gate), the required Sharpe on 220 trades is closer to **3.5–3.8**, not 1.3 — which
means combo 865 *already* fails the proper corrected gate before Probe 3 starts.

Given that, Probe 3's role should be falsification, not confirmation. Recommendations:

- **≤4 session buckets** — {all / RTH / RTH-lunch-exclude / overnight-only}. Seven buckets
  triples the multiplicity burden for no preregistered payoff.
- **Freeze all 28 non-session params.** No "combo 865-neighborhood sweep." That path leaks
  researcher degrees of freedom back into a probe that's supposed to close them.
- **Regime halves as a HARD gate.** Split the 1.48-year test window into halves; both halves
  must pass the multiplicity-corrected Sharpe gate independently OR Probe 3 sunsets the combo.
- **15m as mandatory negative control.** If 15m ever passes a bucket where 1h doesn't, the
  pipeline has a bucket-level artifact and Probe 3 is invalid.
- **Preregistered Option Z sunset trigger**, in writing, before signing: if Probe 3 fails
  the multiplicity-corrected gate AND the regime-halves gate, the combo-865 carve-out
  retires and the Z-score family sunsets on NQ end-to-end.

### The First Principles Thinker — Reframe the probe as an exit ritual

The framing "session-structure sweep to confirm combo 865's edge" is already wrong. The
project's actual state is: combo 865 survived (a) a 13,814-combo sweep, (b) an ML#1-driven
ranker, (c) a cross-TF consistency check, and (d) a 2-gate §4 readout on one test window.
Each filter already concentrated the posterior; adding a 7-bucket session sweep without a
matching posterior update is ceremony, not science.

The real question Probe 3 should answer is narrower and more honest: **"Given combo 865's
1h PASS on one held-out window, what is P(real edge) and what is the minimum additional
evidence that moves that posterior past the paper-trading threshold?"**

Answering that requires three orthogonal probes, not a session sweep:

1. **Multiplicity-adjusted inference** — compute the Bonferroni-corrected Sharpe floor
   against the full decision-tree denominator (13,814 combos × K ranker decisions × 2 TFs).
   Commit that number pre-probe. Gate on it.
2. **Temporal halves as primary evidence** — not as a "side test." If combo 865 only passes
   on one half of the test window, that's the finding; session sweeps will not rescue it.
3. **Parameter-neighborhood stability** — does combo 865's edge survive ±5% jitter on
   `z_band_k`, `stop_fixed_pts`, `min_rr`, `max_hold_bars`? Hard sparse edges are diagnostic
   of overfitting; stable neighborhoods are diagnostic of signal.

Session buckets, if run at all, belong as a supplementary 4-bucket probe after the primary
three gate on PASS. Freeze all non-session params. Flat $500 sizing. Preregister sunset
trigger. 15m is a negative control, not an upside probe.

### The Expansionist — Wide session grid + parameter neighborhood + 1m/5m upside probes

Combo 865 at 1h on the test window delivered Sharpe 2.89, $124k/yr, 220 trades — after
friction, on held-out bars, with a pre-registered gate. That is the single strongest clean
result this project has ever produced, and the council is one peer-review round away from
drafting a probe that treats it like a suspect rather than a lead.

Probe 3 should be **designed for upside discovery**, not just falsification:

- **7 session buckets**: all / RTH-full / RTH-morning / RTH-afternoon / RTH-lunch-exclude /
  overnight-only / Asia+Europe. Session-structure effects are known on NQ; if combo 865's
  edge localizes to one bucket, that doubles or triples its economic value per hour of
  attention.
- **Parameter neighborhood** around 865: ±5% on continuous params, ±1 bar on discrete
  window params. Stable neighborhoods are the ship-readiness signal. This is ~40 combos
  × 4 buckets = ~160 cells; one day of compute.
- **1m and 5m probe** on combo 865's parameter dict. §7.6 of Probe 1 sunsets the
  *family* at intermediate bar-TFs, but combo 865's carve-out is explicitly a single-
  realization exception — the council cleared this path in Probe 2's §4. If 1h's edge
  extends to 5m, that's a live trading multiplier.
- **Kelly-fractional diagnostic**: compute optimal-Kelly sizing on the 220 trades as a
  diagnostic (not for live sizing). If Kelly f* < 0.02, the edge is economically thin
  even if statistically real; if f* > 0.15, the economic case is strong.
- **15m stays in Probe 3** as a live second-TF candidate, not just a negative control —
  the friction explanation is known, the question is whether session filtering unlocks it.

Flat $500 sizing is fine. Preregister an Option Z sunset trigger if the user wants one —
but don't let falsification-discipline optimism-starve the one positive result we have.

### The Outsider — Freeze everything, structure for an outside reader

I don't know this project's history and I'm reading Probe 2 for the first time. What I see:
one combo, one timeframe, one held-out window, PASS — with a footnote that test Sharpe
exceeds train Sharpe, which is *the* classic regime-window artifact pattern. A reader who
didn't grow up with this project would ask three questions before Probe 3 runs:

1. "Is it stable in time?" — The test>train inversion is a red flag. Before Probe 3 burns
   compute on session structure, split the test window in halves and run the gates on each.
   Both halves must pass, or Probe 3 is chasing a window-favorability artifact.
2. "Is it stable in parameters?" — The param dict has 28 knobs. Was combo 865 picked *because*
   it PASSED in a 13,814-combo sweep? (Yes.) Then freeze the knobs and test SESSION only.
   Don't jitter them — that restarts the multiplicity game.
3. "What does a FAIL look like, and who authorizes it?" — If Probe 3 comes back mixed
   (e.g. RTH passes, overnight fails), who decides? The preregistration should write
   the decision rule mechanically, same as Probe 1 §3 and Probe 2 §4. And it should include
   a sunset trigger — the project's V3/V4 precedent is that things shipping clean OOS can
   still fail combo-agnostic audits; paper-trading readiness needs a decommission hook.

Recommendations:
- **≤4 session buckets** (simpler is clearer).
- **Freeze all 28 non-session params** (no neighborhoods).
- **Regime halves as HARD gate** — both halves must pass.
- **15m included as negative control** — not for its own PASS, but to catch bucket artifacts.
- **Flat $500 sizing** — matches the project's stated capital base.
- **Preregister Option Z sunset trigger** in §4, same format as Probe 2.

### The Executor — Four buckets, one week, done

Probe 2 took 2 seconds of compute on sweep-runner-1 and 30 minutes of writeup. Probe 3 can
be the same shape if scope is disciplined. Here's the executable spec:

**Scope**:
- 1h only. Freeze all 28 non-session params. Drop 15m entirely — it failed §4 mechanically,
  keeping it for negative-control value costs a bucket and adds noise.
- 4 session buckets: `session_filter_mode ∈ {0=all, 1=RTH, 2=RTH-lunch-exclude, 3=overnight}`.
- Flat $500 sizing, $5/contract RT friction. Pin `MIN_RR_ADAPTIVE=False` (combo 865 uses
  fixed R:R = 1.847 already).

**Execution**:
- Reuse `scripts/param_sweep.py --start-combo 865 --end-combo 866` with a new
  `--session-filter` override flag. One combo × 4 buckets = 4 runs × ~2s each.
- Wrapper script `tasks/_run_probe3_remote.py` mirrors `_run_probe2_remote.py`.
- Readout `tasks/_probe3_readout.py` mirrors `_probe2_readout.py`. Per-bucket metrics +
  §4 gate application table.

**Gates** (mechanical):
- Per-bucket: `net_sharpe ≥ 1.3`, `n_trades ≥ 50`, `net_$/yr ≥ $5000`.
- For each passing bucket, *also* split test window in halves as a diagnostic (not a gate
  — raising regime halves to a hard gate doubles the thresholds and will kill live buckets).

**Timeline**: preregistration draft half a day; signing ceremony + commit half a day;
remote run 10 minutes; readout 30 minutes; verdict 30 minutes. Total ~1.5 days wall-clock.

**Don't**: add parameter neighborhoods, don't run 1m/5m (§7.6 binds), don't add Kelly
diagnostics (not preregistered, distracting). One probe, one answer, then re-council.

---

## 3. Peer Review (5 reviewers × 3 questions each, anonymized A–E)

**Anonymization mapping** (revealed in this transcript; reviewers saw only A-E):

| Letter | Advisor |
|---|---|
| A | The Expansionist |
| B | The First Principles Thinker |
| C | The Executor |
| D | The Outsider |
| E | The Contrarian |

### Reviewer 1

1. **Strongest: B.** Reframing the question from "confirm edge via sessions" to "what's
   the minimum evidence to move the posterior past paper-trading" is the only response
   that engages with the actual decision under uncertainty. Everyone else answers
   "what session design" — B answers "is session design the right question."
2. **Biggest blind spot: A.** Treats combo 865's Sharpe 2.89 as discovered alpha. After
   post-hoc selection from 13,814 combos + ML#1 + cross-TF filter, the effective N of
   tests that had to clear to get here is ≥9,000. At α=0.05 Bonferroni on 220 trades, the
   required Sharpe to reject the null is **~3.7** — combo 865 already fails. Expanding
   the probe with 7 buckets + param neighborhood + 1m/5m multiplies the researcher DoF
   *after* the multiplicity binding is already violated.
3. **All five missed**: nobody computed the Bonferroni-corrected Sharpe floor explicitly.
   At α=0.05 over 9,000 tests on 220-trade samples, critical Sharpe ≈ 3.7. Combo 865 at
   2.89 is below threshold. Probe 3's preregistration must commit to this number
   quantitatively before the probe runs, or the probe cannot generate usable evidence.

### Reviewer 2

1. **Strongest: B.** It's the only response that says "the probe you're asking for isn't
   the probe you need." Session structure is plausibly irrelevant if the underlying edge
   is regime-driven (B's probe 2: temporal halves) or overfit-driven (B's probe 3: param
   neighborhood). A council that approves a session probe without these first is
   front-running its own evidence.
2. **Biggest blind spot: A.** "1m and 5m" is a direct §7.6 violation. Probe 1
   preregistration binds the bar-timeframe axis at the *family* level, and the Probe 2
   carve-out is explicit: combo 865 at 1h only. A proposes re-opening the bar-TF axis for
   a single combo, which re-multiplies the searched space the §7.6 sunset was specifically
   closing.
3. **All five missed**: the effective N for multiplicity correction isn't just the 3000-
   combo sweep. It's the full decision tree: 3 × 3000 combos (1m/15m/1h) = 9,000 sweep
   tests, then the ML#1 ranker that selected 865, then the cross-TF coherence filter that
   *also* picked 865. That chain has maybe 20,000+ effective tests once the ranker is
   priced in — and nobody wrote it down.

### Reviewer 3

1. **Strongest: B.** Three orthogonal probes (multiplicity / regime / parameter stability)
   each answer a distinct posterior question. A session sweep answers neither of the
   three. The methodological superiority is structural, not just stylistic.
2. **Biggest blind spot: A.** Kelly-fractional "as diagnostic, not for live sizing" is a
   backdoor: once a number gets written down, it pulls downstream decisions toward it.
   Kelly on an unvalidated 220-trade sample is theater — it assumes the edge is real,
   which is exactly what Probe 3 is supposed to test.
3. **All five missed**: P(real edge | Probe 2 PASS) is not written down anywhere. The
   council is implicitly assuming the posterior is high enough that Probe 3 is confirmatory,
   but the multiplicity math (Reviewer 1's point) suggests the posterior is ~2-3% after a
   proper prior. Probe 3's preregistration should commit to a posterior gate too: paper-
   trading authorization requires P(real edge | PASS-all-gates) ≥ 0.5, computed against
   a stated prior.

### Reviewer 4

1. **Strongest: B.** B is the only response that treats Probe 3 as a decision-theoretic
   problem rather than an experimental-design problem. The reframing is not just cosmetic —
   it changes what gets preregistered.
2. **Biggest blind spot: A.** Missing the ML#2 V3/V4 revocation precedent. Both V3 and V4
   shipped on clean OOS evaluation (Sharpe 1.78 and 2.13) and subsequently failed combo-
   agnostic audits (Sharpe 0.31 and −0.42, ruin 53% and 56%). Probe 3 designed as A
   proposes — wide sweep, upside discovery, Kelly diagnostic — recapitulates the V3/V4
   design flaw: confirmatory evaluation on a narrow success metric, with no mechanism to
   catch the combo-specific memorization that actually failed them in production.
3. **All five missed**: Probe 3's preregistration needs a **decommission trigger**, not
   just a ship gate. If combo 865 enters paper-trading and shows Sharpe < 0 on N fresh
   trades, at what point does it get pulled? Without this preregistered, paper-trading
   degrades into an indefinite "watch it for a while" process that will outlast the
   evidence.

### Reviewer 5

1. **Strongest: B.** The "exit ritual for a survivor" framing is the correct mental model
   for a single-combo carve-out after a family-level sunset. Session sweeps are what you
   do when the family is alive. The council was one step from forgetting that.
2. **Biggest blind spot: A.** The test>train Sharpe inversion is a regime-favorability
   signal, and A's response addresses it only by proposing *more* tests in the same
   regime window (1m, 5m, bucket splits). That compounds the exposure rather than
   diagnosing it.
3. **All five missed**: bootstrap the test>train inversion's null probability. Under
   H0 (no real edge, test metrics are noise around the true distribution), P(test_Sharpe
   > train_Sharpe) on 220 trades and 747 trades respectively is computable. If that
   probability is ≥ 10%, the inversion is not a red flag. If it's < 1%, it's diagnostic
   of either regime favorability or selection. This number should be in Probe 3's
   preregistration as a gate or at minimum as a declared observation.

---

## 4. Chairman Synthesis

### Where the Council Agrees

Five of five advisors converge on five of the six axes; only the session-scope ceiling
(ax 1) and the 15m-inclusion question split the room.

1. **Parameter freeze.** 4/5 (B, C, D, E) — freeze all 28 non-session parameters. Only A
   (Expansionist) proposes a ±5% neighborhood sweep. A's position is outvoted on both
   methodological (multiplicity) and procedural (Probe 2 §5.2 "combo 865 parameters
   frozen" was explicit for the carve-out) grounds.
2. **Timeframe lock at 1h.** 4/5 explicit, 1 violating (A's 1m/5m proposal). §7.6 of
   Probe 1 and §5.3 of Probe 2 both bind — A's proposal is inadmissible without a new
   preregistration cycle, not a Probe 3 axis.
3. **Session-bucket cap at ≤4.** 4/5 (B, C, D, E) explicit; A argues 7. All four on the
   cap side cite the same reason: multiplicity burden scales linearly with bucket count,
   4 is minimally informative, 7 triples the correction burden.
4. **Regime-window stress test (temporal halves).** 4/5 advisors treat this as either a
   HARD gate (B, D, E: mandatory pass) or a PRIMARY probe (B promotes it above session
   structure); only C frames it as a diagnostic. The test>train Sharpe inversion in
   Probe 2 is the specific thing this is designed to pressure-test.
5. **Flat $500 sizing and $5 RT friction.** Unanimous non-objection; user directive locks
   this at the project-standard convention. Confirmed.

### Where the Council Clashes

Two genuine disagreements remain after peer review:

1. **Scope ceiling — upside-discovery vs. falsification.** A (Expansionist) wants a wide
   grid (7 buckets + parameter neighborhood + 1m/5m upside probes + Kelly diagnostic);
   B/C/D/E want a narrow probe. The Chairman sides with the 4-1 majority: Probe 3's role
   is to test whether the combo-865 carve-out is real or a post-hoc survivor of a 9,000-
   test pipeline. A's design *expands* the searched space at the moment it should be
   contracting. The Expansionist position is methodologically incoherent with the §7.6
   terminal it would have to override.
2. **15m inclusion.** A/D/E want 15m included (A: as live second-TF candidate; D/E: as
   negative control); C explicitly drops it; B silent. The Chairman's resolution: keep
   15m in Probe 3 as a **hard-fail negative control** (D/E framing), NOT as a PASS-
   eligible bucket (A's framing). If any 15m bucket passes where 1h doesn't, the pipeline
   has a bucket-level artifact and Probe 3 is invalid — that's the negative-control
   value. This costs one extra compute cell (~2s) and provides meaningful falsification
   leverage.

### Blind Spots the Council Caught (Peer Review)

Five reviewers converged on five distinct "all missed" themes. All five must be folded
into Probe 3's preregistration or the probe cannot generate usable evidence:

1. **Multiplicity-corrected Sharpe floor was not computed.** Reviewer 1: at α=0.05
   Bonferroni over ~9,000 tests on 220-trade samples, required Sharpe ≈ **3.7**, not 1.3.
   Combo 865 at 2.89 **already fails** a proper multiplicity-corrected gate. Probe 3
   preregistration must commit to the corrected number quantitatively.
2. **Effective N denominator needs the full decision tree**, not just the sweep.
   Reviewer 2: 9,000 sweep tests × ML#1 ranker × cross-TF coherence filter ≈ 20,000+
   effective tests. The multiplicity denominator is larger than any advisor wrote down.
3. **Posterior P(real edge) not preregistered.** Reviewer 3: under a stated prior and
   the 9,000-test multiplicity, P(real edge | Probe 2 PASS) ≈ 2-3%. Probe 3 needs a
   preregistered posterior gate (paper-trading requires P(real edge | all-Probe-3-PASS)
   ≥ 0.5 against the stated prior) to know what it's for.
4. **ML#2 V3/V4 precedent mandates a decommission trigger, not just a ship gate.**
   Reviewer 4: both V3 (Sharpe 1.78 shipped) and V4 (Sharpe 2.13 shipped) failed combo-
   agnostic audits after clean-OOS ship. Probe 3 must preregister the conditions under
   which paper-trading combo 865 gets decommissioned (e.g. Sharpe < 0 on next 100 trades
   triggers immediate halt).
5. **Test>train Sharpe inversion has a computable null probability.** Reviewer 5:
   bootstrap P(test_Sharpe > train_Sharpe | H0) on (220, 747) trades. If < 1%,
   inversion is diagnostic of regime favorability or selection — preregister as
   an observation or gate.

### The Recommendation

**Run a narrow, multiplicity-corrected probe on 1h** structured as First Principles'
three-stack (multiplicity-adjusted session inference + temporal halves as HARD gate +
15m negative control), **not** Expansionist's 7-axis sweep. Specifically:

1. **Scope**: 1h only. Freeze all 28 non-session params from combo 865.
2. **Session axis**: 4 buckets = {all / RTH / RTH-lunch-exclude / overnight}. Bonferroni
   α=0.05 over 4 tests → required Sharpe per bucket ≈ **1.78** (corrected down for
   effective N — compute the exact number in the math memo).
3. **Regime gate (HARD)**: Split the test window 2024-10-22 → 2026-04-08 into halves.
   Both halves must clear the multiplicity-corrected Sharpe gate independently on any
   passing bucket. If only one half passes, that bucket FAILS.
4. **15m negative control**: Run same 4 buckets on 15m. Any 15m bucket passing where 1h
   does not → mechanical falsification signal → Probe 3 is invalid → Option Z sunset.
5. **Sizing**: flat $500/trade, $5/contract RT friction. Locked per user directive.
6. **Trade-count floor**: `n_trades ≥ 50 per bucket × per half` = effectively 100 per
   bucket on full window. Buckets with fewer than 50 trades in any half drop out (report
   as "insufficient power" — not FAIL, not PASS).
7. **Preregistered Option Z sunset trigger**: if no 1h bucket clears both the
   multiplicity-corrected Sharpe gate AND the regime-halves gate, the combo-865 carve-
   out retires and the Z-score family on NQ/MNQ terminally sunsets (Probe 3 §4 FAIL
   branch writes `tasks/project_sunset_verdict.md`).
8. **Preregistered posterior math**: pre-commit the prior P(real edge) ≈ 0.02-0.03 and
   compute P(real edge | all-Probe-3-PASS). Paper-trading requires ≥ 0.5 posterior.
9. **Preregistered decommission trigger**: paper-trading authorization commits to pulling
   combo 865 if live Sharpe < 0 on next N trades (N = 50, matching §4 trade-count gate).
10. **Test>train inversion bootstrap**: compute P(test_Sharpe > train_Sharpe | H0) on
    (220, 747) trades. Report in §4 as observation; if < 1%, preregister as a FAIL
    trigger.

The Chairman disagrees with the 3-2 split from Probe 2's council (Option Y-gated-on-865)
being read as "session structure probe now." Probe 2's PASS was one combo × one TF × one
window × one gate — the smallest admissible positive result. Probe 3's job is to do what
the user's own discipline has done across V3/V4: test whether the clean-OOS evidence
holds under adversarial gates (multiplicity, regime, parameter-neighborhood precedent,
and decommission triggers). The First Principles reframe gets this right; the Expansionist
design re-enacts the V3/V4 ship-then-fail pattern.

### The One Thing to Do First

**Before drafting `tasks/probe3_preregistration.md`, write a one-page math memo
`tasks/probe3_multiplicity_memo.md` that commits numerically to:**

1. The effective-N denominator for the full decision tree that selected combo 865 @ 1h
   (Probe 1 sweep + ML#1 ranker path + cross-TF filter + Probe 2 gate).
2. The Bonferroni-corrected Sharpe floor on 220-trade and 110-trade (half-window) samples
   at α=0.05 given that N.
3. The prior P(real edge) against which Probe 3's posterior gate will be computed.
4. The bootstrap-computed P(test_Sharpe > train_Sharpe | H0) number from Probe 2's observed
   (2.89, 2.098) on (220, 747) trades.

**Sunset trigger (revised 2026-04-21 UTC after user challenge)**:

The initial Chairman draft leaned on Bonferroni-over-9,000 as the binding multiplicity gate.
Reviewer pushback clarified a structural error: the 9,000-test denominator applies to
**Stage 1 (generation)** — the sweep + cross-TF filter that selected combo 865 on training
data. Probe 2 was **Stage 2 (confirmation)** — a pre-registered single hypothesis tested
once on held-out bars. Stage 2's multiplicity denominator is **1**, not 9,000. Combo 865's
test-partition Sharpe 2.89 is a legitimate OOS measurement, not a selection-inflated
conditional-max.

The revised sunset trigger does NOT rely on Bonferroni-over-9,000. It fires on any **two or
more** of the following:

1. **Regime halves FAIL** — either temporal half of 2024-10-22 → 2026-04-08 shows net
   Sharpe < 1.3 on every candidate bucket (combo 865 requires consistency across at least
   one session bucket for at least two halves).
2. **Parameter neighborhood FAIL** — ±5% jitter on `z_band_k`, `stop_fixed_pts`, `min_rr`
   produces Sharpe < 1.3 on > 50% of the neighborhood (diagnoses sharp-point overfit).
3. **15m negative control FAIL** — any 15m bucket passes where 1h fails → bucket-level
   pipeline artifact → Probe 3 invalid.
4. **Bootstrap test>train FAIL** — `P(test_Sharpe > train_Sharpe | H0)` on (2.89, 2.098)
   × (220, 747) < 1% → regime favorability diagnostic → cautionary, not auto-FAIL.

If 0-1 gates trigger, combo 865 advances to paper-trading at the decommission gate.
If ≥2 trigger, write `tasks/project_sunset_verdict.md`.

**What DID hold from the earlier framing**:
- V3/V4 revocation precedent (both shipped clean OOS, both failed combo-agnostic audits)
  remains the strongest project-specific prior against trusting 2.89 at face value.
- Decommission trigger (live Sharpe < 0 on N=50 halts paper-trading) is still preregistered.
- Winner's curse on training selection (expected test Sharpe < training Sharpe under
  regression to mean) — combo 865's 2.89 > training 2.098 is the *opposite* of the
  expected regression direction, which is diagnostic of either genuine edge, regime
  favorability, or small-sample variance. The three are distinguishable via gates above.

**Memo scope (revised)**:
1. σ_Sharpe estimator noise at n=220 and n=110 (regression-to-mean reference)
2. Bootstrap P(test_Sharpe > train_Sharpe | H0)
3. Parameter neighborhood definition (which knobs, what δ, what decision rule)
4. Regime-halves methodology (where to split, which gates per half)
5. Pre-probe posterior P(real edge) using V3/V4 base rate as the project-specific prior
   — this is more defensible than an abstract α-level threshold

**The memo is still ~2 hours of work**, and it still gates whether Probe 3 runs at all.
The sunset branch fires on the revised criterion, not the withdrawn 9,000-test Bonferroni.

The memo is ~2 hours of work. Writing it before the preregistration saves writing a
preregistration that might turn out to be unrunnable.
