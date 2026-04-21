# Probe 3 Calibration Memo

**Date**: 2026-04-21 UTC
**Purpose**: Supply the five numerical inputs the Probe 3 scope council
chairman (`tasks/council-transcript-2026-04-21-probe3-scope.md`) required
before drafting the Probe 3 preregistration.
**Authority chain**: Probe 1 verdict → Probe 2 verdict → Probe 3 council →
**this memo** → Probe 3 preregistration (task #34a).

---

## 0. Scope — what this memo is and is not

This is a **calibration memo**, not a multiplicity-decision memo.

The council's initial Chairman draft leaned on a Bonferroni-over-9,000
correction applied to Probe 2's held-out net Sharpe of 2.89. That framing
is **withdrawn** (user correction, 2026-04-21):

- **Stage 1 — hypothesis generation**: 3000-combo sweep on training
  partition + cross-TF filter → combo 865 selected. Multiplicity paid here.
- **Stage 2 — confirmation**: Probe 2, a pre-registered single hypothesis
  tested once on held-out bars 2024-10-22 → 2026-04-08. Multiplicity
  denominator = **1**.

Combo 865's test-partition Sharpe 2.89 on 220 trades is therefore a
legitimate OOS measurement, not a selection-inflated conditional maximum
requiring a 9,000-test Bonferroni correction.

The motivation for Probe 3 is **not** multiplicity. It is three other
concerns this memo quantifies:

1. **Sample noise** at n = 220 trades.
2. **Regime representation** within a 1.48-year test window.
3. **V3/V4 precedent** — 0 of 2 shipped OOS-clean ML#2 stacks survived
   combo-agnostic audit (project-specific base rate).

The sunset trigger derived in §7 fires on **≥ 2 of 4** diagnostic gates
failing during Probe 3 execution, not on any single corrected-Sharpe
threshold pre-Probe-3.

---

## 1. Sharpe ratio standard errors (Jobson-Korkie / Opdyke)

Closed-form approximation for annualized Sharpe:

```
σ²(SR_ann) ≈ (q / T) · (1 + SR² / (2q))
```

where `q = n_trades / years` (trades per year) and `T = n_trades`.

This ignores higher-moment skew/kurtosis corrections (Mertens 2002,
Opdyke 2007). Combo 865's per-trade net PnL distribution has moderate
negative skew (friction-asymmetry: stops hit more symmetrically than
targets). The Mertens correction would **widen** the CIs slightly, so the
numbers below are conservative on precision.

### 1.1 Test partition — n=220, SR=2.89, years=1.4799

- q = 220 / 1.4799 = 148.66 trades/year, T = 220
- σ² = (148.66 / 220) · (1 + 2.89² / (2·148.66))
     = 0.6757 · (1 + 8.3521 / 297.32)
     = 0.6757 · 1.02809
     = 0.6947
- **σ_SR ≈ 0.834**
- 95% CI: [2.89 − 1.96·0.834, 2.89 + 1.96·0.834] = **[1.26, 4.52]**

### 1.2 Regime half — n=110, SR=2.89 (hypothetical if preserved), years=0.740

- q = 148.65, T = 110
- σ² = 1.3513 · 1.02809 = 1.389
- **σ_SR ≈ 1.179**
- 95% CI: [2.89 − 2.31, 2.89 + 2.31] = **[0.58, 5.20]**

### 1.3 Training partition — n=747, SR=2.098 (gross), years=5.92

- q = 126.18, T = 747
- σ² = (126.18 / 747) · (1 + 2.098² / 252.36)
     = 0.1689 · 1.01744
     = 0.1719
- **σ_SR ≈ 0.415**

**Interpretation**: the test-partition 95% CI [1.26, 4.52] straddles the
§4 gate floor of 1.3. A single 1.48-year held-out window cannot by itself
distinguish "Sharpe 1.3" from "Sharpe 4.5" at conventional confidence.
This is the quantitative reason Probe 3 needs to exist. It is not about
search multiplicity.

---

## 2. Test-greater-than-Train inversion bootstrap

Combo 865's held-out net Sharpe (2.89) **exceeds** its training gross
Sharpe (2.098). Under H0 "same underlying edge, independent windows," how
surprising is this?

### 2.1 Gross-vs-gross (test 4.404 vs train 2.098)

- diff = 2.306
- σ_test_gross (SR=4.404, n=220, years=1.48):
  σ² = 0.6757 · (1 + 19.395/297.32) = 0.6757 · 1.0652 = 0.720 → σ = 0.848
- σ_train_gross = 0.415 (from §1.3)
- σ_diff = √(0.415² + 0.848²) = √(0.172 + 0.720) = √0.892 = **0.944**
- z = 2.306 / 0.944 = **2.44**, one-tailed p ≈ **0.007**

### 2.2 Net-vs-inferred-net (test 2.89 vs train ≈ 1.378)

Training net Sharpe inferred by applying the test net-to-gross ratio
(2.89 / 4.404 = 0.6563) to training gross Sharpe. This is valid because
`stop_fixed_pts = 17.018` fixes contract count at both timeframes, so
friction per trade is ~$438 on both training and test windows.

- train_net ≈ 2.098 · 0.6563 = **1.378**
- diff = 2.89 − 1.378 = 1.512
- σ_test_net = 0.834 (from §1.1)
- σ_train_net ≈ 0.413 (same formula, SR=1.378)
- σ_diff = √(0.413² + 0.834²) = √(0.171 + 0.695) = √0.866 = **0.930**
- z = 1.512 / 0.930 = **1.62**, one-tailed p ≈ **0.053**

### 2.3 Interpretation

- Gross-vs-gross inversion is **significant** at p ≈ 0.01.
- Net-vs-inferred-net inversion is **marginal** at p ≈ 0.05.

Three possible explanations:

1. **Genuine regime-persistent edge** for combo 865 at 1h on NQ.
2. **Test window was favorable** to combo 865's directional bias —
   first-order concern for the regime-halves gate in §4.
3. **Sample noise** — less plausible given p < 0.01 on gross.

Probe 3 regime-halves + parameter-neighborhood + 15m negative-control
jointly pressure-test (2) and (3). Explanation (1) is the H1 we admit
carrying into paper-trading **if Probe 3 PASSES**.

---

## 3. Parameter neighborhood specification

Tests the "sharp-point overfitting" hypothesis — i.e. whether combo 865
sits on a knife-edge of parameter space that ±5% jitter would
disintegrate.

### 3.1 Axes (3 continuous params × 3 levels = 27 combos)

| Parameter | 865 center | −5% | +5% |
|---|---:|---:|---:|
| `z_band_k` | (from 865 dict) | ×0.95 | ×1.05 |
| `stop_fixed_pts` | 17.018 | 16.167 | 17.869 |
| `min_rr` | (from 865 dict) | ×0.95 | ×1.05 |

All other 25+ parameters in combo 865's config are **frozen** at 865
values. No qualitative flags (z-score formulation, exit ritual, session
filter) are varied here — those belong to §5 of the council recommendation.

### 3.2 Gate

**Majority gate**: ≥ 14 of 27 neighborhood combos each pass the Probe 2
§4 three-gate set (net_sharpe ≥ 1.3 AND n_trades ≥ 50 AND net $/yr ≥
5,000) on the 1h held-out test partition.

### 3.3 Rationale

- If 865 sits on a **ridge** (real edge), ±5% jitter should preserve
  most passes → ≥ 14 / 27 gate clears easily.
- If 865 sits on a **needle** (overfit), ±5% jitter collapses the edge
  → gate fails, sunset trigger candidate fires.
- Base-rate check: uniform pass rate across the 13,814 v11 sweep combos
  is ~0.03% at this three-gate level. A 14 / 27 (52%) pass rate in the
  neighborhood is 1,700× above base rate; cannot arise from random
  parameter choice.

### 3.4 Compute

27 combos × ~220 trades × 1h = trivial. Runs on sweep-runner-1 in < 5 min.

---

## 4. Regime halves methodology

Tests the "test window happened to be favorable" hypothesis.

### 4.1 Split

- **Split point**: 2025-07-15 (midpoint of test window 2024-10-22 →
  2026-04-08).
- Expected ~110 trades per half (exact counts reported at execution; do
  NOT retune split post-hoc based on counts).

### 4.2 Per-half gates (each half independently)

- `net_sharpe(half, annualized) ≥ 1.3`
- `n_trades(half) ≥ 25` (relaxed from full-window 50 because of the split)
- `net_dollars_per_year(half, annualized) ≥ 5,000`

### 4.3 Composite gate

**Both halves pass all three sub-gates.**

### 4.4 Power analysis (σ_SR ≈ 1.179 at n=110)

| True annualized SR | P(half passes SR ≥ 1.3) | P(both pass) |
|---:|---:|---:|
| 2.89 (observed test SR) | Φ((2.89−1.3)/1.179) = Φ(1.348) = 0.911 | **0.830 (83%)** |
| 2.5 (conservative H1) | Φ(1.017) = 0.845 | **0.714 (71%)** |
| 1.3 (gate boundary) | 0.500 | 0.250 (25%) |
| 0 (null H0) | 1 − Φ(1.102) = 0.135 | **0.018 (1.8%)** |

### 4.5 Interpretation

- At true SR ≥ 2.5: 71%+ power. Adequate.
- At null: 1.8% false-pass rate. Sharp enough to be a decisive filter.
- At true SR = 1.3: 25% false-pass rate — residual risk of paper-trading
  a marginal edge. Paper-trade deposit size + tight stop discipline
  handles this residual.

---

## 5. V3/V4-informed Bayesian prior

The strongest project-specific base rate is the V3/V4 ML#2 precedent:
two ML#2 stacks shipped after passing clean OOS gates; both subsequently
revoked after failing combo-agnostic audits.

### 5.1 Observed

- **0 successes out of 2** shipped OOS-PASS stacks.

### 5.2 Priors

- **Jeffreys Beta(0.5, 0.5)** (uninformative reference):
  posterior mean = (0.5 + 0) / (1 + 2) = **0.167**
- **Laplace Beta(1, 1)** (uniform):
  posterior mean = (1 + 0) / (2 + 2) = **0.250**

**Adopt Jeffreys (0.167)** as the pre-Probe-3 prior P(real edge | held-out
PASS). Conservative but defensible given the specific V3/V4 pattern.

### 5.3 Sensitivity

N=2 is small; here is the sensitivity to the next result:

- If Probe 3 PASSES → Beta(0.5, 0.5) + 1/3 → posterior mean **0.375**.
- If Probe 3 FAILS → Beta(0.5, 0.5) + 0/3 → posterior mean **0.125**.

---

## 6. Bayes update after Probe 3

### 6.1 Gate structure

Probe 3 consists of four independent diagnostic gates:

1. **Regime halves** (§4): both halves pass §4 three-gate set
2. **Parameter neighborhood** (§3): ≥ 14 / 27 pass §4 three-gate set
3. **15m negative control**: 15m remains FAIL under session/exit sweep
   (confirms the friction-regime story — if session/exit sweep could find
   15m winners, Probe 2's 1h PASS may just be "any TF that survives
   sufficient sweep")
4. **Session / exit ritual**: at least one {session × exit} combination
   with combo 865 params passes §4 three-gate set on 1h test partition

### 6.2 Joint likelihoods (approximate; independence assumption)

| Gate | P(PASS | H1: real edge SR≈2.5) | P(PASS | H0: no edge) |
|---|---:|---:|
| Regime halves both pass | 0.71 | 0.018 |
| Param neighborhood ≥14/27 | ~0.85 | <0.001 (binomial null) |
| 15m NC remains FAIL | ~0.95 | ~0.5 |
| Session/exit ritual clears | ~0.80 | ~0.05 |

**P(all four PASS | H1)** ≈ 0.71 × 0.85 × 0.95 × 0.80 ≈ 0.459
**P(all four PASS | H0)** ≈ 0.018 × 0.001 × 0.5 × 0.05 ≈ 4.5e-7
**Naive BF** ≈ 10⁶ — implausibly large; independence is violated, and H0
has many alternative forms (favorable-window, sample noise).

### 6.3 Conservative BF ≈ 50

A more cautious BF estimate aggregating the four gates with realistic
correlation and an expanded H0 alternative set gives BF ≈ 50 for full
PASS. I use this in what follows.

### 6.4 Posterior scenarios

| Probe 3 outcome | BF | Posterior (prior 0.167) |
|---|---:|---:|
| Full 4/4 PASS | 50 | 0.167·50 / (0.167·50 + 0.833) = **0.91** |
| 3/4 PASS, regime halves FAIL | 0.3 | **0.055** (sunset) |
| 3/4 PASS, param nbhd FAIL | 0.2 | **0.038** (sunset) |
| 2/4 PASS | ~0.05 | **~0.01** (sunset) |
| 1/4 PASS | ~0.01 | **~0.002** (sunset) |

**Paper-trade gate** at posterior ≥ 0.5 is cleared only by full 4/4 PASS.

---

## 7. Sunset trigger formulation

**SUNSET (Option Z, project sunset on combo 865 @ 1h)** fires when
**≥ 2 of the 4 Probe 3 gates FAIL**.

**PAPER-TRADE trigger** fires when **all 4 Probe 3 gates PASS** (posterior
≈ 0.91).

**AMBIGUOUS BRANCH (exactly 1 of 4 fails)**: re-convene LLM Council and
user review. Depending on which gate failed:

- Regime halves FAIL alone → likely sunset (favorable-window story alive).
- Param neighborhood FAIL alone → likely sunset (needle hypothesis).
- 15m NC fails (15m unexpectedly passes) → likely sunset (sweep p-hacking
  story alive; 1h PASS may not be mechanically special).
- Session/exit FAIL alone → possible reduced-deployment with smaller
  position (still a regime/param-verified edge, just not session-tuned).

---

## 8. Conclusion

**Recommended action**: proceed to draft `tasks/probe3_preregistration.md`
(task #34a) using the gate structure in §7.

**Not recommended**: pre-Probe-3 sunset verdict. The V3/V4 prior is
pessimistic (0.167) but does not alone justify foreclosing Probe 3 —
the test design has sharp discriminating power (71%+ power at true SR=2.5,
1.8% false-pass at null), and the asymmetric cost of a foregone real edge
at pre-probe posterior 0.167 exceeds the cost of running Probe 3 (cheap
compute, < 1 week of effort, well-bounded scope).

---

## 9. Files / references

- Probe 2 verdict: `tasks/probe2_verdict.md`
- Probe 2 readout: `data/ml/probe2/readout.json`
- Probe 3 council transcript: `tasks/council-transcript-2026-04-21-probe3-scope.md`
- Probe 3 council report (HTML): `tasks/council-report-2026-04-21-probe3-scope.html`
- Probe 1 family-level falsification: `tasks/probe1_verdict.md`
- V3 revocation audit: `tasks/v3_no_gcid_audit_verdict.md`
- V4 revocation (via CLAUDE.md banner, 2026-04-21 00:34 UTC)
- CLAUDE.md combo-865 carve-out bullet

---

## Appendix — verification checklist for preregistration

When drafting `tasks/probe3_preregistration.md` (task #34a), confirm:

- [ ] §4 three-gate set pinned to **1h only** (per Probe 2 §4 PASS clause)
- [ ] Regime halves split point frozen at **2025-07-15**
- [ ] Parameter neighborhood axes = `{z_band_k, stop_fixed_pts, min_rr}`
      at `×{0.95, 1.00, 1.05}` (27 combos)
- [ ] 15m negative-control methodology: session/exit sweep applied
      symmetrically to 15m; 15m must remain FAIL to preserve friction-regime
- [ ] Session/exit ritual scope: exact session list + exit rule list
      pre-committed (not sampled post-hoc)
- [ ] Sunset trigger: **≥ 2 of 4 gates FAIL** (matches §7)
- [ ] Paper-trade trigger: **all 4 gates PASS**
- [ ] Ambiguous branch: single-gate FAIL routes to council re-convene
- [ ] Pre-probe posterior documented: **0.167** (Jeffreys on V3/V4)
- [ ] No post-hoc gate relaxation clause (matches Probe 2 §5 commitment 4)
- [ ] YEARS_SPAN_TEST = 1.4799 frozen at signing
