# Probe 3 preregistration — pre-sign reviewing agent report

**Date of review**: 2026-04-21 evening CDT
**Reviewed commit (baseline)**: `4e5a114` — Probe 3 pre-registration drafted (unsigned)
**Review protocol**: `CLAUDE.md` → Reviewing Agent Protocol
**Document under review**: `tasks/probe3_preregistration.md`
**Companion memo cross-referenced**: `tasks/probe3_multiplicity_memo.md`
**Verdict**: **FIX WARNINGS THEN SIGN WITH JUSTIFICATION**

---

## Severity tally

| Severity | Count | Disposition |
|---|---:|---|
| CRITICAL | 0 | — |
| WARN | 4 | Fix in-text, no structural changes. Addressed in commit after `4e5a114`. |
| INFO | 3 | Advisory only; authors may adopt or leave. |

---

## Check matrix

| Group | Check | Status |
|---|---|:---:|
| A — Transcription / arithmetic | A1: §2.1 combo 865 parameter dict matches Probe 2 §2 bit-for-bit | PASS |
| | A2: §2.2 ±5 % neighborhood arithmetic (z_band_k, stop_fixed_pts, min_rr) | PASS |
| | A3: §4.1 split date 2025-07-15 is exact calendar midpoint of test window | PASS |
| | A4: `YEARS_SPAN_TEST = 1.4799` matches Probe 2 signing-commit frozen value | PASS |
| B — Structural integrity | B1: four gates in §4 map 1:1 to four §5 branch inputs (`F ∈ {0…4}`) | PASS |
| | B2: §5 branch routing exhaustive and mutually exclusive over `F` | PASS |
| | B3: §6 irrevocable commitments block all 7 identifiable post-hoc escape routes (gate-relax, split-date retune, timeframe-switch, grid-augment, scope-creep, early-stop, council-as-override) | PASS |
| C — Memo consistency | C1: every §4 `P(PASS \| H1)` / `P(PASS \| H0)` cited in the preregistration is sourced from `probe3_multiplicity_memo.md` §3–§4 | PASS |
| | C2: §5 posterior bounds (F = 0 → 0.91; F ≥ 2 → < 0.06) match memo §6 | PASS |
| D — Reference integrity | D1: all 9 referenced files in §8 exist in the repo and resolve | PASS |

---

## WARN findings (addressed in follow-on commit)

### W1 — §4.1 cites only conservative-H1 power
**Finding**: §4.1 rationale quotes the 71 % power figure at conservative
H1 (SR = 2.5) but does not also cite the 83 % figure at observed H1
(combo 865's Probe 2 SR = 2.89). For a reader auditing the gate's
discriminative capacity, seeing only the conservative figure under-
states the gate's power against the actual observed effect size.

**Why it matters**: Rule 2 of `feedback_council_methodology.md` requires
`P(PASS | H1)` to be transparent. Citing a single conservative value is
defensible, but citing both the conservative and observed values is
more informative and forestalls an auditor's question "what if the
true SR is closer to the observed 2.89 than to 2.5?".

**Recommended fix**: extend §4.1 rationale paragraph to quote both
71 % (conservative H1, SR = 2.5) and 83 % (observed H1, SR = 2.89),
with likelihood ratios 40× and 46× respectively, referencing memo §4.

**Fix status**: applied.

---

### W2 — §4.2 "~1,700× base rate" asserted without derivation pointer
**Finding**: §4.2 rationale states "14 / 27 is ~1,700× base rate" but
does not point the reader at the binomial derivation in memo §3. A
future auditor could question the arithmetic or the choice of H0
per-combo pass rate.

**Why it matters**: The 1,700× figure is load-bearing for the
likelihood-ratio summary that justifies this as the strongest
individual gate in the 4-gate set. Disclosure of inputs (per-combo
H0 / H1 pass rates) and the binomial CDF makes the claim auditable.

**Recommended fix**: extend §4.2 rationale with a binomial calibration
line — per-combo H0 rate ≤ 0.25 → `P(≥ 14 / 27 | H0)` < 0.001;
per-combo H1 rate ≈ 0.85 → `P(≥ 14 / 27 | H1)` ≈ 0.999; LR > 850×.
Reference memo §3.

**Fix status**: applied.

---

### W3 — §4.3 ≤ 2 / 16 threshold lacks explicit H0 per-cell rate
**Finding**: §4.3 rationale states "Expected under H0 (with modest
session/exit search): < 2 passes" without naming the per-cell pass
rate under H0 that underlies the ≤ 2 / 16 threshold calibration. The
memo table gives `P(gate PASS | H0) ≈ 0.5` and LR ≈ 2×, but the
preregistration does not reproduce the per-cell rate (p ≈ 0.15) that
inverts to that figure.

**Why it matters**: A naïve reader might assume p = 0.05 under H0 (a
common default), but at p = 0.05 the binomial gives `P(≤ 2 / 16)` ≈ 0.96,
which would make the gate near-uninformative. The actual calibration
requires p ≈ 0.15 for the memo's LR ≈ 2× to follow. Naming p explicitly
closes the audit loop.

**Recommended fix**: extend §4.3 rationale with a "Power calibration"
paragraph: per-cell H0 pass rate ≈ 0.15; per-cell H1 pass rate ≈ 0;
yielding `P(gate PASS | H0)` ≈ 0.50, `P(gate PASS | H1)` ≈ 0.95,
LR ≈ 2×; reference memo §4.

**Fix status**: applied.

---

### W4 — §5.2 Rule 2 reference does not restate binding character
**Finding**: §5.2 bullets list the council framing requirements
including "Explicit reference to `memory/feedback_council_methodology.md`
Rule 1 and Rule 2". But the preregistration itself does not restate
Rule 2's **binding** character inline — namely, that the post-F=1
council scopes the *next-step probe* and cannot rewrite the current
gate thresholds.

**Why it matters**: Rule 2 is the mechanical load-bearer of the entire
multi-gate design. If a future council (F = 1 branch) interpreted §5.2
as authority to re-score a gate downward, the probe's statistical
guarantee collapses. Making this non-negotiable inline, not as a
citation, protects against that mis-read.

**Recommended fix**: add a "Binding clause (Rule 2 restatement, inline)"
paragraph to §5.2 clarifying that gate thresholds are frozen at signing
and that post-F=1 council output scopes follow-on probes only.

**Fix status**: applied.

---

## INFO findings (advisory, not blocking)

### I1 — §3.2.1 split-date freeze note is strong but buried
The clause "do NOT retune split date post-hoc if the count is unequal"
is excellent anti-p-hacking discipline, but it lives inside the §3.2.1
paragraph. Consider elevating to a standalone sub-bullet under §6
irrevocable commitments (it's currently implicit in §6.5 but not
explicit about the specific 2025-07-15 date).

### I2 — §5 branch table missing σ notation for F = 1 verdict
§5.1 details paper-trade setup; §5.3 details sunset. §5.2 details
council framing. None cite the posterior for F = 1 (the memo §6 gives
~0.5). Consider adding "Posterior per memo §6 ≈ 0.5" to the §5 table
row for F = 1 symmetrically with F = 0 and F ≥ 2.

### I3 — §7 compute budget does not cross-reference Probe 2 actuals
Probe 2 §4 sweep wall-clocks are known (§3.2 in probe2_verdict.md).
§7 here estimates from first principles. Cross-referencing Probe 2
actuals would tighten the estimates, but not essential for signing.

---

## Verdict

**FIX WARNINGS THEN SIGN WITH JUSTIFICATION**

All four WARN items are in-text clarity / disclosure refinements that
do **not** change any gate threshold, branch rule, or irrevocable
commitment. They can be addressed with ~4 small in-text additions and
a re-sign at a fresh commit hash without restructuring the document.

No CRITICAL findings. Structural integrity (B1–B3), transcription /
arithmetic (A1–A4), memo consistency (C1–C2), and reference integrity
(D1) all PASS.

**Post-fix state**: the WARN-fix commit supersedes `4e5a114` as the
signing referent. Signing commit will diff only §9 (signature block)
against the WARN-fix commit — per Phase B3 invariant.

---

## References used by reviewer

- `tasks/probe3_preregistration.md` @ `4e5a114` — document under review
- `tasks/probe3_multiplicity_memo.md` — calibration authority
- `tasks/probe2_preregistration.md` — structural template
- `tasks/probe2_verdict.md` — Probe 2 authority for carve-out
- `memory/feedback_council_methodology.md` — Rules 1 & 2
- `memory/feedback_remote_execution_only.md` — execution policy
- `CLAUDE.md` — Reviewing Agent Protocol specification
