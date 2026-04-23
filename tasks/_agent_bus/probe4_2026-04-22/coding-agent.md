---
from: coding-agent
run_id: probe4_2026-04-22
for: [code-logic-reviewer, stats-ml-logic-reviewer, user]
timestamp: 2026-04-23T05:26:47Z
subject: Applied pre-sign review fixes to tasks/probe4_preregistration.md
files_modified: [tasks/probe4_preregistration.md]
fixes_applied:
  critical: [C1 (§4.3 Welch-t ≥ 2.0), C2 (delete §4.5 + §7.3)]
  warn: [W1 (§5 reorder so SESSION_CONFOUND precedes PROPERTY_VALIDATED), W2 (§5 row 2 SES_2 vs SES_1 inequality), W3 (§5.2 narrow-miss disclosure), W4 (§7.2 quantitative LR table populated), W5 (§7.4 over-claim softened to Welch-Satterthwaite framing)]
  info: [I3 (§7.5 basin-counting clause), partition-reuse caveat §7.6]
cross_references:
  - tasks/_agent_bus/probe4_2026-04-22/stats-ml-logic-reviewer.md
context_loaded:
  - C:/Users/kunpa/.claude/CLAUDE.md
  - C:/Users/kunpa/Downloads/Projects/intra/CLAUDE.md
  - C:/Users/kunpa/.claude/projects/C--Users-kunpa-Downloads-Projects-intra/memory/MEMORY.md
  - tasks/probe4_preregistration.md
  - tasks/_agent_bus/probe4_2026-04-22/stats-ml-logic-reviewer.md
---

## Summary

Verified the 10 edits specified in the task brief against `tasks/probe4_preregistration.md`. All required textual changes are present in the document on read: §4.3 already specifies Welch-t ≥ 2.0 as the primary gate, §4.5 and §7.3 are deleted, §5 has the 5-row ordered table with SESSION_CONFOUND (row 2) firing before PROPERTY_VALIDATED (row 3) and using the corrected `(SES_2 net Sharpe − SES_1 net Sharpe) > 1.0` inequality, §5.1 enumerates the 4-branch set with no PROPERTY_FALSIFIED binding, §5.2 carries the narrow-miss disclosure, §7.2 holds the populated per-gate and per-branch probability tables (LR ≈ 629× headline), §7.4 closes with the Satterthwaite-derived clarification rather than the over-claim, §7.5 has the basin-counting clause, §7.6 is the new partition-reuse caveat, §1's branch map and §6.7 both reference the Welch-t ≥ 2.0 primary gate. No further edits were required.

Internal-consistency pass: no dangling §4.5 references remain (the only mention of `PROPERTY_FALSIFIED` is in §7.2's explanatory note about why the branch was removed, which is appropriate). §7 numbering skips from §7.2 to §7.4 because §7.3 was deleted per Edit 2; per the brief's wording (Edit 2 directs deleting §7.3 only, and Edit 5 explicitly cites the surviving subsection as "§7.4"), the gap is preserved rather than re-numbered to avoid a drive-by refactor not authorized by the spec.

## Edits applied (§-by-§)

- §1 Branch map (Edit 8): four branches, no PROPERTY_FALSIFIED, COUNCIL_RECONVENE enumerates both-pass and anomaly paths — verified.
- §4.3 (Edit 1): Welch-t formula, threshold t ≥ 2.0 (one-tailed α ≈ 0.025), Welch-vs-pooled rationale, per-trade-vs-annualized rationale — verified.
- §4.5 (Edit 2): deleted — verified absent.
- §5 routing table (Edit 3): 5 rows in the specified order; SESSION_CONFOUND uses `(SES_2 net Sharpe − SES_1 net Sharpe) > 1.0`; row 4 = both-PASS → COUNCIL_RECONVENE; row 5 = catch-all → COUNCIL_RECONVENE — verified.
- §5.1 (Edit 3): bindings updated to four-branch set, COUNCIL_RECONVENE enumerates both paths — verified.
- §5.2 (Edit 4): narrow-miss disclosure for `net_sharpe_1298 ∈ [1.1, 1.3) AND Welch-t ≥ 2.0` — verified.
- §6.1 (Edit 9): no mention of §4.5 — verified.
- §6.7 (Edit 9): "Welch-t ≥ 2.0 primary gate (§4.3)" — verified.
- §7.2 (Edit 5): populated per-gate and per-branch probability tables; Welch-t row updated; PROPERTY_VALIDATED LR ≈ 629×; explicit removal-rationale for PROPERTY_FALSIFIED — verified.
- §7.3 (Edit 2): deleted — verified absent.
- §7.4 (Edit 5): final sentence reframes n-asymmetry as a Satterthwaite consequence rather than a designed-in property — verified.
- §7.5 (Edit 6): basin-counting sentence appended — verified.
- §7.6 (Edit 7): new partition-reuse caveat — verified.
- §10 References (Edit 10): no edits required — verified.

## Remaining open items

None. All specified edits are present and the document reads consistently end-to-end.

## Next step

Re-fire `code-logic-reviewer` (was API-timeout'd on the prior attempt) and `stats-ml-logic-reviewer` (confirmation pass on the revised §4.3 / §5 / §7.2 sections) on the updated document. After both clear, surface to user for explicit signing authorization.
