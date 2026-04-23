---
from: coding-agent
run_id: probe4_2026-04-22
pass: 2
for: [user]
timestamp: 2026-04-23T05:33:43Z
subject: Final pre-sign fixes to tasks/probe4_preregistration.md
files_modified: [tasks/probe4_preregistration.md]
fixes_applied:
  critical: [§5.2 routing citation corrected to row 5]
  warn: [W1 ordering disclosure, W2 JSON semantics, W3 execution model, W4 artifacts list]
  info: [none — I4/I5 are verdict-doc concerns]
cross_references:
  - tasks/_agent_bus/probe4_2026-04-22/code-logic-reviewer.md
  - tasks/_agent_bus/probe4_2026-04-22/stats-ml-logic-reviewer-pass2.md
---

## Summary
Five surgical edits applied. CRITICAL §5.2 fix aligns with both reviewers' convergent finding. Four WARNs from code-logic addressed. INFOs I4/I5 deferred to verdict document per reviewer guidance.

## Sign-readiness assessment
Per stats-ml-logic-reviewer-pass2: "Sign-ready after the one-line §5.2 correction." That correction has landed plus additional WARN clarifications. Prereg is structurally complete pending (a) §2.1/§2.2 parameter dict population from the v11 sweep parquet (per §9 step 3) and (b) user signing authorization.

## Remaining open items
- §2.1/§2.2 parameter dicts: still marked "TO BE POPULATED AT SIGNING" per §9 workflow.
- No third reviewer round required under stats pass-2's conditional approval.
