---
deployed_cost: 186
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task376 — data-dependent modulus → row-index permute via Gather

## 2026-07-03 S13 — UNKNOWN triage: AT-FLOOR (int64→int32 lever = MEASURED ZERO)

**Cost (grader mem 144, params 42):** `packed` INT32[30] Gather-index row vector (floor) +
`starts`/`ends` INT64[4] Slice inits. Rule: data-dependent modulus → row-index permute via
Gather(axis=2).

**KILL — false lever:** triage agent flagged "Slice inits INT64→INT32, +0.098". MEASURED
REFUTED: params = element count (`math.prod(dims)`), dtype-agnostic → Δparams = 0 (42→42).
`packed[30]` Gather-index row vector is the irreducible floor. See task103 S13. AT-FLOOR.