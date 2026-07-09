---
deployed_cost: 739
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task215 — FLOOR (2026-07-01)

mem/params at structural floor. 240B ch0 full-width row read needed for per-row bg counts (width varies 12-19, 30 safe bound); Wheres/ReduceMax build free-output row index. Structural.

No source change.

## ADOPTED 20260709T041318Z
- cost: 739 -> 629 (points 18.5559)
- source: candidates/public_dumps/20260709/7261-53-lb-compact-onnx-artifact-starter/nets/task215.onnx
- note: min-merge from nets
