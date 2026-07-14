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

## ADOPTED 20260713T143831Z
- cost: 629 -> 600 (points 18.6031)
- source: candidates/public_dumps/20260713_7281/extracted/task215.onnx
- note: Ryosuke 7281.18 public-LB confirmed per-task min-merge; bundled fail=0

## ADOPTED 20260713T150958Z
- cost: 629 -> 600 (points 18.6031)
- source: candidates/public_dumps/20260713_7281/extracted/task215.onnx
- note: Ryosuke-7281 isolation B; task047 explicitly excluded; bundled fail=0

## ADOPTED 20260713T151940Z
- cost: 629 -> 600 (points 18.6031)
- source: candidates/public_dumps/20260713_7281/extracted/task215.onnx
- note: Kaggle-isolated safe: group delta +2.05 exactly (sub 54651291 minus 54651270); task047 excluded
