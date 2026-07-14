---
deployed_cost: 709
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task269 — FLOOR (2026-07-01)

mem/params at structural floor. 360B native 3x3x10 read floor; 4x4 Pad required for crop_and_resize to span full 3*scale extent (tested: dropping Pad breaks geometry, 5x5 vs 6x6).

No source change.

## ADOPTED 20260713T143830Z
- cost: 709 -> 656 (points 18.5138)
- source: candidates/public_dumps/20260713_7281/extracted/task269.onnx
- note: Ryosuke 7281.18 public-LB confirmed per-task min-merge; bundled fail=0

## ADOPTED 20260713T150958Z
- cost: 709 -> 656 (points 18.5138)
- source: candidates/public_dumps/20260713_7281/extracted/task269.onnx
- note: Ryosuke-7281 isolation B; task047 explicitly excluded; bundled fail=0

## ADOPTED 20260713T151940Z
- cost: 709 -> 656 (points 18.5138)
- source: candidates/public_dumps/20260713_7281/extracted/task269.onnx
- note: Kaggle-isolated safe: group delta +2.05 exactly (sub 54651291 minus 54651270); task047 excluded
