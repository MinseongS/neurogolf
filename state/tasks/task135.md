---
deployed_cost: 200
logged_costs_match: match
migrated: 2026-07-09
---


## S9 (2026-07-03) — mechanism-14 probe: REJECTED (270 > 200)
Incumbent Conv(10,10,2,1) does col-shift+crop FREE via pads (w_begin=-6) + occupancy
subtract. Einsum needs 3×30-wide tables (30-axis grading tax). Floor stands.

## ADOPTED 20260713T143829Z
- cost: 200 -> 180 (points 19.8070)
- source: candidates/public_dumps/20260713_7281/extracted/task135.onnx
- note: Ryosuke 7281.18 public-LB confirmed per-task min-merge; bundled fail=0

## ADOPTED 20260713T150957Z
- cost: 200 -> 180 (points 19.8070)
- source: candidates/public_dumps/20260713_7281/extracted/task135.onnx
- note: Ryosuke-7281 isolation B; task047 explicitly excluded; bundled fail=0

## ADOPTED 20260713T151939Z
- cost: 200 -> 180 (points 19.8070)
- source: candidates/public_dumps/20260713_7281/extracted/task135.onnx
- note: Kaggle-isolated safe: group delta +2.05 exactly (sub 54651291 minus 54651270); task047 excluded
