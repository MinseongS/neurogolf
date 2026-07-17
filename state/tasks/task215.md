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

## ADOPTED 20260715T020447Z
- cost: 600 -> 492 (points 18.8015)
- source: candidates/task215/factor_routes.onnx
- note: exact rank-3/rank-4 factorization of both position-route tensors inside the single FREE-output Einsum; cost 600->492

## ADOPTED 20260715T062756Z
- cost: 492 -> 430 (points 18.9362)
- source: candidates/task215/joint_spatial_basis.onnx
- note: shared exact spatial basis across VrL/VhL inside final output Einsum

## ADOPTED 20260715T063124Z
- cost: 430 -> 414 (points 18.9741)
- source: candidates/task215/joint_basis_min.onnx
- note: absorb identity VhL coefficient into shared spatial basis

## ADOPTED 20260715T105416Z
- cost: 414 -> 372 (points 19.0811)
- source: candidates/task215/joint_basis_reuse.onnx
- note: parameter collapse: drop all-ones Ksel; reuse four-column spatial basis for fixed column selector and both route axes, cost 414->372

## ADOPTED 20260715T111039Z
- cost: 372 -> 260 (points 19.4393)
- source: candidates/task215/quadratic_color.onnx
- note: mathematical rewrite: replace dense 2x10x10 colour tensor with rank-4 threshold polynomial; foreground score 2k*(0.5-(f-k)^2), background delta(f=0), cost 372->260

## ADOPTED 20260715T113500Z
- cost: 260 -> 256 (points 19.4548)
- source: candidates/task215/basis_change.onnx
- note: parameter collapse: unimodular spatial-basis change makes unweighted basis sum equal the fixed column selector, removing basis_pick; cost 260->256

## ADOPTED 20260715T114043Z
- cost: 256 -> 234 (points 19.5447)
- source: candidates/task215/rank3_sign.onnx
- note: mathematical rewrite: shared rank-3 foreground/background sign route with scaled epsilon column; combines channelwise threshold polynomial with basis-pick elimination, cost 256->234

## ADOPTED 20260715T131635Z
- cost: 234 -> 171 (points 19.8583)
- source: candidates/task215/spatial_rank3.onnx
- note: sign-rank spatial collapse: replace fixed-column selector by positive in-grid width, remove independent background route, and share exact period-3 basis; cost 234->171
