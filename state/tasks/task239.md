---
deployed_cost: 945
logged_costs_match: match
migrated: 2026-07-09
---

# task239 — FLOOR (2026-07-01)

mem/params at structural floor. Two 300B planes a=[1,10,30,1],b=[1,10,1,30] are operands of broadcast Less that IS the free 30x30 output; 30-dim is free-output dim, cannot shrink.

No source change.

## S10 (2026-07-03) — crop-to-bound priced FLOOR
Verified generator bound = 12 (in 4×4). Flagged `a`/`b` uint8 [1,10,30,1]/[1,10,1,30] 300B each are the operands of the Less that IS the free 30×30 output — their 30-dims are the output axes and can't shrink. Confirms the existing tasklog floor. FLOOR.

⭐ TRANSFERABLE: crop lever requires a counted ENTRY-read plane; a plane whose oversized dim is the free-output axis is un-croppable (S10 11/11 FLOOR — check output-weldedness before probing).
## 2026-07-08 — ReduceSum spatial profile -> Einsum tail ADOPTED

Applied the public-derived `ReduceSum(input, axes=[2,3], keepdims=0)` to
`Einsum(input, equation='bchw->bc')` rewrite from
`reports/candidates/reducesum_spatial_to_einsum_probe.py`.

Candidate:
`reports/candidates/task239/task239_reducesum_spatial_to_einsum_greedy.onnx`.
Bundled gate after adoption: fail=0, cost `947 -> 945`
(memory `897`, params `50 -> 48`).  Adopted into
`submission/overfit_nets/task239.onnx`.
