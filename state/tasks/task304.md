---
deployed_cost: 1441
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task304 — c3e719e8

## 2026-06-29 label-vs-onehot screen

Current source score: 17.726907 @ mem 1404 params 37.

Rule: from a 3x3 colour grid, find the majority colour and expand only majority
cells into copies of the whole 3x3 pattern, producing a 9x9 output.

The source uses `DepthToSpace` to build a [1,1,9,9] colour label plane and then
`Equal(colors9, color_codes)` to produce a [1,10,9,9] bool one-hot (`y9`, 810 B)
before padding to output.  A full-canvas label route would require a 30x30 uint8
label (900 B) before final `Equal`, so the current 9x9 one-hot is cheaper.

No rewrite adopted.

## ADOPTED 20260709T041322Z
- cost: 1441 -> 1320 (points 17.8146)
- source: candidates/public_dumps/20260709/7261-53-lb-compact-onnx-artifact-starter/nets/task304.onnx
- note: min-merge from nets

## 2026-07-11 CANDIDATE-READY — Kronecker/fractal einsum (NOT yet adopted)

`candidates/task304/task304_kron_einsum.onnx` (builder `build_kron_einsum.py`):
**cost 564 (mem 174 + params 390), 18.665 pts = +0.850** vs deployed 1320/17.815.
Gate PASS 266/266; fresh A/B 2000/2000, candidate != incumbent = 0.

Mechanism: out = modeMask ⊗ grid. Mode one-hot via ReduceSum/ReduceMax/Equal
(the only nonlinearity, 174B chain incl wm2=[onehot;ones] concat); final node
= ONE 12-operand free-output Einsum reading input twice
(`tcz,nzEF,Ed,Fe,tw,nwGH,Gr,Hs,Yr,Yd,Xs,Xe->ncYX`): read1 crop-folds the grid
one-hot (mod-3 fold ≡ crop since input empty beyond 3×3), read2×wm2 gives
[Mmask; occ≡1] per t; W1 = [δ(c,z)−2e0[c]; e0[c]·1] ⇒ ch0 = 1−2·Mmask under
(out>0) decode. ⭐ TRANSFERABLE: kron_fractal_einsum family.

## ADOPTED 20260711T141618Z
- cost: 1320 -> 564 (points 18.6649)
- source: candidates/task304/task304_kron_einsum.onnx
- note: kron_fractal_einsum: fractal stamp at mode-color cells; mode one-hot only nonlinearity, wm2 runtime-stacked mask operand
