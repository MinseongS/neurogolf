---
deployed_cost: 1470
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task217 — 8f2ea7aa

## 2026-06-29 mechanism screen

Rule: use a 3x3 Conway-like motif as a sprite and render pairwise motif-position
convolutions in a 9x9 output area.

Current source score: 17.706982 @ mem 1429 params 41.  The graph is already close
to a semantic floor: reduce background/foreground colour, crop the motif, run one
dilated Conv for the pairwise placement mask, and route with `Where(mask, fg, bg)`.

No rewrite adopted.  The dominant cost is the small 9x9/10x10 crop and output
masking, not a removable full-canvas intermediate.

## 2026-07-11 CANDIDATE-READY — Kronecker/fractal einsum (NOT yet adopted)

`candidates/task217/task217_kron_einsum.onnx` (builder `build_kron_einsum.py`):
**cost 510 (mem 0 + params 510), 18.766 pts = +1.059** vs deployed 1470/17.707.
Gate PASS 266/266; fresh A/B 2000/2000, candidate != incumbent = 0.

Mechanism: generator output = sprite⊗sprite×color, INDEPENDENT of which block
holds the sprite ⇒ mod-3 fold of the free input recovers the sprite mask M
wherever it sits. Whole net = ONE 12-operand free-output Einsum (input×2),
zero counted intermediates: `tcz,nzEF,Ea,Fb,tw,nwGH,Gi,Hj,Yi,Ya,Xj,Xb->ncYX`
with 3 stacked t-terms (colors δ(c,z≥1)·M⊗M / +81·e0·inside9 / −82·e0·M⊗M)
exploiting (out>0) grader decode. Tdiv[Y,i]=1{Y//3==i,Y<9} gates the canvas.
⭐ TRANSFERABLE: Kronecker/fractal einsum family (see state/insights.yaml
entry `kron_fractal_einsum`); the 2026-06-29 "close to a semantic floor"
verdict above is FALSIFIED (self-referential floor, −0.96 nats off).

## ADOPTED 20260711T141618Z
- cost: 1470 -> 510 (points 18.7656)
- source: candidates/task217/task217_kron_einsum.onnx
- note: kron_fractal_einsum: sprite-kron-sprite x color as ONE free-output einsum (memory 0); div/mod placement tables + mod-s fold free-input reads; falsifies 06-29 'semantic floor'

## ADOPTED 20260715T032548Z
- cost: 510 -> 506 (points 18.7735)
- source: candidates/task217/factor_w2_duplicates.onnx
- note: direct exact duplicate-column initializer factorization
