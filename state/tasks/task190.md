---
deployed_cost: 1127
logged_costs_match: true
migrated: 2026-07-09
---

# task190 — 7ddcd7ec (extend diagonal seeds of a 2x2 box into full 45° rays)

**Rule:** A solid 2×2 box of one COLOUR k sits at (row,col) on a fixed 10×10 grid.
Up to three of the four diagonal corners carry a single SEED pixel (same colour) one
cell out from the matching box corner (d0 up-left (row−1,col−1); d1 up-right
(row−1,col+2); d3 down-left (row+2,col−1); d2 down-right (row+2,col+2)). The INPUT
shows box + present seeds (one cell each); the OUTPUT extends each present seed into a
full 45° ray out to the grid edge, box preserved.
**Current:** 17.972685 pts, custom:task190 (closed-form scalar recovery + FREE-output threshold-sign polynomial), mem 620, params 507.
Session start 17.587840 at cost1656; current session gain +0.384846.
**Target tier:** B (colour-index label-map + final Equal). Not S/A: rays are data-dependent
45° diagonals (r−c==Dmain / r+c==Aanti) clipped to a half-plane — not a fixed per-cell
permutation (S) and a single diagonal is not a row⊗col separable rectangle (A).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | colf30 1×1 conv entry + full-plane flag ANDs | B | 9353 | 82 | 15.85 | — | 3600B colour plane + 4×400B flags |
| 2 | ch0-slice occupancy, ReduceMax-k (drop colour plane) | B | 5833 | 85 | 16.31 | — | killed the 3600B entry |
| 3 | flags via per-row occupancy profiles (od/oa→ReduceMax cols) | B | 5193 | 85 | 16.43 | — | 1600B flag planes → 2×200B |
| 4 | Where for od/oa and L (drop ondiagf/onantif/fillf casts) | B | 4593 | 85 | 16.55 | — | −600B |
| 5 | box row/col via 1-D profiles (drop two 9×9 product planes) | B | 4341 | 85 | 16.60 | — | −252B |
| 6 | ondiag/onanti via [1,1,1,10] target vector (drop dval/aval) | B | **3981** | **85** | **16.69** | **200/200** | WIN +0.43 |

## Best achieved
**17.972685 @ mem620 params507, cost1127 — bundled266/266.** Adopted through `ng adopt` and exact-source synchronized.

## Current floor analysis
The old 900B padded label and all three 100B ray/mask planes are gone. Current memory620 is led by
`bg_f`144B and symmetric target features150B plus small coordinate tensors. Params507 are led by
`poly_row_project[10,30]`300 and `poly_spatial_features[3,30]`90. A further +0.1 requires
cost<=1019; the only priced detector rewrite saves about65, so no current byte-proven +0.1 remains.

## OPEN ANGLES (re-attack backlog)
- Coordinate detector: replace `bg_f`/`bg_u8`/`blockmax` only after proving a u8/boolean lowering
  with total cost<=1019 and no new full-plane carrier.
- Row routing: factor `poly_row_project[10,30]` only under threshold semantics and only if the
  finite-state proof preserves every input/background channel sign.
- Do not restore staged 10x10/30x30 masks, exact CP identity routes, or the slow original operand order.

## INSIGHT (transferable)
⭐ **A "seed → grow ray to edge" task is closed-form, not a flood/connectivity wall:** the full
ray = a fixed 45° diagonal (r−c==Dmain or r+c==Aanti, both scalars from the 2×2-box top-left)
clipped to a half-plane (rows ≤ row−1 or ≥ row+2; the box rows split the line). Ray-present
flags are `ReduceMax(occ & ondiag & half) > 0` — scalar bools with NO Gather, reusing the same
masks. The box (the diagonal cells row,col & row+1,col+1) sits in NEITHER half so it never
fires a phantom ray.
⭐ **Build `Equal(rr−cc==D)` as `Equal(rr[1,1,H,1], (cc+D)[1,1,1,W])`** — fold the scalar into a
[1,1,1,W] target row vector so the Equal broadcasts straight to the 2-D bool, eliminating the
fp16 `dval=rr−cc` / `aval=rr+cc` full planes (−400B here).
⭐ **`Where(mask, occf, 0)` and `Where(mask, k, 0)` replace cast+Mul** (drops the bool→f16 cast
plane each time); reduce a detection plane to 1-D row/col profiles BEFORE the index-weighted
ReduceSum to recover (row,col) scalars without two full product planes.

## ADOPTED 20260712T140209Z
- cost: 2355 -> 1656 (points 17.5878)
- source: /Users/minseong/project/neurogolf/dumps/archive_extract/submission7300+/task190.onnx
- note: archive.zip submission7300+ net; fresh 2000/0 fail; mechanism-graft

## ADOPTED 20260715T115101Z
- cost: 1656 -> 1419 (points 17.7423)
- source: candidates/task190/archive_fast_backbone.onnx
- note: runtime-safe output-fold: precontract constant polynomial coefficient/column banks; backbone-ordered 17->13 Einsum operands; remove unused C_u8; 1656->1419 (the adopt CLI note said 12, corrected here after recounting inputs)

## ADOPTED 20260715T120438Z
- cost: 1419 -> 1367 (points 17.7796)
- source: candidates/task190/archive_reordered_backbone.onnx
- note: parameter-cheap runtime fold: retain factorized constant banks, backbone-reorder 17-operand terminal Einsum, remove unused C_u8; 1419->1367

## ADOPTED 20260715T122244Z
- cost: 1367 -> 1215 (points 17.8975)
- source: candidates/task190/route_sign_factor.onnx
- note: threshold route factor: replace dense identity/all-ones route[2,10,10] with shared [1,x,x^2] features and 2x3x3 sign core; 1367->1215

## ADOPTED 20260715T123004Z
- cost: 1215 -> 1163 (points 17.9412)
- source: candidates/task190/color_sign_factor.onnx
- note: threshold color fold: ArgMax/Gather shared quadratic features replace exact one-hot color Concat with epsilon-scaled sign branch; 1215->1163

## ADOPTED 20260715T123721Z
- cost: 1163 -> 1149 (points 17.9534)
- source: candidates/task190/shared_branch_core.onnx
- note: parameter collapse: share E00/quadratic branch core across route and color paths, move epsilon into term coefficients; 1163->1149

## ADOPTED 20260715T124839Z
- cost: 1149 -> 1127 (points 17.9727)
- source: candidates/task190/symmetric_ray_factor.onnx
- note: symmetric ray polynomial: replace repeated `(w-d)^2(w-a)^2` target Cast/Pow factors with u8 `[1,d+a,da]`, shared fp32 `[1,w,w^2]`, and a 2x3x3 core; cap10 preserves all roots on w0..9
