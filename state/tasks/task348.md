---
deployed_cost: 1895
logged_costs_match: match
migrated: 2026-07-09
---

# task348 — db3e9e38

**Rule:** INPUT is a single vertical ORANGE(7) line at column `col`, rows 0..length-1, on a width×height grid (5..10, top-left anchored). OUTPUT draws a triangular pyramid: cell (r,c) colored iff `r + |c-col| < length`; colour = ORANGE(7) if (c-col) even (same parity as col), else CYAN(8). Rest stays background; off-grid stays all-zero.
**Current:** 16.12 pts (prior public net)
**Target tier:** A (closed-form, one 2-D plane) — the pyramid mask `r+|c-col|<length` is the single genuine 2-D tensor; everything else (col, length, width, parity, colour) is 1-D vectors/scalars.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | scalar (col,length,width) -> thresh[c] vector -> single `colored` plane -> Where(colored, colorOH[1,10,1,30], input) into FREE output | A | 4626 | 85 | 16.54 | 200/200 | ADOPT |

## Best achieved
16.54 @ mem 4626 params 85 — beats prior 16.12 by +0.42. fresh 200/200.

## Irreducible-floor analysis
Dominant intermediates: `colored` [1,1,30,30] bool (the pyramid mask, ~900B inferred) and `colsum_all` [1,10,1,30] fp32 (1200B, the row-reduction used to read length/col from channel 7). The `colored` plane is the one true 2-D object (coupled r+|c-col|, not separable) so it cannot be removed. The 10-ch expansion is FREE: Where's else-branch = input already carries correct in-grid background and off-grid zeros, and the per-column colour one-hot is only [1,10,1,30].

## OPEN ANGLES (re-attack backlog)
- `colsum_all` 1200B could in principle be avoided if length/col were read from a cheaper signal, but ReduceSum(input,axes=2)→[1,10,1,30] is already the minimal full-input reduction; slicing channel 7 first costs 3600B. Net not worth it.

## INSIGHT (transferable)
A row/col-COUPLED region mask `r + |c-col| < length` (pyramid/triangle/diamond) is NOT separable, but it is ONE plane via `Less(rowramp[1,1,30,1], thresh[1,1,1,30])` after folding all per-column geometry (|c-col|, length, in-grid width gate) into the tiny thresh VECTOR. Then a per-COLUMN colour one-hot `[1,10,1,30]` (parity-dependent) as the Where value-branch with input as the else-branch routes the full 10-ch expansion into the FREE output with zero extra full planes — off-grid and background fall out of the unchanged input automatically.

## 2026-07-07 — initializer dedupe micro-overlay

`reports/candidates/task348/task348_dedupe_initializers.onnx` rewired duplicate
initializer `pad_val->u8_250`.  Bundled gate fail=0.  Cost: 1896 -> 1895
(params 72 -> 71).

## ADOPTED 20260709T041321Z
- cost: 1895 -> 1706 (points 17.5581)
- source: candidates/public_dumps/20260709/7261-53-lb-compact-onnx-artifact-starter/nets/task348.onnx
- note: min-merge from nets

## ADOPTED 20260712T041155Z
- cost: 1706 -> 1288 (points 17.8392)
- source: candidates/task348/rankeinsum.onnx
- note: Manhattan pyramid scalar-emission rebuild: single sign condition t(c)-r>0 (t=length-|c-col|, no A*B product so no false corner); 10-wide geometry space via identity-embed M[30,10] shared compress/expand (no 30-wide runtime plane); separable occ=ingrow*ingcol; one free-output einsum ed,pm,epg,gk,rm,ck,b->bdrc, d=7/8 parity + d=0 bg; 1706->1288 (+0.28); fresh 2500 div 0

## ADOPTED 20260712T051838Z
- cost: 1288 -> 842 (points 18.2642)
- source: candidates/task348_codex/cand.onnx
- note: codex-worktree absorb: scalar-moment sign-decode refinement of my 1288 pyramid net 1288->842 (+0.425); fresh 2500 fail=0, divergence vs my deployed 348=0; gate 265/265 fail=0
