# task340 — d687bc17

**Rule:** Same-size grid. A coloured border frame is drawn: top row = ec0, bottom
row = ec2, left col = ec3, right col = ec1 (edgecolors = 4 distinct random colours;
corners stay background). Interior contains scattered coloured pixels (plus 1-3
garbage pixels of a non-edge colour). In the output every interior pixel whose
colour equals one of the four edge colours is PROJECTED onto the inner ring toward
its matching edge: ec0→output[1][col], ec1→output[row][W-2], ec2→output[H-2][col],
ec3→output[row][1]. Garbage pixels (no edge-colour match) vanish; the border is
preserved. The four projection sets and the border ring occupy DISJOINT cells.
Width,height ∈ [10,20].
**Current:** 14.69 pts, custom:task340 (prior version, mem 28784), pending (not adopted).
**Target tier:** B (label map) — output colours COPY arbitrary input edge colours
(random per instance) so Tier S/A colour-routing is blocked; the per-cell colour is
a deterministic function (single colour-index plane) ⇒ label-map + final Equal is
the natural minimal tier.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | prior file: cg32 colour-Conv + 4 dir Convs + 2 MatMuls | B | 28784 | 1307 | 14.69 | 200/200 | baseline (correct but heavy) |
| 1 | drop cg32 (H/W via ReduceMax[1,3]/[1,2]); 2 Convs only | B | 24274 | 708 | 14.87 | 40/40 | improve |
| 2 | fuse the 2 placement MatMuls into ONE A[30,8]@B[8,30] (no Add) | B | 20674 | 708 | 15.03 | 40/40 | improve |
| 3 | derive 4 edge one-hots from rowsum/colsum (kill row0all/row1all slices) | B | 19408 | 692 | 15.09 | 40/40 | improve |
| 4 | contract 10-ch axis with small MatMuls (kill all [1,10,30] Mul planes) | B | 16904 | 715 | 15.22 | 200/200 | **best** |

## Best achieved
**15.22 @ mem 16904 params 715 — 200/200 fresh, evaluate ok.** Beats prior 14.69 by
**+0.53** (and a large floor-break 28784→16904). Adopt-recommend **Y**.

## Irreducible-floor analysis (at this structure)
Dominant intermediates: og [30,30] fp16 = 1800 (the packed placement MatMul output —
a colour-index plane is unavoidable for a copy-colour label map); the two depthwise
count Convs colsum32/rowsum32 [1,10,1,30]/[1,10,30,1] fp32 = 1200 each (conv outputs
are forced fp32; BOTH axes are needed — top/bottom use per-column, left/right use
per-row); their [10,30] fp16 working copies rsr/csc = 600 each; the three label
planes og_u8 / L / gm_b = 900 each (uint8/bool — the in-grid sentinel mask gm_b is
load-bearing because off-grid og=0 would otherwise emit ch0=background as TRUE).
All per-channel selection/Mul planes ([1,10,30,*], the old ~4800B bloat) were removed
by contracting the channel axis with tiny MatMuls ([10,2], [2,30]) instead of
Mul+ReduceSum. The remaining ~16.9KB is near the structural floor for a copy-colour
label map that needs per-row AND per-column channel counts.

## OPEN ANGLES (re-attack backlog)
- Kill one of the two count Convs: if per-row counts could be derived from per-col
  counts (they can't in general — different reductions), or if a single 2-D
  integral image served both, ~1200-2400B could drop. No clean route found.
- Fuse rsr/csc reshape with the conv cast to avoid the duplicate fp16 view
  (currently conv32→colsum16→csc16, two 600B tensors per axis). Reshape-before-cast
  trades fp16 for fp32 at equal cost; no net win located.
- Merge og_u8 into the Where (Cast then Where currently = 1800B for two planes).
  A fp16 Where carrier is 1800 (worse); uint8 path is already minimal.
- Theoretical Tier-A would need the output to be a row⊗col separable one-hot, but
  the placement is a UNION of 8 thin lines on different rows/cols whose colours are
  arbitrary input copies — only the disjoint label map collapses it, so B is the cap.

## INSIGHT (transferable)
⭐ For tasks that need per-channel per-row AND per-column statistics, do NOT build
`Mul(counts[1,10,*], onehot)` then `ReduceSum` over channels — that materializes a
[1,10,30] plane (600 fp16 / 1200 fp32) for every selection. Instead RESHAPE the
depthwise-count tensors to [10,30] once and CONTRACT the 10-channel axis with tiny
MatMuls: `counts[10,30] @ selector[30,k]` picks border lines, `onehotT[k,10] @
counts[10,30]` reads per-line per-axis profiles, `krow[1,10] @ onehot[10,k]` reads
colour indices — all outputs are [≤10, ≤k] (tens of bytes). This collapsed ~4800B of
selection planes to a handful of small MatMuls here (19408→16904). Also: H/W and the
in-grid mask come free from `ReduceMax(input, axes=[1,3]/[1,2])` (bg ch0=1 in-grid,
0 off-grid) — no colour-index Conv needed at all. And N disjoint thin output lines
pack into ONE placement MatMul A[30,2N]@B[2N,30] (row-selectors×col-values for
column-indexed lines stacked with row-values×col-selectors for row-indexed lines),
eliminating a separate Add of two MatMul grids.
