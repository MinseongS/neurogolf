# task55 — 272f95fa

**Rule:** The grid is partitioned into a 3x3 arrangement of variable-size blocks (block sizes random 1..10) by two full horizontal cyan(8) lines and two full vertical cyan(8) lines (all four always present). The input already contains the cyan cross. Output keeps the input cyan lines and fills 5 blocks (a plus shape) with FIXED colours by (rowband,colband): (0,1)=red2, (1,0)=yellow4, (1,1)=magenta6, (1,2)=green3, (2,1)=blue1. Corner blocks + off-grid stay background 0.

**Current:** 14.83 pts (prior net)
**Target tier:** A — separable rows×cols band partition + fixed colour LUT, routed into FREE BOOL output; no flood-fill, no NonZero.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | CumSum band + double-MatMul LUT, fp32 planes, occ ReduceSum gate | A | 18900 | 51 | 15.15 | — | passes 263/263 |
| 2 | fp16 downstream + separable rowin/colin gate | A | 9660 | 46 | 15.82 | — | passes |
| 3 | drop uint8 cast, fp16 Equal directly into output | A | 8760 | 46 | 15.92 | 200/200 | FINAL |

## Best achieved
15.92 @ mem 8760 params 46 — adopted? pending (left at src/custom/task55.py). Beats prior 14.83? Y (+1.09).

## Irreducible-floor analysis
Three [1,1,30,30] fp16 value planes (Lband, Lg, L = 1800 B each = 5400) dominate. The (rowband,colband)->colour LUT is rank>1 (a sum of 5 rank-1 filled blocks) so the double-MatMul genuinely produces one full plane; the line override (hline_r OR vline_c) and the off-grid override (NOT(rowin AND colin)) are 2-D OR/AND of row & col vectors, so each Where materialises its own 30x30 selection. Band cumsums and occupancy gates are kept as ~120 B row/col vectors — no full occupancy plane.

## OPEN ANGLES (re-attack backlog)
- Fold the line override into the double-MatMul by augmenting the band one-hots with hline/vline basis vectors. Blocked: a line cell still carries its band one-hot, so the bilinear term adds LUT[k][cb] that cannot be cancelled to a clean 8 — would need the band term zeroed, which the band one-hot does not provide. Could try a 5-component row/col factor where the line component DOMINATES additively (e.g. line weight 1000) then read back by magnitude bands in the final Equal threshold, collapsing Lg into Lband (~−1800 B, ~+0.2 pts).
- Merge offgrid+line into a single override plane (disjoint regions) to drop one Where — but building `8*isline + 10*offgrid` costs as many full planes as it saves.

## INSIGHT (transferable)
A data-dependent rows×cols block grid drawn by FULL separator lines is a fully separable partition: the per-axis band index is the EXCLUSIVE CumSum of the line indicator sampled from the line colour along the first column/row (which is never itself a separator). A non-rank-1 (rowband,colband)->colour map is then the double-MatMul LUT idiom (Ronehot @ LUT3x3 @ Conehot), and preserving the input separator lines is free: just overlay the line colour (8) where the line indicator is set before the final Equal. No flood-fill, no NonZero — closed-form Tier A. ⭐ Reusable for any "fill specific cells of a line-delimited grid with fixed colours" task.
