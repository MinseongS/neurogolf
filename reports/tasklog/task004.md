# task004 — 025d127b

**Rule:** Grid (size 8..16, variable H,W) holds 0..4 axis-stacked slanted parallelogram
outlines, one DISTINCT random colour each, separated by a full blank row (shapes are
ROW-SEPARABLE, never share a row). The output un-slants each shape's outline: every shape
row shifts RIGHT by +1, EXCEPT the shape's BOTTOM row (shift 0) and the rightmost pixel of
the SECOND-TO-LAST row (also shift 0). Colours simply copy the input colours. Reformulated
to a fully-local per-cell partition (verified exact 3000/3000, zero collisions):
rowany[r]=ReduceMax(occ,cols); below[r]=rowany[r+1]; below2[r]=rowany[r+2];
is_bottom=rowany∧¬below; is_2ndlast=rowany∧below∧¬below2;
special[r,c]=occ[r,c]∧is_2ndlast[r]∧occ[r+1,c] (occ pixel directly below);
copy_cell=occ∧(is_bottom∨special); shift_cell=occ∧¬copy_cell;
L_out=shiftR1(colf·shift_cell)+colf·copy_cell.
**Current:** 14.08 pts, gen:thbdh6332, mem 54000, params 1020
**Target tier:** A — output colours COPY arbitrary input colours (Tier S route blocked: a
fixed Conv can't route random per-instance colours), but the whole map is a separable
per-row shift collapsed into ONE colour-index value plane, no [1,10,H,W] product.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | full-30 fp16 planes, single L plane + ingrid mask | A | 43926 | 32 | 14.31 | n/a | works, MARGINAL (+0.22) |
| 2 | crop all working planes to 17×17 (active-region escape) | A | 25152 | 38 | 14.87 | 200/200 + 500/500 | ADOPT (+0.78) |

## Best achieved
14.87 @ mem 25152 params 38 — recommend adopt: Y. Beats prior 14.08? Y (+0.78, ≥ +0.3).

## Irreducible-floor analysis
Dominant intermediates: colf30 (3600B fp32 Conv entry — the irreducible 10→1 colour-index
plane), ingrid30 (3600B fp32 ReduceMax-over-channels, needed to distinguish in-grid-bg from
off-grid since both have colf=0), L30 (1800B fp16 padded value plane) and its int32 cast
L30_i32 (3600B, forced because opset-10 Equal rejects fp16/uint8 — only int32/int64/bool).
Everything else is a 17×17 fp16/bool working plane (≈300–600B each). The 17×17 active-region
crop (generator bounds H,W ≤ 16, +1 col of shift headroom) is the lever that took it from
44k → 25k.

## OPEN ANGLES (re-attack backlog)
- Drop the separate ingrid30 ReduceMax (3600B): fold in-grid detection into the colour plane,
  e.g. a single Conv with a marker weight that makes off-grid==exactly 0 while in-grid-bg ≥
  some sentinel — would remove one full 30×30 fp32 plane (~3600B → score ~+0.15).
- The L30_i32 cast (3600B) exists only because opset-10 Equal needs int32. If the final
  expansion could run as Pad(uint8 one-hot)→FREE output instead of Equal(int32 plane), the
  3600B int32 plane disappears — but Pad rejects bool and casting the [1,10,W,W] one-hot to
  uint8 then padding (FREE output) measured WORSE here (two 2890B planes). Re-check with a
  smaller W.
- Could the two row-shift Pads (below/below2) be merged into one 18-row pad + two slices to
  shave a 62B plane (marginal).

## INSIGHT (transferable)
⭐ A per-shape geometric SHEAR/"un-slant" that looks like it needs shape segmentation is
fully LOCAL when shapes are row-separable: classify each row by 1-cell vertical occupancy
neighbours (rowany[r], rowany[r±1,2]) and characterise edge-case pixels by their immediate
vertical neighbour (here the special "rightmost of 2nd-to-last row" = an occupied cell with
an occupied cell directly below in a 2nd-to-last row) — NO flood-fill, NO rightmost/argmax
scan. Then a colour-COPY remap collapses to ONE colour-index value plane
L=shiftR1(colf·shiftmask)+colf·copymask (masks partition the occupied cells with zero
collisions, so Add == Or). Pairs with the active-region crop (generator size bound) for the
big byte win — the 30×30 floor only really bites the two fp32 entry planes and the final
int32 Equal plane.
