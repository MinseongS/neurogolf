# task138 — 5daaa586

**Rule:** Input H×W grid (H,W~10..26) with a rectangular box of four FULL lines —
left/right vertical (cols `left`,`right`, every row) and up/down horizontal
(rows `up`,`down`, every col) — coloured colors[0..3]=(left,right,up,down) and
drawn in a random `draworder` (corners take the last-drawn line's colour). Scattered
single-cell pixels of one `drawcolor` (== exactly one of the four line colours,
otherwise distinct) emit a RAY in a single global direction toward the matching
wall (left if drawcolor==colors[0], right==colors[1], up==colors[2], down==colors[3]),
painting drawcolor from the pixel up to (not into) the wall. Output = the box region
[up..down]×[left..right] moved to the top-left of a fresh canvas (the four edge lines,
the pixels and their rays; corners = the input box corners).

**Current:** 14.223 pts, ext:biohack_new, mem 47770, params 125
**Target tier:** B (output colours COPY arbitrary input colours → Tier-S Conv routing
blocked; the variable crop + ray fill is a label-map+Equal job).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | label-map+Equal, 30×30 fp32 planes, 4 tri-matrices | B | 60913 | 3720 | 13.92 | 200/200 | correct but worse (bug fixed: line colour via interior-masked max, not raw ReduceMax — corners leak larger colours) |
| 2 | all downstream fp16, 2 tri-matrices | B | 39171 | 1919 | 14.38 | 200/200 | marginal |
| 3 | line colours by single-cell Gather (no colf_ir/colf_ic planes) | B | 35375 | 1921 | 14.47 | 200/200 | marginal +0.25 |
| 4 | 26×26 working canvas (slice colf, Pad label to 30 at end) | B | 31331 | 1475 | 14.602 | 500/500 | ADOPT (+0.38) |

## Best achieved
14.602 @ mem 31331 params 1475 — beats prior 14.223 by +0.379 (≥+0.3). Fresh 500/500
on a disjoint seed range.

## Irreducible-floor analysis
Entry: Conv→colf30 (3600 fp32) sliced to colf 26×26 (2704 fp32) = 6304B for the one
value plane; fp32 is forced (ORT ReduceSum/Conv on the 10-ch one-hot). Everything
downstream is fp16 26×26 (1352B) or bool/uint8 (676B). The ray costs Mc+Mr (matrix
selects, 1352 each) + rayH+rayV (matmuls) + rayc (axis select) ≈ 6760B — the dominant
removable cluster. Slicing to 26×26 (max grid dim verified 26 over 8000) was the big
lever (~25% off every plane), partly offset by the 2704 slice plane.

## OPEN ANGLES (re-attack backlog)
- Ray select: Mc/Mr/rayc are 3 fp16 planes for a single active axis. A transpose-based
  unification (vertical = (seed^T @ M)^T) could drop to one matmul + 2 transposes, or a
  data-independent-attr CumSum (reverse handled by a constant flip-Gather) might cut the
  matrix inits — net byte payoff uncertain (~1-2k), would push ~14.6→~14.7.
- intv (drawcolor) is a full 1352B fp16 plane; drawcolor = unique interior colour could
  perhaps come from a 1-D row/col interior-max pair instead of a 2-D mask-then-max.
- Tighter working canvas: median grid is ~17×17; a data-dependent slice would trip the
  symbolic-dim trap, so 26 is the safe static bound.

## INSIGHT (transferable)
⭐ A "box + axis-aligned rays + variable crop" task is closed-form Tier-B, NOT a
detection/connectivity wall: (1) FULL boundary lines are recovered EXACTLY by per-row/
col occupancy COUNT == H/W (a full line hits every cell; scattered pixels never reach
H — 0 collisions/5000); (2) the uniform-direction ray fill = ONE triangular boolean
MatMul (seed@UT/LT for horizontal, LT/UT@seed for vertical), the triangle chosen by the
recovered direction scalar; (3) the input value plane `colf` already carries the lines
WITH correct corner overlaps (draworder baked in by the generator), so V = Where(ray,
drawcolor, colf) needs no corner logic; (4) variable crop → Gather(axis2,arange+up)·
Gather(axis3,arange+left) shift-to-origin + a row<oh & col<ow keep-mask → sentinel. Two
gotchas: read line colours from a NON-corner line cell (single-cell Gather at (up+1,left)
etc.) — a raw per-column ReduceMax leaks a larger-valued corner colour; and clamp the
working canvas to the generator's max grid dim (26) to take ~25% off every plane.
