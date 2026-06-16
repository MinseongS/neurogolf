# task094 — 41e4d17e

**Rule:** Grid size 15, background cyan(8). Input holds 1-2 blue(1) 5x5 box *outlines*
(square perimeters) centred at (r,c), r,c in 3..11, boxes well separated (>=3 gap each
axis, crosshairs never touch). Output keeps the blue boxes and paints, for each centre
(r,c), the entire row r and column c pink(6). Blue is drawn after pink, so blue overwrites
pink at overlaps.
**Current:** 15.767 pts, ext:biohack_new, mem 10115, params 112
**Target tier:** A (separable label-map) — the crosshair mask is row-profile OR col-profile
(separable), output is a 3-colour label map → final Equal into free BOOL output.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 5x5 outline-Conv (no pad) → centre profiles → label map | A | 7575 | 289 | 16.03 | (offset bug) | Conv top-left aligned, crosshair off by (-2,-2) |
| 2 | + pads=[2,2,2,2] to centre the Conv response | A | 7575 | 289 | 16.03 | 265/265 | correct, MARGINAL (+0.26) |
| 3 | drop dead 15x15 resp slice; ReduceMax full resp → crop 1-D profiles | A | 6690 | 287 | 16.15 | 200/200 | beats P by +0.38 ✓ |

## Best achieved
16.15 @ mem 6690 params 287 — adopted? N (orchestrator gates). Beats prior 15.767? Y (+0.38).

## Irreducible-floor analysis
Dominant intermediate: the Conv response `resp` [1,1,30,30] fp32 = 3600B. Conv output is
forced to 30x30 because the input is 30x30; cannot shrink without first slicing the 10-ch
input (a [1,10,15,15]=2250B fp slice, plus its own conv — net worse). Second: the blue-mask
fp slice [1,1,15,15]=900B. The 5x5 outline detector is the cheapest exact centre finder
(peak=16 vs <=8 elsewhere) and reduces to two 1-D profiles immediately.

## OPEN ANGLES (re-attack backlog)
- Cast resp to fp16 before the two ReduceMax (1800B) — but Cast adds a tensor; only helps if
  the fp32 resp can be elided (it can't, it's the Conv output). Net neutral/worse.
- Detect centres without a 30x30 Conv: blue verticals at the centre row sit at cols c±2 with a
  3-cell gap — a 1-D row-profile of "blue count == box-perimeter signature" might isolate the
  centre row from cheap ReduceSum profiles (no 2-D Conv). Could drop the 3600B resp to ~120B
  vecs → potential tier-A ceiling jump (~17+). Not yet tried; the 2-axis ambiguity (which
  centre-row pairs with which centre-col) when two boxes are present needs the 2-D Conv to
  bind them — but the rule paints FULL rows and FULL cols independently, so row-set and col-set
  need NOT be paired. 1-D profiles alone may suffice → strong open lever.

## INSIGHT (transferable)
⭐ Box/ring CENTRE detection = one Conv with the ring's exact perimeter kernel; the response
peaks at the perimeter's pixel-count at the centre and is strictly lower elsewhere, so a single
Greater(thr) isolates centres with NO flood-fill. CRUCIAL: a default (no-pad) Conv aligns the
peak to the window's TOP-LEFT, not its centre — add `pads=[k,k,k,k]` (SAME) to land the peak on
the geometric centre. When the rule paints full independent rows AND full independent cols, the
2-D crosshair is SEPARABLE: reduce the centre plane to is_row[1,1,H,1] OR is_col[1,1,1,W] and
let the broadcast happen in the free final ops.
