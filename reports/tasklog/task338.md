# task338 — d5d6de2d

**Rule:** size x size grid (size=5*mult, mult in [2,5] -> size in {10,15,20,25}), background
black(0). Several non-overlapping (separated by >=1) solid red(2) rectangles; each box's INTERIOR
((tall-2)x(wide-2) inner block) is reset to black, so every box is a 1-cell-thick red ring around a
black hole. OUTPUT: interior holes -> green(3); red ring + outside background -> black(0).
**Current:** 14.285 pts, ext:kojimar6275, mem 43200, params 1816
**Target tier:** A — closed-form column ray-cast (parity of horizontal-wall crossings); no flood-fill.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 4-dir any-red prefix/suffix-OR enclosure | A | 25750 | 1289 | — | 191/200 | FAIL: gap cell surrounded by 4 separated boxes (global OR merges) |
| 2 | horizontal red-run-start parity (left scanline) | — | 27575 | 1301 | — | 86/200 | FAIL: solid top/bottom EDGE rows leak parity rightward |
| 3 | Hm=3-consec-red horizontal-wall, parity of Hm ABOVE (task204 ray-cast) | A | 15765 | 666 | 15.293 | 200/200 | works |
| 4 | cast red->fp16 once, Conv in fp16 (kill 2x fp32 2500B planes) | A | 14515 | 667 | 15.372 | 300/300 | adopted-as-best |

## Best achieved
15.372 @ mem 14515 params 667 — adopted? N (per instructions). Beats prior 14.285? Y (+1.09). Fresh 300/300.

## Irreducible-floor analysis
Dominant: the fp32 red slice red_f ([1,1,25,25]=2500B, Slice preserves input fp32 — the irreducible
entry plane) plus ~4 fp16 full planes (red, c1h(Conv), Hm, cnt(=Tl@Hm), par(=cnt mod 2) @ 1250B each).
The MatMul ray-cast (cnt) + Mod (par) are intrinsic to the column parity; W=25 (size<=25) is the active
canvas (can't crop tighter — boxes can sit anywhere). This sits just above task204's ~15.2 floor (its
W=20 makes its planes 36% smaller).

## OPEN ANGLES (re-attack backlog)
- Fold notred into greenb without a separate Not plane (~625B, ~+0.04).
- The two ReduceMax in-grid profiles are tiny; gridb is a 625-bool full plane only for the off-grid
  sentinel — could be folded into the Pad sentinel if off-grid cols/rows were sliced exactly, but size
  is data-dependent (10/15/20/25) so a fixed W=25 slice can't auto-drop the off-grid background.
- A data-dependent W-slice (Gather by size scalar) would shrink every plane ~size^2/625 but trips the
  symbolic-dim "could not be measured" trap — not worth it.

## INSIGHT (transferable)
⭐ "Fill the hollow interior of each axis-aligned red box" is the SAME structure as task204 and is NOT a
flood-fill wall: the WINNING discriminator is the column ray-cast — Hm = a HORIZONTAL-WALL cell (red with
red BOTH left & right, via a 1x3 sum-Conv + bias -2 + Relu, so it fires ONLY on top/bottom edges, never
on 1-wide vertical side walls or isolated/gap reds), then interior = PARITY of Hm cells strictly ABOVE
(tril MatMul + Mod-2). Two WRONG approaches that look plausible but fail: (a) 4-direction any-red
prefix/suffix-OR enclosure merges separated boxes when a gap cell happens to have a box on each of its 4
sides; (b) horizontal red-run-start left-scanline parity leaks rightward on solid edge rows (one
unmatched crossing). The Hm restriction to genuine horizontal walls is exactly what makes the crossing
count local and edge-row-safe.
