# task278 — b27ca6d3

**Rule:** Restore green "olives" from their two red centers. The generator draws an
olive as a pair of 4-adjacent cells, a 3x3 green block around EACH, then sets the
two centers to red; in the INPUT every green cell is erased to black, so an olive
appears only as two 4-adjacent red pixels. "static" red pixels are isolated
(remove_neighbors) so they have NO red 4-neighbor. Transform: olive_red = a red
cell with >=1 red 4-neighbor; paint green on the 3x3 Chebyshev-1 block around each
olive_red (centers stay red); everything else passes through. INPUT colors are only
{black, red}; OUTPUT colors are {black, red, green}. Grid size 15..18, anchored
top-left. Verified transform 500/500 against the generator.
**Current (prior):** 14.7 pts (mislabeled-infeasible; re-triage = FEASIBLE).
**Target tier:** B (label-map + final Equal) — output color is content-dependent
(needs neighbor red-count + a dilation), so not pure Tier-S; but the label is built
from ONE conv plane, landing well below the usual label-map floor.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 2-stage convs + bool masks + 3 float Wheres + Equal, 30x30 | B | 53100 | 135 | 14.12 | — | works, fat |
| 2 | crop to 18x18 (slice 10ch) + Pad + Equal | B | 27900 | 147 | 14.76 | — | slice-10ch dominates |
| 3 | conv-collapse-then-slice (1ch) + 2 convs + 3 Wheres | B | 22140 | 147 | 14.99 | — | two 30x30 convs dominate |
| 4 | drop ~red (red Where overrides) + uint8 label + fp16 dilation | B | 14580 | 148 | 15.40 | — | 2 fp32 convs left |
| 5 | **single banded Conv** (in-grid/red/olive in one plane) | B | 9684 | 138 | 15.81 | 200/200 | superseded |
| 6 | slice input ch{0,2} 18x18 BEFORE conv (kills 3600B full conv) | B | 8676 | 70 | 15.92 | — | +slice saves ~1000B |
| 7 | slice ONLY ch2 + in-grid from 1-D ReduceMax profiles (drop ch0 from conv) | B | 7800 | 73 | 16.03 | 200/200 | win |
| 8 | threshold profiles to bool BEFORE the 18-slice | B | **7716** | **73** | **16.04** | 200/200 | **best** |

## Best achieved
**16.04 @ mem 7716, params 73** — evaluate ok (265/265 stored), ISOLATED fresh 200/200.
adopted? N (build-only). Beats prior custom 15.81 by +0.23; beats the DEPLOYED kojimar
net (15.96) by **+0.08** only → MARGINAL (below the +0.3 bar). 2026-06-19 session note:
re-golfed the prior custom net by (a) slicing the FREE input to JUST the red channel
(ch2) over the 18x18 active block before the conv — this killed the dominant 3600B
full-30x30 conv plane AND the 2nd (ch0) slice channel — and (b) recovering the in-grid
rectangle from two 1-D ReduceMax occupancy profiles (thresholded to bool before slicing)
instead of an in-grid conv band. mem 9684 -> 7716.

## Irreducible-floor analysis (UPDATED 2026-06-19 — floor is now 7716, not 9684)
2-stage 2D-neighborhood task → structural floor ~7716B:
- Detection 3240B (firm): redblk slice [1,1,18,18] f32 1296 (Slice can't emit f16; cast
  adds a plane → no win) + ocount conv f32 1296 (ONE conv: red band 500, olive = red+nbr
  count 501-504 — red AND olive from one plane) + redb 324 + oliveb 324. Conv on the free
  input = 3600B (worse); shift-OR olive detection costs more; uint8/bool Conv is ORT-rejected.
- Dilation 1620B (firm): olivef f16 cast 648 (Conv needs float; bool→f16 cast unavoidable)
  + dil f16 conv 648 + dilb 324. Olive-ness is NONLINEAR (AND of two reds) so it cannot
  fold into the red conv; the 3x3 spread is a 2nd neighborhood op.
- Output 1872B (firm): 4-way uint8 label (3 Wheres 972) + 30x30 uint8 Pad carrier 900 (the
  only route to the FREE bool output; Pad rejects bool). + in-grid 660 + greenb 324.
mem+par must be ≤6234 for +0.3 (16.26); floor is 7789. The +0.3 is structurally infeasible.

## OPEN ANGLES (re-attack backlog)
- Single-Conv mem-0 (Tier-S, task344 class) is impossible: green needs a 5x5
  receptive field AND a non-linear product (red AND adjacent-red), so depth>=2.
- Could the dilation be folded into the banded conv via a second 3x3 conv pass that
  reuses ocount thresholds? Marginal; the 3600 fp32 plane is the real floor.

## INSIGHT (transferable)
⭐ **BANDED SINGLE-CONV: pack multiple boolean predicates into ONE conv plane by
giving each its own disjoint magnitude band, separated by thresholds.** Here a 3x3
conv yields `100*center_bg + 500*center_red + 1*(#red 4-neighbors)`: off-grid 0..4,
in-grid-bg 100..104, static-red 500, olive-red 501..504. Three `Greater` thresholds
(>50, >250, >500.5) recover in-grid / red / olive from the SAME plane — eliminating
the separate in-grid Conv (+slice) entirely (−4896 B). The key constraint: the
center-only tags (in-grid, red) must use weights large enough that the neighbour
count (which leaks onto off-grid cells adjacent to in-grid red) can never cross the
lower threshold. ⭐ Also reusable: (1) when INPUT has a restricted colour set,
"in-grid" = (ch0 OR the used colours), foldable into the same conv; (2) crop to the
generator's max grid (here 18, top-left-anchored) + uint8 label + fp16 dilation +
Pad-with-sentinel(10) keeps the whole label pipeline at 18x18 / uint8.
