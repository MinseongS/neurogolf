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
| 5 | **single banded Conv** (in-grid/red/olive in one plane) | B | **9684** | **138** | **15.81** | 200/200 | ADOPT-READY |

## Best achieved
**15.81 @ mem 9684, params 138** — beats prior 14.7 by **+1.11**. evaluate ok
(265/265 stored), ISOLATED fresh 200/200. adopted? N (build-only per instructions).

## Irreducible-floor analysis
Dominant intermediate = the single 30x30 fp32 Conv output `ocount30` (3600 B): one
must read the per-cell colour/neighbour info from the fp32 input at full resolution
before cropping. Slice (1296) + Pad-back (900 uint8) are the next costs. Everything
else is 18x18 (324–648 B). The 3600 fp32 conv is irreducible (fp32 input -> fp32
conv; can't feed fp16 conv without a 18000 B input cast).

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
