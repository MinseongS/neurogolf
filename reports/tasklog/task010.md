# task010 — 08ed6ac7

**Rule:** 9x9 grid. Four gray (colour 5) bars hang from the bottom at columns
{1,3,5,7} (column = order[bar]*2+1, order a permutation of 0..3), each with a
distinct height in 1..9. The OUTPUT keeps the identical bar shapes but recolours
each bar by its height RANK: tallest -> colour 1, 2nd -> 2, ..., shortest -> 4;
background stays 0.
**Current:** 16.48 pts (public label-map), beaten.
**Target tier:** B (count-parametric label map) — output colour couples columns
(rank needs a global pairwise comparison) and the bar shape couples r&c
(`r+h[c]>=9`), so it is neither single-Conv (S) nor row⊗col separable (A). But
the entire output is reconstructible from a [9] column-height vector, so the
label map is the only full plane.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | ReduceSum(input,[2]) heights + count-rank + count-parametric fill, fp32 | B | 3885 | 50 | 16.72 | 200/200 | marginal (+0.24) |
| 2 | replace row-sum with one no-pad Conv (W[1,10,30,1], ch5) -> 120B | B | 2685 | 349 | 16.98 | 200/200 | +0.50 |
| 3 | + cast all [9,9] working planes to fp16 (ints<=9 exact) | B | 2109 | 349 | 17.19 | 200/200 | ADOPT (+0.71) |

## Best achieved
17.19 @ mem 2109 params 349 — beats prior 16.48 by +0.71. Fresh 200/200.

## Irreducible-floor analysis
Dominant intermediate is the 30x30 uint8 label map L (900B) materialised by the
Pad before the final `Equal(L, arange)`. The output is 30x30 so a per-cell
colour-index carrier is required; the active region is only 9x9 but Pad to 30x30
to off-grid sentinel 10 is unavoidable for the BOOL output (Pad rejects bool, so
the small-canvas carrier cannot itself be the bool output). Second cost is the
300-element Conv weight (params, cheap in ln). No fp16 helps here: dtype tricks
don't shrink the 30x30 carrier and ORT upcasts; uint8 900B is already the floor.

## OPEN ANGLES (re-attack backlog)
- Trim the 300-param Conv: any single-op channel-5 row-sum with fewer elements
  (e.g. exploiting that only rows 0..8 are ever nonzero) would save ~250 params
  (~+0.1) but every alternative (Slice/Gather channel 5) materialises a 3600B
  full plane — net loss. Conv is the cheapest known.
- A separable route is blocked: fill `r+h[c]>=9` couples r&c, so any separable
  formulation still needs a 30x30 fill plane = the same 900B.

## INSIGHT (transferable)
"Bottom-anchored solid bars recoloured by height-RANK" is closed-form tier-B with
ZERO occupancy plane: per-column height = a no-pad Conv row-sum on channel-k
([1,10,30,1] kernel -> [1,1,1,W], 120B, dodging the [1,10,1,W] ReduceSum
intermediate); rank = pairwise-Greater over the tiny height vector (count
function, no sort/argmax); and the bar SHAPE rebuilds from height alone via
`r+h[c]>=9` (no per-cell gray mask read). Cast the height vector to fp16 once and
run all the tiny [K,K] rank/fill planes in fp16 (ints<=9 exact) to halve them.
