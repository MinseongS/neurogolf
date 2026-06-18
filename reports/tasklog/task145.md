# task145 — 6455b5f5

**Rule:** A grid (<=20x20, top-left anchored on the 30x30 canvas) is recursively
guillotine-bisected into axis-aligned rectangular leaf regions; every cut line is painted
red(2). The INPUT shows ONLY the red lines (all other in-grid cells = bg 0; off-grid cells
have all channels 0). The OUTPUT keeps the red lines, fills the leaf region(s) of MINIMUM
area cyan(8) and the region(s) of MAXIMUM area blue(1), everything else 0. Because every
region is a SOLID RECTANGLE bounded by red/border, area = (horizontal free-run length) *
(vertical free-run length) per cell — NO flood-fill / labeling / global-argmax-loop needed.
Run length = (nearest wall right) - (nearest wall left) - 1; a "wall" is red, off-grid, or
the grid border. Then amin/amax are two scalar reductions and selection is two Equal masks.

**Current:** 13.90 pts (blank-note "confirmed-infeasible", ~66KB Gather/Sum net)
**Target tier:** B/detection — connectivity+global-argmax form, but the rectangle structure
collapses it to closed-form per-cell area + two reductions (well above the ~13.4 floor).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | fp32 cummax wall-dist, Where→[1,10,30,30] route | det | 209464 | 317 | 12.75 | - | correct but huge |
| 2 | +Conv colf, single colorindex→Equal output, crop W=20 | det | 79944 | 305 | 13.71 | - | better |
| 3 | uint8 cummax/min via Where(Greater/Less), uint8 tail | det | 56424 | 306 | 14.05 | - | |
| 4 | slice ch0/ch2 (input only has colours 0,2) — drop full-30 Conv/ReduceSum | det | 49624 | 304 | 14.18 | - | |
| 5 | drop area_for_max (walls=area1<max), drop blue free-And | det | 48424 | 303 | 14.21 | 200/200 | ADOPT-CANDIDATE |

## Best achieved
14.206 @ mem 48424 params 303 — beats prior 13.90 by +0.306. ISOLATED fresh 200/200.

## Irreducible-floor analysis
Dominant: the four uint8 log-doubling cumulative-max/min chains (nearest-wall positions)
= ~48 uint8 [1,1,20,20] (400B) planes + ~30 bool comparators (400B). 4 chains x 5 doubling
steps (W=20 -> ceil(log2 20)=5) x (slice+pad+Greater/Less+Where). This is the genuine 2-D
segmented-scan cost; uint8 (1B) is the dtype floor (ORT has no uint8 Max/Min, so
Where(Greater,...) is used; uint8 Sub/Mul are unsupported so the rw*rh area is fp16).

## OPEN ANGLES (re-attack backlog)
- Replace Slice+Pad shift with a single Gather over a once-augmented sentinel array to drop
  the per-step `sliced` plane (~5KB). I prototyped this but the per-step re-augment Concat
  ate the savings; a cleaner version that Gathers from a fixed augmented `cur` could net
  ~3-4KB (-> ~14.3).
- Pack the two horizontal chains (LM cummax / RM cummin) onto the batch axis so one set of
  shifts serves both (different ops though — max vs min — so only the shift is shared).
- Fold the two `-1` subtractions and 4 casts into fewer fp16 planes (~1.6KB).

## INSIGHT (transferable)
⭐ "Segment into components + global argmax + variable crop" is NOT the ~13.4 floor when the
components are RECTANGLES from a guillotine partition: per-cell component AREA = (row free-run)
x (col free-run), turning the global argmin/argmax into TWO scalar ReduceMin/Max + two Equal
masks — no flood-fill, no labeling, no NonZero. Nearest-wall distance via uint8 log-doubling
cumulative max/min: ORT has no uint8 Max/Min but `Where(Greater(a,b),a,b)` / `Where(Less,...)`
work and run at 400B (W=20) vs 800B fp16. Bake the grid BORDER into the scan as the cummax
init (0 => pos -1) and cummin pad-fill (W+1 => pos W). Wall cells get area 1, so amax =
ReduceMax(area) needs no free-mask (generator guarantees a region of area>1), but amin still
must mask walls. Big mem wins came from: input only has colours {0,2} -> slice ch0/ch2 on the
WxW active region instead of a full-30 Conv/ReduceSum; and a single uint8 colour-index plane
-> Pad(uint8) -> Equal(arange) BOOL output (no [1,10,30,30] intermediate ever materialised).
