# task379 — ecdecbb3

**Rule:** 1-2 full-width horizontal CYAN(8) lines; each column has ≤1 RED(2) dot.
Each dot shoots a ray (painting red along its column) toward the NEAREST line
above AND the NEAREST line below it (a line strictly between blocks the farther
one). Where a ray reaches a line at (L,c): paint the inclusive column segment
[dot..L] red, stamp a 3×3 CYAN box centred at (L,c), then set the box centre RED.
Paint priority: cyan-lines < ray-red < box-cyan < centre-red. `xpose` flips the
whole figure (lines become vertical).
**Current (public):** 13.82 pts, mem ~70k.
**Target tier:** B (closed-form masks; per-cell reconstruction routed into the
free BOOL output) — full Tier S impossible (output colours are fixed but the
geometry is a per-column ray + stamp, not a pure copy/permutation of input cells).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | dual-branch fp32 closed-form | B | 140k | 53 | 13.15 | — | correct, too big |
| 2 | single branch (select inputs by xpose flag) | B | 92.8k | 53 | 13.56 | — | correct |
| 3 | + fp16 whole pipeline | B | 67.2k | 54 | 13.88 | — | +0.06 |
| 4 | + CROP-TO-ACTIVE 20×20 + pad-back | B | 41.7k | 61 | 14.36 | — | +0.54 |
| 5 | + uint8 nested-Where label (no fp32 L planes) | B | 38.5k | 64 | 14.44 | — | |
| 6 | + combined-channel Slice + bool in-grid | B | 29.3k | 68 | 14.71 | — | |
| 7 | + ArgMax dotrow (no rred plane) | B | 28.7k | 68 | 14.73 | — | |
| 8 | + scalar line-rows (no dval/uval full planes) | B | 26.5k | 68 | 14.81 | 200/200 | **best** |

## Best achieved
14.81 @ mem 26511 params 68 — adopted? N (build only). Beats prior 13.82? Y (+0.99).

## Irreducible-floor analysis
Dominant survivors (all 1600 B = fp32 20×20): 3 entry colour slices (red/cyan/bg,
free input → slice is counted), the fp16 casts of the red/cyan masks needed for
the float ReduceSum (cyan-per-row line detect) / ReduceMax (red-per-col presence),
and the box dilation (bool→fp16 cast + 3×3 MaxPool — MaxPool needs float). The
orientation transpose/select are now uint8 (400 B). ReduceSum/ReduceMax reject
uint8/bool so a colour count needs ≥1 fp16 full plane per axis — that's the
remaining structural cost, plus the unavoidable 3 entry slices.

## OPEN ANGLES
- Merge the box dilation into the colour Conv route to drop the bcF cast + boxF
  MaxPool pair (~3.2k) — would need a float-free 3×3 dilation.
- Compute red-per-col presence and cyan-per-row count from ONE shared fp16 plane
  instead of two casts (redf + cyf) — saves ~1.6k if a single reduction suffices.

## INSIGHT (transferable)
⭐ A "ray + iterative stop-on-cyan + 3×3 stamp" generator that LOOKS like a flood
is fully CLOSED-FORM: the stop-on-cyan reduces to "reach the NEAREST line in each
direction" → per-column scalar `Ldown=min line>dot`, `Lup=max line<dot`. With
≤2 lines these come from a tiny `Where(lineB, rowidx, ±BIG)` ReduceMin/Max over
the **[1,1,WK,1] line vector** (NO WK×WK candidate plane) plus a 2-level Where
chain on the [1,1,1,WK] dot-row vector. Ray = row-vs-{dot,L} range masks; box =
3×3 MaxPool of the line-intersection mask; compose by nested **uint8** Where
(priority order) → no fp32 colour-value plane. xpose handled by uint8
transpose+select of the masks once (uint8 Where works; bool Where does NOT).
⭐ ArgMax works on uint8 (ReduceSum/ReduceMax do not) — use it for "row index of
the unique marker per column" to avoid a full coord×mask product plane.
