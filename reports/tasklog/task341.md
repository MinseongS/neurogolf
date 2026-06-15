# task341 — d6ad076f

**Rule:** Two solid colour blocks. A SHORT block (colour c0, thickness 2-4, length 4-6) and a LONG block / bridge (colour c1, length 6-9); the short block's column span is strictly NESTED inside the long block's span. A vertical CYAN(8) bar fills the gap *rows* between the two blocks, spanning the short block's columns shrunk by one cell on each side (its interior). `apply_gravity` then transposes/flips the whole figure into one of 4 cardinal orientations, so the two blocks may be stacked vertically or horizontally, in either order. INPUT = the two blocks; OUTPUT = input + the cyan bridge (cyan only overwrites background).
**Current (public):** 15.37 pts, ext:biohack_new
**Target tier:** detection (gap-fill between two blocks, 4 orientations) — reformulated to a clean reduction-based label-map (B-ish). Not separable globally (gap axis varies per instance), but per-instance it is rowmask⊗colmask, recoverable from 1-D occupancy + band reductions.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | reduction label-map: detect gap axis via occupancy hole; cyan rect = gap span (gap axis) × nested-intersection interior (perp axis); L=V+8·cyan; final Equal | B/detection | 10109 | 62 | 15.77 | 200/200 | WIN (+0.40) |

## Best achieved
15.77 @ mem 10109 params 62 — adopted? N (agent writes file only). Beats prior 15.37? **Y (+0.40)**.

## Irreducible-floor analysis
Dominant intermediate = the colour-index Conv output **V32 [1,1,30,30] fp32 = 3600 B** (the Conv is forced to emit full 30×30 from the fixed 30×30 one-hot input). The 10×10 working-canvas masks (M, band products Mabove/Mbelow/Mleft/Mright, cyan, Lcol) are ~400 B each (≈4–5 kB total); L padded back to [1,1,30,30] uint8 = 900 B; output [1,10,30,30] is FREE. So the floor here is "one 30×30 colour-index Conv + a label map". Cannot drop below ~3600 without abandoning the colour-index Conv (input is fp32 so an fp16 Conv would need an 18 kB fp16 cast of the whole one-hot input — strictly worse).

## OPEN ANGLES (re-attack backlog)
- Convert the 10×10 working-canvas float planes (M, the 4 band products, cyan, Lcol) to fp16/bool to shave ~1.5–2 kB → ~+0.15–0.2 pts. Care needed: all dtypes must match across Where/Mul/ReduceMax (ORT ReduceMax rejects bool; Where cond must be bool). Marginal; deferred.
- Eliminate the 4 band-product [1,1,10,10] planes (Mabove/Mbelow/Mleft/Mright) by computing band column/row extents without the full Mul (e.g. masked-min/max via index arithmetic on `colocc` restricted to the band) — would remove ~1.6 kB.
- Per-cell single-Conv (Tier S) is blocked: the cyan width is the *nested-interior* of one block, which is non-local (needs both blocks' spans + the gap), so no local hyperplane recovers it.

## INSIGHT (transferable)
`apply_gravity` in ARC-GEN is NOT physical gravity — it is a transpose/reflection applied IDENTICALLY to input and output, so the input→output rule is orientation-equivariant; handle it by computing BOTH axis branches and selecting via "which axis has the occupancy hole". ⭐ For two-block gap-fill tasks: the gap axis is the one whose 1-D occupancy has an internal empty run ((extent_len) > (#occupied lines)); the perpendicular cyan span is the nested intersection of the two flanking bands' extents (max-of-mins, min-of-maxes), shrunk by 1 — recoverable purely from ReduceMax/ReduceMin/Where on band-masked occupancy, no per-channel block identification needed.
