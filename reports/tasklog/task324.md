# task324 — d07ae81c

**Rule:** Background is a two-colour stripe pattern (base bg0=bgcolors[0] fills the
grid; stripe bg1=bgcolors[1] fills full horizontal stripes `brows` and full vertical
stripes `bcols`). A few seed "dots" (2-3 cells, exactly two distinct dot colours
colors[0]/colors[1]) sit on the grid. The OUTPUT draws BOTH 45° diagonals through
every seed (main r-c=const AND anti r+c=const); each diagonal cell over the base
background becomes colors[0], over a stripe becomes colors[1]. Seed cells are fixed
points (their input value already equals the correct recolour). So
`out = colors[0] if (on_diag & input==bg0) else colors[1] if (on_diag & input==bg1) else input`.
**Current:** 14.5319 pts, mem 33639, params 1538 (public CumSum/one-hot net)
**Target tier:** B (label-map + Equal; the per-cell colour-index copy plane is required).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | diag spread via two 39×39 line Convs, fp32 planes, Mul-collapse colf | B | 68208 | 3087 | 13.83 | — | [1,10,30,30] Mul intermediate (36000B) |
| 2 | colf via 1×1 Conv (kill [1,10,30,30]); fp32 planes | B | 32208 | 3107 | 14.53 | — | tied; 30×30 planes + 39×39 kernels dominate |
| 3 | fp16/bool planes; in-grid rect from H/W (drop tot30); Where col-split | B | 20616 | 3136 | 14.93 | 200/200 | beats by +0.39 |
| 4 | 2-pass reach-10 (21×21) diag Convs instead of one 39×39 | B | 22216 | 976 | **14.948** | 300/300 | params 3136→976, +2 planes; net +0.02 |

## Best achieved
**14.9484** @ mem 22216 params 976 — adopted? N (do not adopt per instructions).
Beats prior 14.5319 by **+0.42**. GENERALIZES: yes (200/200 + 300/300 isolated fresh).

## Irreducible-floor analysis
Dominant intermediates: `colf30` 3600B (30×30 fp32 colour-index plane from the 1×1
Conv — the real 3600B floor; Conv must emit 30×30 since input is 30×30) + its 20×20
fp32 slice `g32` 1600B. These are required because non-diagonal cells COPY the input
colour, so a full per-cell colour-index plane is unavoidable (escape (1) spatial-copy
and (2) separable-output don't apply — the recolour mask is a non-separable diagonal
union). Everything else is fp16/bool on the 20×20 active canvas (~10×800B + ~22×400B).
Diag-spread params reduced to 976 by the two-pass small-kernel trick.

## Key exact detectors (verified 20000/20000 + 6000/6000 reconstruction in numpy)
- **bg0 (base)** = the unique background colour (count ≥5; dot colours ≤3 each) present
  in the top-left 2×2 block, which is NEVER on a stripe (first stripe index ≥2).
  bg1 = the other background colour. (Corner [0,0] alone fails ~1% because a dot can
  land there; the 2×2-block rule is exact because bg1 never appears in the 2×2 region.)
- **seeds** = in-grid cells that are neither bg0 nor bg1.
- **stripe row/col** = in-grid row/col with ZERO bg0 cells.
- **colors[1]** = max seed value on a stripe; **colors[0]** = max value of base seeds.
- **on_diag** = seed plane spread along each diagonal by a full-reach diagonal-line
  Conv (SAME pad) thresholded >0 — no flood-fill, NonZero, or scan.

## OPEN ANGLES (re-attack backlog)
- The 3600B `colf30` plane: the only escape is a smaller active canvas, but the colour
  Conv is forced to emit 30×30 (input is 30×30) and slicing the 10-ch input first costs
  14400B. If a future trick lets Conv emit a cropped colour plane this drops to ~15.5.
- Diagonal spread is genuinely memory-optimal as a Conv (few named tensors); a
  shift/doubling-OR formulation EXPLODES memory (~40 named intermediates) because
  calculate_memory sums EVERY intermediate, not peak — confirmed not worth trying.

## INSIGHT (transferable)
⭐ "Draw both 45° diagonals through scattered seeds" is closed-form tier-B, NOT a
connectivity/flood wall: spread the seed plane along main and anti diagonals with two
full-reach diagonal-LINE Convs + >0 (reuse the task037/task019 idiom on the full grid,
not just bounded segments). ⭐ When a "background vs stripe" 2-colour split must be
ordered (which colour is base), the top-left K×K corner block is guaranteed non-stripe
(stripe indices start ≥2) so the unique high-count colour appearing there is the base —
exact where corner-cell-alone fails (~1%) because a dot can occupy the corner.
⭐ A two-pass small diagonal Conv (reach R each, 2R≥needed) cuts kernel params ~5× vs
one full-reach kernel at the cost of only 2 extra small planes — net win when params
dominate. ⭐ `Where(mask, g, 0)` for a masked-max replaces Cast(mask)+Mul(g) (saves one
fp16 plane per masked reduction).
