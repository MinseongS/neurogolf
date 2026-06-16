# task255 — a64e4611

**Rule:** Input = dense random noise (one non-green color at ~50% density over a black=0 background)
overwritten by several solid-black axis-aligned rectangles ("artery" + "veins"). Each rectangle is
drawn solid black, then its 1-cell-eroded INTERIOR is painted green; in the INPUT the green is converted
back to black (so the whole box reads as solid black), in the OUTPUT the interior reads green(3).
Boxes routinely extend OFF-GRID (row=-1, col=-1, wide=size, tall=size+2) and the grid may be transposed
and/or flipped. So the task: recolor to green every black cell that is the interior of a maximal SOLID
black rectangle (the boxes), leaving the 1-cell outline and all noise black. Green always touches a grid
edge (boxes run off-grid), so off-grid must be treated as black.
**Current:** stored 13.95, fresh 0/60 (OVERFIT — contributes ~0 to real LB).
**Target tier:** detection / BAIL — connectivity + maximal-solid-rectangle vs. dense abutting noise.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 3×3 erosion of black (off-grid=black) | — | — | — | — | — | FN=0 but ~17 FP/inst (noise-black abutting outlines + grid-edge noise) |
| 2 | erosion + 4/8-run-length coverage (K=5..24) | — | — | — | — | — | FP persists near boxes; long K adds FN (short veins) |
| 3 | erode3 + has-eroded-neighbor (4/8 conn) | — | — | — | — | — | FN=0, FP unchanged (chance black clusters ≥4×4 survive) |
| 4 | "no colored cell within Chebyshev r" r=1 | — | — | — | — | — | identical to erode3 (FN=0, FP~19); r=2 → massive FN |
| 5 | morphological reconstruction seed=5×5 erode, mask=erode3, 35 iters | — | — | — | — | — | FN=0 but FP UNCHANGED — leak: noise FP cells are 8-connected to the box interior through the leaky mask |

## Best achieved
No generalizing encoding. erode3 (the unique FN=0 detector) carries ~17 FP/instance that cannot be
removed by any bounded local op or reconstruction. NOT adopted.

## Irreducible-floor analysis
The only FN=0 detector is 3×3 erosion of the black set with off-grid treated as black (the box outline is
exactly 1 cell, so the interior is precisely the black cells with no colored 8-neighbor). Because the noise
is ~50% BLACK and is drawn UP TO the box outline, the box's solid outline-black + adjacent noise-black form
chance 3×3 all-black patches just OUTSIDE the true outline → ~17 false-positive eroded cells per instance,
and they sit at the grid edges too (off-grid=black is mandatory since boxes run off-grid). These FP cells
are 8-CONNECTED to the true interior, so morphological reconstruction (bounded geodesic dilation) leaks
straight into them rather than removing them. Cleanly separating the box's maximal solid axis-aligned
rectangle from the abutting dense noise requires global connected-component / maximal-rectangle analysis on
a VARIABLE-size region over the full 30×30 canvas — a Loop/Scan/flood-fill class that is banned and pins at
the ~14.2 fill-floor for everyone (per BUILD_PROMPT WALL note, task198 analogue).

## OPEN ANGLES (mostly exhausted)
- Exact rectangle fit per box (detect 4 clean corners of a solid-black rect) — corners are locally
  indistinguishable because dense noise-black abuts the outline; no local corner test is exact.
- If a future op budget allowed bounded connected-component labeling, reconstruct from an artery-only
  thick seed with a NON-leaky mask — but no leak-free local interior mask exists here.

## INSIGHT (transferable)
"Erode the black set / no-colored-within-1" is the exact interior detector for solid-rect-with-1px-outline
tasks AND has FN=0 — but when the background noise shares the box color's complement (here noise is colored
on a BLACK bg, so the box IS black and noise is ~50% black too), the noise abuts and merges with the box
outline into chance-eroded blobs that are CONNECTED to the true region. That connection defeats
morphological reconstruction. ⭐ Discriminator: solid-rect-interior tasks are closed-form ONLY when the
shapes sit on a noise-free / differently-colored background; dense same-as-box-color noise abutting the
outline makes it a true connectivity WALL.
