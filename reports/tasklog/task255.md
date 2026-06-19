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

## GAP-CLOSER RE-ATTACK (2026-06-19) — deployed net = ext:kojimar7113 (341 nodes, mem 58710, scores ~95% fresh)
Re-attacked under the gap-closer framing (deployed net scores 190/200, fixing→100% would adopt). VERDICT: INFEASIBLE confirmed by a stronger, decisive test than the prior round.

**New characterization of the deployed net's 5% failure mode** (measured over 2000 fresh):
- 3.3–3.5% failure rate. Of 69 failures: **19 edge-only, 50 INTERIOR-involving**. So it is NOT a fixable
  off-grid/edge bug — most failures are genuine interior rectangle-reconstruction mistakes (the FP are
  contiguous off-grid-vein extent errors, ±1–2 rows on box boundaries near grid edges, plus interior leaks).

**Structural facts newly established:**
- Green region is ALWAYS a single 8-connected component (300/300) — all veins attach to the artery.
- erode3 set decomposes into: avg **1.00 "mixed" component** (green LEAKED to abutting noise via 8-conn
  bridge) + avg **1.24 pure-noise components** (separable). Reconstruction kills the pure-noise blobs but
  the 1 leak/inst is unfixable by any bounded local op.
- NO input→output collisions in 20000 instances ⇒ output IS a deterministic function of input (NOT a pure
  information wall); the wall is that the function is statistically non-identifiable by the box-defining feature.

**DECISIVE TEST — global maximal-solid-rectangle ORACLE (full numpy, NO op-set constraints):**
mark the 1-eroded interior of EVERY all-black rectangle of box size (w≥6, h≥3, off-grid padded). Result:
**FN=0, avg 16 FP/instance, 0/80 exact.** Even an unconstrained global solver with perfect rectangle-fitting
fails 100% — because in 50%-black noise, box-sized all-black rectangles occur ~16×/instance BY CHANCE. So
the master-key (bounded-iteration unrolling / flood / label-prop) CANNOT help: the blocker is not
Loop/Scan-expressibility, it is that "solid black rect with 1px outline" does not statistically separate
true boxes from chance noise rectangles.

**Angles tried this round (6 distinct, all fail):** (1) erode3 FN=0/19FP; (2) run-length/max-run thresholding
— noise runs ≥23 exceed box widths; (3) component bbox-fill solid-rect filter — leak destroys rect property,
277 FN; (4) seed-erode + geodesic reconstruction in erode-mask — 100% leak; (5) **global max-solid-rect oracle
— 16FP, 0% (decisive)**; (6) artery-anchored reconstruction — leak bridges flood artery→noise, 17FP, 0%.

**Why the deployed kojimar net still hits 95%:** it does artery-anchored disambiguation with structured
outline/profile heuristics (ArgMax/ConvTranspose/Mod over row-col structure) that exploit the generator's
*placement* regularities — but it is itself inexact (50/69 interior failures), so it is fundamentally a
heuristic, not an exact rule. An EXACT generalizing net would have to recover the same placement priors the
generator used, which are not present in the input pixels (only their stochastic black realization is).

**FINAL: INFEASIBLE** — no closed-form / bounded-unrolling / candidate-enumeration net can beat 95% here.
The connectivity-wall verdict stands and is now backed by the global-oracle 0%-exact proof.
