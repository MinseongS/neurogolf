# task349 — db93a21d ("death stars")

**Rule:** Input shows only MAROON(9) square centers, one per death-star; each is a
solid 2r×2r block (WIDTH always exactly 2r — `col∈[0,size-2r]`; HEIGHT may be
clipped at the top/bottom edge). Output redraws, per star: the MAROON center
(copied), a GREEN(3) halo = the center block Chebyshev-DILATED by r (→ a 4r×4r
square centered on the block), and a BLUE(1) beam filling the center's columns
from just below the block to the bottom edge. Per-cell priority MAROON>GREEN>BLUE>bg.

**Current:** 14.31 pts, ext:kojimar6275, mem 40500, params 3579
**Target tier:** detection/B — variable-per-object dilation + downward beam over a
full-size (10..30) canvas; no crop (grid IS the data region).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | conical grayscale dilation of a radius field (R=Σ ge(2k)) | B | 79890 | 1868 | 13.69 | — | correct but heavy |
| 2 | green = OR_d dilate_d(run≥2d) via single fused MaxPool/d; inclusive-blue | B | 39600 | 960 | 14.39 | 200/200 | best |

## Best achieved
14.39 @ mem 39600 params 960 — adopted? N. Beats prior 14.31? +0.083 → **MARGINAL** (<+0.3).

## Irreducible-floor analysis
GREEN dominates: it needs per-radius detection AND per-radius dilation.
- `green = OR_{d=1..5} dilate_d( run-length-of-maroon ≥ 2d )`. The cleanest form
  fuses spread+2D-dilate into ONE MaxPool per d:
  `cv_2d = Conv(maroon, ones[1,1,1,2d])` (window sums) and
  `green_d = MaxPool(cv_2d, kernel[2d+1,4d], pads[d,3d-1,d,d]) ≥ 2d`
  (a width-2d window sums to ≤2d, so "≥2d within reach" == "a full 2d run lies
  within Chebyshev d"; horizontal reach window x'∈[x0-3d+1, x0+d]).
- That is **5 conv fields + 5 maxpool fields = 10×900 fp16 = 18000B**, and this is
  the measured floor: radius r∈{1..5} all occur at size 30 (verified by fresh
  histogram), so all 5 levels are mandatory. Element count (not dtype) is the wall
  — fp16 is already the min for Conv/MaxPool (uint8 rejected); stacking d on the
  channel axis keeps the same element count. Conical-dilation and cumulative-
  dilation reformulations BOTH cost MORE (they un-fuse the spread from the dilate).
- Plus one mandatory fp32 entry plane (3600): `colf = Conv(input, w)` with
  w[ch0]=1, w[ch9]=2 → {0:off-grid, 1:in-grid bg, 2:maroon} gives BOTH the maroon
  mask and the in-grid mask from a single 10→1 reduction.
- The remaining ~16k is: m(fp16,1800), colcum blue-OR (fp16,1800), the 5-way bool
  OR for green (9×900=8100), and the uint8 compose chain (~3600).

## OPEN ANGLES (re-attack backlog)
- Halve the GREEN dilation by computing it at 15×15 and upscaling ×2 — green
  regions are ≥4×4 solid squares, but their EDGES are not 2-aligned, so naive
  downsample/upsample loses exactness; would need an edge-correction pass.
- Cheap run-length R via CumSum (run isolated) to drop to a single dilation pass —
  blocked: CumSum gives no segment-reset; cummax/cummin not in opset-11, and the
  centered-window-sum conv is contaminated by horizontally-near same-row blocks
  (verified: K=11..19 all fail).
- Collapse the 9-plane bool green-OR: every alternative (fp16 Max-tree after
  per-d Sub, conv-bias normalisation, ReduceMax over channel-stacked gp) costs ≥
  as much (Max/Sub planes are fp16 1800 vs bool 900). Truly 5-way OR = 9 tensors.

## INSIGHT (transferable)
- ⭐ **Variable-radius square dilation `dilate_r(run≥2r)` in ONE MaxPool per r**:
  `MaxPool(Conv(m, ones[1,2r]), kernel[2r+1, 4r], asym pad)` ≥ 2r fuses the
  "full-run detection (erode+spread)" and the "2D dilate-by-r" into a single pool.
  Beats conical (5 iters of pool−1+max) and cumulative-dilation whenever the per-r
  radius classes are needed — those un-fuse the spread, costing strictly more.
- ⭐ **One fp32 1×1 Conv can carry BOTH a colour mask and the in-grid mask**: when
  the input alphabet is tiny (here {bg=0, maroon=9}), weight ch0→1, ch9→2 so
  colf∈{0:off-grid,1:bg,2:maroon} — one 3600B plane replaces a separate
  ReduceMax-occupancy plane.
- ⭐ **Inclusive prefix-OR beats strict** for a "beam below an object" when a
  higher-priority Where overwrites the object's own cells: drop the shift entirely
  (cells above the object have no object above → stay 0; object cells get
  overwritten by the object colour). Saved the slice+pad shift planes.
- FLOOR: a multi-object task that needs BOTH per-object size detection AND
  per-object-size dilation over a full uncroppable canvas floors at ~2·5·900 fp16
  green planes ≈ 18kB; +0.3 is not reachable without a sub-resolution trick.
