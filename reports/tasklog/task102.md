# task102 — 44d8ac46

**Rule:** 12x12 input with ONLY black(0) + gray(5): axis-aligned gray 1px-outline
rectangles on black. A box's black interior is recoloured RED(2) in the output IFF
the gray outline is a SQUARE (side s in {3,4,5,6}) with an all-black (gray-free)
interior; every other cell (gray rings, black bg, off-grid) copies the input.
(Verified fresh 0-mismatch / 2000+: red interior <=> strictly-interior cell of an
empty square gray ring.)
**Current (was):** 15.673 pts, ext:biohack_new import, mem 10715, params 524
**Target tier:** B/detection — local square-ring detection + interior stamp; not
a closed-form scalar (positions/sizes are data-dependent), but well above the
public floor once the active region (12x12) and gray-only signal are exploited.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | public import (6 sizes, 2ch conv, 10ch concat) | det | 10715 | 524 | 15.67 | 200/200 | baseline |
| 1 | 4 sizes only + Where-route output | det | 9970 | 246 | 15.77 | 200/200 | drop dead sizes 7/8 |
| 2 | uint8 carrier Concat + Pad | det | 9286 | 380 | 15.82 | 200/200 | zeros init costs params |
| 3 | gray-ONLY detection (perim+1/interior-10) + Where | det | 7198 | 153 | 16.10 | 200/200 | black chan unneeded |
| 4 | gray-only + carrier route | det | 7702 | 286 | 16.01 | 200/200 | carrier worse than Where |
| 5 | stamp value=2 -> colour-index plane -> ONE Equal | det | 6730 | 152 | **16.16** | 500/500 | ADOPT |

## Best achieved
16.163 @ mem 6730 params 152 — beats prior 15.673 by **+0.49** (Y). fresh 500/500.

## Irreducible-floor analysis
Dominant intermediates: (a) one 30x30 uint8 colour-index plane = 900B — required
because the output is genuinely 30x30 and the 10-channel expansion is routed FREE
via Equal(lab, arange); (b) the fp32 ch5 slice = 576B — Slice inherits the fp32
input dtype, irreducible entry cost for a single 12x12 channel; (c) four
square-ring sizes x ~3 fp16 12x12 planes (score/hit-cast/stamp ~288B each) — the
ring side s is data-dependent over {3,4,5,6} so each needs its own kxk perimeter
Conv + (s-2)x(s-2) ConvTranspose interior stamp; they cannot share a kernel.

## OPEN ANGLES (re-attack backlog)
- Merge the 4 score Convs into ONE Conv[4,1,6,6] (uniform 6x6, smaller rings
  anchored top-left) — same memory (1152) but fewer ops; no score gain expected.
- Single grouped ConvTranspose (groups=4) for the 4 stamps to drop the 3-op Add
  chain (~864B) — risky alignment, marginal.
- True closed-form interior test (H-run == V-run gated by gray walls) FAILS
  (927/1500 mismatch) due to nested mini-squares + junk boxes faking equal runs;
  the ring Conv is the robust discriminator, so this wall stands.

## INSIGHT (transferable)
⭐ For 2-colour inputs, detection often needs only ONE colour channel (the other
is its complement) — halves the fp32 entry-slice cost and lets the detection Conv
run on 1 channel. ⭐ Make the interior ConvTranspose STAMP carry the output
colour-label value directly (kernel value = 2.0) so the summed stamps ARE the
red-label plane — add 5*gray and you have a single uint8 colour-index plane,
expanded FREE via Equal(lab, arange); this beats both a separate fill-mask+Where
(needs 2x 30x30 planes) and a 10-channel uint8 carrier Concat. ⭐ Pad the
colour-index with a SENTINEL (99, not 0) so off-grid cells map to all-channels-
false under Equal, while in-grid black-bg (lab=0) correctly lights ch0.
