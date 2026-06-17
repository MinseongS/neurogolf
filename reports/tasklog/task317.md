# task317 — ce22a75a

**Rule:** size=3 fixed, so the grid is ALWAYS 9x9 = a 3x3 array of 3x3 blocks. A
subset of blocks are marked by a single GRAY(5) pixel at the block CENTRE
(3r+1, 3c+1). The output fills the ENTIRE 3x3 block of every marked block SOLID
with BLUE(1); unmarked/background cells stay 0.
**Current:** stored ~18.2 pts but FRESH-RATE 0.00 (does NOT generalize, real LB ~0).
**Target tier:** A (closed-form block-dilate; not S because output colour-index
needs a 30x30 uint8 carrier to land the 10-way one-hot expansion).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | gray-slice 30x30 + 3x3 ones Conv(pad1) dilate + Where->L30 + Equal | A | 9000 | 924 | 15.80 | 200/200 | works |
| 2 | crop gray slice to 9x9, Conv on 9x9, Where->L9, Pad(99-sentinel) to 30x30, Equal | A | 1710 | 40 | 17.53 | 200/200 | best |

## Best achieved
17.53 @ mem 1710 params 40 — adopted? leave as candidate. Beats prior real(0)? Y
(huge: +17.53 vs the non-generalizing 0). Generalizes (fresh 200/200).

## Irreducible-floor analysis
Dominant intermediate is the Pad'd 30x30 uint8 colour-index label `L` (900B) —
needed so `Equal(L, arange10)` produces the [1,10,30,30] BOOL output. The 9x9
working planes (gray slice, conv resp, blue mask, L9) are all <=324B. uint8 is
the minimal carrier dtype; bool can't be Pad'd (ORT Pad rejects bool), fp16
doubles it. The blue-block mask is NOT row/col separable (data-dependent block
subset), so it cannot be routed into the free output via separable broadcast.

## OPEN ANGLES
- The 900B Pad'd carrier is the floor for the colour-index route. Only escape is
  if some clever separable factorization of the block mask existed — it doesn't
  (arbitrary subset of 9 blocks). 17.53 is effectively at the achievable ceiling.

## INSIGHT (transferable)
A gray seed sitting at the EXACT centre of each disjoint 3x3 block makes
"fill the marked block" a single 3x3 all-ones Conv(pad=1) dilation — no
flood-fill, no NonZero. Cropping the working plane to the generator's FIXED
active region (9x9) BEFORE the conv, and only Padding the tiny uint8 LABEL out
to 30x30 at the very end, dropped mem 9000->1710 (15.80->17.53). ⭐ Crop-early /
pad-the-label-late is the standard recipe for fixed-small-grid upscale tasks.
