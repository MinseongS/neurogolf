# task230 — ARC-AGI 95990924

**Rule:** Input has several non-overlapping 2x2 GRAY (colour 5) blocks at top-left
(r,c). Output copies the gray blocks and adds four single coloured pixels at the
diagonal-outer corners of each block: out[r-1][c-1]=1, out[r-1][c+2]=2,
out[r+2][c-1]=3, out[r+2][c+2]=4. Satellites always fall on background (in-grid,
no collisions). Fully local closed-form stamp — NOT detection/connectivity.
**Current:** base net stored ~18.2 but FRESH-RATE 0.00 (does not generalize, real ~0).
**Target tier:** A — local separable conv stamp; output colours are FIXED {1..5}
so a label plane + Equal one-hot routes the 10-ch expansion into the FREE output.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | fp32: slice ch5, 2x2 count-conv ==4, 4x4 satellite-stamp conv, L=5g+sat, in-grid gate, int32 Equal | A | 37800 | 37 | 14.46 | 200/200 | works |
| 2 | same but fp16 chain (cast entry plane, Greater>3.5 instead of int Equal) | A | 25200 | 37 | 14.86 | 200/200 | adopted-candidate |

## Best achieved
14.86 @ mem 25200 params 37 — adopted? candidate (do-not-self-adopt). Beats prior
real 0 (non-generalizing base net)? YES by ~+14.86 generalizing.

## Irreducible-floor analysis
Three fp32/int32 full planes dominate (3600 each = 10800):
- g32 — fp32 channel-5 slice (entry plane; the 10->1 reduction must be fp32).
- occ — ReduceSum over channels for the in-grid mask (ReduceSum rejects bool/uint8,
  must be fp32). Needed: off-grid cells have g=0,sat=0 -> L=0 -> would wrongly hit
  output channel 0; the sentinel-10 Where gate suppresses that.
- Li — int32 cast feeding the output Equal (opset-10 Equal rejects float/fp16, only
  int32/int64/bool/uint8). Li itself counts (3600); the `output` it feeds is FREE.
Remaining ~6 fp16 full planes at 1800 each (g, cnt, tl, g5, sat, L0/L).

## OPEN ANGLES (re-attack backlog)
- Eliminate `occ` (3600 fp32): if the in-grid gate could be folded into the output
  one-hot for channel 0 ONLY (channels 1-5 are already auto-zero off-grid because
  L=0 there), a cheaper bool-only construction might drop the full-plane ReduceSum.
- Drop the satellite to a smaller active canvas: generator size is 10 or 15, so the
  active region is <=15x15; slicing planes to 15x15 before the convs would cut every
  full plane ~4x. Blocked only by needing a data-dependent crop (size 10 vs 15);
  worth a separable row/col-occupancy bound attempt for a possible jump toward ~16.
- Fuse g5+sat+L0 to shave 1-2 fp16 planes (~+0.05-0.1, marginal).

## INSIGHT (transferable)
A "place fixed-colour markers at fixed offsets from a detected local shape" task is a
pure two-Conv pipeline: (1) a small all-ones Conv + threshold detects the shape's
anchor cell; (2) ONE weighted Conv whose kernel encodes each marker's COLOUR at its
relative OFFSET stamps all markers as distinct values in a single label plane (no
per-marker shift/add army). Disjoint markers + body never collide so the label is
just `body_colour*body + stamp`. opset-10 Equal needs int32 (float/fp16 both rejected
by shape-inference) so the final value plane must be cast to int32 before the FREE
output Equal — but the upstream arithmetic can all run in fp16 (half the bytes).
