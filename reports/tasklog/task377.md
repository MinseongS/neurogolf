# task377 — eb5a1d5d

**Rule:** Input is N strictly-NESTED axis-aligned rectangles painted in order; rect 0
fills the whole grid, each later rect sits strictly INSIDE the previous with its
top-left corner moving strictly SE, in colours[0..N-1] (adjacent differ, non-adjacent
may repeat). OUTPUT is a (2N-1)x(2N-1) concentric-square-ring target at the top-left:
out[r][c] = colours[ min(r,c,S-1-r,S-1-c) ], S=2N-1, all-zero elsewhere. The whole
output is determined by the scalar N and the length-N colour SEQUENCE.
**Current:** 15.69 pts, ext:kojimar7113, mem 10887, params 180
**Target tier:** A (closed-form scalar+sequence recovery + tiny ring) — no flood-fill.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | prior local build (horizontal depth-scan + 15x15 ring) | A | 15639 | 1004 | 15.28 | — | worse than deployed |
| 1 | kojimar recovery (dilated colour Conv + per-row ArgMax/TopK) + SMALL ring tail (W=15, int32 ring) | A | 9171 | 138 | 15.86 | — | +0.17 |
| 2 | W=13, int32 ring math (drop fp16 ring plane) | A | 8581 | 134 | 15.93 | 500/500 | +0.24 MARGINAL |
| x | drop `curr` by gathering off 27-row grid (Pad+shift) | A | 8713 | 140 | 15.91 | — | reverted: Pad plane costs more than it saves |

## Best achieved
15.927 @ mem 8581 params 134 — adopted? N (build-only). Beats prior 15.69? Y by +0.237 (MARGINAL, < +0.3).

## Irreducible-floor analysis
- `grid_f` fp32 27x27 = 2916B: the 10->1 channel colour-index reduction off the fp32
  input forces fp32 output (Conv/MatMul/ReduceSum inherit input dtype); corners reach
  row/col 26 so the recovery window can't crop below 27x27. kojimar pays the same.
- Recovery scan: `curr`,`prev`,`row_delta` = 3 x 702B uint8 (2106B). `curr` already does
  double duty (subtraction operand + GatherElements colour source); `prev` is the other
  subtraction operand. Equal->bool would need a Cast back to uint8 for ArgMax (net worse).
- Value tail: `Lpad` uint8 30x30 = 900B is the minimal full-canvas value plane feeding the
  Equal->BOOL output; Gathering a one-hot at the small ring + bool-Pad is strictly bigger
  (10*W^2). `ring_i` int32 W^2 (676B at W=13) is the Gather-index floor.
The architecture floors ~8.4-8.6KB; the +0.3 target (~8.3KB) is just out of reach.

## OPEN ANGLES (re-attack backlog)
- A recovery that avoids the fp32 27x27 colour grid entirely (1-D-profile-only depth +
  sequence) would unlock < 6KB, but reading the per-corner COLOUR VALUE needs the column
  position of each corner, which a 1-D profile loses -> appears structurally 2-D.
- W=13 covers N<=7 (max over 500k samples). LB tail N=8 (S=15, prob ~0) would silently
  miss; bump to W=15 (mem 8755, +0.197) for full N<=8 safety if a fresh failure ever shows.

## INSIGHT (transferable)
- SMALL-RING TAIL beats full-canvas index Gather for "bullseye/target from O(1) scalars":
  when a crowd net builds the per-cell index at FULL 30x30 (kojimar `ring_raw` int32 3600B)
  only to Gather a colour table into output, rebuild the index on a TINY W x W canvas
  (W = 2*Nmax-1), Gather the VALUE plane there, then uint8-Pad into 30x30 + Equal->BOOL
  output. Removes the 3600B full-canvas plane for ~+0.24 with NO recovery change.
- A dilated 2x2 Conv with only the [0,0] tap nonzero (`dilations=[30-GW,30-GW]`) reads the
  colour-index AND crops to the top-left GWxGW window in ONE op (kojimar `color_decode_w`).
- uint8 `Sub`/`Min` and bool `Pad` require opset >= 14/18 under ORT_DISABLE_ALL — opset-11
  rejects them. Bump the model opset to 18 to use kojimar-style uint8 row-delta scans.
- `curr`/`prev` slices: the subtraction's two operands are BOTH irreducible planes; reusing
  one operand as the colour-read source is the only saving (gathering off the full grid via
  a Pad-shift adds a bigger plane than it removes — net negative).
