# task355 — de1cd16c

**Rule:** The grid is a 2×2 tiling of solid-colour rectangular blocks (block size 5..10
per side, fully filling a ≤20×20 canvas, anchored top-left). The 4 block colours are
DISTINCT (`sample`). A unique `pcolor` (not any block colour) is scattered as `counts[idx]`
single specks over block idx, the counts a DISTINCT sample of range(6). The 1×1 output =
`mostest`, the colour of the block that received the MOST specks (unique max count).
(The hand-written `validate()` test uses 2×3 blocks with repeated colours, but `generate()`
— what generalization is scored against — is always 2×2 with distinct colours.)

**Current:** 15.624 pts, gen:thbdh6332, mem 11706, params 89
**Target tier:** A — closed-form per-channel bbox count; blocked from a higher tier by the
one irreducible fp32 30×30 gathered-pcolor plane.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | per-channel bbox (first/last) + fp16 MatMul | A | 17588 | 44 | 15.22 | — | too many [1,10,*] planes |
| 2 | occupancy-IS-the-band (drop first/last/fill), fp16 | A | 11388 | 72 | 15.65 | — | +0.03, big simplification |
| 3 | same, fp32 MatMul (no speck fp16 copy) | A | 10168 | 72 | 15.77 | 200/200 | best; +0.14 = MARGINAL |

## Best achieved
15.766 @ mem 10168 params 72 — adopted? N (build agent does not adopt). Beats prior 15.624? YES by +0.14, but BELOW the +0.3 bar → MARGINAL.

Approach: cnt=ReduceSum(input,[2,3]); pcolor=argmin nonzero count; speck=Gather(input,pcolor)
[1,1,30,30]. Key simplification: a SOLID block fills every row/col in its range, so the 1-D
per-channel occupancy `ReduceMax(input,axis)` IS the bbox band directly — no first/last/fill
needed (verified 800/800). boxcnt[k] = rowband_k @ speck @ colband_k via two batched MatMuls
(broadcast 10 vs 1, never materialises the 10×30×30 product). Zero the pcolor & absent
channels, ArgMax → mostest. Output = And(row0, And(col0, onehot(mostest))) routed into the
FREE bool output (associative broadcast, no full label plane).

## Irreducible-floor analysis
Dominant intermediates (fp32): speck Gather [1,1,30,30] = 3600 (the 10→1 colour-index entry
floor — cannot go below fp32 30×30 per FLOOR_RESEARCH); rowocc+colocc ReduceMax = 1200+1200;
the two transposed bands + proj = 3×1200. The 3600 speck is the wall: counting pcolor specks
per candidate-colour box REQUIRES isolating the pcolor channel first (Gather), because doing
it without the gather couples input-channel × candidate-colour into a [10,10,30] ≈ 36KB
intermediate. To beat +0.3 needs mem+params < 8763; removing any 3600 fp32 entry has no exact
route. fp16-everything-downstream is a wash: it halves the 3 downstream planes (−1800) but
forces a fp16 speck copy (+1800).

## OPEN ANGLES (re-attack backlog)
- Quadrant approach: exploit the exact 2×2 layout. Contract speck to a [2,2] quadrant
  histogram via tiny row/col split masks (split at r0=talls[0], c0=wides[0]) — kills ALL
  per-channel [1,10,*] planes. BUT mapping winning-quadrant→block-colour needs a 2nd colour
  readout; naive route adds a colf [1,1,30,30]=3600 plane (two 3600 planes = same wall).
  Untried: read the 4 block colours as 4 SCALARS (ArgMax of input at 4 quadrant-centre cells
  via Gather), avoiding a 2nd full plane — could land both speck-count and colour in scalars
  and drop below the wall. Worth a focused attempt.
- Active-region (≤20×20) slicing of the bands/proj after a fp16 cast — but the speck Gather
  still emits the full 30×30 (3600), so no net win on the entry.

## INSIGHT (transferable)
⭐ "per-region count of a sparse marker colour, distinct solid block colours" needs NO bbox
first/last/fill: a solid block occupies EVERY row & col in its range, so the raw 1-D
per-channel occupancy `ReduceMax(input,[3])`/`[2]` IS the bbox band — feed it straight into
`band_k @ marker @ band_kᵀ` (two batched MatMuls, broadcast 10-vs-1, no 10×H×W product). This
collapsed an 8-plane bbox pipeline to 2 occ planes (17588→10168). The residual wall is the
one fp32 Gather of the marker channel: any "count marker pixels per candidate colour" needs
the marker isolated to ONE channel first, else input-ch × candidate-colour couples to a
[10,10,W] plane.
