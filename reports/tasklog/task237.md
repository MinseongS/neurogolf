# task237 — 99fa7670

**Rule:** A (width×height) grid (both ∈ 3..9) holds scattered seed pixels, at most one per row
(seed rows strictly increase by 2 or 3; seed cols ∈ [0, width-2]). For each seed (r,c,color):
(a) horizontal fill rightward — `output[r][col]=color` for col∈[c, width-1]; (b) vertical fill down
the last column — `output[row][width-1]=color` for row∈[r, height-1]. Vertical fills are applied in
ascending seed-row order, so the last column at row `row` carries the color of the seed with the
LARGEST seed-row ≤ row (forward-fill, most-recent wins). Off-grid stays background.
**Current:** 15.43 pts (P). 
**Target tier:** A (separable row-fill + tiny forward-fill on length-≤9 vectors; routed into FREE bool output).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | suffix-fill MatMul + plateau forward-fill, 10-ch slice + Mul+ReduceSum | A | 11284 | 218 | 15.65 | 200/200 | MARGINAL (+0.22) |
| 2 | replace Mul+ReduceSum with 1×1 Conv; rowin/colin via ReduceMax(input,axes=[1,3])/[1,2] (no 2-D occ plane) | A | 7960 | 222 | 15.99 | 200/200 | +0.56 |
| 3 | slice channels 1..9 only for colour Conv (ch0 contributes 0) | A | 7636 | 221 | 16.03 | 500/500 | ADOPT (+0.60) |

## Best achieved
16.03 @ mem 7636 params 221 — beats P (15.43) by +0.60. fresh 500/500, stored 5/5.

## How it works
- Channel-0 sentinel recovers grid extent: convert_to_numpy sets ch0=1 only for IN-GRID bg cells;
  off-grid is all-zero across channels. `rowin=ReduceMax(input,axes=[1,3])`→[1,1,30,1],
  `colin=ReduceMax(input,axes=[1,2])`→[1,1,1,30] give the solid H×W mask with NO 2-D occupancy plane.
- colf = 1×1 Conv(input[:,1:10,:9,:9], weight=[1..9]) → [1,1,9,9] colour-index, no [1,10,9,9] Mul plane.
- Horizontal fill = `colf @ Suf`, Suf[csrc,c]=[csrc≤c] (one seed/row ⇒ exact suffix fill); gate by colin.
- Last-column forward-fill on tiny vectors: srow=ReduceSum_col(colf); sany=srow>0; pc=LowTri@sany
  (prefix seed count); E[row,r]=[pc[row]==pc[r]]; ff=E@srow picks the unique plateau seed (most-recent).
- Place ff at last in-grid col (islast=colin∧¬colin_shift), Hfill elsewhere; off-grid→sentinel 99 so
  Equal(final30_f16, arange[0..9])→BOOL output (FREE) is all-False off-grid (= all channels 0).

## Irreducible-floor analysis
Dominant: the 9-channel 9×9 fp32 input slice (3240B) — reading colour requires a fp32 multi-channel
region; fp16/uint8 cannot shrink it (ORT upcasts; Conv/Mul reject uint8). Second: the 30×30 fp16 output
carrier for the final Equal (1800B) — Equal must emit a 30×30 plane to broadcast into the [1,10,30,30]
output. Everything downstream is fp16 9×9 (162B) or length-9 vectors. mem+params≈7857 → ~16.03.

## OPEN ANGLES (re-attack backlog)
- Drop the slice's 3240B by contracting channels on the FULL input via a clever Conv then slicing colf30
  (3600B) — net worse here; would need a sub-3240 single-channel colour read at 9×9 (not currently possible).
- If a future task shares the "forward-fill / most-recent-wins" need, the plateau trick (E=Equal of prefix
  counts, then E@values) is the closed-form replacement for a Scan.

## INSIGHT (transferable)
⭐ Forward-fill (carry most-recent nonzero down a sparse axis, later-wins) is closed-form with NO Scan:
prefix-count the seeds `pc=LowTri@sany`, then `ff = Equal(pc_row,pc_col) @ values` — the equality matrix
groups each plateau, and `values` (nonzero only at the plateau's seed) selects it. Exact on length-≤K axes.
⭐ "Fill rightward from a single per-row seed to the row's end" = `colf @ triu` (suffix-fill MatMul), exact
because one seed/row means the row-sum has one term — no overlap, no argmax.
