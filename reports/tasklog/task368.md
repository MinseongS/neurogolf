# task368 — e76a88a6

**Rule:** A 10x10 grid holds 3-4 IDENTICAL solid HxW rectangles (H,W in {3,4}, never both 4)
at random non-overlapping positions. Each rectangle is the same 2-colour pattern P (palette of
2 colours, gray excluded). Exactly ONE sprite (the first drawn) is shown in its real colours;
every other sprite is shown all-gray (colour 5). OUTPUT = redraw EVERY sprite (gray ones and the
coloured one) in the real pattern P, aligned to each sprite's own top-left corner.
**Current (stored):** 14.69 pts, gen:thbdh6332, mem 29948, params 84.
**Target tier:** B (label-map). NOT S (output colour is non-local — depends on the cell's offset
within its sprite + the revealed pattern). NOT A-separable (P is an arbitrary 2-colour pattern, so
the per-cell colour is NOT rowcond⊗colcond). The per-cell offset is recoverable LOCALLY though, so
the whole thing collapses to a label map + final Equal.

## Key structure
Every sprite is a SOLID HxW rectangle, so for any occupied cell (r,c) its offset within its own
sprite is a LOCAL run-length:
  dr = (#consecutive occupied cells upward incl. self) - 1   (0..3)
  dc = (#consecutive occupied cells leftward incl. self) - 1 (0..3)
computed by the product-chain of shifted occupancy (resets at gaps; sprites are separated so a run
never spans two sprites). key = dr*4+dc (0..15). The colour at offset (dr,dc) is identical for every
sprite and is revealed by the ONE coloured sprite. Binary-partition the 2 palette colours lo/hi by
channel index; learn tableHi[dr,dc] = "coloured sprite's cell at (dr,dc) is the hi colour" as a 4x4
histogram via a double-MatMul over offset one-hots of the (sparse) hi-colour cells:
  tableHi = dr_ohw @ dc_ohT   ([4,N]@[N,4] -> [4,4]),  dr_ohw already isHi-weighted via a -1 sentinel.
Then per occupied cell outHi = Gather(tableHi_flat[16], key). Label L = occ ? (outHi?hi_idx:lo_idx) : 0,
Pad to 30x30 with sentinel 99 (off-grid -> all-channels-off), final `output = Equal(L, arange)` (BOOL, free).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | runlen dr/dc + matmul-histogram + Gather + Where(occ,Where(outHi,hi,lo),input) | B | 100268 | 341 | 13.48 | 200/200 | works but heavy (full [1,10,30,30] inner Where + full-input channel contract) |
| 2 | label-map L + final Equal (drop the inner Where 36000) | B | 31356 | 364 | 14.64 | 200/200 | correct (needed sentinel-99 pad so off-grid = all-channels-off) |
| 3 | fp16 working canvas everywhere (occ/dr/dc/histogram/label all fp16) | B | 21676 | 364 | 15.00 | 200/200 | |
| 4 | colour-index via 1x1 Conv (Sum k*input_k) -> occ=colf>0, isHi=colf==hi_idx (drop inp10+hi_sel 8000) | B | 17056 | 362 | 15.23 | 200/200 | |
| 5 | fold isHi into dr via -1 sentinel (drop standalone dr_oh/dc_oh + Transpose) | B | 15856 | 369 | 15.31 | 200/200 | |
| 6 | one Pad+3 Slices per axis for the run-length shifts (drop 4 Pad planes) | B | **14936** | **314** | **15.37** | **500/500** | FINAL |

## Best achieved
**15.37 pts @ mem 14936, params 314 — 265/265 stored, fresh 500/500.** Adopted? **N** (orchestrator
gates). Beats stored 14.69 by **+0.68 (Y, generalizes both train+test+arc-gen).**

## Irreducible-floor analysis
Dominant intermediates:
- **colf30 3600 B fp32 [1,1,30,30]** — the 1x1 Conv colour-index plane. fp32 because any linear combo
  of the FREE fp32 input is fp32; casting input->fp16 would materialise an 18000B one-hot first. This
  is the input-read floor (same "conv-3600 entry floor" lesson as task358).
- **L30 900 B uint8 [1,1,30,30]** — the label feeding the free final Equal; output spans 30x30, uint8
  is the cheapest dtype, sentinel-99 pad off-grid. Irreducible for any per-cell colour rewrite.
- **histogram operands ~2400 B**: dr_ohw [4,100] + dc_ohT [100,4] fp16 (800 each) + their bool [4,100]
  (400 each). N=100 because the coloured sprite can sit anywhere in the 10x10; the table is genuinely
  4x4 (P is an arbitrary 2-colour pattern, not separable). fp16 is exact ({0,1}, sums<=16).
- run-length shifts ~1700 B fp16 (two Pad-by-3 planes + 6 sliced occ + 6 products). fp16 forced (ORT
  Mul rejects uint8; products need a float type).
Floor ~= 3600 + 900 + 2400 + 1700 + smalls ~= 14900 -> ~15.37.

## OPEN ANGLES (re-attack backlog)
- Trim the run-length shift chain (~1700 B): a single Conv can't compute the run-length PRODUCT
  (sum doesn't reset at gaps), but a cleverer cumulative-with-reset (e.g. cumulative-max of a
  position ramp masked by occ, then subtract) might do dr/dc in fewer planes. ~0.1 pt, untried.
- The colf30 3600 fp32 is the binding constraint; no free-input fp16 colour-index path exists
  (shared wall with task358). If a future op-fusion lets Conv emit fp16 from fp32 input, this drops.
- Tier A is blocked: P is an arbitrary 2-colour pattern, so the per-cell output colour is NOT a
  row-cond AND col-cond product (the 4x4 table is irreducibly 2-D). Tier S blocked (non-local).

## INSIGHT (transferable)
⭐ "Recolour every gray stamp using the one coloured stamp" (identical solid rectangles at random
non-overlapping positions) is NOT a shape-correspondence BAIL. The offset of a cell within its own
solid-rectangle sprite is a LOCAL run-length (product-chain of shifted occupancy, resets at gaps),
so key = dr*4+dc is computable per cell with no flood-fill. The colour-by-offset map is a 4x4
histogram learned from the single coloured sprite via a double-MatMul over offset one-hots
(weight the hi-colour cells by folding isHi into the offset with a -1 sentinel so one Equal does
the mask+one-hot), then a 1-D Gather propagates it to all sprites. Whole thing lands as a label-map
+ final Equal (Tier B). The recurring 3600B fp32 colour-index Conv is the input-read floor.
