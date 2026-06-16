# task359 — e26a3af2

**Rule:** The grid (width=sum(wides) x height, top-left anchored) is tiled with consecutive
VERTICAL stripes — band i spans `wides[i]` columns in a fixed stripe colour. Sparse noise pixels
(prob 0.1, colours 1..9) are stamped on top in the INPUT. If `xpose=1` the whole figure is
transposed (stripes run horizontally). OUTPUT = clean stripes (every cell restored to its band's
stripe colour). Every in-grid cell is a colour 1..9; background 0 appears only OFF the grid.

**Current (prior public/stored):** ~14.68 pts, tier A
**Target tier:** A (~17) — output[cell] = stripe colour of the cell's column-band (or row-band if
transposed), i.e. a per-LINE (row/col) constant. Not Tier S because the colour is data-dependent
(per-instance random stripe colours, recovered by a per-line argmax) and orientation must be
selected, so no single fixed Conv/permute routes it.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | per-line argmax of ReduceSum counts + peak-sum orientation select + separable in-grid mask, routed into FREE Equal->output | A | 6190 | 12 | 16.27 | 200/200 | WIN (+1.59 vs ~14.68) |

## Best achieved
16.27 @ mem 6190 params 12 — adopted? N (orchestrator gates). Beats prior ~14.68? YES (+1.59),
GENERALIZES (fresh 200/200 + 2000/2000 numpy prototype, both train+test pass).

## Irreducible-floor analysis
Dominant intermediates: the two count tensors colcount[1,10,1,30] and rowcount[1,10,30,1] (fp32,
1200 B each = 2400 B) — both stripe orientations must be probed since xpose is data-dependent, and
ORT ReduceSum requires float, so they cannot be uint8. Next: the [1,1,30,30] uint8 label plane `L`
(900 B) and the [1,1,30,30] bool `ingrid` (900 B) and `selected` uint8 (900 B). The three 30x30
planes (~2700 B) plus the two reductions set the ~6.2 KB floor.

## OPEN ANGLES (re-attack backlog)
- Fuse `selected` + `ingrid` + `L` into fewer 30x30 planes. The off-grid sentinel could be folded
  into `selected` itself if the empty-column argmax produced 10 instead of 0 (e.g. add a tiny
  bias channel), removing the separate `ingrid` And-plane (~ -900 B -> ~16.4).
- Shrink the working canvas: rows only go to height<=15, columns to <=30, so the row-axis planes
  could be cropped to 15 tall before broadcasting (final Pad), trimming the 30x30 planes ~50%.
  Potentially pushes toward ~16.7-17 (true tier A).
- Replace ReduceSum counts with a per-channel batched MatMul against a ones vector to contract the
  free input directly without a 1200 B fp32 reduction (operand-order trick, task025 idiom).

## INSIGHT (transferable)
Orientation (xpose) can be recovered with ZERO per-cell planes by comparing the two axes' total
"peak" match mass: peak_col = sum_c max_ch colcount[.,c] vs peak_row = sum_r max_ch rowcount[r,.].
The correct stripe axis maximises peak mass (the wrong axis mixes several band colours per line).
Also: `Where(scalar[1,1,1,1], A[1,1,1,W], B[1,1,H,1])` broadcasts to [1,1,H,W] in ONE op — it
simultaneously SELECTS orientation and BROADCASTS the chosen per-line vector across the canvas,
avoiding building two full candidate planes. And remember convert_to_numpy leaves OFF-grid cells
with ALL channels = 0 (NOT channel-0 = background), so the in-grid mask must be "any channel set",
never "channel 0 == 0".
