---
deployed_cost: 1089
logged_costs_match: true
migrated: 2026-07-09
---

# task069 — 321b1fc6

**Rule:** Grid is always 10x10. `num_boxes` (4) IDENTICAL 4-connected sprites (an arbitrary
connected subset of pixels inside a small box, width 2..4 height 2..3) are placed at random
non-overlapping positions (bounding boxes separated with margin 1). Exactly ONE sprite (first
drawn) is shown in its real per-pixel COLOURS; every other sprite is shown all-cyan (8). OUTPUT =
erase the coloured sprite to black, and redraw EVERY cyan sprite in the real colour pattern,
aligned to each sprite's own bounding-box top-left. Verified (500 fresh): output nonzero <=> input
== cyan(8); coloured box -> black; colour at a cyan cell = colourmap[(r-bbox_top, c-bbox_left)]
where the colourmap (offset -> colour) is revealed by the single coloured sprite, same for all.
**Current (stored):** 13.84 pts.
**Target tier:** B (label-map). NOT S (non-local: cell colour depends on its offset within its own
sprite + the revealed pattern). NOT A-separable (P is an arbitrary 2..4-colour pattern; the colour
table is irreducibly 2-D over offset). Per-cell offset IS recoverable locally -> label-map + Equal.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 8-conn min-prop (3x3 minpool) for bbox-top | B | 40896 | 84 | 14.38 | 200/200 | WRONG: merges corner-adjacent sprites (key 12+ OOB). Margin-1 bboxes can touch at a corner. |
| 2 | 4-conn PLUS-min (vert 3x1 + horiz 1x3 minpool, elementwise Min), 7 iters | B | 30098 | 104 | 14.68 | 200/200 | correct; min-prop in negated space (MaxPool, no per-step Neg) |
| 3 | histogram via dr/dc double-MatMul ([3,N]@[N,4] fp16) instead of [N,12] one-hot | B | 25950 | 99 | 14.83 | 200/200 | dropped 4800B keyoh plane |
| 4 | col_w = colf*(colf!=8) directly (drop occf/is_col) | B | **25150** | **99** | **14.86** | **500/500** | FINAL |

## Best achieved
14.86 pts @ mem 25150, params 99 — 264/264 stored, fresh 500/500. Adopted? **N** (orchestrator
gates). Beats stored 13.84 by **+1.02 (Y, generalizes)**.

## Irreducible-floor analysis
- **colf30 3600 B fp32 [1,1,30,30]** — the 1x1 colour-index Conv; output must be 30x30 fp32 (any
  linear combo of the FREE fp32 input is fp32). Entry-read floor (shared with task368/358).
- **bbox-anchor plus-min ~11 KB** — 7 iterations x 2 axes x ~4 small fp16 planes (vert minpool,
  horiz minpool, Max, re-mask Where) at 200 B each. Iterations: a 4-connected sprite <=3x4 has a
  longest plus-propagation path of 7; radius MUST be 1 (a wider minpool reaches across the margin-1
  gap and merges neighbour sprites). This is the dominant *non-entry* cost and is fundamental to
  recovering each cell's bbox top-left for ARBITRARY (holed) shapes under 4-connectivity.
- **L30 900 B uint8 [1,1,30,30]** — output label feeding the free final Equal; uint8 is cheapest,
  sentinel-99 pad off-grid. Irreducible for any per-cell colour rewrite.
- histogram ~2.7 KB (dr/dc one-hots [3,N]/[N,4] fp16+bool); N=100 since the coloured sprite can
  sit anywhere; table genuinely 3x4 (arbitrary pattern).

## OPEN ANGLES (re-attack backlog)
- The 7-iter plus-min is the binding non-entry cost. A non-iterative bbox-anchor would need either
  (a) a connectivity-respecting closed form for arbitrary holed shapes (none known cheaper than
  propagation), or (b) anchor-correlation: recover the coloured sprite as a small KxK stamp + find
  each cyan box's top-left by correlation, then stamp. Risky (bbox top-left corner can be an EMPTY
  cell for L-shapes, so the anchor isn't a pixel) — untried, maybe ~0.2 pt.
- colf30 3600 fp32 is the entry floor; no free-input fp16 colour-index path (shared wall).

## INSIGHT (transferable)
⭐ "Recolour every marker-coloured sprite from the one revealed sprite" generalises beyond SOLID
rectangles (task368) to ARBITRARY 4-connected shapes: the per-cell offset is `(r,c) - bbox_top_left`
of the cell's own sprite, recovered by propagating the MIN row/col index over the sprite via an
iterated PLUS-shaped min (NOT 3x3 — generator margin-1 bboxes touch at CORNERS, so 8-connectivity
merges distinct sprites; use 4-connectivity = elementwise Min of a vertical 3x1 and horizontal 1x3
min-pool). Do min-prop in NEGATED space so each step is a plain MaxPool with no per-step Neg.
Build the offset->colour table as a dr/dc double-MatMul ([3,N]fp16 @ [N,4]fp16, colour-weighted)
instead of a [N,12] key one-hot (saves ~4.8 KB). The whole thing lands as a label-map + final Equal.
⭐ Plus-min radius MUST be 1: a wider min-pool reaches across the inter-sprite gap and merges them.

## ADOPTED 20260709T041314Z
- cost: 3905 -> 2949 (points 17.0108)
- source: candidates/public_dumps/20260709/7261-53-lb-compact-onnx-artifact-starter/nets/task069.onnx
- note: min-merge from nets

## ADOPTED-MECHANISM NOTE 2026-07-09 (public autopsy)
- 우리 tasklog의 OPEN ANGLE (b) "anchor-correlation (risky, untried)"를 공개가 그대로 구현해 승리 —
  runtime-template QLinearConv 상관검출 + flip-stamp. L-shape empty-corner 리스크는 correlation
  peak counting으로 해결됨. insights.yaml: runtime_template_qlinearconv_correlate_stamp.

## ADOPTED 20260709T061944Z
- cost: 2949 -> 2919 (points 17.0210)
- source: candidates/task069/kcollapse.onnx
- note: kernel-collapse: single-position Conv kernel collapse after public/regime overlays

## ADOPTED 20260715T133248Z
- cost: 2919 -> 2530 (points 17.1640)
- source: candidates/task069/poly_convinteger.onnx
- note: opset14 polynomial ConvInteger fold: outplane Pad Equal -> [x,x^2,1] logits

## ADOPTED 20260715T135229Z
- cost: 2530 -> 1827 (points 17.4896)
- source: candidates/task069/compact_exact.onnx
- note: exact compact composition: occupancy Sign profiles + colored peak Div + shifted 2-feature QLinear tail

## ADOPTED 20260715T140107Z
- cost: 1827 -> 1597 (points 17.6241)
- source: candidates/task069/slice_dynamic_stamp.onnx
- note: dynamic Slice crop/reverse plus runtime quadratic two-feature stamp

## ADOPTED 20260715T141141Z
- cost: 1597 -> 1432 (points 17.7332)
- source: candidates/task069/packed_modular_crop.onnx
- note: packed marker255 Div/Mod plus keepdims ArgMax and fixed modular Gather crop

## ADOPTED 20260715T142003Z
- cost: 1432 -> 1384 (points 17.7673)
- source: candidates/task069/int32_crop.onnx
- note: int32 modular crop indices after int64 ArgMax scalars

## ADOPTED 20260715T142414Z
- cost: 1384 -> 1297 (points 17.8322)
- source: candidates/task069/packed_binary_corr.onnx
- note: packed idx at 1/255 against binary runtime footprint; remove marker plane

## ADOPTED 20260715T142847Z
- cost: 1297 -> 1212 (points 17.9000)
- source: candidates/task069/dynamic_anchor_bias.onnx
- note: binary footprint self-count plus dynamic full-overlap QLinearConv bias

## ADOPTED 20260715T143235Z
- cost: 1212 -> 1210 (points 17.9016)
- source: candidates/task069/bias4d.onnx
- note: direct 4D singleton bias; remove scalar Reshape

## ADOPTED 20260715T145249Z
- cost: 1210 -> 1134 (points 17.9665)
- source: candidates/task069/signed_shared_plane.onnx
- note: signed shared colour/marker plane with all-INT8 zero-point anchor and output chain

## ADOPTED 20260715T150258Z
- cost: 1134 -> 1124 (points 17.9754)
- source: candidates/task069/unit_marker_bias.onnx
- note: unit signed marker with direct one-minus-N anchor bias

## ADOPTED 20260715T151901Z
- cost: 1124 -> 1104 (points 17.9933)
- source: candidates/task069/u8_crop_indices.onnx
- note: UINT8 modular crop arithmetic with final INT32 Gather indices

## ADOPTED 20260715T152427Z
- cost: 1104 -> 1102 (points 17.9951)
- source: candidates/task069/slice_alias.onnx
- note: alias identical reverse Slice starts and axes

## ADOPTED 20260715T152726Z
- cost: 1102 -> 1089 (points 18.0070)
- source: candidates/task069/gap_coded_square.onnx
- note: gap-coded colours with direct INT8 square feature and complete class0..9 decoder
