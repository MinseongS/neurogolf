---
deployed_cost: 350
logged_costs_match: true
migrated: 2026-07-09
---

# task271 — ae4f1146

**Rule:** A 9x9 black grid holds four NON-overlapping 3x3 cyan(8) boxes; each box
carries some blue(1) pixels at local positions. The generator samples 4 distinct
blue-pixel counts from `range(9)`, sorts them ascending, and assigns them to boxes
0..3 — so the four boxes carry strictly different blue counts and box 3 (the last)
has the MOST. The output is the 3x3 content of the box with the most blue pixels:
output[r][c] = blue(1) where that box has a blue pixel, else cyan(8). Winner is
unique (0 ties over 2000 fresh samples). Grid is always 9x9.

**Current:** 19.1420668 pts at cost350, memory238, params112; bundled267/267.
**Target tier:** A/B — argmax-select-among-4-boxes then crop a separable 3x3
window. Output color is per-cell deterministic given the winning box, so it lands
in the label-map family but with a tiny working canvas (9x9/7x7), beating B.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 3x3 sum-convs (blue/occ) over 9x9 → score=blue·isbox → ReduceMax argmax → Gather 3x3 window → label-map Equal | A/B | 2880 | 70 | 17.01 | 500/500 | WIN |

## Best achieved
19.1420668 @ cost350, mem238, params112 — adopted through `ng adopt` at
20260715T145416Z. The 2026-07-15 task-local sequence improved cost1074 ->350
(+1.121212 points) with bundled267/267.

## Historical irreducibility claim — falsified 2026-07-15
Dominant intermediates at that time: `L` = the 30x30 uint8 label plane (900B) — then treated as unavoidable
because the output must broadcast to [1,10,30,30] and the smallest per-cell label
dtype is uint8; and the two fp32 9x9 channel slices `blue9`/`cyan9` (324B each =
648B) — Slice preserves the fp32 input dtype so the fp32 footprint is paid before
the fp16 Cast. The rest is 7x7/3x3 fp16 + scalar argmax machinery (~600B). No
[1,10,*,*] plane is ever materialised; the 10-way expansion lands in the FREE bool
`output` via the final Equal.

## OPEN ANGLES (re-attack backlog)
- The two fp32 9x9 slices could collapse to one if box-detection used ch0
  (occ = 1 − ch0_9x9) instead of ch8 — but blue (ch1) is still a second slice, so
  no net win. A single 1×1 colour-index Conv would emit a [1,1,30,30]=1800B fp16
  plane (worse than 648B of two 9x9 slices), so that direction loses.
- The old analysis treated L=900B as a lower bound for a 30x30 uint8 label and
  considered only a fixed single-Conv escape. The 2026-07-15 free-input window
  Einsum plus direct 3x3 patch read falsified that premise by avoiding L entirely.

## INSIGHT (transferable)
"Emit the 3x3 box with the most X pixels" = (a) a 3x3 all-ones sum-Conv over a
small slice gives per-top-left counts AND a box-validity count in one pass;
(b) score = count gated by (occ-conv==9) via Where; (c) UNIQUE-argmax position
recovers as scalar (minrow,mincol) = ReduceMax(iswin·rowramp) / ReduceMax(iswin·
colramp) — no NonZero/argmax-op needed; (d) data-dependent crop = Add scalar
offset to a [0,1,2] index const, chained Gather(axis=2)·Gather(axis=3). Whole
select-and-crop pipeline stays at ~9x9/7x7 fp16, below the old 900B 30x30 label-plane cost.

## ADOPTED 20260713T143829Z
- cost: 1187 -> 1074 (points 18.0209)
- source: candidates/public_dumps/20260713_7281/extracted/task271.onnx
- note: Ryosuke 7281.18 public-LB confirmed per-task min-merge; bundled fail=0

## ADOPTED 20260713T150957Z
- cost: 1187 -> 1074 (points 18.0209)
- source: candidates/public_dumps/20260713_7281/extracted/task271.onnx
- note: Ryosuke-7281 isolation B; task047 explicitly excluded; bundled fail=0

## ADOPTED 20260713T151939Z
- cost: 1187 -> 1074 (points 18.0209)
- source: candidates/public_dumps/20260713_7281/extracted/task271.onnx
- note: Kaggle-isolated safe: group delta +2.05 exactly (sub 54651291 minus 54651270); task047 excluded

## ADOPTED 20260715T110459Z
- cost: 1074 -> 682 (points 18.4750)
- source: candidates/task271/free_input_window.onnx
- note: free-input sliding-window Einsum removes 2x9x9 crop+quantized conv; direct tiny patch read

## ADOPTED 20260715T120657Z
- cost: 682 -> 586 (points 18.6267)
- source: candidates/task271/maxpool_convinteger_fold.onnx
- note: MaxPool winner index plus compact ConvInteger renderer removes dual argmax and crop8 Pad tail

## ADOPTED 20260715T132945Z
- cost: 586 -> 455 (points 18.8797)
- source: candidates/task271/direct_conv_score.onnx
- note: direct FREE-input 3x3 Conv with negative end pads replaces dense window3 basis; preserves MaxPool/ConvInteger tail

## ADOPTED 20260715T135609Z
- cost: 455 -> 411 (points 18.9814)
- source: candidates/task271/int32_signed_renderer.onnx
- note: int32 MaxPool coordinates plus shared-zero-point one-feature signed ConvInteger renderer

## ADOPTED 20260715T142423Z
- cost: 411 -> 366 (points 19.0974)
- source: candidates/task271/encoded_score_payload.onnx
- note: encode winning 3x3 payload in score low bits; remove MaxPool indices and dynamic Slice

## ADOPTED 20260715T145416Z
- cost: 366 -> 350 (points 19.1421)
- source: candidates/task271/bitwise_where_decoder.onnx
- note: decode score payload with UINT16 BitwiseAnd and centered UINT8 Where

## ADOPTED 20260715T155323Z
- cost: 350 -> 283 (points 19.3546)
- source: candidates/task271/hash_chd_payload.onnx
- note: bundled rank1 hash + CHD perfect lookup + seven-lane INT64 payload packing
