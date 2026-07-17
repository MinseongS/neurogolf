---
deployed_cost: 921
logged_costs_match: true
migrated: 2026-07-09
---

# task062 — 2bcee788

**Rule:** Size is ALWAYS 10x10. A connected 3x3 sprite (always with a column-0
cell, `unshown>0` so it has a c>0 cell too) is drawn in `color` (color in
{1,4,5,6,7,8,9}) at offset (row 1..6, col 4..6). Each sprite cell (r,c) paints
`grid[row+r][col+c]=color`; the MIRROR cell `grid[row+r][col-c-1]` gets a red(2)
marker only for c==0 (c>0 mirror cells stay background=0 in the input). The
OUTPUT paints BOTH the sprite and its full mirror across the axis at col-0.5 in
`color`, on a green(3) background. A random flip_horiz / transpose is then
applied to grid and output identically, so the reflection axis is vertical or
horizontal. Solve: C=colored mask, R=red mask; d = C-centroid − R-centroid;
VERTICAL iff |dcol|>|drow|; axis-sum s = 2·r_active + sign(d_active); reflect C
across that axis (x→s−x); M = C ∪ refl; colour = the one non-{0,2,3} channel;
output = green in-grid except M-cells = colour, off-grid all-zero.
**Current:** 18.174540 pts, mem 633, params 288, cost 921, bundled 267/267.
**Target tier:** A. The semantic rule copies an arbitrary runtime colour and performs
a nontrivial reflection, but the bounded generator admits a proved local separator;
the former global extrema/Gather representation was not a floor.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | crop work to 10x10, BGSEL 9000-param const, Where output | A | 10238 | 10270 | 15.07 | — | passes, +0.12 (under thr) |
| 2 | derive colour profiles from C10, single combined Slice, drop 2 convs | A | 8798 | 9656 | 15.18 | — | +0.23 (under thr) |
| 3 | ⭐ Pad(incolor10[1,10,10,10]) AS the free output → kills 9000-param BG + 30x30 mask pad | A | 9898 | 666 | 15.73 | — | +0.78 |
| 4 | incolor10 + colour-onehot fp16, output declared fp16 | A | 7918 | 666 | 15.94 | 200/200 | adopt |
| 5 | `[L,L²,1]` polynomial ConvInteger FREE-output tail; duplicate class-0 channels preserved | A | 1965 | 184 | 17.3272 | bundled 267 thresholded A/B | adopt |
| 6 | signed cropped Conv + shifted `[L,L²]` QLinear FREE-output tail | A | 1609 | 82 | 17.5669 | bundled 267 thresholded A/B | adopt |
| 7 | direct signed-code profiles + compact mask-to-feature broadcast | A | 1432 | 83 | 17.6768 | bundled 267 thresholded A/B | adopt |
| 8 | uint8 reflection arithmetic with signed Gather extension | A | 1392 | 83 | 17.7036 | bundled 267 thresholded A/B | adopt |
| 9 | encode shifted object colour directly in signed Conv plane | A | 1336 | 73 | 17.7494 | bundled 267 thresholded A/B | adopt |
| 10 | one-cell zero sentinel clamp + axis-column overlap orientation | A | 1315 | 73 | 17.7644 | bundled 267 thresholded A/B | adopt |
| 11 | dynamic one-feature signed QLinear FREE-output tail | A | 1232 | 61 | 17.8353 | bundled 267 thresholded A/B | adopt |
| 12 | identity-fallback direct reflection Gathers | A | 1040 | 57 | 17.9997 | bundled 267 thresholded A/B | adopt |
| 13 | 8x8 pre-Pad signed code + dynamic ConvInteger tail | A | 1003 | 55 | 18.0359 | bundled 267 thresholded A/B | adopt |
| 14 | one shared INT8 cast for profiles, colour, and overlap | A | 965 | 55 | 18.0724 | bundled 267 thresholded A/B | adopt |
| 15 | generator-complete local 11x11 QLinear reflection mask + UINT8 zero-point tail | A | 633 | 288 | 18.1745 | complete 12,384-state proof + bundled 267 thresholded A/B | adopt |

## Best achieved
18.174540 @ mem 633 params 288, cost 921. The 2026-07-15 adoption chain changed
2524 -> 2149 -> 1691 -> 1515 -> 1475 -> 1409 -> 1393 -> 1388 -> 1293 -> 1097
-> 1058 -> 1020 -> 921 and added +1.008140 points overall (+0.847298 from the
requested cost-2149 incumbent and +0.339260 in the final 1293->921 follow-up).

## Current residual profile
The former separate black/red planes, 10-channel colour-presence selector,
`color_grid10/30`, terminal `Equal`, geometry profiles, coordinate extrema,
reflection Gathers, and mask unions are gone. The largest counted tensor is the
FLOAT `cell_code` crop at 256B; the proved local classifier's two INT8 polarity
channels cost128B, padded sign code100B, and INT8 crop/sign code64B each. Dynamic
class matching and UINT8 ConvInteger weights cost10B each. The 242-element INT8
11x11 kernel plus compact quantization constants bring params to288 while total
memory falls to633B.
The exact source in `src/custom/task062.py` rebuilds deployed SHA
`6bc314ba3021b14e74273657ae7fb015d642cf200bebdf0ac71923c17b43caa5`
byte-for-byte.
All twelve historical/current task062 acceptance regressions pass after the
local-QLinear adoption. The stored proof exhausts 172 sprites, 12,384 generator
states, and 45,345 distinct local patches with object/background margins +1/-1.
No other task model was modified by this session.

## INSIGHT (transferable)
⭐ **`Pad(small_plane) AS the graph output beats a full-canvas BG constant.** When
the active grid is a fixed small region (here 10×10) sitting in the top-left of
the 30×30 canvas and off-grid output cells are all-zero, build the entire
coloured result at 10×10 (`Where(M10[1,1,10,10], colvec[1,10,1,1],
bgvec[1,10,1,1])` → [1,10,10,10]) and make the **final op** `Pad(...,
value=0)` to 30×30 — the 30×30 expansion lands in the FREE `output` and the
off-grid zero-fill is exact. This removed a 9000-param background constant AND
the 30×30 mask planes (−9000 params, −2700B mem in one move: 15.18→15.73).
⭐ Combined with output declared **fp16** (colour one-hot is {0,1}, exact; harness
reads `out>0`), the colour expansion halves (15.73→15.94). General lever for any
"small-active-grid recolour on a fixed bg" task.
⭐ The harness leaves **off-grid output cells ALL-ZERO** (no channel set), NOT
channel-0=1 — verified the hard way; a BG plane that sets ch0 off-grid fails.
⭐ **Polynomial FREE-output decode preserves intentional multi-hot class collisions.**
The channel classes are `[0,1,0,3,4,5,6,7,8,9]`, not `range(10)`. Building each
weight row from that vector gives `1-(L-k)^2`, so label 0 activates both channels
0 and 2, label 2 activates neither, and implicit ConvInteger padding activates none.
⭐ **One signed categorical plane can serve geometry, mask, marker, and colour.**
The cropped 1x1 Conv encodes red as -1, background as 0, and each object colour as
its shifted positive label. `Greater(0)` yields the object mask, `ReduceMin/ArgMin`
finds the red axis, `ReduceMax/ArgMax` finds object extents, and a global
`ReduceMax` returns the dynamic output colour. This deletes separate red/black
planes and the 10-channel presence selector without adding a counted carrier.
⭐ **Clamp wrapped reflection indices to a one-cell zero sentinel.** uint8 coordinate
arithmetic wraps negative indices above 127; after orientation selection,
`Min(index,8)` maps every negative/too-large value to the appended zero cell. This
permits one-cell padding and a direct uint8→int32 Gather cast. Orientation itself
is cheaper as `obj_col_occ[axis_col] == 0`: the marker axis column is disjoint from
the object only for left/right reflection.
⭐ **A fixed-background runtime-colour tail needs only one signed feature.** Encode
object/background as raw uint8 0/2 with activation zero-point1; implicit padding is
the zero-point and therefore effective0. Runtime weights use raw0 for the selected
class, raw2 for green, and raw1 for neutral rows with weight zero-point1. The
products are object `(-1)*(-1)`, background `(+1)*(+1)`, and padding0. Building the
match against shifted classes `[1,2,1,4,5,6,7,8,9,10]` preserves duplicate class0
rows0/2 while deleting the 200B quadratic feature carrier.
⭐ **An inactive Gather may return identity instead of zero when the final merge is
set union.** Replace wrapped out-of-range and inactive reflection indices with the
destination identity index. Those fallback pixels are exactly `obj_any`, so
`Or(ref_h,ref_v)` already contains the original object and the active reflection.
This deletes both one-cell Pad sources and the separate `ref` union carrier.
⭐ **Encode the three padding states before the spatial Pad.** `Where(obj_all,0,255)`
creates a 64B signed code, Pad fills the 10x10 border with255, and terminal
ConvInteger treats implicit padding as zero-point1. Runtime weights 0/255/1 yield
selected-object +1, green-background 64516, and padding/neutral0. Direct float
class comparison preserves duplicate shifted code1 rows0/2 and removes the scalar
colour Cast, QLinear scale, and 100B bool mask.
⭐ **Cast an integral FLOAT crop once, then share its INT8 profile bank.** When a
FREE-input FLOAT Conv must remain FLOAT but its values are exact small integers,
one 64B INT8 cast can feed multiple reductions and scalar comparisons. Here four
length-8 FLOAT profiles became INT8 (128B→32B), while the colour and axis-overlap
scalars became one byte each. The shared cast cost 64B, so the exact net saving
was 38B: memory1003→965, params55 unchanged, bundled267/267. Comparing the INT8
colour against `[1,2,1,4,5,6,7,8,9,10]` preserves duplicate class-0 rows0/2.
⭐ **A finite generator can replace global reflection bookkeeping with a proved local
separator.** Exhaust the complete bounded sprite/position/orientation generator,
deduplicate every receptive-field patch, reject opposite-label collisions, and
verify a stored INT8 separating kernel with strict margins. Here two QLinearConv
nodes replace every profile, ArgMin/ArgMax, coordinate, Gather, and union node.
The signed mask uses object127/background-128; a supported UINT8 `Where` builds
zero-point128 tail weights 129/127/128 for selected/green/neutral, preserving
duplicate shifted class1 rows0/2 through mixed INT8/UINT8 ConvInteger.

## ADOPTED 20260715T132331Z
- cost: 2524 -> 2149 (points 17.3272)
- source: candidates/task062/poly_convinteger_tail.onnx
- note: polynomial ConvInteger FREE-output tail; preserves selector [0,1,0,3,4,5,6,7,8,9] thresholded multi-hot

## ADOPTED 20260715T134249Z
- cost: 2149 -> 1691 (points 17.5669)
- source: candidates/task062/signed_conv_qlinear.onnx
- note: signed cropped Conv geometry + shifted two-feature QLinear FREE-output tail; preserves duplicate class-0 multi-hot

## ADOPTED 20260715T134957Z
- cost: 1691 -> 1515 (points 17.6768)
- source: candidates/task062/mask_feature_fold.onnx
- note: compact object-mask polynomial features + direct signed-code object profiles; thresholded multi-hot preserved

## ADOPTED 20260715T135727Z
- cost: 1515 -> 1475 (points 17.7036)
- source: candidates/task062/u8_signed_indices.onnx
- note: u8 reflection-index arithmetic with u8->i8->i32 sign extension; thresholded multi-hot A/B 267/267; preserves duplicate class-0 mapping

## ADOPTED 20260715T140129Z
- cost: 1475 -> 1409 (points 17.7494)
- source: candidates/task062/color_coded_cell_conv.onnx
- note: color-coded signed geometry Conv removes 10-channel presence selection; thresholded multi-hot A/B 267/267; duplicate class-0 decode preserved

## ADOPTED 20260715T140807Z
- cost: 1409 -> 1393 (points 17.7608)
- source: candidates/task062/sentinel_clamped_indices.onnx
- note: one-cell zero-pad sentinel clamp removes signed index extension; thresholded multi-hot A/B 267/267; duplicate class-0 decode preserved

## ADOPTED 20260715T141107Z
- cost: 1393 -> 1388 (points 17.7644)
- source: candidates/task062/axis_overlap_orientation.onnx
- note: axis-column object overlap replaces max-bound orientation test; thresholded multi-hot A/B 267/267; duplicate class-0 decode preserved

## ADOPTED 20260715T142611Z
- cost: 1388 -> 1293 (points 17.8353)
- source: candidates/task062/dynamic_sign_qlinear.onnx
- note: dynamic one-feature signed QLinear FREE-output tail; thresholded A/B 267/267; duplicate class-0 mapping preserved

## ADOPTED 20260715T144641Z
- cost: 1293 -> 1097 (points 17.9997)
- source: candidates/task062/identity_fallback_reflection.onnx
- note: identity-fallback direct reflection Gathers; thresholded A/B 267/267; duplicate class-0 mapping preserved

## ADOPTED 20260715T144845Z
- cost: 1097 -> 1058 (points 18.0359)
- source: candidates/task062/compact_sign_convinteger.onnx
- note: 8x8 pre-Pad signed code plus dynamic ConvInteger FREE-output tail; thresholded A/B 267/267; duplicate class-0 mapping preserved

## ADOPTED 20260715T150522Z
- cost: 1058 -> 1020 (points 18.0724)
- source: candidates/task062/shared_int8_profiles.onnx
- note: shared INT8 code-plane profiles; thresholded A/B 267/267; duplicate class-0 rows 0/2 preserved

## ADOPTED 20260715T154301Z
- cost: 1020 -> 921 (points 18.1745)
- source: candidates/task062/generator_complete_local_qlinear.onnx
- note: generator-complete local QLinear reflection mask; thresholded 267/267; duplicate class-0 rows 0/2 preserved; UINT8 zero-point tail
