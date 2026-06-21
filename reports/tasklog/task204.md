# task204 — 868de0fa

**Rule:** Several non-overlapping (gap >= 1) HOLLOW blue square boxes of side L in [3,10]
on a size x size grid (size = 10..20). Each box's (L-2)x(L-2) interior is filled orange (7)
if L is odd, red (2) if L is even; the 1px blue outline is preserved; in-grid bg stays 0,
off-grid stays all-zero. (Equivalently: interior = parity of horizontal-wall crossings above
a cell; colour = side-length parity = (topwall_row+botwall_row) mod 2.)
**Current (deployed kojimar):** 15.62 pts, uint8 perimeter-anchor net, mem 11364, params 454
**Target tier:** detection-but-closed-form — uint8 anchor detection, no fp arithmetic needed.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | prior parity net (fp16 TriL MatMul + 2 MaxPool colour) | A | 16540 | 483 | 15.26 | 268/268 | correct but fp16 planes too fat; BELOW deployed |
| 2 | anchor detection (deployed) + label+Equal output | det | 12304 | 436 | 15.55 | 268/268 | Where/Greater output path heavier than deployed conv |
| 3 | anchor detection + 1x1 QLinearConv output on [grid,blue,red,orange] | det | 10604 | 454 | 15.69 | 268/268 | **beats deployed**: drop bg fp32 slice, occupancy grid plane |
| 4 | + uint8 Min(rowb,colb) grid (drop bool And + uint8 Cast) | det | 10244 | 454 | **15.72** | 3000/3000 | **ADOPTED** |

## Best achieved
15.72 @ mem 10244 params 454 — adopted. Beats deployed 15.62 by **+0.10**. fresh 3000/3000 exact.

## Irreducible-floor analysis
Dominant intermediates: out_state Concat [1,4,20,20] uint8 (1600B) + blue fp32 slice (1600B,
the entry floor — Slice inherits the fp32 input dtype) + the 8 perimeter anchors (~1724B,
shrinking valid-conv outputs) + 8 dilation fills (8x324=2592B). The anchor/fill bank is the
size-3..10 detector and is inherent (all 8 sizes occur). The 4-channel out_state is the minimum
for a linear output conv emitting ch0=grid-blue-red-orange, ch1=blue, ch2=red, ch7=orange.
fp32 blue slice is unavoidable (any channel slice of the fp32 input is fp32).

## OPEN ANGLES (re-attack backlog)
- Reduce the 8 size-anchors: a size-independent corner+parity detector + bounded clipped fill
  would cut ~4300B of anchor/fill planes, but the clip is a flood (expensive). Unclear net win.
- Pack red+orange into one out_state channel (needs a value-carrying fill, blocked: QLinearConv
  sharp threshold forces fire-value=1, can't also encode 2 vs 7).
- The parity reformulation (attempt 1) floors ~14-15k because MatMul/Mod/Add force fp16 full
  planes; the uint8-only anchor route is structurally cheaper for this task.

## INSIGHT (transferable)
⭐ When a deployed uint8-anchor net assembles its 10-ch output via Concat[bg_slice, ...] + a
final 1x1 conv, you can often shave the fp32 BACKGROUND slice (1600B+cast) by replacing it with
a 1-D-occupancy `grid` plane (ReduceMax row/col profiles -> Greater -> uint8 `Min` broadcast,
~440B) and letting the output conv compute ch0 = grid - blue - red - orange (int8 out_w, uint8
output clamps the subtraction at 0). uint8 `Min` broadcasts both axes and builds the grid plane
directly, skipping the bool `And` + uint8 `Cast` (saves ~360B). The whole 10-ch expansion lives
in the FREE output conv — no per-cell Where/Equal label plane. (task204 11364 -> 10244, +0.10.)
⭐ QLinearConv perimeter-anchor SHARP threshold: weight-scale = 1/(2*peri-1) makes conv==peri
round to 1 and conv==peri-1 round to 0 (one missing ring pixel kills the anchor) — exact box
detection with no comparison op.
