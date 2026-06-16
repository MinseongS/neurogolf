# task064 — 2c608aff

**Rule:** One solid `boxcolor` axis-aligned rectangle (>=3x3) plus sparse `dotcolor`
pixels (density 0.02, never inside the box) on a `b` background; grid is top-left
anchored and width,height in [8,24] (so active canvas <= 24x24). Each dot whose ROW
lies in the box's row-span shoots a horizontal ray toward the box, filling every bg
cell from itself up to (but not into) the box edge with dotcolor; each dot whose
COLUMN lies in the box's col-span does the same vertically. A dot in a corner region
(both row- and col-offset nonzero, i.e. diagonal to the box) stays a single pixel.
Equivalently: in a box-row the fill is `[leftmost-left-dot .. box_left-1] U
[box_right .. rightmost-right-dot]` (rays all terminate at the box, so per-side runs
union to one span); same per box-column. The box itself and the original dots remain.
**Current:** 14.99 pts, custom:task064, mem 22104, params 66.
**Target tier:** detection/fill (4-direction box-blocked prefix/suffix). NOT S
(output color per cell is not a fixed linear/permutation of input — it is a
data-dependent directional fill). NOT A (row⊗col separable): horizontal fill in a
box-row depends on the DOTS in that row, which differ row-to-row, so the column
mask is not shared across rows — genuinely 2-D per direction.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | (incumbent) 30x30 fp16 dyn-Conv run-signal + 4 box-blocked prefix/suffix Convs (-30 sentinel) + per-row/col span-threshold Greaters, Where->output | fill | 22104 | 66 | 14.99 | 200/200 | baseline, already optimized |
| 2 | same algo on 24x24 active-canvas slice (E0->fp16-30->Slice24), cum/Greater at 24, fp16 Pad fill back to 30 | fill | 22188 | 60 | 14.99 | (eval ok 267/267) | WASH — pad-back tax (fp16-30 pad 1800 + cast 1152 + bool30 900) exactly cancels the canvas savings on the 4 cum planes |

## Best achieved
14.99 @ mem 22104 params 66 — adopted? **N** (== current P, no gain). Beats prior
14.99? **NO** (MARGINAL/at-floor).

## Irreducible-floor analysis
Memory (22104) is a tight sum of 11 full-canvas planes that cannot be removed
without breaking correctness:
- 3600 E0f fp32 [1,1,30,30] — the entry run-signal from the dynamic-weight 1x1
  Conv (the documented 3600B fp32 colour/value-plane floor; the Conv on the fp32
  input must output fp32).
- 1800 E0 fp16 [1,1,30,30] — cast of E0f so the 4 prefix/suffix Convs run in fp16.
- 7200 = 4 x 1800 cumL/cumR/cumT/cumB fp16 [1,1,30,30] — the FOUR directional
  box-blocked sweeps are intrinsic (a dot can be left/right/above/below the box;
  all four are needed and none is a flip-reuse of another without an extra plane).
- 2400 = 2 x 1200 R,C fp32 [1,10,30,1]/[1,10,1,30] — ReduceMax keep-channel needed
  for the EXACT box discriminator (count == nrows*ncols AND count>=9). Verified
  empirically that max-count-non-bg is NOT the box in 125/5000 fresh instances
  (dots can outnumber the box), so the rectangle test (hence R,C) is mandatory.
- 6300 = 4 x 900 Greater bools + 3 x 900 OR bools — the threshold + 4-way union.
  Consolidating via Max(cumL,cumR) trades a saved bool for a new fp16 Max plane of
  equal cost (wash), so this layer is already minimal.

Reaching the +0.3 bar (15.29) needs mem+params <= ~16450 — i.e. dropping ~5700 B,
roughly three full fp16 planes. There is no reformulation that removes a direction,
the entry plane, or the exact rect detector, so the floor for a CORRECT net sits at
~21-22k => ~14.99. This is the public/incumbent floor and it is already reached.

## OPEN ANGLES (re-attack backlog)
- Drop R,C (2400) only if a single small Conv could prove "solid rect" exactly —
  it cannot (a 2x2/3x3-solid Conv response is not an exact discriminator vs. rare
  adjacent dots; the bbox-area==count test is the only exact one and needs R,C).
  Even if removed, mem ~19700 => ~15.11, still < +0.3 (MARGINAL).
- 24x24 active canvas: tried (attempt 2) — pad-back to the mandatory 30x30 Where
  mask cancels the gain. The final mask must be [1,1,30,30] (broadcast vs input),
  and ORT Pad rejects bool, so the 24-route always pays a fp16-30 pad (1800) +
  bool30 (900) that exceeds the per-plane 30->24 savings (1800->1152).
- No Tier-A: per-row horizontal fill is not a shared col-mask (dots differ per row).

## INSIGHT (transferable)
⭐ **A box-directed ray-FILL ("dots aligned with a box shoot rays to it") is a
4-direction box-blocked prefix/suffix sum, NOT a connectivity wall** — build a run
signal `E0 = +1 at dots, -30 at box cells` (one dynamic-weight 1x1 Conv: weight
vector = dot_onehot - 30*box_onehot), then four all-ones Convs with asymmetric SAME
pads give cumL/R/T/B; `cum>0` means "a dot precedes with no box in the prefix" (the
-30 sentinel poisons any prefix crossing the box). Span-gate via
`thr = Where(rowspan, 0.5, 1e4)` from `ReduceMin(E0)<-1` so only box-rows/cols fill.
⭐ **But it sits at a ~21-22k floor** because four full directional planes + the
fp32 entry + the exact rect detector are all irreducible. **Negative result for the
24x24-active-canvas lever:** when the final op is a `Where` whose MASK must be the
full 30x30 output shape, cropping the intermediate sweeps to the active region does
NOT help — the mandatory pad-back of the fill mask to 30x30 (and ORT's bool-Pad
rejection forcing an fp16 pad + re-threshold) costs as much as the cropping saved.
The active-canvas lever only pays off when the FINAL output plane is itself the
small canvas, not when it must be padded back for a 30x30 broadcast.
