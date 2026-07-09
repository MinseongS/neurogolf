# task071 — REGIME-CRACK verdict: FLOOR (2026-07-09)

Deployed `submission/overfit_nets/task071.onnx`: pass 265/0, memory 2627 + params 55 = **2682**.

## Rule (verified numpy on all 265 bundled)
Sprite is bilaterally symmetric about a vertical axis; a box marker (different colour)
occludes part of it; drop the marker and mirror-complete the sprite in its own colour S.
- axis `2a = first_fg + last_fg column of the topmost foreground row` (top row is un-occluded).
- `M[h,w] = F[h,w] AND F[h,2a-w]` (F = foreground incl. marker). AND recovers the true shape
  because the shape is symmetric and the marker's mirror is sprite.
- S = single global non-background colour of the output. Output = S on M, background elsewhere.
All grids are 16x16; output bbox is exactly rows2:12, cols1:13 (the deployed crop rows2:13,cols1:14).

## Why the free-output Einsum lever prices HIGHER than deployed
Deployed is already a tight *cropped* net: reflection is a 13-col Gather on an 11x13 bool crop
(143B), placement is Pad, channel-expansion is a FREE `Equal(color30, channel_values)`.
Non-free tensors: bg_active fp32 11x13 **572** (the forced fp32 channel-0 read = floor, cf task033),
color30 u8 30x30 **900** (canvas placement), color16 u8 16x16 **256**.

Every regime-crack reformulation costs MORE:
1. **Placement into 30x30** = ~900B (Pad) OR ~720 params (Prow[11,30]+Pcol[13,30] matrices).
   Pad(900 mem) already beats the matrices; folding placement into the free Einsum needs the
   matrices as params.
2. **Background channel-0** must cover the FULL 16x16 grid (the ring rows{0,1,13-15}/cols{0,14,15}
   is always in-grid background → channel0=1), not just the crop. Unifying the full-grid bg term
   with the cropped-shape term in ONE free Einsum forces either a p-axis that **doubles** the
   placement matrices (RowP[2,11,30]+ColP[2,13,30] = 1440 params) or a stacked [2,30,30] plane
   (1800B) — both exceed deployed's color16+color30+color_active (1156).
3. **color16 (256B) is load-bearing, NOT redundant**: the two Pads use different constants —
   inner value 0 (in-grid ring → channel0) then outer value 255 (off-grid → all-zero via Equal).
   Merging to one Pad breaks correctness (verified: fail 265/265). Off-grid needs 255, the ring
   needs 0, at different radii → two constants → two Pads mandatory.
4. Reflection cannot be a polynomial `1-E^2` predicate (fragile on horizontal runs; verified dead)
   and a dynamic 30x30 reflection matrix as an fp32 Einsum operand is 3600B (> whole deployed net).

Net: deployed's cropped-Gather + two-constant Pad + free Equal is the cheapest known mechanism.
The 572B fp32 read and the 900B canvas placement are independent-of-reformulation floors here.

## Falsification / reopen triggers
Negative verdict = "no cheaper mechanism found by hand-analysis on 2026-07-09 vs deployed 2682".
Reopen if: (a) a public/cristianoc net for task071 shows sub-2000 mem; (b) a way to produce the
in-grid background channel-0 without a full-grid plane/matrices (e.g. cheap input-routing that
keeps 10-channel intermediates out of node outputs); (c) a cheaper-than-572B fp32 foreground read.
