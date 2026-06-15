# task020 — 11852cab

**Rule:** Complete a symmetric "blast" of 4 concentric diamond rings around a center: ring0=center
(colors[0]); ring1=4 diagonal neighbors (colors[1]); ring2=4 axis cells at dist2 (colors[2]); ring3=4
diagonal cells at dist2 (colors[3]). INPUT shows all rings full EXCEPT one ring which keeps only one cell;
OUTPUT completes all rings. Task = detect center + 4 ring colors, re-stamp the full fixed pattern.
**Current:** 15.71 pts (custom:task020, adopted from gen:vyank6322 14.14), mem 10708, params 160
**Target tier:** B (label-map). Center+colors detection is non-local-ish but bounded; floor ~3600.
10708 leaves headroom.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | center via stamp-correlation==10, ring color via max-over-ring, restamp uint8 L on 10×10 crop → Pad → Equal | B-ish | 10708 | 160 | 15.71 | 200/200 (+1500/1500) | ADOPTED (+1.57) |

## Best achieved
15.71 @ mem 10708 params 160 — adopted? **Y**. Beats prior 14.14? Y (+1.57).

## Irreducible-floor analysis
NOT at floor. mem_profile: **4000B fp32 input crop [1,10,10,10]** dominates (1000 elem × 4) + 900 uint8 L
Pad + 3×400B fp32 Conv + ~25 × 200B fp16 planes (correlation/color reductions). The 4000B crop is one-hot
(values 0/1) held in fp32 → 4× waste; the fp16 planes are already lean.

## OPEN ANGLES (re-attack)
- **Crop input as uint8 or fp16, not fp32** (one-hot is 0/1). 4000→1000 (uint8) or 2000 (fp16). Conv needs
  float, but ORT runs Conv in fp32 internally regardless of declared dtype — feed a fp16 crop, or do the
  correlation Conv straight on a smaller-dtype crop. ⇒ mem ~7–8k ⇒ ~16.2 pts.
- **Avoid the full [1,10,10,10] crop**: center detection only needs the present-mask (ReduceMax over
  channels → [1,1,10,10], 100 elem). Color reads need per-ring argmax but only at 13 fixed offsets — gather
  those 13 cells, don't carry the whole 10-channel crop. ⇒ could drop the 4000 entirely ⇒ ~17 pts.
- The 3 fp32 Convs (400 each) → fp16 (200) if exact (counts ≤13, colors ≤9 — fp16-exact). −600B.

## INSIGHT (transferable)
⭐ Carrying the full [1,10,H,W] one-hot crop in fp32 is the single most common avoidable cost. Two fixes:
(1) downcast the crop (one-hot ⇒ uint8/fp16 exact); (2) better, REDUCE to a present-mask [1,1,H,W] for
geometry and only Gather the few cells whose COLOR you actually need. Detection rarely needs all 10
channels carried full-size.
