# task208 — 890034e9

**Rule:** Fixed 21x21 random 2-colour field (rare black pixels) with two identical
`h x w` all-black rectangular holes. One hole already has a 1-cell-outside frame of
`boxcolor` (box0); the other has none (box1). Output draws the same frame around the
un-framed hole. `boxcolor` is the rarest non-bg colour (appears only in box0's frame);
`h,w` = box0 frame inner dims; locate EVERY `h x w` all-black rect (generator guarantees
exactly the two holes are `h x w` all-black) and stamp a frame around each (box0 re-stamp
is idempotent, so box1 need not be singled out).
**Current:** 15.13 pts, custom:task208, mem 19057, params 236
**Target tier:** A (closed-form: separable bbox reductions + corner-Conv + ConvTranspose ring)

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | crop count+ring to 21x21 active, fp16 ring, slice B | A | 18699 | 236 | 15.151 | 200/200 | dup fp32 slice+cast cancelled gain |
| 2 | reductions on full B (no slice), fp16 count path, holef pad->25 + ConvTranspose->31 + slice[1:31] | A | 16853 | 248 | 15.253 | 200/200 | best validated, MARGINAL |

## Best achieved
15.253 @ mem 16853 params 248 — adopt? N (MARGINAL). Beats prior 15.13? +0.12 only (< +0.3 bar).
Stored eval pass 266/266, fresh 200/200.

## Irreducible-floor analysis
Dominant intermediates:
- **B = boxcolor 1x1 Conv plane, 3600B fp32 [1,1,30,30]** — the entry colour plane that
  yields `h,w` via tiny row/col ReduceMax+ramp reductions. Forced to 30x30 by the 30x30
  input (Conv preserves input spatial). Every 1-D-profile alternative for h,w is STRICTLY
  WORSE: `ReduceSum(input,axis=3)`→[1,10,30,1] is 1200B per axis PLUS a 1200B Mul (or a
  1200B Transpose for a MatMul channel-contraction), i.e. ~4800B for both axes vs 3600B for
  the single Conv. So B is the floor for h,w extraction.
- **black slice 1764B fp32 + cnt 1764B fp32 [1,1,21,21]** — Slice of input ch0 is mandatorily
  fp32 (Slice preserves dtype); the count Conv on it is the second full plane. Casting black
  to fp16 first (882) makes cnt fp16 (882) but then black0(1764)+black16(882)+cnt(882)=3528 ==
  the fp32 path (1764+1764). No net gain — the fp32 slice is the price either way.
- **ring path ~5.9KB** (holef0 882 + holef25 1250 + ringbig31 1922 + ringbb31 961 + ringb30 900):
  the -1 frame offset (frame sits one cell OUTSIDE the hole, above/left of the corner) cannot be
  produced by ConvTranspose alone (no negative output offset), so a +1-padded ConvTranspose plus a
  `Slice[1:31]` is required, forcing a >=30 fp16 plane + its bool.

To beat +0.3 needs mem+params <= ~14328 (shave ~2770B); the only >2KB lever is B, and removing
it costs MORE. Conclusion: task is pinned ~15.25, MARGINAL.

## OPEN ANGLES (re-attack backlog)
- Eliminate the ring -1 slice by shifting the holeb CORNER to (y-1,x-1) inside the count Conv
  (asymmetric pads) so a 24-padded ConvTranspose lands at exactly 30x30 with no post-slice —
  saves ~1KB on the ring (ringbig31->30, drop ringbb31). Tried direction analysis only; the
  pad arithmetic to move the fire-point -1 (needs a top CROP, not pad) was not closed out.
  Even if it works (~15.32) it stays MARGINAL.
- Fuse black-count + boxcolor into ONE multi-output Conv (would still be >=2 full planes; the 1x1
  boxcolor + 5x5 black kernels differ, so a single Conv can't serve both — likely a dead end).

## INSIGHT (transferable)
⭐ Cropping full-canvas planes to a generator-fixed smaller active region (here 21x21) only pays
off for planes whose op OUTPUT can be sliced WITHOUT first materialising the full plane. A Conv/
ConvTranspose output is locked to its input spatial size, so slicing it AFTER still pays the full
30x30 fp32 (the pre-slice `Bfull` dominates). The win only lands when you can feed the op a
smaller input (e.g. a 21x21 single-channel SLICE of input for the count Conv) — but a one-channel
Slice is still fp32 (Slice preserves dtype), so the "fp16-after-entry" lever buys nothing when the
fp32 slice itself is the entry plane. Net lesson: count the ENTRY fp32 plane, not the post-cast
copy — casting to fp16 after a mandatory fp32 slice ADDS a plane rather than shrinking one.
