# task319 — ce602527

**Rule:** Grid (15-19 square-ish) bg color + two conway sprites (3-5 wide/tall). Sprite idx0 ("magnified")
is drawn small in colors[0]; a 2x-upscaled copy of sprite0 (each pixel→2x2 block) is drawn elsewhere in
`magcolor`, deliberately placed so exactly one 2-cell edge strip is off-grid (`some_hidden` required; can also
collide with other pixels). Sprite idx1 is a distractor in colors[1]. Exactly 3 non-bg colors always present.
**Output** = sprite0's small copy rendered in its own (wide×tall) bbox, in colors[0]. So the whole task is:
identify WHICH of the two small sprites is the magnified one, output it in its own color.
**Current:** 14.58 pts, gen:thbdh6332, mem 33406, params 101 (a bloated full-canvas correlation net; also a
GAP-ATTRIBUTION task — scores less on Kaggle held-out).
**Target tier:** detection / correspondence — would need exact shape-correspondence.

## Attempts (numpy feasibility ceilings, ISOLATED fresh generator)
| # | angle | accuracy | outcome |
|---|---|---|---|
| 1 | identify magcolor by max pixel-count, sprite0 by count/size heuristic | ~60-90% | FAIL (mag!=max-count 7/500) |
| 2 | bbox-doubling match (2*spriteH/W ∈ {magH, magH+2}) unique winner | 302/500 unique&correct | FAIL (ambiguous, both sprites pass ~40%) |
| 3 | downsample mag blob /2, cross-correlate vs each sprite (all parities/shifts, asym penalty) | 263/300 ≈ 88% | FAIL (conway sprites coincide; clipping+collision noise) |
| -  | exact 2x2-clean-block test on mag blob | 469/500 | FAIL — mag blob not always clean (partial-cell clip / collisions) |

## Best achieved
No exact net. Best achievable discriminator ≈ 88% fresh — far below 200/200.

## Irreducible-floor analysis
Two compounding correspondence walls, neither closed-form:
1. **Magnified-blob identity is not a clean scalar.** Max-pixel-count picks the wrong color ~1.4% (the
   distractor sprite1 can out-count a heavily-clipped/colliding magnified blob). The "all 2x2 blocks uniform"
   test that would anchor it exactly fails ~6% because the off-grid clip can remove a partial (non-block-aligned)
   strip and the magnified copy can collide with pre-existing sprite pixels.
2. **sprite0 vs sprite1 needs true shape correspondence.** Both are conway sprites of overlapping size (3-5),
   so size/count/bbox give no separating margin; the only signal is matching the (occluded, possibly-collided)
   2x blob shape against each small sprite — a cross-correlation + argmax. Even an exhaustive numpy matcher
   (every parity, every shift, asymmetric mismatch penalty) tops out ~88%. Expressing that in ONNX is exactly
   the full-canvas correlation the public net already does (mem 33406) — bloated AND still wrong on held-out.
No separable row⊗col / count→pattern / bounded-unroll reformulation exists: the answer depends on a 2-D shape
identity, not on any per-axis profile or scalar. PARTIAL is also impossible — there is no exact sub-component
(both color readout AND geometry hinge on first solving the unsolvable identification).

## OPEN ANGLES (exhausted to INFEASIBLE)
- None that reach exactness. The matching is intrinsically a noisy 2-D template correlation; no closed form.

## INSIGHT (transferable)
⭐ "Magnified-sprite correspondence" (output = the small sprite whose 2x-upscale appears elsewhere) is a TRUE
correspondence wall when (a) the upscaled copy is deliberately edge-clipped AND can collide with other pixels,
so it isn't a clean 2x2-block image, and (b) the two candidate sprites are same-distribution conway shapes with
no size/count margin. Both the anchor (which color is magnified) and the selection (which small sprite matches)
require full 2-D cross-correlation — best fresh accuracy ~88%, no separable/scalar/bounded-unroll escape.
This matches the BUILD_PROMPT "shape-correspondence + global argmax across data-dependent components" floor.
