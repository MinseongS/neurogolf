# task012 — 0962bcdd

**Rule:** size=12 grid, two distinct colours c0,c1 (1..9). Two centres placed; a
gravity reflect/transpose is applied EQUALLY to input and output, so the
input→output map is gravity-INVARIANT (gravity is just a symmetry of the whole
example, not a per-cell motion). INPUT: each centre is a 5-cell plus — centre=c0,
the 4 orthogonal neighbours=c1. OUTPUT: each centre grows a 5x5 stamp — c0 at the
centre and the 8 diagonal cells (dist 1 & 2: (0,0),(±1,±1),(±2,±2)); c1 at the 8
orthogonal cells (dist 1 & 2: (±1,0),(0,±1),(±2,0),(0,±2)). The two stamps never
overlap (centres 6 rows apart, stamps reach ±2). Output colour per cell is a
deterministic LOCAL function of input.

**Current:** 16.64 pts, label-map + structural centre Conv, mem 4170, params 103
**Target tier:** B (label map + final Equal). Tier S blocked: output colours
c0,c1 are random per instance, a fixed Conv cannot route them to the correct
output channel. Tier A blocked: the 5x5 X/plus stamp is not a row⊗col separable
rectangle.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | colour-image Conv→A30, centre=Equal(A,c0), stamp Convs, label-map | B | 9066 | 102 | 15.88 | 200/200 | works, A30 [1,1,30,30] f32 = 3600 dominates |
| 2 | structural centre detector (slice ch0 12x12, count nonbg orth nbrs ≥4), drop A30 | B | 5610 | 102 | 16.35 | 200/200 | A30 eliminated |
| 3 | fp16 for all Conv-path 12x12 planes | B | 4170 | 103 | 16.64 | 200/200 | BEST |

## Best achieved
16.64 @ mem 4170 params 103 — adopted? N (build-only per task brief). Beats prior
15.56? Y (+1.08).

## Irreducible-floor analysis
Two intermediates dominate: **L [1,1,30,30] uint8 = 900** (the single-channel
output label map that drives the final Equal — irreducible for a label-map at the
30x30 output footprint; uint8 is already the smallest dtype) and **bg [1,1,12,12]
f32 = 576** (channel-0 slice; Slice preserves the f32 input dtype). Everything
else is fp16 (288) / bool (144) / uint8 (144) on the 12x12 active canvas. To go
lower you must leave the label-map family entirely, which the data-dependent
output colours forbid (S/A blocked). 4170 is at/near the label-map floor for this
rule.

## OPEN ANGLES (re-attack backlog)
- bg slice as fp16: Slice keeps f32; a Cast adds a node but bg stays 576 either
  way (the Slice output is the f32 plane). Could try Conv ch0-selector directly
  to fp16 — marginal (~288 saved → ~16.7).
- Fold L0/L12 into one Where via arithmetic label (c0st*c0 + c1st*c1) to drop one
  144B uint8 plane — sub-0.1pt.
- The 900B L is the hard floor; only escapable by a non-label representation,
  which the random per-instance colours block.

## INSIGHT (transferable)
⭐ **Gravity in ARC-GEN apply_gravity is a SYMMETRY of the whole example, not a
per-cell force.** It reflects/transposes input AND output identically, so any
input→output rule is gravity-invariant — build it in the canonical frame and it
holds in all 4 gravities for free, IF the detector is reflection/transpose
symmetric (a plus/X kernel is). **Structural, colour-independent center detection
(count non-background orthogonal neighbours via a 4-orth plus Conv, threshold ≥4)
beats colour-matching: it removes the [1,1,30,30] colour-value image (3600B) — the
single biggest win here (15.88→16.35), because the only 30x30 plane you truly need
is a single-channel SLICE of input ch0 (576B at 12x12), not a full colour Conv.**
fp16 on every small Conv-path 12x12 plane (values 0..4, exact) then halves them
(16.35→16.64).
