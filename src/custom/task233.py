"""task233 (ARC-AGI 97a05b5b) — INFEASIBLE for a +0.3 closed-form improvement.

RULE (derived fully from the generator tasks/task_97a05b5b.py):
  The input holds ONE large solid RED (=2) rectangle (the "red box", size tall x wide,
  at random position (brow,bcol)) PLUS 1..5 "sprites" scattered OUTSIDE the box.  Each
  sprite k is a 3x3 block painted in a distinct colour `colors[k]`, with `counts[k]`
  of its 9 cells overwritten by RED (its shape); counts are common.sample(range(4,9))
  so EVERY sprite has a DISTINCT pixel-count in {4..8} (=> distinct colour<->count).
  Inside the red box the SAME sprite shape (after one of 4 rotations) is carved as BLACK
  (=0) holes at a random (irows[k],icols[k]) offset; the 3x3 inside FOOTPRINTS are kept
  non-overlapping but may be EDGE-ADJACENT (common.overlaps spacing=0).

  OUTPUT = the red-box crop (size wide x tall, placed TOP-LEFT of a fresh 30x30 canvas,
  rest = colour 0), with every carved hole-footprint REPAINTED: the 3x3 footprint becomes
  the matching sprite's COLOUR where there is no hole, and stays RED where the hole is —
  i.e. an exact copy of that sprite's outside 3x3 block.  The colour of a given inside
  footprint is found by MATCHING shapes (= matching the distinct pixel-count) to the
  outside sprite of the same count.

WHY INFEASIBLE (cannot beat the deployed kojimar 13.76 by >=+0.3 with allowed ops):
  1. Data-dependent localize + crop + reposition of a variable-pos/size rectangle to the
     top-left (feasible alone, but heavy: scalar-offset Gathers, WORK<=20).
  2. MULTI-OBJECT SHAPE CORRESPONDENCE: up to 5 sprites, each a RANDOM shape, one of 4
     rotations, at arbitrary offsets, must each be matched to its carved hole-footprint
     and recoloured.  Per-cell colour assignment requires EITHER connected-component
     labelling (NonZero/Unique/Loop — all BANNED) OR a runtime-template correlation bank
     of ~5 sprites x 4 rotations = ~20 passes whose Conv weights are the runtime-extracted
     sprite shapes, each materialising full-grid planes.  Inside footprints can be
     edge-adjacent, so a fixed-window hole-count sum is corrupted by neighbours (no clean
     per-cell count without grouping).
  3. The closest documented analog (task158, single CANONICAL sprite, NO rotation
     ambiguity, separable stamping) floored at mem 152769 / 13.01 pts.  task233 is strictly
     harder (5 RANDOM shapes, 4 rotations each, PLUS a data-dependent crop+reposition on
     top), so an EXACT net necessarily carries more full-grid correspondence planes than
     the analog and cannot land below the existing 13.76 by +0.3.  The deployed net already
     solves it exactly with TopK+ScatterND+ArgMax+5 Conv (492 nodes ~75KB) — a near-optimal
     kojimar pipeline for this structure.

  Determinism is guaranteed by construction (output is a pure deterministic function of the
  input grid), so this is NOT a stochastic-collision wall — it is an OP/COMPLEXITY wall:
  the exact rule needs banned component-ops or a heavy runtime-template correlation bank,
  and no encoding gets a strictly-smaller EXACT net than the deployed one by the required
  margin.  Any approximate (count-window / partial) build would LEAK on the private set.

This module intentionally provides NO build(): the task is declared INFEASIBLE.
"""

INFEASIBLE = True
REASON = (
    "multi-object shape-correspondence (up to 5 random sprites x 4 rotations matched by "
    "pixel-count to edge-adjacent carved hole-footprints) PLUS a data-dependent crop+"
    "reposition; exact recolouring needs banned component ops or a heavy runtime-template "
    "correlation bank; deployed kojimar net (492 nodes, ~75KB, 13.76) is already near "
    "optimal and cannot be beaten by +0.3 without a strictly-smaller EXACT net, which the "
    "task158 analog (single canonical sprite) already shows floors near 13.0."
)
