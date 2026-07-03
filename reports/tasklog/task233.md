# task233 — 97a05b5b

## Current live

Exact-preserve baseline: `memory=59147`, `params=565`, `points=14.002711715792984`.
Source is now live-exact via `reports/scripts/live_to_exact_source.py`.

## Semantic rule discovered

The generator creates:

- one large red rectangle; the output is that rectangle cropped out;
- up to 5 outside 3×3 sprites, each with one non-red background colour and red shape pixels;
- inside the red rectangle, black pixels mark the rotated red-shape pixels;
- output starts as all red, then each matched 3×3 sprite bbox is filled with its background colour,
  and the matched shape pixels remain red.

Important corrections from the initial hypothesis:

- The transform group is rotation-only (`rotates in {0,1,2,3}`), not full dihedral.  This halves the
  template bank vs an 8-orientation matcher.
- The red rectangle bbox can be found by the dominant red row/column block.  Naive full-grid column
  threshold sometimes over-includes outside sprite columns; robust detection should first isolate the
  dominant red row block, then compute the dominant column block inside those rows.
- Sliding 3×3 matching creates false positives.  A better semantic representation is:
  1. extract outside sprite masks/colours;
  2. find black connected components inside the red rectangle using 8-neighbor connectivity;
  3. for each component, test full 3×3 consistency against the 4 rotated sprite masks;
  4. scatter the corresponding coloured 3×3 patch.

## Reference progress

- Python reference with rotation-only component matching passes stored 4/4.
- It passed fresh prefix 100/100 with simple bbox detection.
- Larger fresh run exposed a bbox edge case: one failure at fresh 26 where `components=6` but
  `sprites=4`; immediate cause was over-wide red-box column detection, not the sprite matcher.
- `reports/scripts/task233_reference_probe.py` now captures the reference solver.
  Bbox mode comparison:
  - `simple`: stored 4/4, fresh failed at 118 (`pred=(18,20)`, target `(18,18)`).
  - `row_first`: stored 4/4, fresh failed at 107 (`pred=(11,9)`, target `(19,9)`).
  - `iter`: stored 4/4, fresh 200/200.  Robust bbox is:
    first dominant red row block, then dominant red column block inside those rows,
    then recompute row block inside those columns, then recompute columns.
- 2026-06-29 larger fresh showed `iter` is still not sufficient: a black hole can
  split the true red rectangle rows and the lower block wins (`pred=(10,8)`,
  target `(14,8)` in one reproduced case).  Added `iter_bounded_span`, which only
  fills holes inside the initial dominant row/column block.  It passes stored 4/4
  and improved bbox stability, but fresh still failed at 271/300 with the same
  output shape (`pred=(13,19)`, target `(13,19)`) and different content.  The
  remaining issue is not bbox; it is component/template assignment.

## ONNX compiler direction

Potential lower-memory rewrite:

- use row/column red counts to crop the box and output via final Pad/Equal;
- outside sprite extraction can be represented as a small fixed set of 3×3 gather windows after
  locating non-box coloured components;
- black component handling is the hard part.  But because there are at most 5 sprites and every
  component fits in 3×3, a scan-free compiler can avoid full flood-fill by enumerating 3×3 black
  windows and requiring exact full-window match against 4 rotations.

Expected win if implemented: current graph has many repeated ScatterND/Gather/Where chains and
full-canvas planes. A direct 4-rotation 3×3 matcher plus one uint8 label plane should plausibly cut
memory by tens of KB, making this a high-value semantic rewrite target.

Current gate: do not implement ONNX yet.  The Python semantic reference must pass
large fresh content checks first; 200/200 was not enough for this generator.

## 2026-06-30 deep reauthor attempt → FLOOR (confirms prior "assignment is hard")

Re-verified baseline: ok=True, pass=266, fail=0, **mem 59147, params 565,
points 14.0027**.

New lever found but proven insufficient: `counts = common.sample(range(4,9), k)`
⇒ every sprite has a DISTINCT popcount (4..8), so COLOUR assignment is trivial
(hole-cluster popcount n ↔ the unique outside sprite with n red pixels). This
removes the rotation-hash→colour matching (the `[5,324]` mul/equ/or planes ≈16 KB,
mat45/wrot/pow2w/scorew). BUT colour is not the cost driver — exact PLACEMENT is,
and the popcount trick does nothing for it.

Stateless numpy reauthors built and pushed to 9/80 wrong (best). The residual
failures are structural and match the prior session's "component/template
assignment" wall — three generator-allowed configs defeat every stateless form:
1. **count-4 shapes with <3×3 bbox** (generator only blocks 2×2-for-4, 2×3-for-6):
   two 3×3 windows contain the holes; only the shape orientation picks the right
   one → mis-placement.
2. **Adjacent patches** (`overlaps(..., margin 0)`): patches may touch, so any
   isolated-window / 5×5-ring test drops or merges a sprite.
3. **Disconnected shapes** (pixels sampled from all 9 cells): 8-connected components
   split one patch into popcount-1 fragments.
p() handles all three only via its ordered, consume-once two-pass `popitem()`
matching — i.e. exactly the incumbent's 2-pass-TopK + 9× ScatterND placement/scatter
unroll (400 nodes). No cheaper stateless graph reaches 0/1500.

Floor structure: con1 `[1,1,30,30]` fp32 = 3600 (per-cell colour read; sprites can
be anywhere in ≤30×30 → unavoidable). Exact placement keeps a ~324-wide 3×3-hash
plane and the stateful per-sprite scatter unroll regardless. Safe micro-golf nil:
the 17 Gathers' int64 indices feed ScatterND (must stay int64) or int64 arithmetic
(converting adds Cast planes for ≤160B tensors → net-negative); big planes already
fp16/uint8/bool; params already 565.

**Verdict: FLOOR. Incumbent kept (59147 / 565 / 14.00). Lowest mem reached by any
correct candidate: none below incumbent (stateless forms top out ~11% wrong on
fresh).**

## S8 (2026-07-02, late) — counting-model re-encode (+0.223) ADOPTED, bit-identical
Sprite-window detector (11 planes ~6.2KB) → ONE Conv (w=16·[v≥1]−[v==2], sprite ⟺ v>135.5);
{0,2} profiles feed ReduceMax directly (comparators doubled); 4D Gathers for crop; hole hash
via Cast+Conv(−2^k, b=511) fp16; scorew deleted (TopK asc-index tie-break = scan order);
4 chained ScatterND → 1 (sequential last-wins verified). 32796+446 vs 40808+722 → +0.223.
Fresh 2500×2 + 400 re-run div 0; 600 vs live onnx div 0. Walk-einsum proper N/A (TopK/argmin
rounds on [5,3] 60B planes = not a walk polynomial; memory was in parallel mask parades).
Floors: con1 3600, mul103 3240+equ97 1620 (TopK feed), vspr 3136.

## S9 (2026-07-03) — fold 2nd pass: FLOOR re-confirmed (incl. crop lens)
13a N/A (no walk chain; output = ScatterND sprite placement). fp16 recast of vspr/con1
net-negative (Conv dtype-match: input cast 6000B or output cast +1568). Hash-matcher
Equal→Cast(f16)→TopK minimal (324 positions inherent). pub bundled-override machinery
MEASURED load-bearing: pruned cand fails exactly the 3 rotated bundled examples;
pub 1387B+165p < 4-rotation matcher blowup (+~9700B) — pub already optimal encoding.
Crop lens checked by orchestrator: generator width=wide+randint(2,10), wide≤20 → grids
reach 30×30. NOT croppable. Floor final. DO NOT re-probe.

## S11 (2026-07-03) — signed-priority overlay (playbook 15) scout: KILL — output = content-matched 3x3 sprite stamping (rotation-hash assignment); cost = 3600B detection read + 3136B sprite-window Conv + ~4860B hash-match/TopK planes + 9x ScatterND placement. No label/priority carrier to delete. S9 FLOOR stands under the new lens.
