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
