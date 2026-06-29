# Task066

## Mechanism

Generator `2dd70a9a` is a marker-routed hidden path task.

- Input preserves red endpoint pair `2`, green endpoint pair `3`, and cyan marker/noise `8`.
- Hidden blue path cells are black in input and must become green in output.
- In canonical orientation the red pair is on the right side and the green pair is on the left side.
- Two families exist:
  - `S`: right vertical from red pair, horizontal bridge, left vertical to green pair; deliberate cyan turn markers at `(mid-1,left)` and `(mid,right+1)`.
  - `U`: right vertical down from red pair, bottom bridge, left vertical up to green pair; deliberate cyan markers at `(bottom+1,left)`, `(bottom,right+1)`, plus anti-side-entry marker at `(red_top,right-1)`.
- Flip/transpose produce four effective orientations.

## 2026-06-28 semantic probe

Python solver using dihedral normalization, endpoint pairs, zero-only path painting, and marker checks:

- stored validation: reached 3/4 initially; after duplicate-corner fix reached 4/4 before strict generator-range filters.
- fresh random: about 99.7%+ exact before final tie-break work.
- main failure mode: random cyan noise can accidentally satisfy the wrong U/S marker pattern, so marker checks alone are insufficient.
- useful tie-break: include generator parameter constraints and candidate-family priority, but validation examples are not perfectly captured by strict random-range filters, so range checks must be soft rather than hard.

## Current graph bottleneck

`task066` current exact-preserve graph is already a semantic-ish path scorer, but expensive:

- 301 nodes, roughly 19 KB profiled intermediates.
- top memory contributors are full `20x20` input slices/reductions (`Slice` channel plane, row/col `ReduceMax`) and boolean path masks.
- Simple dtype conversion cannot produce a large win; the real win requires compiling only a small set of S/U candidate masks and avoiding broad scan-style preserved graph machinery.

## Reusable insight

Register as `marker_routed_hidden_path_compiler`: solve endpoint-pair hidden path tasks by generating a bounded set of candidate path masks from endpoints and deliberate marker pixels, then choose with soft generator constraints to suppress random-noise false markers.

## Open angle

Build a source-owned ONNX compiler for the canonical S/U families:

1. Reduce to four generator orientations, not full arbitrary graph replay.
2. Extract endpoint row/col scalars from row/col presence.
3. Build S and U masks from row/col comparisons.
4. Validate marker pixels with `GatherElements`.
5. Use soft legality/tie-break instead of strict range rejection.
6. Emit a one-channel uint8 label plane and final output, avoiding early 10-channel expansion.
