# High-score frontier analysis

Generated after the 2026-06-28 global inventory pass.

## Score frontier

Current total: `7178.329899498155` across 400 tasks, average `17.945824748745387`.

The current distribution shows that very high per-task scores are possible:

- 2 tasks score `25.0` with `mem=0, params=0`.
- 6 tasks score at least `23.0`.
- 26 tasks score at least `21.0`.
- 48 tasks score at least `20.0`.

The common property of the 20+ group is not “slightly smaller graphs”; it is
near-elimination of intermediate tensors:

- `25.0`: total counted cost `0`.
- `23.39`: counted cost `5`.
- `22.70`: counted cost `10`.
- `21.60`: counted cost `30`.
- `20.0`: counted cost about `148`.

So a 20+ transition requires `mem + params <= ~148`, and a 23+ transition
requires single-digit counted cost. A full `[30,30]` uint8 label plane is already
900 bytes and therefore structurally caps the task below the 20+ band.

## Implication for the 8000 path

Capping every task at 900 bytes would only raise the total by about `+10.67`.
Capping every task at 500 bytes would only raise the total by about `+40.90`.
Capping every task at 148 bytes would raise the total by about `+246.56`.
Capping every task at 64 bytes would raise the total by about `+554.47`.
Capping every task at 32 bytes would raise the total by about `+872.46`, enough
to cross 8000.

Therefore, 8000 is not reachable by shaving kilobytes from a few worst tasks.
It requires a broad mechanism that moves many tasks from full-canvas internal
state to scalar/vector/final-output-only execution.

## Current high-score mechanisms to mine

The 20+ group is dominated by:

- `Gather` single-op or tiny-LUT models.
- `Einsum` single-op symbolic models.
- `Conv` single-op local stencils.
- `RoiAlign` / `MaxRoiPool` tiny-parameter special ops.
- Tasks where one-hot expansion happens only as the final graph output.

These are the mechanisms that can plausibly create explosive score movement.

## Low-score candidates with high-score upside

Not every low-score task is equally valuable. Prioritize candidates where the
current score is low but the semantic rule may fit a high-score mechanism.

High-upside candidates:

- `task092`: simple axis-aligned stick filling. Current graph is only 16.145
  because it materializes several `[30,30]` label/scatter planes. If it can be
  compiled as final-output-only predicates or a tiny symbolic op chain, this is
  a model for many scatter/line tasks.
- `task233`: semantic breakthrough already found. Rule is in-context 3x3 sprite
  reconstruction under four rotations inside a dominant red box. Current graph
  is exact-preserve and huge. A source-owned compiler may convert a 14.0 task to
  a much higher band, but it is not a guaranteed 20+ task unless component logic
  avoids full-canvas scans.
- `task349`, `task367`, `task396`, `task074`, `task138`, `task202`, `task204`:
  tagged as local-stencil/conv-heavy. If the rule is truly local and output is
  thresholded, single `Conv` / `QLinearConv` floors can move these toward the
  20+ family.
- `task025`, `task064`, `task017`, `task222`, `task328`, `task208`, `task055`:
  one-hot-final candidates. The question is whether the full 10-channel planes
  can be delayed until graph output, not merely reduced by a few hundred bytes.
- `task366`: template/dot reconstruction with LUT/scatter structure. There is
  likely medium upside, but reaching 20+ requires deleting full template/label
  planes, not only dtype tuning.

Lower high-score probability:

- `task286`: true flood-fill/checkerboard maze propagation. Existing bit-parallel
  representation is already the right family. It may still yield incremental
  savings, but it is unlikely to become 20+ without a new flood-fill primitive.
- ambiguity/information-loss wall tasks such as `task118`, `task209`, `task018`
  should not be first targets for 20+ frontier work unless a public teacher
  reveals a hidden generator shortcut.

## Revised priority rule

Do not sort purely by lowest score. Sort by:

1. Can the task avoid all full-canvas intermediate tensors?
2. Can the final graph output be the only full `[1,10,30,30]` object?
3. Can the task be expressed by one of the known high-score families:
   `Gather`, `Einsum`, `Conv/QLinearConv`, `RoiAlign/MaxRoiPool`, or tiny final
   `Equal`?
4. If yes, deep-dive even when the current score is not among the bottom 20.

The immediate research seed should be `task092`, because its semantics are
simple enough that failure would reveal a real ONNX/scorer floor, while success
would propagate to many scatter/line/one-hot tasks.
