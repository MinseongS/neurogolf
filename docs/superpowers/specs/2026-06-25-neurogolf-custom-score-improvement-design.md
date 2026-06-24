# NeuroGolf custom score improvement design

Date: 2026-06-25

## Goal

Improve the current NeuroGolf score through local custom solver work, not by wholesale adoption of a newer public base.

The work has two tracks:

1. Recover and validate the highest-ROI known candidate, starting with task017.
2. Search for genuinely new algorithmic primitives that can beat current public nets, instead of only replaying known tasklog wins.

Existing dirty user changes in `src/custom/task064.py`, `src/custom/task110.py`, `src/custom/task198.py`, and `src/custom/task370.py` are out of scope and must not be overwritten.

## Current context

The latest repo notes say public-base adoption has been the reliable leaderboard lever, but the user explicitly selected custom/research work.

The manifest still has several `ext:kojimar7113` tasks where local tasklogs mention prior wins or near-wins. Among them, task017 is the best first target:

- Current manifest: `task017`, external method, about 15.30 points.
- Tasklog: a 15.62-point candidate, about +0.32, fresh 500/500, marked as an adopt candidate.
- Current source tree has no `src/custom/task017.py`, so the candidate is not present as a rebuildable custom solver.

`/tmp/arc-gen` is missing in this session, so fresh-generator verification requires restoring the generator source before any final adoption claim.

## Track A: task017 recovery

Implement `src/custom/task017.py` from the documented algorithm:

- Recover one of 106 valid `(mod, length, offset)` parameter tuples by template matching against fixed sample cells.
- Use a compact sample set, with the tasklog's safe floor being 15 samples.
- Rebuild the 21×21 periodic formula directly into a one-channel label plane.
- Use a uint8 padded sentinel carrier and final `Equal` into the free `output` tensor.
- Keep the graph within allowed ONNX constraints accepted by the current project harness.

Acceptance criteria:

- Stored evaluation passes.
- Fresh generator verification passes after `/tmp/arc-gen` is restored.
- `python -m src.adopt 017` adopts the solver against the current network.
- The resulting points improve task017 by at least +0.25 stored points and pass the fresh/adopt gates, or the result is reported as superseded if the current base already wins.

## Track B: new algorithm search

The new-algorithm work should avoid repeating known dead ends. The search target is not “highest memory task first” by itself. Many top-memory tasks are proven connectivity, ambiguity, or correspondence walls.

Prioritize tasks matching at least one of these patterns:

1. A tasklog says an old “infeasible” verdict was wrong, but the current manifest still shows an external method.
2. The incumbent has a large forced fp32 entry plane plus multiple medium planes, where a different primitive could remove a whole carrier.
3. The rule has data-dependent copy/stamp/crop behavior that might be expressible by runtime Conv weights, Gather-based coordinate maps, or single-channel label carriers.
4. The task is deterministic and fresh-verifiable; ambiguous generators are not candidates for exact score improvement.

Candidate primitive families to explore:

- Runtime-kernel Conv: extract a small reference pattern and use it as a dynamic Conv kernel for matching or stamping.
- Coordinate-map Gather: replace repeated shift/stamp planes with direct source-coordinate maps.
- Sentinel label carrier: collapse 10-channel output assembly into one uint8/int label plane plus final `Equal`.
- Parameter-template recovery: recover a small set of global scalars by matching fixed probes, then rebuild closed-form output.
- Directional carry with opset >=12 uint8 MaxPool where allowed, replacing fp16/fp32 scans.
- Generator-bound active-canvas cropping only when the crop avoids larger dtype conversion or padding tails.

Initial new-search candidates after task017:

- task182: tasklog records a sizable exact generalizing custom win, but current repo state needs reconciliation.
- task025: ledger says not infeasible, stalled previously, but current gain may be marginal.
- task367 or task198: prior notes say false infeasible and closed-form exists, but task198 is currently dirty and must be avoided unless the user explicitly allows merging with those changes.

## Verification

Every accepted solver must pass, in order:

1. Stored harness evaluation.
2. Fresh generator verification in an isolated process after generator source is available.
3. `src.adopt` against the current installed network.
4. Manifest/network diff review to ensure only intended task files changed.
5. Submission zip creation only after local adoption succeeds.

No Kaggle submission should be made without explicit final confirmation of the file being submitted.

## Error handling

- If `/tmp/arc-gen` cannot be restored, stop before claiming a fresh-generalizing win.
- If task017 no longer beats the current base, record it as superseded and move to task182/new-search.
- If a new candidate is ambiguous or requires connectivity/flood unrolling above the incumbent memory floor, bail and document the reason.
- If verification touches dirty user files, stop and ask before proceeding.
