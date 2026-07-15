---
deployed_cost: 849
logged_costs_match: match
migrated: 2026-07-09
---

# task124 — 53b68214

**Rule:** Input is an H×10 grid (H 5..8) holding the TOP of a vertically (and optionally
diagonally) periodic sprite tiling. A sprite of height `tall` (1..3) repeats vertically with
period `tall`; if it repeats diagonally it also shifts right by `shift=(wide-1)*diag` (0..2) every
period. The 10×10 OUTPUT extends the same pattern over all 10 rows:
`out(r,c) = P[r%tall][c - shift*(r//tall)]` (0 if source col OOB), P = first `tall` input rows.
**Current:** 18.255941 pts, memory779, params70, cost849; rank-4/hash centered QLinear renderer
**Target tier:** B (closed-form periodic extension; pure index Gather of a recovered value plane)

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 9-cand 2D-mismatch detect + 10x10 value plane + Pad30 fp32 carrier | B | 23604 | 141 | 14.92 | 267/267 | working but heavy |
| 2 | 3-cand (leftcol-derived shift) + fp16 + uint8 carrier + padded-gather | B | 12834→10334 | ~120 | 15.32→15.75 | ok | iterating down |
| 3 | 8-row crop + 9-ch slice entry + per-cand scalar mismatch | B | 9540 | 122 | 15.82 | 267 | closing in |
| 4 | **1-D occupancy-bitmask consistency** `bm[r]==(bm[r-t]·2^s) mod 1024` | B | 7694 | 119 | **16.04** | 200/200 | ADOPTED |

## Best achieved
18.255941 @ **memory779 params70 cost849**. The 2026-07-15 recursive chain lowered the live
cost 1953->1409->1078->1057->945->939->849, with bundled267/267 at every adoption and final
fresh2000 raw/sign divergence0/off-grid positives0. Final SHA:
`4e4bafbb3d65046a1ec08a211de6c9951705b613777a2a3ece9f4c73f6041b25`.

## Irreducible-floor analysis
This is not an absolute floor. The largest remaining counted tensor is the load-bearing FREE-input
channel-0 `Slice [1,1,5,10]` at 200B. The final geometry has no per-cell INT32 index plane, and the
renderer has no valid plane, scalar colour crop, full 30x30 label carrier, or terminal Equal.
Rank preservation removed flat row carriers, and bounded scalar QLinear fingerprints removed the
cellwise p3 reduction. At cost849 the next +0.1 requires cost<=768, so a new exact
QLinear-preserving route must remove at least81 cost. A ConvInteger probe saved only one parameter
and remains outside scope because direct uint8 QLinearConv output is an explicit task invariant.

## OPEN ANGLES (re-attack backlog)
- Fuse the 200B float channel-0 crop and foreground predicate into a counted<=119B exact producer
  while preserving pinned ONNX/ORT legality.
- Jointly remove at least81 cost from the remaining compact period detector/row-bank path; scalar
  and initializer folds below this threshold are useful only when composable.
- Re-run the centered-mask/row-Slice signatures if a new public teacher or legal fused crop op lands.

## INSIGHT (transferable)
⭐ **A runtime-colour two-state mask can use quantized padding as a third zero-effective state.**
Store background/foreground as uint8 0/2 with x_zero_point1, so they become -1/+1 while implicit
QLinearConv padding is 0. Scatter a stored2 into a `[0,1,...,1]` weight base at w_zero_point1 to get
background -1, selected +1, others0. This writes the FREE one-hot output directly. Upstream, when a
small fixed number of fixed-length rows comes from a compact bank, split runtime starts and issue
one Slice per row instead of broadcasting `start + columns` into a full INT32 Gather map; keep the
bank rank-preserving when flattening is semantic-free. For bounded rows, a collision-free uint8
QLinearMatMul fingerprint can replace cellwise equality/reduction when the full reachable domain
and nonsaturating accumulator are proved.

## 2026-07-08 rescan — direct free-output label route rejected by measurement

Context: active overfit task124 is no longer the stale tasklog incumbent. Current
`submission/overfit_nets/task124.onnx` is bundled-clean at **memory 1879, params 74,
cost 1953, points 17.422878**.

Probe: `reports/candidates/free_output_label_runtime_probe.py` flagged task124 as a
2-value compact label tail (`color_crop` [1,1,10,10] -> `Pad(color_full)` -> `Equal`).
Built candidates in `reports/candidates/task124/build_free_output_variants.py`:

- foreground-only free-output `Einsum`: **rejected**, bundled fail 267/267 because it
  omitted the required channel-0 background inside the 10x10 output grid.
- corrected `bg+delta` free-output `Einsum`: bundled pass 267/267, but **cost worsened
  1953 -> 3024** (memory 1879 -> 2248, params 74 -> 776). The 10x30 row/column
  projection params plus two-term background handling cost more than the removed 900B
  label carrier.
- bool `Pad` variant: ONNX checker/ORT rejected bool Pad for this opset/path.

Negative verdict fields: what was run = task124 ONNX surgery plus bundled
`src.harness.evaluate`; tool/date = `build_free_output_variants.py`, 2026-07-08;
reopen trigger = a cheaper dynamic channel placement primitive, a bool-Pad-compatible
opset/output path, or a way to avoid the 10x30 projection params; falsification history =
the generic "900B label floor" was falsified by task295, so this rejection applies only
to task124's 10x10 background-bearing label route, not the whole mechanism family.

## ADOPTED 20260715T132706Z
- cost: 1953 -> 1409 (points 17.7494)
- source: candidates/task124/dynamic_qlinear_tail.onnx
- note: [mask,valid] runtime-colour padded uint8 QLinearConv; wzp=1 signed rows; off-grid zero

## ADOPTED 20260715T134753Z
- cost: 1409 -> 1078 (points 18.0171)
- source: candidates/task124/centered_qlinear_tail.onnx
- note: centered mask 0/2 with shared x/w zero-point 1; valid plane removed; padding state 0

## ADOPTED 20260715T135619Z
- cost: 1078 -> 1057 (points 18.0368)
- source: candidates/task124/scatter_weight.onnx
- note: ScatterElements runtime-colour weight; remove Cast Equal Where and channel ids

## ADOPTED 20260715T141226Z
- cost: 1057 -> 945 (points 18.1488)
- source: candidates/task124/centered_slice_geometry.onnx
- note: reuse centered uint8 top plane; replace 200B Gather index plane with five dynamic row Slices

## ADOPTED 20260715T141855Z
- cost: 945 -> 939 (points 18.1552)
- source: candidates/task124/shared_initializers.onnx
- note: share QLinear scale threshold, Slice step row index, and crop ends reshape shape

## ADOPTED 20260715T144530Z
- cost: 939 -> 849 (points 18.2559)
- source: candidates/task124/rank4_qlinear_hash.onnx
- note: rank-4 row bank plus exact uint8 QLinear row hash
