# Task124 Further Optimization Plan — completed

**Goal:** Preserve the direct padded uint8 QLinearConv renderer while reducing the adopted
`[mask,valid]` cost1409 graph and checking the next +0.1 opportunity.

**Constraints:** task124 only; test first; official gate before every adoption; `ng adopt` only;
bundled fail0; fresh/off-grid diagnostics0; no pack or submit.

## Completed stages

- [x] Replace `[mask,valid]` with one centered uint8 0/2 mask at shared x/w zero-point 1.
  Result: memory999, params79, cost1078; gate267/267; fresh2000 divergence0.
- [x] Replace runtime-colour `Cast->Equal->Where` with one `ScatterElements` weight update.
  Result: memory988, params69, cost1057; gate267/267; fresh2000 divergence0.
- [x] Reuse centered5 for detection, source rows, and rendering; replace the 200B broadcast
  Gather index plane with five fixed-length dynamic row Slices.
  Result: memory878, params67, cost945; gate267/267; fresh2000 divergence0.
- [x] Share existing initializers: QLinear scale as the one-hot threshold, Slice step as the
  row-1 Gather index, and crop ends as the bottom reshape shape.
  Result: memory878, params61, cost939; gate267/267; fresh2000 divergence0.
- [x] Preserve the compact bank and five Slice outputs as rank4; replace ten-cell p3
  Equal/Cast/ReduceMin with two nonsaturating scalar QLinearMatMul hashes.
  Result: memory779, params70, cost849; gate267/267; fresh2000 raw/sign divergence0,
  off-grid positives0, and exhaustive legal-state hash mismatch0/1428.
- [x] Rebuild `src/custom/task124.py` byte-identically and adopt every passing stage through
  `ng adopt`; synchronize DISCOVERY, insight, task, lever, and STATE records.

## Residual result

The next +0.1 from cost849 requires cost<=768. The remaining 200B float input crop is load-bearing;
the next route must remove at least81 cost while preserving the requested direct uint8 QLinearConv
output. The previously recorded "no exact >=90 fold" was falsified by composing rank preservation
(-81) with a bounded QLinear row hash (-9); below-bar folds must continue to be priced jointly.
