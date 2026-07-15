---
deployed_cost: 1389
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task297 — bd4472b8

## 2026-06-29 final-expansion screen

Current source score: 17.763661 @ mem 1350 params 39.

The graph builds a compact [1,9,14,6] bool one-hot tensor (`compact`, 756 B) and
pads it directly to the free graph output.  This beats the alternative label-map
route: a [1,1,30,30] uint8 label alone would cost 900 B before the final `Equal`.

No rewrite adopted.  This is a useful counterexample to the usual label-map rule:
when the active footprint is very small and only non-black colour channels are
needed, compact one-hot + final Pad can be cheaper than full-canvas label + Equal.

## ADOPTED 20260712T060946Z
- cost: 1389 -> 965 (points 18.1279)
- source: candidates/signed_poly_wave/global_scan/task297/candidate.onnx
- note: signed header-band free-output predicate; fresh 1600/1600 div0 in isolated batches

## ADOPTED 20260715T031632Z
- cost: 965 -> 939 (points 18.1552)
- source: candidates/task297/factor_poly_exact.onnx
- note: direct exact selector-rank poly factorization

## ADOPTED 20260715T051632Z
- cost: 939 -> 938 (points 18.1563)
- source: candidates/task297/factor_poly_fast_order_unsqueeze.onnx
- note: runtime-safe exact factor: operand-order search 8x faster plus Reshape(shape_one)->Unsqueeze saves 1 param
