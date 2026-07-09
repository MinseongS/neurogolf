---
deployed_cost: 3967
logged_costs_match: match
migrated: 2026-07-09
---

# task377 — eb5a1d5d

## S5 win — TopK-width re-fit (LANDED +0.008)
**Before:** mem 8230, params 150, total 8380, pts 15.966.
`TopK` width K=13 (theoretical gen max). K-dependent: top_values/top_idx/inner_colors/active_inner
([1,1,13,1]) + colors4/colors (14) + colors41 (41=K+28).
**Measured:** instrumented `inner_top_rows` selection over 310k fresh instances → max true selected
rows = 6 (depth 6 seen 12×; 7 never). Bundled max 4.
**Change:** K 13→**9** (empirical 6 + margin 3); resized all K-dependent value_infos. TopK feed kept fp16.
**After: mem 8162, params 150, total 8312, pts 15.975.** evaluate fail 0; `fresh_verify 377 "" 3000`
fail 0. See [[neurogolf-topk-width-refit]].

## FLOOR siblings (same lever, no room): task46 (K16, empirical max 16/160k = tight),
task361 (K15, already BELOW safe 16 — under-provisioned), task285 (K33 = max+1, tight).

## S9 (2026-07-03) — TopK K=9→6 shrink (+0.006) ADOPTED; einsum angle FLOOR
200k fresh: max depth 7 (not 6 per old 310k note), active_rows=depth−1; but codeK core
11×11 caps correct geometry at depth 6 → K margin above 6 = pure waste, zero private-LB
downside. K=6 bit-identical on every realizable input (div 0/4000 vs live onnx; 6000+800
uncached fresh 0/0/0). mem 8162→8111. Angle (a) floor: grid_f 2916 = 27²×4 native
detection read (already cropped); alternatives ≥5832B. DO NOT re-probe reformulation.

## 2026-07-08 — output-coupled fp16 recast ADOPTED (+0.140846 local)

Candidate: `reports/candidates/task377/task377_fp16_output_coupled.onnx`, built by
`reports/candidates/task377/build_fp16_output_coupled.py`.

Current active task377 is a free-output `Einsum` graph, not the old source-exact
TopK graph. The remaining fp32 planes flagged by
`reports/candidates/fresh_sweep/scan_deployed_fp16.py` were output-coupled:
`OHf/Xs/X` [1,5,10] and `R` [1,5,30]. Recast `OHb->OHf` and `ab->R` to fp16,
converted `SK` to fp16 so `OHf*SK` stays homogeneous, and made the graph output
fp16. This is valid in 8000 mode because `output` is free and scorer thresholds
`>0`.

Gate:
- incumbent: memory 3552, params 1015, cost 4567, points 16.573388, bundled fail=0.
- candidate: memory 2952, params 1015, cost 3967, points 16.714235, bundled fail=0.
- full active manifest after adoption: 400/400, local 7274.668531.
- `scan_unsigned_topk.py submission/overfit_nets`: clean.

Adopted into `submission/overfit_nets/task377.onnx`; backup:
`reports/candidates/task377/adopt_backup_727465_rescan/task377.onnx`.
Packed and submitted as Kaggle **54462993**, completed at displayed publicScore
**7274.79**, with message
`active 7274.668531 task377 fp16 output-coupled tensors cost 4567->3967 after task295 rescan fail=0 topk clean`.

Mechanism note: the S9 "einsum angle floor" was too broad for the deployed graph.
The detector read may still be structural, but output-coupled fp32 masks feeding
the final free `output` can be recast when all values are fp16-exact and homogeneous
operands are adjusted.
