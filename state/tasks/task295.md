---
deployed_cost: 393
logged_costs_match: match
migrated: 2026-07-09
---

# task295 — bbc9ae5d

## 2026-06-29 final-expansion screen

Current source score: 17.619744 @ mem 1557 params 47.

Dominant tensor is `full_label` [30,30] uint8 = 900 B, followed by the 9x18
working label/mask planes.  This is already the cheaper final-output route for
this rule: replacing the label plane with a compact one-hot before padding would
materialize roughly 9x18x10 bool cells, larger than the single uint8 label plane.

No rewrite adopted.

## 2026-07-03 S12 — UNKNOWN-bucket dossier

**Rule:** a 9×18 working region's per-column/row label is expanded to the final 30×30 output; the net builds a full label plane then one-hots it.

**Cost (grader mem 1557, params 47):** ops Where×4/ReduceSum×3/Cast×3/Less×3/ArgMax/Mul/Add/Equal/Pad. Counted intermediates: `full_label` [30,30] uint8 900B (full-canvas label), `filled` [9,18] bool 162B, `default_label`/`small_label` [9,18] uint8 162B each. Params: `col_index` [1,18] fp16 36B, `pad_spec` [4] int64 32B, `row_index` [9,1] fp16 18B. Output [1,10,30,30] bool 9000B is FREE.

**Blocker class:** full-output-carrier. The 900B `full_label` [30,30] uint8 is the dominant counted plane. The prior log established that a compact one-hot before padding would materialise ~9×18×10 bool cells (>900B) — the single uint8 label plane is already the cheaper final-expansion route.

**Lever:** no lever visible (log-confirmed floor; index params already fp16).

## 2026-07-08 — REGIME CRACK ADOPTED: direct free-output label route

The prior "full-output-carrier floor" verdict was false under the new regime-crack
mechanism.  The 900B `full_label` Pad and final Equal can be removed without building
the 9x18x10 one-hot plane: keep the scalar/reduction prefix (`color_minus_one`,
`in_height`, `in_width`, `filled`), build only a dynamic 10-channel colour vector via
`ScatterElements`, and write the final 30x30 one-hot directly to the FREE graph `output`
with Einsum masks for the 9x18 active area.

Artifact: `reports/candidates/task295/regime.onnx`, builder
`reports/candidates/task295/build_regime.py`.  Backup:
`reports/candidates/task295/adopt_backup_727188/task295.onnx`.

Gate: isolated bundled `evaluate()` fail=0.  Cost `1604 -> 393`
(memory `1557 -> 174`, params `47 -> 219`), points
`17.61974421157354 -> 19.02619038813074`.  Full active manifest after adoption:
**400/400**, local **7274.210520**, unsigned TopK clean.  Submitted as **54461590**;
Kaggle completed at displayed publicScore **7274.65**.

Reopen/falsification note: this directly falsifies the 2026-06-29/2026-07-03
"label Pad is floor" analysis for tasks where the final label map is a low-rank
function of masks plus one dynamic colour scalar.  Reopen similar label-pad tasks
when the label can be expressed as a small sum of separable masks and dynamic channel
vectors, without materializing a counted 10-channel small one-hot.

## 2026-07-08 — ⭐ REGIME CRACK ADOPTED (batch 3, 900B mask → free-output Einsum)
- 1604→393 (+1.41, batch best). One 9-operand Einsum with input itself as operand (bshw): per-channel counts read free, kills ArgMax/Equal color-extraction; bounded black run = rank-2 product of linears.
- Bundled fail=0, fresh-gated, unsigned-TopK clean, deployed-gated. Candidate: reports/candidates/task295/regime.onnx. Backup: reports/candidates/fatmid_adopt_backup/task295.onnx.bak.
- ⭐ Memory neurogolf-regime-crack-freeoutput-einsum. ⚠️ Concurrent-session collision risk on candidate dirs — always re-measure on-disk before adopt.

## ADOPTED 20260715T021858Z
- cost: 393 -> 377 (points 19.0678)
- source: candidates/task295/factor_v_g_rank2.onnx
- note: exact combined rank2 factorization of final-Einsum V[4,10] and G[4,2,3]; no added nodes, two latent operands

## ADOPTED 20260715T114710Z
- cost: 377 -> 343 (points 19.1623)
- source: candidates/task295/runtime_winner.onnx
- note: B2 square + shared rank-2 factors; rotate terminal Einsum operands for 1.17x isolated runtime
