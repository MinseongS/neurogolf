---
deployed_cost: 1329
logged_costs_match: match
migrated: 2026-07-09
---

# task281 — b548a754

**Rule:** Input has a rectangular box (1px outer-colour border + inner-colour interior) near an edge plus a single cyan(8) dot offset along the box's axis (within the box's perpendicular span). Output stretches the box so its edge reaches the dot's line: the output rectangle = the bounding box of ALL non-background cells (box ∪ dot), drawn as a 1px outer frame with an inner-colour interior; the cyan dot is removed. xpose/flip applied to both grids — "union-bbox → framed rect" is orientation-independent.
**Current:** 16.01 pts, separable bbox + count-based colours, mem 7315, params 745
**Target tier:** A (separable rect/interior masks routed into the FREE Equal output; colours are tiny [1,10,1,1] arithmetic — no 2-D colour/neighbour plane).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | colf 13x13 plane + 3x3 deep-conv inner-colour + triangular masks | A | 17688 | 739 | 15.18 | 200/200 | exact but 6760B multi-ch slice killed it |
| 2 | Conv→colf30 fp32, slice 13x13, fp16 downstream | A | 11462 | 752 | 15.59 | 200/200 | 3600B colf30 dominates |
| 3 | drop 2-D colf entirely; colours via cnt==area; 1-D occ | A | 10421 | 806 | 15.67 | 200/200 | bbox Where planes (rlo/rhi/clo/chi) heavy |
| 4 | colours via MIN/MAX count (ring always > interior), fp16 rc/cc | A | 8291 | 745 | 15.89 | 200/200 | beats +0.3 |
| 5 | Conv occupancy on fp32 rc_f directly (drop fp16 rc/cc planes) | A | 7315 | 745 | 16.01 | 500/500 | ADOPTED |

## Best achieved
16.01 @ mem 7315 params 745 — beats prior 15.54 by **+0.47**. 266/266 stored, 500/500 fresh.

## Irreducible-floor analysis
Dominant intermediates are the two per-channel 1-D spatial reductions `rc_f`/`cc_f`
([1,10,30,1] fp32 = 1200B each). They must be fp32 (ReduceSum of the fp32 input) and
feed the [0,1,…,1] channel-Conv that yields non-bg row/col occupancy. Using the 1-D
[1,10,30,1] shape (not a [1,10,30,30] plane) keeps each at 1200B, beating the
"two fp32 reductions ≈15.8" ceiling. The 900B uint8 30×30 label map (Pad target for the
output shape) is next; Pad rejects bool so the carrier must be uint8 before Equal.

## OPEN ANGLES (re-attack backlog)
- Fuse the two occupancy reductions: a single channel-Conv on input gives [1,1,30,30]
  (3600B) serving both axes — WORSE than 2×1200, so not pursued. A genuine 1-D-only
  occupancy that avoids the 30-length per-channel plane would drop ~1200B → ~16.3.
- The five 338B 13×13 fp16 work planes (rect_f/intr_f/Lcol/L_outer/L_diff) could fold
  into fewer ops (~−1000B → ~16.2) but each is cheap.

## INSIGHT (transferable)
⭐ "stretch a bordered box to a marker dot" = the union bounding box of all non-bg
cells, drawn as a framed rect — orientation-free, no flood-fill. For inner-vs-outer
colour of a box, the OUTER ring count > INNER interior count for ALL box sizes in
[3,5]² (perimeter 2(t+w)−4 vs area (t−2)(w−2)), so inner = MIN-count / outer = MAX-count
present non-cyan colour — pure [1,10,1,1] arithmetic, no bbox-area Where planes and no
2-D colour plane at all. Rect mask = non-strict triangular prefix∧suffix-OR; eroded
interior = the SAME with STRICT triangulars (k<r, k>r) — erodes the run 1 cell each end
for free, no separate erosion Conv.


---

## 2026-07-08 — 8000-mode adoption (public re-mine, LB-confirmed 7264.26)

**+0.4921 LB** (mem 1208, par 140; from franksunp-7249.50). Mechanism: wide ArgMax/TopK index-plane decode (53 nodes) replaced by **arithmetic coordinate decode** (76 nodes: Log/Floor/Div/Reshape). More nodes but far smaller COUNTED tensors — bit-pack/arithmetic index math instead of materialized one-hot/index planes. This is the bit-pack-encoding lever realized.

Gate: isolated `evaluate` bundled fail=0 + strictly cheaper than our incumbent + uint8-TopK scan clean.
Artifact: `submission/overfit_nets/task281.onnx` (backup in `.minmerge_backup/`). Source dumps + notebook code under `reports/candidates/public_mine_20260708/`.
This is a direct 8000-mode ONNX overlay (constant dataset => permanent). Source `src/custom/task281.py` NOT regenerated: the public net is optimizer-emitted (base64 in-notebook), no per-task Python builder to lift; the mechanism above IS the transferable insight.

## 2026-07-08 — zero-compare bool-cast tail ADOPTED

Re-ran `reports/candidates/zero_compare_to_bool_cast_probe.py` on the current active
overfit set.  `Greater(pres, 0)` is a nonnegative presence test, so replacing it
with `Cast(pres -> BOOL)` preserves bundled semantics and removes the now-unused
scalar zero initializer.

Artifact: `reports/candidates/task281/task281_zero_compare_tail.onnx`.
Bundled gate: `266/266`, fail=0.  Cost: `1348 -> 1344` (memory `1208 -> 1204`,
params `140` unchanged).  Active backup:
`submission/overfit_nets/.zero_compare_tail_backup/task281.onnx`.

## 2026-07-08 — 8000-mode public tails after regime crack
- `jackelysia/neurogolf-7250-25-v23-lucifer-unscored-carry` 11:39 first improved the
  active overfit overlay: cost `1344 -> 1340` (memory `1204 -> 1200`, params unchanged
  `140`), fail=0. Submitted as **54460887**, completed at **7270.82**.
- `boristown/neurogolf-chatgptloop` 11:51 then improved it again: cost `1340 -> 1329`
  (memory `1200 -> 1188`, params `140 -> 141`), points
  `17.79957510705504 -> 17.807817941286753`, fail=0. Source:
  `reports/candidates/public_mine_20260708/boristown_chatgptloop_1151/submission/task281.onnx`.
- Applied to `submission/overfit_nets/task281.onnx`; backups in
  `reports/candidates/public_mine_20260708/jackelysia_7250_25_1139/adopt_backup_726828/`
  and `submission/overfit_nets/.minmerge_backup/task281.onnx`. Included in submission **54461084**.

## ADOPTED 20260713T144125Z
- cost: 1329 -> 1304 (points 17.8268)
- source: candidates/public_dumps/20260713_highroi/king77578_neurogolf-udit22-single-zips-public/task281/task281.onnx
- note: Udit22 public-LB min-merge; bundled fail=0

## ADOPTED 20260713T150907Z
- cost: 1329 -> 1304 (points 17.8268)
- source: candidates/public_dumps/20260713_highroi/king77578_neurogolf-udit22-single-zips-public/task281/task281.onnx
- note: isolated residual-public LB probe; bundled fail=0

## ADOPTED 20260715T081653Z
- cost: 1329 -> 1304 (points 17.8268)
- source: candidates/public_dumps/20260715_refresh/lucifer19_chimera-safe-boost-caddies/submission_black_cat_bbi_v3/task281.onnx
- note: lucifer19 black-cat BBI v3 public min-merge; bundled fail=0
