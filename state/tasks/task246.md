---
deployed_cost: 94
logged_costs_match: match
migrated: 2026-07-09
---

# task246 — a2fd1cf0

**Rule:** `common.hpwl`. Red dot (2) at (r0,c0), green dot (3) at (r1,c1) on a
black grid (size 10–20). Output keeps both dots and draws a cyan (8) L-path:
horizontal along row r0 strictly between the two columns, then vertical along
col c1 from r0 (inclusive) to r1 (exclusive). Corner (r0,c1) is cyan.
**Current:** 20.456705 pts, custom:task246 (single 69-operand fp32 Einsum), mem 0, params 94
**Target tier:** A (separable bilinear → free bool output) — path is a union of two
rank-1 (row⊗col) regions plus two single-pixel dots, so a single colour-index
plane suffices; the 10→1 reduction routes into the FREE output, no per-cell colour army.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | single-ch colour plane L=R@M@C, 3 feats [a,b,c], Equal→bool | A | 4740 | 1219 | 16.31 | 0/200 | off-grid ch0 wrongly True (no in-grid mask) |
| 2 | +sentinel feats [a,b,c,in,ones], 99 off-grid | A | 5820 | 1295 | 16.13 | 200/200 | correct but 4 convs + f32 concat heavy |
| 3 | width-18 convs, fp16 per-feature concat | A | 4320 | 815 | 16.46 | 200/200 | downstream fp16 |
| 4 | 1 conv/axis (v∈{0,1,2}, w=clip, g=v−w); fp16 clip/maxpool | A | 4080 | 457 | 16.58 | 200/200 | −2 convs (−360 params), all ops fp16 |
| 5 | inclusive `b` both axes (strict cols via bc−wc folded into M) | A | 4020 | 457 | 16.59 | 200/200 | drops the col Sub; +0.30 boundary |
| 6 | feed `v` directly (a=2w−v, c=v−w in M), drop g Subs | A | 3900 | 457 | 16.62 | 500/500 | ADOPTED |

## Best achieved
16.6205 @ mem 3900 params 457 — adopted? Y. Beats prior 16.29384? Y (+0.327).
evaluate() ok pass 266/266; ISOLATED fresh 200/200 and 500/500.

## Irreducible-floor analysis
Dominant intermediate is the single colour-index plane L=[1,1,30,30] fp16 = 1800B
(the per-cell entry plane; cannot go below fp16 per FLOOR_RESEARCH). Everything
else is tiny: two [1,1,5,30] fp16 MatMuls/Concats (300B each) and the f32 floor of
4 input-reading ops (2 ReduceMax in-grid + 2 colour Convs, 120B each). The two
ReduceMax (in-grid 0/1 mask) MUST be f32 max-reductions (a Conv sum would give a
data-dependent COUNT that breaks the −99·in·in / +99·ones·ones sentinel
cancellation), so the 480B f32 floor is structural for this construction.

## OPEN ANGLES (re-attack backlog)
- Eliminate the +99/in/ones sentinel pair (2 of the 5 features, ~120B of MC/concat
  + the 2 ReduceMax 240B f32) if off-grid could be masked without an in-grid term —
  but in-grid-bg and off-grid both have all path-features 0, so a constant (ones)
  AND an in-grid product seem unavoidable for a single-plane Equal route.
- The 1800B L plane is the hard floor; only a Tier-S spatial-copy escape (N/A here,
  the path is synthesized not copied) would beat it.

## INSIGHT (transferable)
⭐ An L-shaped / Manhattan path between two marked points is a SEPARABLE bilinear:
collapse the whole thing to ONE single-channel colour-index plane L = R@M@C with
per-axis features [w=occupancy, v∈{0,1,2} endpoint value, b=between-inclusive band,
in=in-grid occ, ones], and route via Equal(L, arange)→bool — strictly cheaper than
a 10-channel double-MatMul (1800B plane vs 2400B R10). Two levers stacked here:
(1) feed the raw conv VALUE v (1=red,2=green) as a feature so endpoint identities
a=2w−v, c=v−w fall out of M's integer coefficients — no Sub planes;
(2) after the one mandatory f32 entry (Conv/ReduceMax read the f32 input), Cast to
fp16 and run ALL downstream vector ops (Clip/MaxPool/Mul) in fp16 — ORT_DISABLE_ALL
runs fp16 Clip/MaxPool/Mul/Sub fine (only the 10→input cast is forbidden, 18000B).
Off-grid masking for a single-plane Equal route needs a +K·(ones·ones − in·in)
sentinel (value ∉ {0..9}) so off-grid matches no channel.


## S15 (2026-07-06) — ADOPTED from urad public bundle 7225.82 (sub 54367833): 2753 -> 2637 (+0.043)
Mechanism: Einsum/value_info crop. Gate fresh_verify 1500: inc=0/cand=0 (CLEAN). Source-owned via live_to_exact_source --write-src, re-measured fail=0. See [[neurogolf-urad-7225-bundle-vein]].
## 2026-07-09 — ⭐ REGIME CRACK ADOPTED (batch 5, opus)
- 2634→2226 (+0.17). L-path (2-marker↔3-marker, color-8). Global-state 4 scalar positions; mask = rank-2 separable H-segment Rr(h)·cHq(w) + V-segment q_row(h)·Gc(w) with signed quadratics q=(x-a)(b-x); channel mix Dfull[3,10,10]. One free-output Einsum, no 30×30 plane.
- Bundled fail=0, deployed-gated, TopK clean. Candidate reports/candidates/task246/regime.onnx. Confirms opus handles structured cracks. Memory neurogolf-regime-crack-freeoutput-einsum.

## ADOPTED 20260709T041323Z
- cost: 2226 -> 2067 (points 17.3661)
- source: candidates/public_dumps/20260709/7261-53-lb-compact-onnx-artifact-starter/nets/task246.onnx
- note: min-merge from nets

## ADOPTED 20260709T054751Z
- cost: 2067 -> 1128 (points 17.9718)
- source: candidates/task246/cand.onnx
- note: regime vein batch7: 11-operand free-output Einsum — out[c]=input[c]+m(h,w)*D[c]; endpoint-protected direction-symmetric quadratic (x-a+s/2)(b-x) bands, coefficient-vs-basis [1,h,h^2] trick (interval predicates = 12B coefficients); 500 fresh 0-fail. TRANSFERABLE: inclusive-start/exclusive-end quadratic kills dot-overwrite sign hazard without Min/Max

## ADOPTED 20260715T104647Z
- cost: 1128 -> 1126 (points 17.9736)
- source: candidates/task246/dtype.onnx
- note: dtype tail: output-coupled scalar Cast removal/recast, cost 1128->1126

## ADOPTED 20260715T112037Z
- cost: 1126 -> 1003 (points 18.0892)
- source: candidates/task246/free_input_gatefold.onnx
- note: FREE-input gate fold: delete Q/R2/G3 carriers; stage row factor to preserve 11-operand ORT plan; collapse [1,x,x^2] basis via repeated [1,x] operands

## ADOPTED 20260715T114534Z
- cost: 1003 -> 734 (points 18.4015)
- source: candidates/task246/direct13_permutations/perm_11.onnx
- note: direct FREE-input row/col gate fold; operand-order tuned 13-way output Einsum (50-run 0.449s vs original 6.99s), removes staged 360B row carrier; bundled 266/266 fresh 500/500

## ADOPTED 20260715T115013Z
- cost: 734 -> 590 (points 18.6199)
- source: candidates/task246/direct15_permutations/perm_13.onnx
- note: inline 72B row/col coefficient cores into tuned 15-way FREE-output Einsum; operand-order perm13; bundled 266/266 fresh 500/500, profile50 0.513s

## ADOPTED 20260715T115551Z
- cost: 590 -> 476 (points 18.8346)
- source: candidates/task246/channel_sign_cp.onnx
- note: rank-4 sign CP collapses dense E/F channel router; bundled 266/266, fresh 500/500; cost 590->476

## ADOPTED 20260715T115854Z
- cost: 476 -> 416 (points 18.9693)
- source: candidates/task246/axis_free_sum.onnx
- note: remove 60-param axis selector via full-grid sums and positive term rescaling; bundled 266/266, fresh 500/500; cost 476->416

## ADOPTED 20260715T121035Z
- cost: 416 -> 375 (points 19.0731)
- source: candidates/task246/rank3_flat_parabola.onnx
- note: rank-3 flat-parabola sign code plus coordinate-selector reuse; bundled 266/266, fresh 500/500; cost 416->375

## ADOPTED 20260715T121259Z
- cost: 375 -> 351 (points 19.1392)
- source: candidates/task246/coeff_rank2.onnx
- note: factor Cr/Dc through shared rank-2 term selectors; bundled 266/266, fresh 500/500; memory 154->130 cost 375->351

## ADOPTED 20260715T121447Z
- cost: 351 -> 330 (points 19.2009)
- source: candidates/task246/power_pair_basis.onnx
- note: factor P[3,30] into repeated [1,x] plus 3x2x2 map and reuse coordinate selector; bundled 266/266, fresh 500/500; cost 351->330

## ADOPTED 20260715T121836Z
- cost: 330 -> 302 (points 19.2896)
- source: candidates/task246/shared_color_basis.onnx
- note: share rank-3 black/red/green channel basis across routing and marker selectors; bundled 266/266, fresh 500/500; cost 330->302

## ADOPTED 20260715T121953Z
- cost: 302 -> 272 (points 19.3942)
- source: candidates/task246/symmetric_color_basis.onnx
- note: reuse one symmetric D/Z/constant basis on input and output channels; bundled 266/266, fresh 500/500; cost 302->272

## ADOPTED 20260715T122303Z
- cost: 272 -> 263 (points 19.4278)
- source: candidates/task246/marker_protected_quadratics.onnx
- note: marker-zero channel basis permits strict vertical and inclusive horizontal quadratics without Sign; bundled 266/266, fresh 500/500; cost 272->263

## ADOPTED 20260715T122433Z
- cost: 263 -> 259 (points 19.4432)
- source: candidates/task246/signed_power_map.onnx
- note: fold quadratic signs into static power-pair map, removing two scalar Neg tensors; bundled 266/266, fresh 500/500; cost 263->259

## ADOPTED 20260715T122637Z
- cost: 259 -> 257 (points 19.4509)
- source: candidates/task246/difference_selectors.onnx
- note: difference selectors jointly encode marker, coordinate, and inclusive endpoint offsets; bundled 266/266, fresh 500/500; cost 259->257

## ADOPTED 20260715T124650Z
- cost: 257 -> 157 (points 19.9438)
- source: candidates/task246/fully_direct_permutations/perm_02.onnx
- note: inline all coordinate reads and root quadratics into operand-ordered FREE-output Einsum; memory 116->0, cost 257->157; bundled 266/266, fresh 500/500

## ADOPTED 20260715T124851Z
- cost: 157 -> 133 (points 20.1097)
- source: candidates/task246/root_mux_factor.onnx
- note: factor direct polynomial mux into reused 2x2 root-difference mux; memory0 params133 cost157->133; bundled266/266 fresh500/500

## ADOPTED 20260715T125120Z
- cost: 133 -> 130 (points 20.1325)
- source: candidates/task246/flipped_green_map.onnx
- note: reuse red channel map for green through 3-value basis sign flip; memory0 params130 cost133->130; bundled266/266 fresh500/500

## ADOPTED 20260715T133957Z
- cost: 130 -> 117 (points 20.2378)
- source: candidates/task246/joint_term_permutations/perm_00.onnx
- note: joint term/root/channel selector factor: collapse 3-state d13 maps into reused root mux + 2x3 channel term; cost130->117; 30 bounded orders; bundled266/266 fresh500/500

## ADOPTED 20260715T141949Z
- cost: 117 -> 104 (points 20.3556)
- source: candidates/task246/quadratic_colour_permutations/perm_12.onnx
- note: quadratic 2D reachable-colour basis with exact marker-zero determinant gate; 16 bounded operand orders; bundled266 fresh500

## ADOPTED 20260715T150025Z
- cost: 104 -> 94 (points 20.4567)
- source: candidates/task246/joint_reused_colour_permutations/perm_00.onnx
- note: cost104->94: reuse 2x2 quadratic gate as channel map; synthesize e0/e1/swap from direct_root_mux; bundled266 fresh500

## ADOPTED 20260715T153719Z
- cost: 94 -> 92 (points 20.4782)
- source: candidates/task246/sign_free_root_permutations/perm_00.onnx
- note: synthesize all joint_axis_sign factors as shared-Z direct_root_mux diagonals; cost94->92; 16 bounded orders; bundled266 fresh500
