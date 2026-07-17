---
deployed_cost: 758
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task267 — 2026-07-01 historical claim, falsified 2026-07-11

A 2026-07-01 note incorrectly treated the 490B `Where[1,10,7,7]` one-hot carrier and channel-0 background fill as irreducible. The 2026-07-11 free-output Einsum adoption removed that carrier, so this is retained only as a falsified historical premise.

No source change.

## 2026-07-03 S12 — UNKNOWN-bucket dossier

**Rule:** 7×7 grid holds one multi-cell shape in colour A plus a single "key" colour marker at the bottom-left corner (row6,col0); output = the same shape recoloured to the key colour, marker erased.

**Cost (grader mem 724, params 42):** ops Slice×2/Pad×2/Greater/Cast/Equal/Where. Counted intermediates: `block` [1,10,7,7] uint8 490B (the 10-channel one-hot carrier), `ch0in` [1,1,5,5] fp32 100B, `mask7` [1,1,7,7] bool 49B. Params dominated by two [8] int64 pad specs (64B each). Output [1,10,30,30] uint8 9000B is FREE.

**Historical blocker hypothesis (falsified):** the 490B `block` one-hot was assumed necessary to emit channel-0 background across the 7×7 grid. The later free-output Einsum showed that this carrier can be folded into the output expression.

**Historical lever assessment (falsified):** no lever was visible under the old carrier-based formulation; fp16 recast alone could not help because the one-hot and Pad were uint8. The later free-output reformulation changed the representation and removed the carrier.

## ADOPTED 20260711T135833Z
- cost: 758 -> 358 (points 19.1195)
- source: candidates/task267/einsum_fold.onnx
- note: 343-class free-input einsum fold: single free-output Einsum 'bjrc,tk,tr,tc,tj->bkrc' with t-axis merging bg-fill+recolor terms; separable inner-5x5 mask (60 params); fixed marker read (6,0) generator-proven; falsifies the old self-referential 490B carrier floor; fresh A/B 2000: cand=inc identical

## ADOPTED 20260715T062803Z
- cost: 358 -> 306 (points 19.2764)
- source: candidates/task267/joint_rc_basis.onnx
- note: shared exact spatial support basis across row/column masks

## ADOPTED 20260715T063126Z
- cost: 306 -> 298 (points 19.3029)
- source: candidates/task267/joint_basis_min.onnx
- note: remove two identity coefficients around shared row/column basis

## ADOPTED 20260715T105948Z
- cost: 298 -> 270 (points 19.4016)
- source: candidates/task267/gathernd_dedupe.onnx
- note: fixed-marker GatherND removes Slice+Reshape; identical row/col spatial basis dedupe

## ADOPTED 20260715T120655Z
- cost: 270 -> 220 (points 19.6064)
- source: candidates/task267/marker_mix_fold.onnx
- note: fold marker-e0 subtraction into terminal Einsum with 2x2 channel mixing; remove GatherND/Sub carrier

## ADOPTED 20260715T132939Z
- cost: 220 -> 180 (points 19.8070)
- source: candidates/task267/marker_affine_factor.onnx
- note: factor marker+e0 through shared J basis; replace 80B Concat carrier with 40B affine marker vector

## ADOPTED 20260715T134456Z
- cost: 180 -> 86 (points 20.5457)
- source: candidates/task267/free_input_corner_fold.onnx
- note: read 3*background+marker directly from FREE input corners via shared R complement; remove Slice/Add and all marker metadata

## ADOPTED 20260715T142013Z
- cost: 86 -> 60 (points 20.9057)
- source: candidates/task267/moment_sign_fold.onnx
- note: replace spatial/channel bases with FREE-input boundary and row moments

## ADOPTED 20260715T144211Z
- cost: 60 -> 30 (points 21.5988)
- source: candidates/task267/one_vector_underflow.onnx
- note: share boundary and row moments through one binary phase vector with pinned float32 inner-support underflow
