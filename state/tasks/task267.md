---
deployed_cost: 758
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task267 — FLOOR (2026-07-01)

mem/params at structural floor. 490B Where=[1,10,7,7] one-hot carrier; ch0 bg must be 1 across in-grid 7x7 (background-channel carrier floor). Already documented FLOOR.

No source change.

## 2026-07-03 S12 — UNKNOWN-bucket dossier

**Rule:** 7×7 grid holds one multi-cell shape in colour A plus a single "key" colour marker at the bottom-left corner (row6,col0); output = the same shape recoloured to the key colour, marker erased.

**Cost (grader mem 724, params 42):** ops Slice×2/Pad×2/Greater/Cast/Equal/Where. Counted intermediates: `block` [1,10,7,7] uint8 490B (the 10-channel one-hot carrier), `ch0in` [1,1,5,5] fp32 100B, `mask7` [1,1,7,7] bool 49B. Params dominated by two [8] int64 pad specs (64B each). Output [1,10,30,30] uint8 9000B is FREE.

**Blocker class:** full-output-carrier. The 490B `block` one-hot is the emission floor: background channel-0 must be set to 1 across the whole in-grid 7×7 (per-channel carrier), so the plane cannot shrink below G²×10 for a colour-replacement whose output size = input size. Already logged FLOOR.

**Lever:** no lever visible (background-channel carrier is structural). fp16 recast blocked — one-hot is uint8 already, Pad needs uint8.

## ADOPTED 20260711T135833Z
- cost: 758 -> 358 (points 19.1195)
- source: candidates/task267/einsum_fold.onnx
- note: 343-class free-input einsum fold: single free-output Einsum 'bjrc,tk,tr,tc,tj->bkrc' with t-axis merging bg-fill+recolor terms; separable inner-5x5 mask (60 params); fixed marker read (6,0) generator-proven; falsifies the old self-referential 490B carrier floor; fresh A/B 2000: cand=inc identical
