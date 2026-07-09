# Regime-crack vein worklist (2026-07-09)

Lever: fold a 900B [30,30] output-routing mask into a free-output Einsum. See memory
`neurogolf-regime-crack-freeoutput-einsum` for the full recipe + ~18 sub-recipes.
Gate: bundled fail=0 + cost < DEPLOYED (`submission/overfit_nets/taskNNN.onnx`). Model policy:
opus for proven sub-recipes; Fable only for genuinely novel/hard. Candidate scanner:
`reports/scripts/mask_dominance_scan.py` → `reports/mask_dominance.json`.

## Confirmed LB trajectory this session
7264.67 → 7268.28 → 7271.88 → 7275.04 → 7279.17 → 7279.34 (+ task190 pending). ~+14.7 LB.

## DONE (cracked + adopted)
329,303,141,033,341 (b1); 260,075,240,159,345,109 (b2); 295,392,398,061,051 (b3);
287,094,063,199,273,224 (b4); 246,190 (b5/retry). Plus fp16 adoptions 251,268,205,222.

## FLOORS (confirmed, do NOT re-spend — positioned-content / dense-recolor)
112,163,099,102,062,012,348,124,354. Boundary rule: mask = (arbitrary motif CONTENT) ⊛
(data-dependent 2-D POSITION), OR a data-dependent run/coupling (task354 SameRun) → 900B
u8 index-plane is the cheapest encoding; every Einsum factorization prices higher.

## batch 6 (opus) — RESULT: DRY (7/8 FLOOR, confirms the easy vein is harvested)
FLOORED (do not re-attempt): 354 (gray-run segmented-max = data-dependent coupling), 090 (mask
only 30% — rectangle FINDER owns 68%, fold can't touch it), 390+154 (variable-hinge reflection —
900B is Equal-DECODE not routing; can't ADD in-place remap + spatial fold in one Einsum), 034
(diagonal staircase at DATA-DEPENDENT seed = positioned), 168 (2-3 arrows arbitrary pos/dir), 381
(per-row run placement). 071 reflection pending (likely floor). ⇒ NO batch-6 adoptions.

## SHARPENED CRACK CONDITION (from batch 6 — three necessary properties)
A 900B-mask task cracks ONLY if ALL hold; else FLOOR:
1. GLOBAL-STATE routing (predicate from ≤~16 scalars: offset/size/color/count), NOT content at a
   data-dependent position, NOT a data-dependent run/coupling.
2. The 900B plane is a Where ROUTING mask, NOT an Equal DECODE discriminator (label30→Equal→10ch;
   those are the dense-recolor floor — 124/354/390).
3. Mask FRACTION high (≥~45%). Low-fraction (~30%, e.g. 090) loses: non-mask compute dominates.
   AND the fold's fp32 operands (forced fp32 by co-binding the free input) must stay < the 900B
   bool mask — needs genuinely low-rank/static routing, not per-cell placement matrices (Sh/Sw
   [9,30]+colour-map blow params past the free bool Where). This fp32-co-bind economics is why
   168/381/034 floor even though structured.

## NEXT TARGETS (triage-classified STRUCTURED, not yet attempted)
268 (frame + diagonal arms), 086 (ring/frame stamp + color-remap). Caveated/near-floor
(deprioritize): 251 (already fp16-adopted, near-floor), 333 (thin margin), 042 (cross-scale
detection false-positive risk). Lower-fraction (<0.30) masks: re-run mask_dominance_scan with
--min-frac 0.20 for the long tail.

## SEPARATE ARSENAL (unproven, potentially big — probe next)
CONV-FP32 family: 074,080,198 (fp32 Conv 3600B `color_f`/`G` planes), 216,162,002 (fp32 Slice
1600-3200B), 349 (QLinearConv u8 4350B). These are NOT bool/u8 Where-masks — need a new fold
(collapse the Conv/Slice into a free-output contraction, or the fp32 plane is the free-input
image → dtype-welded floor unless restructured). Biggest single-task upside if the arsenal works.

## RESIDUAL LEVER flagged mid-vein
Mixed-dtype Einsum: many adopted regime nets still carry a fp32 input-co-bound carrier (~600B,
e.g. task063). If a carrier can go fp16 despite sharing an Einsum with the fp32 free input,
many adopted nets get ~300B cheaper. ORT Einsum binds all operands to one dtype — need an
escape (separate the input-contraction from the carrier-contraction).
