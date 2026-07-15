---
deployed_cost: 2751
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task397 — fcc82909

## 2026-06-29 mechanism screen

Rule: for each 2x2 colour box, draw a green 2-wide shadow whose height equals the
number of distinct colours in the box.

Current source score: 17.045628 @ mem 2648 params 200. The graph detects 2x2 boxes
with a compact `TopK`/`Gather`/`ScatterElements` pipeline. The largest tensor is
`cond30` [1,1,30,30] bool = 900 B, used by `Where(cond30, green, input) -> output`.

No rewrite adopted. The obvious attempt to avoid `cond30` by composing in a 10x10
crop first is worse, because it materializes a counted [1,10,10,10] one-hot crop
before padding. Current routing keeps the 10-channel result as the free graph
output and pays only a one-channel full-canvas condition.

## 2026-07-01 (S7 re-run) — FLOOR re-confirmed
mem 2648/17.05; cond30 900B bool=optimal Where(cond,green,FREE input) delta-route (must be bool 30x30), code_f 320B forced-fp32 box detector. No safe reduction; all dominant intermediates structurally forced (fp32 entry crop / int32-64 index buffer / full-canvas routing mask).

## S10 (2026-07-03) — bobmyers7186 teacher ADOPTED (+0.001)
**Mechanism (op-census diff):** One redundant `And` removed (6→5; 58→57 nodes). −3B.
**Old→new:** mem 2585→2582, params 200→200.
**Gate:** bundled cand fail=0; fresh N=2000 inc_fail=0 cand_fail=0. No TopK reject.
Backup `reports/retired_networks/task397_pre_s10.onnx`; source `public_candidates/bobmyers7186/task397.onnx`. Gate data: scratchpad/gate_small/results.jsonl.
No transferable mechanism — minor trim.

## 2026-07-07 — local-only REJECTED signed INT8 TopK feed probe

`reports/candidates/task397/task397_int8_topk_greedy.onnx` recast `min4_f`
TopK feed to signed INT8.  Bundled gate fail=0.  Cost: 2782 -> 2717 (memory
2582 -> 2516, params 200 -> 201).

Follow-up pruning removed dead initializer `zero_f`. Cost: 2717 -> 2716.

## ADOPTED 20260715T071227Z
- cost: 2751 -> 2685 (points 17.1046)
- source: candidates/task397/signed_topk_cast.onnx
- note: signed int8 TopK carrier: replace bool/u8->fp16 feed without changing indices/presence

## REPAIRED 20260715T073700Z
- cost: 2685 -> 2751 (points 17.0803)
- source: candidates/task397/kaggle_safe_fp16_topk.onnx
- note: Kaggle safety repair after ref54716353 ERROR: INT8 TopK -> FLOAT16
## CORRECTION 2026-07-15 — the `signed_topk_cast` ADOPTED entry above is NOT the live state
- ran: board-wide `neurogolf.topk.find_unsigned_topk` over all 400 deployed nets, plus a
  direct check of this task's `submission/.backups/` chain.
- verdict: the `signed_topk_cast.onnx` adoption recorded above fed **signed INT8 into TopK**
  (elem_type=3). `src/neurogolf/topk.py` classes this as a Kaggle GRADER-KILLER: the grader
  errors the WHOLE submission, it is invisible to local ORT/onnx.checker, and `ng pack`
  refuses to zip such a net. It was established for unsigned ints on 2026-07-02, for signed
  INT8 by task233 submission 54418836, and RE-CONFIRMED by full submission 54716353 on
  2026-07-15 (today). The net was reverted on disk the same day; the ADOPTED block above was
  left behind and reads as live. It is not. Board scan now: **0/400 violations, packable.**
- reopen: none — do not re-adopt any `signed_topk_cast` family member. If a cost win is
  wanted from this direction, the feed must be fp16/fp32 (verified acceptable), never int8
  or any unsigned int. Re-run the board scan before every `ng pack`:
  `uv run python -c "from neurogolf.topk import find_unsigned_topk; ..."` over
  `submission/overfit_nets/*.onnx`.
