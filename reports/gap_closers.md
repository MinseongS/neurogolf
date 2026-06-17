# Gap-closer investigation — CONCLUSION (2026-06-17)

## VERDICT: no big buildable gap-closers beyond the known walls.
A single-process fresh_pass scan (2026-06-17) FALSELY flagged 220/230/282/317 as fresh=0.00.
ROOT CAUSE: arc-gen generators share a module-level `common` state; running fresh_pass for many
tasks in ONE process pollutes that state → bogus failures. The authoritative method runs ONE
process per task (genverify.py uses multiprocessing Pool maxtasksperchild=1; src.adopt runs each
task standalone). Re-checked in isolation: 220/230/282/317 all pass 120/120 (generalize at 18.2).

## The real non-generalizing set (reliable genverify.json, n=40 isolated): 11 tasks
- WALLS (infeasible, do not build): 219 (info bottleneck, fresh 0.00), 255 (connectivity, 0.00)
- small overcounts 0.3-1.7pt (low value): 209, 118, 2, 90, 157, 366, 251, 18, 101
These were already known. No hidden +15 gap-closers exist. The 61.27 stored-LB gap is structural
(per RESUME): overcounted/non-generalizing base nets on tasks no function can solve, + opset>10
harness memory-measurement divergence. NOT recoverable by re-encoding.

## Lesson (also in project memory neurogolf-gap-closers): to scan generalization across tasks,
ALWAYS use isolated processes (`python -m src.genverify`), never a single-process loop.
The proven engine remains the pending-pool grind (lowest-points-first, ~+0.3-0.9/win, ~70% hit).
