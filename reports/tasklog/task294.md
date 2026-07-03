
## 2026-06-30 (leak-audit recovery) — FIXED broken net (real 0 -> 17.05)

Leak audit (fresh_pass n=3000) flagged task294 at **100% fresh failure** AND it
failed all 265 stored examples: the deployed/source net implemented the WRONG rule
(full 3x3 9-cell gray block), so it scored ~0 on the real LB while the manifest
counted a fictional 18.19 (`local_stored_ok:False`).

True rule (cristianoc task294, verified 500/500 vs generator): 10x10 grid; a cell
-> 2 iff gray(5) AND all 4 ORTHOGONAL in-grid neighbours gray(5) (5-cell plus).

New net: Slice gray ch5 to native 10x10 -> cross Conv (count gray over the plus)
-> Equal==5 -> Pad to 30x30 -> Where(interior, color2, FREE input).
- mem=2800, params=37, **points=17.049**, stored 265/265, **fresh 5000/5000**.

## S10 (2026-07-03) — knife-edge flip CONFIRMED but epsilon fix REFUTED (not adopted)
task294 flips ALL-fail in dirty-process eval like 220/230 (prior same-shape-Conv evaluate
in process). BUT the −0.5 score_bias shift (on ≥ +0.5 / off ≤ −0.5 margins) does NOT fix
it: candidate passes fresh-process (265/0) yet still fails 265/265 dirty, ANY pollutant
(220 / 230,220 / 220,230). ⇒ 294's dirty divergence is NOT sub-epsilon float drift —
something bigger changes under arena pollution (different conv path? output deltas ≥0.5 —
UNMEASURED). Candidate parked at reports/harden_candidates/task294_cand.onnx.
DO NOT adopt an unverified "fix" here (this task's history: leak-audit false-positive
"fix" was pure loss). Next step: dump dirty-vs-clean output deltas to size the divergence.
Grader currently passes the incumbent (LB clean incl. 7213.63) — risk is private-LB env drift.
