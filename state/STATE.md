# STATE - NeuroGolf live handoff (updated 2026-07-15; LB 7424.42, local 7425.0466)
> Replace this file at session end; do not append. History lives in git, `state/tasks/`,
> `state/submissions.md`, and `state/levers.yaml`.

## Confirmed State
- 🏆 **RECORD LB 7424.42 — submission 54654166.** The active goal is 7470.
- ⚠️ **Local manifest is 7425.0466, nets 400/400, `ng verify --hash` = HASH-OK.**
  It contains two adopted improvements not yet submitted as a full board:
  task302 cost 1773 -> 1385 (**+0.24697**) and task054 cost 20131 -> 12034
  (**+0.51452**), for **+0.76150 task-local points** over the record deployment.
  The calibrated full-board expectation is roughly 7425.18, leaving about 44.82 to 7470.
- task302 was committed in `7562afe`; its `Concat -> Max` mechanism and exhausted
  board-wide transfer scan are recorded in `state/tasks/task302.md` and
  `state/levers.yaml`.

## Task054 — Leaderboard-Confirmed Win
- Candidate `candidates/task054/relational_renderer.onnx`, SHA256
  `a90259a9050599856df3bd714f3eff8758aa0ad1061af973379e857ca439956f`.
- Isolated submission **54689861 = 15.60**, proving Kaggle execution and scoring.
- Official `ng adopt` gate: **266/266, fail=0**, cost **20131 -> 12034**,
  points **15.08998 -> 15.60451**. Adoption timestamp `20260714T151024Z`.
- The successful mechanism replaces the old sparse-edit floor with a bounded relational
  renderer and exact three-box cancellation. Details are in `state/tasks/task054.md` and
  `.superpowers/sdd/task054-relational-report.md`.

## Task285 — Correct Math, Non-Deployable Execution
- The affine-reflection rule and low-degree pivot certificate are exact, but the two-Einsum
  cost-9225 candidate did not finish locally and isolated submission **54688350 = ERROR**.
- The pivot stage was reduced from 45 operands to a bounded 8-operand polynomial and runs
  locally; the remaining 32-operand renderer is the blocker. Do not repeat the same probe.
  Reopen only with a bounded-temporary renderer below incumbent cost 18674.

## Direct-Discovery Status
- Repaired self-Einsum search commit `0832952` passed its test suite, but independent review
  found malformed resume rows are under-validated at `tools/self_einsum_search.py:1261`.
  Task2 remains incomplete until that validation is fixed and the planned all-400
  depth-8/beam-3000 run is executed.
- Multi-task cohort search completed all 400 at depth5 and cohorts 0/1/2 at depth6.
  It found no new exact or cost<=150 correction; only controls 067/179/241. Do not repeat
  identical absolute-wrong-cell cohort settings. Reopen triggers are in
  `.superpowers/sdd/cohort-multitask-einsum-report.md`.
- Task349 analysis eliminated the incremental QConv route on cost grounds. The paused next
  design is a three-gate bounded relational renderer: endpoint/radius detector -> relation
  skeleton -> priority composition. Resume from
  `.superpowers/sdd/task349-relational-report.md` and `candidates/task349/`.
- Task173's exact generator rule is recorded in `state/tasks/task173.md`; the natural
  free-output compiler still needs a bounded contraction plan.

## Next Session Order
1. Check Kaggle submissions, `uv run ng pack`, verify 400 entries, then submit the full board
   containing task302 + task054 and poll to completion.
2. Fix the Task2 resume-row schema bug with a regression test, rerun independent review, then
   launch the specified all-400 deep search with safe per-candidate child timeouts.
3. Continue task349's staged relational renderer; reject immediately if any stage cannot keep
   the total below cost 9033 or fails `ORT_DISABLE_ALL` runtime/memory checks.
4. Preserve task285 as a transferable algebraic insight, not an active candidate, until its
   renderer is fundamentally factorized.

## Operational Guardrails
- Adoption only through `ng adopt`; submission flow is `ng pack` -> `ng submit`.
- Keep onnx==1.21.0 / onnxruntime==1.26.0 pinned. Candidates stay under `candidates/`.
- Single-task probes are authoritative for ambiguous runtime viability; restore the canonical
  400-file `submission.zip` immediately afterward.
- Main checkout: `/Users/minseong/project/neurogolf`.
