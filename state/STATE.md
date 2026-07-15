# STATE - NeuroGolf live handoff (updated 2026-07-15; LB 7424.66, local 7424.5321)
> Replace this file at session end; do not append. History lives in git, `state/tasks/`,
> `state/submissions.md`, and `state/levers.yaml`.

## Confirmed State
- Best leaderboard score is **7424.66**, submission **54705903**.
- Local deployment is **7424.5321**, 400/400. It contains the adopted task302
  Concat-to-Max improvement (cost 1773 -> 1385).
- The task054 relational renderer previously scored 15.60 in isolated submission 54689861,
  but its gitignored ONNX and reconstruction source were lost. The deployed task054 is the
  cost-20131 incumbent. Git objects, reflog, stashes, local ZIPs/backups/caches, and temporary
  directories contain no copy of expected SHA256 `a90259a...`. Do not count its +0.5145 gain.

## Work Completed This Session
- Hardened `tools/self_einsum_search.py` resume validation against malformed persisted rows;
  `tests/test_self_einsum_search.py` passes **39/39**.
- Fresh kernel-collapse scan found no candidates. Mask-dominance scan only returned previously
  adjudicated carriers; no candidate was adopted. Public kernel poll found no new dump.
- Task349 was re-audited: deployed cost 14892; reaching <9033 requires eliminating both the
  3600B label plane and 4350B radius bank. The staged report was also lost, and no bounded
  multi-object association contraction was recovered. No candidate was built.

## Next Session Order
1. Run the planned all-400 self-Einsum search at depth 8 / beam 3000 using the now-validated
   resume file and safe per-candidate child timeouts; gate any hits before adoption.
2. Reconstruct task054's bounded relational renderer from the recorded endpoint/radius ->
   relation -> fixed-three-box cancellation mechanism, or recover submission 54689861 after
   signing into Kaggle in a browser that exposes submission-file download.
3. Reopen task349 only with a bounded endpoint/radius association design that prices below
   9033 before build and completes under `ORT_DISABLE_ALL`.
4. Continue public frontier polling and mandatory public-insight generalization for any win.

## Operational Guardrails
- Adoption only through `ng gate` -> `ng adopt`; submission is `ng pack` -> `ng submit`.
- Keep onnx==1.21.0 / onnxruntime==1.26.0 pinned. Candidates stay under `candidates/`.
- Check Kaggle submissions before submitting; current 54705903 already represents the local board.
