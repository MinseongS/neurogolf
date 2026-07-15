# STATE - NeuroGolf live handoff (updated 2026-07-15; local 7431.0339)
> Replace this file at session end; do not append. History lives in git, `state/tasks/`,
> `state/submissions.md`, and `state/levers.yaml`.

## Confirmed state
- Deployment is **7431.0339**, **400/400**, bundled fail=0; `ng verify` completed.
- Best confirmed leaderboard score is **7431.10**, submission **54719275**
  (submitted from local 7430.9666).
- Submission **54718839** (local 7430.7937) scored only **7413.31**. The roughly 17-point
  loss appeared after task080/task138 and concurrent wins. Task080's 6686-cost safe backup is
  `submission/.backups/task080_20260715T084829Z.onnx`, but it is more expensive than the live
  5099-cost net and therefore cannot be restored through the mandatory cheaper-only `ng adopt`
  gate. Do not bypass the gate; treat task080 as the leading hidden-failure/timeout suspect.

## Final adopted batch
- task080: 6686 -> 5099 (16.4632), task138: 11647 -> 11567 (15.6441).
- task350: 418 -> 410 (18.9838), task356: 436 -> 420 (18.9597).
- task374: 1085 -> 1071 (18.0237), task388: 1102 -> 994 (18.0983).
- Public BBI-v3 min-merges were adopted for task281/324/323/206/381/184; task324 was
  subsequently improved to cost 8094. Concurrent gated wins also include task173 and task216.
- All deployments above went through `ng gate`/`ng adopt`; exact Python sources were regenerated
  for the changed live graphs.

## Diagnostics and unresolved items
- Submission 54719275 did not repeat the prior ~17-point loss and established the 7431.10
  leaderboard record. Keep aggregate runtime risk in mind for future large batches.
- task054's previously scoring relational renderer remains lost; deployed task054 is the
  cost-20131 incumbent. Kaggle-browser recovery was unavailable because the browser session was
  not signed in.
- Depth-3 self-Einsum probes on tasks266/248/176 produced no improvement. task353's search was
  stopped after more than 13 minutes without a checkpoint; retry only with within-task
  checkpointing or a bounded contraction search.
- Existing public dumps were refreshed and min-merged; no additional cheaper bundled-safe nets
  were found after the adopted set.

## Invariants
- Goal remains 8000; default mode is 8000-overfit (bundled fail=0 + cheaper). Fresh checks are
  diagnostic only.
- Adoption must use `ng adopt`; submission must use `ng pack` then `ng submit`.
- Keep onnx==1.21.0 and onnxruntime==1.26.0 until a complete 400/400 revalidation authorizes change.
