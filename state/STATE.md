# STATE - NeuroGolf live handoff (updated 2026-07-14; LB 7424.42, local +0.247 unsubmitted)
> Replace this file at session end; do not append. History lives in git, `state/tasks/`,
> `state/submissions.md`, and `state/levers.yaml`.

## Confirmed State
- 🏆 **RECORD LB 7424.42 — submission 54654166.** This is +13.75 over the
  7410.67 starting record and leaves **+5.58 to the 7430 target**.
- ⚠️ **Local manifest is now 7424.5321 (`400/400`, `ng verify --hash` = HASH-OK) —
  +0.2469 AHEAD of the submitted record and NOT yet submitted.** The delta is task302
  alone (cost 1773 -> 1385, 17.5196 -> 17.7665), adopted 20260714T143609Z. Next submit
  should pick this up; it is a pure per-task cost cut, no other net changed.
- task302 = `Concat -> Max` single-plane collapse; mechanism + transferable rule in
  `state/tasks/task302.md`, lever `concat-max-collapse` in `state/levers.yaml`.
  A board-wide rescan found only 3 further carriers worth ~0.04 total, so that lever is
  spent until a new graft lands — do NOT fan out on tasks 160/169/204.
- Record delta: task047 +1.6118 and task338/127/048 +0.4626 on the confirmed
  7422.34 base. Preserve those four LB-isolated ONNX files.

## Task110 / Task061 High-Yield Probe Result
- Approved task110 contraction-reorder design was implemented in candidate scratch.
  The source has **30 operands** (the earlier 31 count was corrected). Structural tests
  preserve every operand/equation-term pair.
- task110 first-inference benchmark: `col_then_row` 2.49s and
  `pair_support_first` 6.19s succeeded; two variants timed out at 10s and one errored.
- `col_then_row` official local gate: cost340, pass59/fail207. Per user direction it was
  still submitted as a single-task leaderboard probe because public train+test was 4/4.
  Submission **54682630 = 0.00**. It is not adopted.
- `pair_support_first` public train+test was also 4/4, but submission **54682796 = 0.00**
  after the long scoring run. It is not adopted.
- Related task061 cost130 self-Einsum probe **54683163 = ERROR**. It is not adopted.
- Deployed task110 cost2751 and task061 cost636 remain unchanged. Candidate artifacts and
  benchmark JSON are under `candidates/task110/`; the lever verdict is in `state/levers.yaml`.

## Next Meaningful Lever
- A static sweep of 302 cached candidate ONNX files found no untried >=+0.45 candidate
  beyond task110/task061. Other apparent hits are already-known Kaggle-zero hash-scatter
  task076/173 or fail-all placeholders task133/198.
- Do not spend time on <+0.1 byte tails. Reopen the variadic self-Einsum lane only with a
  new fast contraction/factorization, or resume the live `public-minmerge` /
  `public-insight-generalize` lanes when a genuinely new frontier artifact appears.

## Operational Guardrails
- Adoption only through `ng adopt`; leaderboard probes did not mutate deployment.
- Submission flow remains pack then submit; pin onnx==1.21.0 / onnxruntime==1.26.0.
- Check Kaggle submissions before every submit; both task110 contraction-order probes are
  now confirmed 0.00 and must not be repeated without a genuinely new factorization.
- After this sync, start the next session from the main checkout:
  `/Users/minseong/project/neurogolf`.
