# NeuroGolf 7470 Score-Climb Design

**Date:** 2026-07-14  
**Baseline:** public LB 7424.42, submission 54654166, local manifest 7424.2851  
**Goal:** reach a confirmed public leaderboard score of at least 7470 while preserving the 7424.42 board as the recovery baseline.

## 1. Strategy

The climb is a direct-discovery program, not a public-model mining program. The required gain is approximately 45.58 points, which is too large for the current mechanical byte-tail queue. Work is therefore allocated as follows:

- **60% — global mechanism discovery:** find new representation regimes that can remove large classes of counted tensors across many tasks.
- **35% — high-yield task deep dives:** attack individual tasks where a semantic rewrite can plausibly gain at least 0.5 points, or where one discovery is likely to transfer to a cohort.
- **5% — opportunistic public frontier check:** poll and min-merge new public artifacts only when they appear, without allowing this lane to displace direct research.

The program optimizes actual public LB points. Fresh evaluation is diagnostic evidence for hidden validity and mechanism quality, while the mandatory adoption gate remains bundled fail=0 plus lower deployed cost as specified by the repository instructions.

## 2. Global Discovery Loop

Each iteration begins from the deployed 400-task board and produces a ranked queue based on more than raw tensor size.

1. Build a per-task structural inventory: deployed cost/points, dominant tensors, operator chain, source mechanism, prior attempts, known wall, and independent lower bound where one exists.
2. Remove stale scanner hits that have already been adopted or adjudicated by the task and lever ledgers.
3. Cluster the remaining tasks by the reason they pay memory or parameters, such as detection planes, routing planes, recurrence/flood planes, dynamic kernels, assignment workspaces, early one-hot expansion, or spatial lookup tables.
4. Rank clusters by `estimated gain × number of transferable tasks × probability of a new construction`, not by one tensor's byte count alone.
5. Select one mechanism family and up to three representative tasks: an easy witness, a high-payoff target, and a boundary case.
6. After any success, register the mechanism in `state/insights.yaml`, rescan all 400 tasks, and apply it to every credible sibling before switching families.

The first discovery wave will focus on underexplored ways to keep full-canvas state inside a free graph input/output or a single operator:

- direct-output contractions beyond the existing mask-dominance taxonomy;
- runtime-parameterized Conv/QLinearConv/ConvTranspose kernels;
- scan or recurrence operators that replace repeated propagation planes;
- low-rank, signed, or threshold-only formulations that exploit output `>0` decoding;
- compact scalar/profile programs that reconstruct spatial output only in the final free node;
- legal ONNX operator families whose attributes provide spatial computation without counted tensors;
- representation changes that delete an entire working regime rather than recasting its dtype.

## 3. High-Yield Task Deep Dives

A task enters the deep queue when at least one of these conditions holds:

- a credible construction can gain at least 0.5 points;
- the incumbent pays at least one large tensor that has no independent-minimum justification;
- the task is a representative of a cluster with several siblings;
- the source is an exact-preserve or borrowed implementation whose semantics have not been recompiled;
- a previous floor verdict was tied to one tool or representation and a genuinely new mechanism can falsify it.

For each selected task, the investigation must inspect the generator/oracle, `src/custom/taskNNN.py`, deployed ONNX, task ledger, tensor profile, and relevant insights before building. Candidates remain under `candidates/taskNNN/`. A build is worthwhile only if byte accounting predicts a meaningful win or if it tests a reusable capability boundary.

Up to three task families may be researched concurrently, as required by the repository's fan-out loop. Agents receive disjoint task directories and do not adopt or submit. The primary agent reviews every result, performs the official gate, registers insights, and controls deployment.

## 4. Plateau Response

If two consecutive waves produce less than 0.5 aggregate adoptable points, the next wave must change the representation lens rather than repeat the same scanner queue. The response order is:

1. Rebuild the global cluster inventory and identify which cost regimes still dominate the missing 45.58 points.
2. Run a focused ONNX capability probe for a new operator or dtype/shape behavior that could erase one of those regimes.
3. Re-open only ledger entries whose explicit trigger is satisfied by that new fact.
4. Choose new representative tasks and repeat the witness/high-payoff/boundary-case loop.

Small optimizer tails below 0.1 points are deferred unless they arise for free during a larger rewrite. This prevents the session from mistaking activity for progress toward 7470.

## 5. Opportunistic Public Lane

Public artifacts are checked at low priority, normally once per major direct-research wave or when Kaggle shows a new/updated kernel. A strictly cheaper bundled-valid task may be adopted through the normal gate, but every material public win must also be reverse-engineered into source and an insight before it is considered complete.

No direct-research slot waits on public downloads, and a zero-win public poll immediately returns control to the direct-discovery queue.

## 6. Validation and Deployment

Every candidate follows this sequence:

1. isolated functional evaluation, with fresh evaluation used to measure divergence risk;
2. `uv run ng gate candidates/taskNNN/<candidate>.onnx --task NNN`;
3. source and insight registration for reusable mechanisms;
4. `uv run ng adopt ...` only after the gate passes;
5. full-board verification and unsigned-TopK safety scan;
6. batch `ng pack` and submission only after checking current Kaggle submissions for concurrent work.

Submissions are grouped by mechanism provenance. High-risk or unusually large gains are isolated before composition so a zero/error can be attributed without sacrificing the confirmed board. The 7424.42 composition remains recoverable until a higher composed board is confirmed.

## 7. Success and Stop Conditions

The success condition is a completed Kaggle submission with public score **>=7470.00**. Local projected points alone are not sufficient.

Before success, a wave may stop only to record a dated four-field negative verdict and immediately select the next live or newly reopened mechanism. The overall session does not declare the board exhausted based on one scanner, one operator family, or one task cohort.

At success, replace `state/STATE.md` with the confirmed board, submission id, adopted mechanisms, current invariants, and the next-session start procedure; update `state/submissions.md`; verify 400/400 and hashes; and commit the exact tracked state.
