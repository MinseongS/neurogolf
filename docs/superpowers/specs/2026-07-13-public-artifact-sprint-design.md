# NeuroGolf 7430 Public Artifact Sprint Design

## Objective

Raise the confirmed Kaggle leaderboard score from 7410.67 to at least 7430.00.
Local score improvements are evidence for candidate selection, not the success
criterion. Success requires a completed Kaggle submission at or above 7430.

## Strategy

Use the newest public `submission.zip` artifacts as the primary source of large
gains. The current leaderboard already exceeds 8100 and several public notebooks
published new submission artifacts on 2026-07-13, after the archive that produced
the 7410.67 record. This makes fresh public min-merge much higher ROI than local
byte-tail work.

The sprint order is:

1. Mine new public artifacts against the protected 7410.67 deployment.
2. Submit a bundled-clean all-in merge without using freshgate as an adoption
   blocker.
3. If the first merge does not reach 7430, use its structural deltas to select
   only high-yield local rewrites. Do not resume sub-0.5-point task work while a
   plausible +5 batch remains.

## Public Inputs

Start with the public kernels confirmed to expose `submission.zip` after the
previous archive cache:

- `lucifer19/neurogolf-agi-circuit-forge`
- `lucifer19/neurogolf-agi-compression-core`
- `kaiwalyaatulraut/neurogolf-championship-solution`
- `lucifer19/chimera-safe-boost-caddies`
- `hoangvux/neurogolf`

Also poll the latest competition kernels once immediately before acquisition so
that a newer artifact is not omitted.

## Artifact Pipeline

### 1. Acquire and normalize

Download each kernel output into a dated directory below
`candidates/public_dumps/`. Preserve source kernel, version, creation time, and
artifact hash. Reject malformed archives, archives without 400 task models, and
exact duplicate artifacts before model evaluation.

### 2. Isolated per-task scoring

For every public model, evaluate it in an isolated process with the pinned
ONNX/ORT versions. A model enters the merge pool only when:

- bundled failures are zero;
- its measured cost is strictly lower than the deployed task model;
- it contains no unsigned TopK hazard that can invalidate the submission.

Fresh generator results are diagnostic metadata only. A fresh failure does not
remove a bundled-clean model from the all-in experiment because the 7410.67
record proved that freshgate is stricter than the real leaderboard suite for this
artifact family.

### 3. Build the all-in frontier

Choose the cheapest eligible model independently for every task across the
7410 deployment and all new artifacts. Produce a report containing task id,
source artifact, old and new cost, local point delta, fresh diagnostic, and hash.

Do not mutate the protected deployment while measuring. Once the complete merge
is known, apply wins only through `ng adopt`, then rebuild the manifest.

### 4. Submission gate

Before packing:

- confirm 400/400 deployed models and manifest hashes;
- run bundled verification for all changed tasks in isolated processes;
- scan all 400 models for unsigned TopK;
- check current Kaggle submissions for concurrent work;
- pack only through `ng pack`.

Submit the whole eligible all-in merge as one experiment. The existing 7410.67
record remains protected, so leaderboard regressions do not require a defensive
fresh-filtered submission first.

## Decision Thresholds

- If projected local gain is at least +20, submit immediately after the normal
  packaging checks.
- If projected gain is +5 to +20, still submit the all-in merge, then use its
  per-task source deltas to identify the missing high-yield mechanisms.
- If projected gain is below +5, do not spend the session polishing the merge.
  Pivot to public-teacher autopsy and the highest-cost tasks, requiring either
  at least +0.5 expected gain per task or at least +5 for a mechanism batch.
- Micro-golf below +0.1 is recorded but not pursued during the 7430 sprint.

## Secondary High-Yield Lane

If public min-merge is insufficient, compare the cheapest new public graphs with
the current high-cost tasks, led by 233, 366, 054, 133, 285, 158, 286, and 018.
Prioritize whole-representation collapses, especially public Einsum forms, rather
than initializer or dtype tails. A public adoption must be followed by source
reconstruction, insight registration, and a 400-task rescan after the leaderboard
experiment; these ownership tasks must not delay the first high-value submission.

Task023 and task233 local refits remain excluded from mixed submissions unless a
new public artifact supplies a materially different proven model. Their previous
two-task refit submission scored 7394.55 despite local gates.

## Failure Handling

- Malformed or incomplete public archive: quarantine it and continue with other
  sources.
- Candidate fails bundled examples: exclude only that task model.
- All-in leaderboard regression: retain the 7410.67 deployment and split the
  changed tasks into large-delta source cohorts for diagnostic submissions.
- Local/LB mismatch: Kaggle is authoritative; do not redefine completion using
  the manifest total.
- Concurrent submission detected: reconcile hashes and wait rather than consume
  a duplicate daily submission slot.

## Verification and Completion

The sprint is complete only when Kaggle reports a completed submission with
`publicScore >= 7430.00`. After that result, reconcile the winning ONNX files to
`src/custom/taskNNN.py`, refresh manifests and state, record the submission, and
commit only the in-scope artifacts while preserving unrelated dirty files.
