# Task285 Residual Optimization Design

## Objective

Reduce the deployed task285 cost below 15,982 without weakening the mandatory
adoption gate. Every adopted stage must pass all 265 bundled examples with zero
failures and be strictly cheaper than the then-current deployed artifact. The
next full +0.1 target is cost 14,461 or lower. Fresh evaluation remains
diagnostic under the repository's 8000-overfit policy.

## Current Evidence

The deployed graph costs 15,982: 15,748 bytes of counted intermediates and 234
initializer elements. The largest remaining costs are the 3,600-byte entry
colour plane, the 1,800-byte FLOAT16 TopK feed, and four 900-byte label/reshape
carriers.

Read-only instrumentation across all 265 stored examples found that the
`target_base -> target10 -> target_cap -> v81_safe` suffix never changes `v81`.
There were two negative target lookups, but both already carried an invalid
`-1` update from the existing membership/pivot gate. Therefore the four
81-element INT8 tensors are redundant on the authoritative corpus. Removing
them also removes their one-use scalar initializer `i89`, for a 325-cost
first-stage reduction.

## Considered Approaches

### 1. Remove the redundant target-sentinel suffix

Connect `v81` directly to the final `ScatterElements` update input and delete
the four suffix nodes and any now-unused initializers. This is the recommended
first stage because it is a local graph rewrite with a measured 325-cost saving
and no new operator or dtype.

Expected endpoint: exactly 15,657. A different measured cost stops the stage for
an accounting investigation. This is below the incumbent but does not reach the
next +0.1 threshold.

### 2. Continue exact sparse-graph micro-folds

After stage 1, inspect only shape/index/fallback chains whose outputs are small
and whose semantic equivalence can be proved independently. Candidate examples
include avoiding duplicated rank adapters around the selected pivot indices or
sharing an existing scalar/shape initializer. Each rewrite receives its own
cost assertion and bundled equivalence test. No batch of speculative rewrites
is combined before the paying change is isolated.

Expected yield: tens to low hundreds of additional cost. This lane is useful
for strict-cheaper adoptions but is unlikely by itself to reach 14,461.

### 3. Replace a large carrier family

The only credible path to another full +0.1 is deleting either the K=31
enumeration family or at least two 900-byte label/output carriers. A new
bounded-pivot plus FREE-output topology may qualify only if all of the following
are proved before a full bundled gate:

- static cost is at most 14,461;
- every wrong colour has a strict negative logit margin, including padding;
- all TopK inputs remain FLOAT16 or FLOAT32;
- fresh-process first inference is below one second with optimizations disabled;
- no counted tensor exceeds the existing 3,600-byte entry plane.

The rejected 14,415 renderer is a control, not a retry seed: its sign baseline
and colour delta were incompatible, and first inference was 2.503682 seconds.
Operand rotation alone or the same polynomial colour basis must not reopen it.

## Execution Design

Work proceeds in three independent stages:

1. Add a regression that proves the current sentinel suffix is redundant on all
   265 stored examples and that a source-built candidate is expected to cost
   less than 15,982.
2. Build the suffix-pruned candidate under `candidates/task285/`, run ONNX
   checker and strict shape inference, compare thresholded output against the
   incumbent on all stored examples, then run `ng gate` and `ng adopt`.
3. Re-inventory the adopted graph. Apply further exact micro-folds one at a time.
   Attempt the large-carrier lane only after a paper/static price at or below
   14,461 and separate margin/runtime probes satisfy the stop rules above.

The deployed file is never edited directly. The canonical source is regenerated
only after an authorized adoption and must rebuild byte-identically to the
deployed artifact.

## Failure and Stop Rules

- Any bundled mismatch rejects the candidate; the gate is never bypassed.
- Any integer TopK input rejects the candidate before gate.
- Any large-lane first inference at or above one second stops that topology
  before full bundled evaluation.
- Any large-lane static cost above 14,461 stops it unless it also contains an
  independently verified strict-cheaper substage that can be adopted alone.
- A negative result is recorded only in the four-field lever ledger with a
  concrete reopen trigger. No task floor is claimed.
- The known reconstructed-fresh K=31 neighbor-Gather out-of-bounds case remains
  a diagnostic limitation and is not hidden by weakening the generator.

## Verification and Adoption

Each candidate must pass, in order:

1. focused proof/regression tests;
2. `onnx.checker.check_model(full_check=True)` and strict shape inference;
3. static counted-memory and initializer accounting;
4. isolated bundled evaluation, 265/265 with fail 0;
5. `uv run ng gate <candidate> --task 285`;
6. `uv run ng adopt <candidate> --task 285 --note <mechanism>`;
7. isolated `uv run ng score 285` and exact source/deployed SHA synchronization.

After a reusable mechanism is found, it is registered in `state/insights.yaml`
and rescanned across all 400 deployed graphs. Packaging and Kaggle submission are
outside this task.
