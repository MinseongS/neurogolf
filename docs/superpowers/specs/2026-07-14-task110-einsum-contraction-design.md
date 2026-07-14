# Task110 Einsum Contraction Redesign

## Objective

Produce a task110 model that preserves bundled correctness, is strictly cheaper than the
deployed cost 2751, and completes the mandatory `ng gate` within its 600-second isolated
evaluation timeout. The first target is the existing bundled-specific one-node Einsum
candidate at estimated cost 340, whose algebra is compact but whose current contraction
order makes even one inference impractically slow.

This work is successful only after:

1. `ng gate <candidate> --task 110` reports bundled fail=0 and a lower cost.
2. The candidate is adopted through `ng adopt`, never by copying over the deployed file.
3. An isolated Kaggle submission on top of the confirmed 7424.42 composition preserves all
   other task points and realizes the expected task110 gain.

## Scope

In scope:

- Reorder the operands and matching equation terms of the existing 30-operand task110
  Einsum without changing its mathematical expression.
- If reordering alone is still too slow, split the expression into small comparison-profile
  Einsums followed by one free-output renderer.
- Measure runtime, bundled correctness, scorer cost, and Kaggle contribution independently.
- Record the contraction mechanism for later task061 fanout.

Out of scope for this iteration:

- Fresh-gate optimization. Fresh evaluation is diagnostic; bundled fail=0 is the adoption
  requirement for this leaderboard-oriented task.
- Changes to ONNX/ORT pins.
- Micro-optimizations under +0.1 before the contraction mechanism is resolved.
- Implementing task061 before task110 proves the mechanism.

## Considered Approaches

### A. Algebra-preserving operand reordering (first choice)

Keep a single output-only Einsum and the same initializers, but interleave each repeated
`input` factor with the masks/selectors that constrain its indices instead of listing all
15 repeated inputs first. The equation labels and operand list are permuted together, so
the scalar expression is unchanged and the target scored cost remains 340.

Advantages:

- Preserves the full theoretical gain, about `ln(2751/340) = +2.09`.
- No counted intermediate tensors.
- Smallest code and easiest algebraic equivalence review.

Risk:

- ORT may canonicalize or choose the same poor contraction path regardless of operand order.

Kill criterion:

- A representative inference still exceeds 10 seconds, or the full bundled gate cannot
  finish inside 600 seconds.

### B. Staged comparison profiles plus free-output renderer (fallback)

Split the 30-operand product into the eight logical row/column comparison factors already
visible in the equation. Each factor emits only the indices required by the final render;
the builder rejects any proposed factor whose materialized output is larger than a
length-30 fp32 profile. A final Einsum multiplies those profiles with the free input and
emits `output` directly.

Advantages:

- Bounds intermediate rank and prevents a catastrophic cross-product contraction.
- Runtime behavior is explicit and independently measurable per factor.
- Still has room for a meaningful gain if total cost stays at or below 1668
  (`ln(2751/1668) = +0.5`).

Risks:

- Materialized fp32 profiles reduce the score gain.
- An incorrect split can change shared-index semantics from a joint sum into a product of
  independent sums.

Kill criterion:

- Bundled fail is nonzero, total cost exceeds 1668, or runtime remains too slow for the
  full gate.

### C. Fixed-output bitset/table renderer (rejected)

The existing task110 bitset candidate recognizes only the four unaugmented examples and
fails 262 transformed bundled examples. Extending it would require explicitly encoding the
augmentation action and would no longer be the shortest or safest route. It is not part of
the implementation plan.

## Data Flow and Correctness Constraints

The bundled-specific expression detects row and column phase relationships from repeated
reads of the free input, then reconstructs the periodic output. Every selector index shared
by two input operands must remain inside the same contraction factor until both reads have
been combined. It is invalid to reduce a selector against one input first and then multiply
by an independently reduced second input; that changes a sum of products into a product of
sums.

For approach A, correctness is structural: only operand order and corresponding equation
term order change. Initializer bytes, index labels, output labels, and values remain
identical, so the scored cost remains 340.

For approach B, each extracted factor must be checked against the equivalent partial
contraction of the original equation. The final renderer must remain the graph output so its
30x30x10 tensor is free under the scorer.

## Implementation Sequence

1. Copy the existing task110 bundled Einsum builder into a new candidate builder under
   `candidates/task110/`; never edit the deployed model directly.
2. Generate several semantically identical operand-order variants, prioritizing mask and
   selector proximity to their repeated input pairs.
3. Benchmark session creation and one inference. Discard variants over 10 seconds before
   running the full task gate.
4. For the fastest variant, compare thresholded output with the deployed model on a small
   representative subset, then run the authoritative `ng gate`.
5. If no reordered variant satisfies the runtime criterion, build approach B factor by
   factor and profile each counted output before the full gate.
6. Adopt only a gate-passing candidate with cost below 1668. Pack and submit it in isolation
   on the 7424.42 base.
7. Keep the 7424.42 deployment backup until the task110 leaderboard delta is confirmed.

## Verification

- ONNX checker and ORT session load under onnx 1.21.0 / onnxruntime 1.26.0.
- No forbidden operators and no unsigned TopK.
- Representative inference below 10 seconds; complete isolated gate below 600 seconds.
- Bundled pass count equals the full task110 set and fail=0.
- Candidate cost below 1668, with the single-Einsum target near 340.
- `ng verify --hash` before packing the submission.
- Kaggle isolation score compared with record submission 54654166 (7424.42). A successful
  cost-340 result should add about +2.09; a staged result must add at least +0.5.

## Failure and Rollback Policy

Failed variants stay under `candidates/task110/`. A local gate failure, timeout, or cost
miss never changes `submission/overfit_nets/`. If a locally passing candidate Kaggle-zeros,
restore task110 from the automatic `ng adopt` backup, update the manifest through normal
verification, and retain submission 54654166 as the protected record.

## Follow-on

After task110 is Kaggle-confirmed, apply the same contraction analysis to task061's existing
18-operand cost-130 self-Einsum candidate. Task061 is a separate implementation and
submission probe; task110 success supplies the mechanism, not automatic authorization to
adopt task061.
