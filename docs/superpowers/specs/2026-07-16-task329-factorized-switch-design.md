# Task329 Factorized-Switch Global Lowering Design

## Objective

Improve only task329 beyond the deployed cost105 model.  The primary target is a
strictly proved cost93--95 construction, rather than stopping as soon as a marginal
gain appears.  A cost97 exact lowering is an acceptable secondary result only after
the cost93--95 search lane has received its bounded evaluation.

The work must not repeat the failed rank2 Adam/hinge search.  It must use exact
algebra, a different tensor topology, or a bounded global solver, and it must prove
all `4 * 30 * 10 * 10 = 12000` logical classifier states before writing ONNX.

## Incumbent and cost budget

The deployed graph has memory12, params93, and cost105:

| component | elements/bytes |
|---|---:|
| `colour_features[10,2]` | 20 params |
| `shared_core[2,2,2]` | 8 params |
| `swap[2,2]` | 4 params |
| `one_state` | 1 param |
| `route_r[2,30]` | 60 params |
| `cell_count_keep`, `state_vec` | 12 memory bytes |

Keeping the 20-element colour table, 60-element spatial route, one state constant,
and 12-byte dynamic-state path leaves a base cost of93.  Therefore:

- no additional initializer gives cost93;
- one 2-vector core gives cost95 and `ln(105/95) = 0.100083`;
- one 2-by-2 selector gives cost97 and `ln(105/97) = 0.079174`.

The route and memory terms are working lower bounds for this design, not universal
impossibility claims.  A genuinely different operator family may beat cost93.

## Exact factorized-switch classifier

Let the input colour be `k`, candidate output colour be `c`, and choose float32
constants `a`, `b`, and `lambda` with

```text
a + b = 0.5
F0(x) = a - x
F1(x) = x + b
```

For state cell count `n` and output column `w`, store a route value `z(w)` and define

```text
u = n * z(w)
v = 1 + lambda * (u - 1)
score(k,c,v) = (F0(k) + v*F1(c)) * (F1(k) + v*F0(c)).
```

Columns1--4 store the reciprocals of legal counts9,25,49,81 respectively.  All other
columns may store zero.  Thus the one target column for a state has `u = v = 1`.
At that column,

```text
score(k,c,1) = (0.5-k+c) * (0.5+k-c)
             = 0.25 - (k-c)^2.
```

This is positive exactly for `c == k` and at most `-0.75` for every wrong colour.
Away from the target, the two leading colour factors can make only output colour zero
positive when `|v|` is sufficiently large.

A read-only float32 feasibility prototype used

```text
a      = 0.3244965076
b      = 0.1755034924
lambda = 74.24807739
```

and zero on the 26 default route columns.  Across all 12000 logical states it measured
`desired_min = 0.24999097`, `wrong_max = -0.75`, margin `0.24999097`, with finite
scores.  This proves the classifier semantics are feasible; it does not by itself
prove that a cost95 or cost97 ONNX lowering exists.

## Lowering search, in priority order

### Lane A: core-free or core2 contraction, target cost93--95

Search terminal-Einsum factorizations that reuse the same two colour features and the
same two spatial features multiple times.  Candidate equations may use implicit
diagonal contractions, repeated latent indices, complements following from
`F0(x)+F1(x)=0.5`, and axis permutations.  At most one learned or solved 2-vector may
be introduced.

Each topology receives a bounded global solve over its coefficients.  Suitable
solvers include differential evolution followed by deterministic polishing, SMT over
rationalized coefficients, or exact symbolic elimination.  Adam, hinge-loss seed
extensions, and the previously rejected rank2 route family are excluded.

A topology is accepted only if its recovered coefficients are independently
re-evaluated in float32 over all 12000 states.  Solver objective values or sampled
training accuracy are never sufficient.

### Lane B: four-element selector lowering, conditional target cost97

If Lane A closes without a solution, allow one dense `2 x 2` selector.  Its job is to
assemble the two affine sums inside the terminal contraction without materializing a
colour or spatial intermediate.  The design must account for every initializer and
every non-output node tensor before implementation.  Cost97 is only valid if the
selector is the sole addition to the base cost93; an extra transform, swap, or hidden
route tensor invalidates that estimate.

This lane is valuable because the factorized-switch formula removes the incumbent's
separate preserve/background sum.  It is not adopted merely from the algebraic
classifier proof: the actual tensor network must first satisfy the same exhaustive
proof and scorer accounting as Lane A.

### Lane C: exact self-routing fallback, target cost101

As a low-risk fallback, solve a new shared core whose own axis permutations or
transpose generate the reflected colour branch, eliminating the four-element swap
from the incumbent.  This would reach cost101.  It is attempted only after the larger
Lane A opportunity, and it must not be mistaken for evidence that cost95 is
impossible.

## Proof-first builder contract

Any implementation builder must expose `--prove-only --json`.  Proof-only mode must
not create or modify an ONNX file.  Before `build()` is allowed to serialize a model,
the report must establish:

1. exactly 12000 logical states were enumerated;
2. every desired score is strictly positive;
3. every wrong score is nonpositive;
4. `min(desired_min, -wrong_max) >= 0.008` in float32;
5. every score and initializer is finite;
6. predicted memory, params, and total cost match the proposed graph;
7. the target lane's cost ceiling is met.

The proof must evaluate the recovered tensor-network equation itself, not only the
high-level factorized formula.  It must also test target/default route partitions and
all cross-state ratios such as `49/81` and `81/49`, which are the closest non-target
ratios to one.

## Implementation and evaluation sequence

1. Add focused RED tests for proof-only no-write behavior, all-state margin, predicted
   cost, and finite values.
2. Implement only the proof/search representation for the selected topology.
3. Run the bounded Lane A search and record every topology family and result so a
   rejected family is not silently retried.
4. Create ONNX only after a concrete coefficient set passes the full float32 proof.
5. Run full ONNX checker, strict shape inference, and isolated scorer measurement.
6. Run `uv run ng gate <candidate> --task 329`.
7. Run fresh1500 as a diagnostic and record candidate/incumbent divergence.
8. Adopt only through `uv run ng adopt`; never copy into deployment or edit the
   manifest directly.
9. Make `src/custom/task329.py` reproduce the adopted protobuf byte-for-byte and keep
   historical task329 builders idempotent.

The deployed cost105 model remains live while the bounded large-gain lane runs.  If
Lane A fails but Lane B proves cost97, cost97 may be adopted as real progress while
the remaining lower-bound question stays open.  Rank4 cost270 remains the independent
historical safety fallback.

## Scope and failure handling

- Modify only task329 candidates, tests, source, and task329-specific records.
- Do not use sparse initializers or change the pinned ONNX/ORT versions.
- Do not pack, submit, or scan/modify other tasks.
- Do not write ONNX after a failed identity, sign, margin, finite-value, or cost check.
- Do not adopt if bundled failures are nonzero or measured cost is not strictly cheaper
  than the deployed graph.
- A failed bounded topology is recorded as a dated, solver-scoped negative result.  It
  does not become a universal floor claim.

## Alternatives rejected for this pass

- Re-running 16-seed rank2 Adam/hinge training: the exact equality sign-rank lower
  bound already closes that spatial route family, and repeating the optimizer changes
  no premise.
- Generating the 60-element route through counted operators: known lowerings
  materialize at least 120--240 bytes and lose to storing the route.
- Cropping, padding, Gather/Scatter, or a full mask: their counted intermediates are
  much larger than the current 12-byte dynamic path.
- Sparse initializer accounting: rejected by the pinned strict-shape/scorer path and
  explicitly closed in project history.
