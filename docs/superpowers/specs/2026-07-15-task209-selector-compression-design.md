# Task 209 Selector Compression Design

## Objective

Reduce task 209 from the adopted cost of 6089 to at most 5509, which is the
next strict +0.1-point boundary, while preserving:

- bundled failures: 0/266;
- divergence from the current incumbent: 0 on a fresh 5000-case holdout;
- the exact first-winner semantics of the incumbent scale, row-phase, and
  column-phase selectors;
- the existing source-owned and `ng gate`/`ng adopt` workflow.

The current incumbent is the exact quotient selector plus two-dimensional
parabolic feature placement and terminal signed `QLinearConv` output fold. Its
cost is 6089 = 5683 memory + 406 parameters. A qualifying candidate must save
at least 580 bytes.

## Scope and Constraints

This pass attacks the selector, not the terminal output decoder. The primary
target is the 576-byte `BmCell` tensor and the surrounding four-color selector
chain. Candidate and scratch artifacts stay under `candidates/task209/`.

The following previously completed or disproven routes are out of scope:

- QA/QS exact splitting and the uint8 geometry/index chain;
- full-input uint8 recasting, which cost 14981 because the input copy became
  counted;
- sparse initializers, which fail strict `Einsum` rank-0 handling;
- another terminal output-fold redesign unless the selector routes fail and a
  combined, independently priced design can still reach cost 5509;
- direct edits to `submission/overfit_nets/task209.onnx`;
- `ng pack` or `ng submit` during this session.

The existing terminal decoder's reachable labels are
`{0, 1, 2, 3, 4, 8, 255}`. Label 255 must continue to decode to zero in every
output channel. These facts are invariants, not new search targets.

## Current Selector Boundary

The current graph converts the yellow-box crop `Bw[1,1,12,12]` into four
per-color equality planes, `BmCell[1,4,12,12]`. Row and column reductions form
four-channel profiles. Exact depthwise quotient pools for divisors 2, 3, and 4
are correlated against the sprite profiles; `ArgMax` chooses scale and then
the first row and column phases. Those three winners feed the already compact
feature placement and output decoder.

The replacement boundary is therefore deliberately narrow:

- inputs: the incumbent's already available geometry, crop, colors, and sprite
  support/profile values;
- outputs: exactly the incumbent `(scale_index, row_phase, column_phase)` with
  the same tie order;
- downstream consumer: unchanged positioned feature renderer and terminal
  decoder.

Keeping this boundary lets selector candidates be tested independently before
rebuilding the full ONNX graph.

## Approach A: Winner-Preserving Low-Rank Color Metric

This is the recommended first route. It treats the incumbent selector as a
teacher and searches for the smallest integer feature basis that preserves its
winners, rather than preserving every intermediate four-color score.

### Data extraction

A source-controlled analysis tool under `candidates/task209/` will record, for
each example:

- the four per-color correlation components for every legal scale and phase
  candidate;
- the incumbent first-winning scale, row phase, and column phase;
- all score margins against candidates that appear earlier or later in the
  incumbent ordering;
- the geometry/profile scalars already present in the graph.

Constraints will be harvested from all 266 bundled cases and exactly 20,000
deterministic fresh discovery cases. A separate deterministic 5000-case
holdout will not participate in the search. The two non-overlapping generator
seed ranges will be written into the analysis result so the split is
reproducible.

### Exact inequality search

For every teacher winner `w` and competitor `c`, the replacement score must
obey an order-aware inequality:

- if `c` appears before `w`, require `score(w) > score(c)`;
- if `c` appears after `w`, require `score(w) >= score(c)`.

This encodes ONNX `ArgMax` first-index tie behavior directly. The search starts
with one nonnegative integer color embedding and then tries two integer basis
features only if one dimension is infeasible. Coefficients, zero points,
accumulator bounds, padding contributions, and requantization rounding are
enumerated in integer arithmetic before any ONNX candidate is built.

The search result is acceptable only if it supplies a compact ONNX lowering,
not merely a mathematical embedding. The lowered graph must compute the basis
without recreating a four-channel 12x12 tensor or another tensor of comparable
cost. Candidate operators may include `Equal`, integer bitwise features,
`QLinearConv`, or `ConvInteger`, but the exact ORT 1.26 behavior must be checked
with micro-tests for every rounding or padding assumption.

### Static price gate

Before full-model construction, a tensor-and-parameter ledger will price the
proposed replacement against the adopted graph. The full predicted task cost
must be at most 5509. A design whose decoding, packing, or basis construction
erases the 580-byte saving is rejected without building a candidate.

The preferred result removes `BmCell` and enough adjacent selector tensors to
save 600-900 bytes while leaving the downstream renderer unchanged.

## Approach B: Compact Geometry/Profile Classifier

If no one- or two-dimensional metric satisfies the exact winner inequalities,
the second route predicts the teacher's three selector outputs from already
available small values. Candidate features include `Hm`, `Wm`, crop offsets,
sprite height/width, row and column supports, run lengths, and quotient-period
residues.

The classifier must be a deterministic rule expressible with small integer
ONNX tensors. Search may use decision tables or bounded arithmetic during
discovery, but the final lowering may not introduce a large lookup table that
merely moves the incumbent memory elsewhere. As with Approach A, the static
full-model estimate must be at most 5509 before construction.

This route has higher savings potential because it can delete most of the
four-color correlation stack, but also higher distribution risk. It therefore
requires exact agreement with the teacher on the full discovery corpus and the
untouched 5000-case holdout before it can proceed to `ng gate`.

## Approach C: Combined Partial Fold

This is a last resort, not a default continuation. A scalar renderer or a
one-plane terminal affine decoder is known to be structurally unable to
separate all five regular states under arbitrary neighborhoods. It may be
reconsidered only if a proved selector saving plus independently verified
micro-folds jointly reaches cost 5509. No bundled-fit-only renderer will be
adopted under the strict fresh-divergence requirement.

## Candidate Workflow

1. Build a selector instrumentation harness under `candidates/task209/` and
   confirm that its teacher winners reproduce the incumbent output on bundled
   and sampled fresh cases.
2. Harvest order-aware inequalities and search Approach A embeddings.
3. Exhaustively validate integer accumulation, padding, zero-point, and
   requantization behavior in Python, then reproduce each assumption in an ONNX
   Runtime micro-test.
4. Produce a static tensor/parameter price. Stop the route if the predicted
   full cost exceeds 5509.
5. Add focused tests before changing the candidate builder. Tests cover tie
   order, all reachable colors, every divisor, boundary padding, and teacher
   agreement.
6. Build the full candidate only after those tests fail for the absent
   implementation and the price gate passes.
7. Run bundled validation and compare against the incumbent on the untouched
   fresh 5000 holdout in fresh processes.
8. Run `uv run ng gate candidates/task209/<candidate>.onnx --task 209` only when
   bundled failures are zero, fresh divergence is zero, and measured cost is at
   most 5509.
9. Run `uv run ng adopt candidates/task209/<candidate>.onnx --task 209 ...`
   only after a gate PASS. Never copy the candidate into the deployed directory
   directly.

If Approach A is infeasible, retain its scripts and results as evidence and
pivot to Approach B. A negative result is recorded with the concrete command,
scope, date, reopen trigger, and falsification history required by the project
ledger; it is not described as a permanent floor.

## Verification and Adoption

A successful candidate must pass all of the following in fresh processes:

- existing task 209 regression tests;
- new selector unit and ONNX micro-tests;
- bundled 266/266 with zero failures;
- zero output divergence from the adopted incumbent on the untouched 5000-case
  holdout;
- measured cost at most 5509;
- `uv run ng gate ... --task 209` PASS.

Only then may `ng adopt` run. After adoption, rebuild the exact source artifact,
verify its SHA matches the deployed model, update `DISCOVERY.md` and the insight
registry, and run the required 400-task inventory/candidate rescan. The session
will not pack or submit.

## Failure Handling

Analysis and search failures must distinguish mathematical infeasibility from
an unavailable compact ONNX lowering. ORT discrepancies are reduced to a
minimal micro-test before changing the search model. Any mismatch on bundled or
fresh examples records the first counterexample and its teacher/candidate
winner tuple so the next iteration adds a concrete constraint.

The deployed incumbent remains unchanged unless every adoption criterion is
satisfied. Unrelated dirty worktree files are preserved and never staged.
