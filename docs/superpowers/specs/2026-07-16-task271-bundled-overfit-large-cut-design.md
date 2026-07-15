# Task271 Bundled-Overfit Large-Cut Design

## Objective

Optimize only task271 from the deployed cost350 graph. The primary target is a large
one-shot reduction, preferably cost316 or lower, but every strictly cheaper bundled-perfect
fixed point remains eligible for adoption. The project-wide 8000 overfit gate now applies:
bundled fail must be zero, while fresh behavior is diagnostic and must be reported rather
than used as a rejection gate.

The deployed baseline is pinned at cost350 = memory238 + params112, bundled267/267, with
candidate/deployed/network/source SHA
`a9a0d70fe35d4f9360883188b62fa8244b473242502739f27d696391ba5ad51b`.

## Scope and invariants

- Modify only task271 artifacts and shared records whose edited clauses concern task271.
- Candidate and scratch artifacts stay under `candidates/task271/`.
- Keep `onnx==1.21.0` and `onnxruntime==1.26.0` unchanged.
- Never edit `submission/overfit_nets/task271.onnx` directly. Adoption uses `ng gate` followed
  by `ng adopt` only.
- Preserve the single FLOAT input and exact 10-channel 30x30 thresholded output contract.
- For every MaxPool-index design, retain the pinned-ORT 49-position flattening/storage-order
  proof. A bundled-trained scorer may change which position wins, but index-to-coordinate
  geometry may not change.
- Do not repeat dual ReduceMax/ArgMax, `cyan_patch/crop8/Pad`, full-canvas Cast/QLinearConv,
  INT8 Where, sparse Conv initializers, or the rejected dense CP lowering of the direct
  `2^count` dominance construction.

## Accepted risk change

The former fresh-divergence-zero success condition is relaxed by explicit user approval.
The hard correctness gate is now bundled267/267 with fail0. Fresh generation still runs on
every candidate that reaches the official gate, and the final handoff must state candidate
failures and raw/sign divergence versus the incumbent. No fresh regression may be hidden or
described as exact generalization.

The 49-position synthetic test remains a runtime geometry test, not a claim that a learned
partial-window scorer generalizes to every possible patch.

## Findings that bound the search

The current score core costs 286 of 350: a dense FLOAT `Conv` kernel with 90 elements and a
FLOAT `[1,1,7,7]` score carrier with 196 bytes. The remaining value-only MaxPool, UINT16
payload decoder, centered UINT8 feature, and terminal ConvInteger renderer cost 64.

The exact `2^count` dominance identity is valid, with a minimum strict margin of one over all
four distinct counts. However, its 7-by-3-by-30 addition relation has input-mode rank nine.
A dense CP implementation therefore needs at least
`9 * (7 + 3 + 30) = 360` parameters before channel mapping or rendering and cannot beat the
incumbent. All 49 top-left positions are generator-feasible, so cropping the score grid is
also invalid. ONNX subgraphs are rejected by the official scorer, and sparse Conv weights are
rejected by strict checking.

These are dated representation rejections, not a claim that task271 is at a global floor.

## Approaches

### 1. Bundled-trained compact selector with coordinate extraction — primary

Start from the retained cost411 MaxPool-index/dynamic-Slice graph because it extracts the
selected full 3x3 patch from the FREE input and therefore does not need to encode nine payload
bits in the score. Replace its exact 3x3 scorer with a bundled-trained smaller-footprint scorer.

For each footprint in `(1,1)`, `(1,2)`, `(2,1)`, `(1,3)`, `(3,1)`, `(2,2)`, `(2,3)`, and
`(3,2)`, build the bundled ranking constraints in memory. Each example contributes the desired
winner position and all competing 7x7 positions. First test a shared linear Conv score using
linear programming with a strict margin. If infeasible, test the smallest explicitly priced
nonlinear composition whose predicted total cost is below350.

The unchanged coordinate tail means a plain smaller Conv can beat cost350 only when its score
kernel and any new tensors save more than the cost411-to-350 gap. Static pricing is therefore a
precondition: no persistent candidate is built when the predicted cost is at least350. A
cost316-or-lower architecture is preferred, but any measured strict win may be adopted and used
as the next baseline.

### 2. Bundled payload compression — secondary

Keep the current value-only MaxPool but search for a lower-cost code on the observed winner
patch set. Candidate encodings may use fewer directly decoded bits only when the missing bits
are reconstructed by a fully priced standard-ONNX function and the total predicted cost is
below350. Collision checks cover all bundled winner patches before ONNX construction.

This lane is secondary because the current decoder is only 64 cost and random generator patches
make simple eight-bit projections likely to collide. It is retained because a bundled-specific
Boolean relation could produce a small win without reintroducing coordinates.

### 3. Low-rank FREE-input/FREE-output learned contraction — high-upside fallback

Search a bounded family of separable linear or low-degree contractions that bind the FLOAT input
directly and make the graph output terminal, avoiding both the 196-byte score grid and the
renderer. Every family is priced from its actual factor shapes before fitting. A family whose
parameter lower bound is at least350 is rejected without optimization.

This lane has the highest possible payoff but the lowest probability because winner selection is
nonlinear. It runs only after the compact-selector feasibility sweep, and it may overfit the
bundled set but must still produce exact thresholded outputs on all 267 cases.

## Search components and data flow

The implementation will separate four concerns:

1. A read-only bundled corpus extractor identifies the expected winner patch/position and emits
   feature differences for scorer fitting.
2. A feasibility/pricing module searches architectures in memory and records solver status,
   training margin, predicted tensor bytes, initializer elements, and total cost.
3. A builder exists only for an architecture with bundled feasibility and predicted cost below
   350. It pins the incumbent SHA and produces a deterministic two-pass fixed point under
   `candidates/task271/`.
4. Focused tests independently cover solver evidence, ONNX checker/runtime capability,
   49-position MaxPool index geometry, bundled output equality, measured cost, and fixed-point
   bytes.

Exploratory fitting results are disposable. The adopted artifact must be reproducible from a
source-controlled builder and then from `src/custom/task271.py`.

## Error handling and pivot rules

- Solver infeasible or zero/non-strict margin: reject that architecture; do not weaken bundled
  correctness.
- Multiple acceptable winner positions: accept only positions whose extracted 3x3 thresholded
  output equals the bundled target; otherwise treat the example as a hard ranking constraint.
- Pinned ORT unsupported operator/dtype: reject the operator family immediately.
- Predicted or measured cost at least350: do not gate or adopt; pivot to the next representation.
- A load-bearing MaxPool tie: reject the candidate. Every bundled example must have a strict
  selected-score margin in the builder's exact dtype.
- Candidate cost below350 but bundled fail nonzero: retain only the falsifying test/evidence, not
  a deployment candidate.
- Fresh regression: record exact counts and divergence in task271 handoff material; it does not
  block adoption under this approved mode.

Negative results are task-local dated grammar results. If the bounded search is dry, record the
concrete families, tool/date, reopen trigger, and falsification history in the live lever ledger;
do not declare an absolute floor.

## Verification and adoption

Before any candidate file implementation, a failing test must specify the selected architecture,
strict bundled margin, predicted cost below350, 49-position geometry, and deterministic build.
After implementation:

1. Run the new focused test and all existing task271 focused tests.
2. Run strict ONNX shape inference and full checker under the pinned environment.
3. Run an isolated official score and require bundled267/267, fail0, and measured cost below350.
4. Run `ng gate` and require PASS.
5. Run fresh comparison as a diagnostic and preserve its exact output.
6. Adopt only through `ng adopt`.
7. Rebuild exact source and require candidate, deployed, network, and source-rebuilt SHA equality.
8. Update the task271 discovery/task ledger and any genuinely reusable insight. Do not pack or
   submit unless separately requested.

If multiple strict wins appear, adopt the cheapest verified candidate rather than stopping at the
first `+0.1` boundary. A smaller intermediate win may be adopted when it is already fully verified,
then the search continues against the new incumbent.
