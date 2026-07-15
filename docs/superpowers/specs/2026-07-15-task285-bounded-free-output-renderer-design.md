# task285 Bounded FREE-Output Renderer Design

## Objective

Improve the deployed task285 graph beyond cost **16,038** while preserving bundled
**265/265** correctness and all Kaggle-safety constraints. The primary score target is
the next +0.1 threshold, cost **<=14,511**. Adoption must use `ng gate` followed by
`ng adopt`; `submission/overfit_nets/task285.onnx` is never edited directly.

## Current Baseline and Budget

The deployed graph uses the exact sparse 3-pivot, 5x5-shape pipeline and costs
**16,038 = 15,793 memory + 245 parameters**. The graph prefix through the bounded
shape-slot representation costs **12,382 = 12,268 memory + 114 parameters**. Its
current suffix therefore costs **3,656**.

Reaching cost 14,511 requires saving at least 1,527. If the prefix is unchanged, the
replacement suffix has an all-inclusive budget of **2,129** counted elements/bytes.
Micro-optimizing the current suffix cannot plausibly meet that bar; the renderer must
be replaced as one unit.

## Considered Approaches

### 1. Bounded direct FREE-output renderer — selected

Keep the deployed sparse pivot and shape extraction through the three 5x5 shape
slots. Replace coordinate materialization, destination gathers, sentinel scatter, the
900-byte label grid, and final equality expansion with a bounded analytical renderer
whose graph output is the task's BOOL NCHW tensor.

The renderer reads the FREE input directly for the identity branch. The stamp branch
uses at most 3 x 9 active shape slots and four reflected destinations per creature.
Row and column spatial relations are contracted separately or factored into small
static bases. No serialized Einsum may have more than ten operands, and the builder
must reject any contraction whose prefix live-index estimate exceeds the configured
bound. Every counted intermediate must be at most 3,600 bytes; the total new suffix
must be at most 2,129 to retain the +0.1 path.

This is selected because it attacks the only suffix large enough to pay the target
saving while reusing the already-fast and exact sparse discovery front end.

### 2. Direct Scatter output bridge — fallback

Retain current sparse destination/value calculation but try to make ScatterND or
ScatterElements write a representation that is consumed directly by the FREE output.
This is lower mathematical risk, but known ONNX layouts require either a counted
900-byte reshape or 81x4 int64 ScatterND indices, so it is unlikely to fit 2,129.
It is attempted only if a schema-valid terminal bridge is found with a calculated
cost at or below 14,511 before full evaluation.

### 3. Pivot extractor replacement — deferred

Replace K31 neighbour enumeration with the exact hash/LUT or bounded quadratic pivot
plane. Existing probes are either too expensive or slower than one second on first
inference, and the current K31 path is already exact and compact. This approach is
reopened only if the selected renderer falls short by an amount a priced pivot change
can recover.

## Architecture and Data Flow

1. The existing graph computes labels, enumerates visible cells, finds up to three
   pivots, recovers orientation, and emits bounded local shape slots.
2. The new renderer consumes only the smallest available slot tensors plus the FREE
   input. It must not rebuild a dense pivot plane or a dense label grid.
3. A row relation and a column relation encode the affine destinations `A-i` or
   `A+1+i`, and `B-j` or `B+1+j`. Off-grid coordinates contribute no positive output.
4. The colour branch derives the destination quadrant colour for each stamp. Missing
   colours and padded creature slots contribute zero.
5. The terminal contraction combines the preserved input and non-destructive stamp
   logits directly into the FREE BOOL output. Multiple creatures may overlap only
   through an OR-equivalent, nonnegative accumulation.

The implementation may use Fourier/Kronecker factors, polynomial equality features,
or small Scatter operators, but it must preserve this bounded interface and budget.

## Correctness Invariants

- Every production TopK input remains fp16 or fp32. Integer TopK is forbidden.
- Existing visible cells remain unchanged.
- Each active local shape offset is reflected into all four legal quadrants and takes
  its destination pivot colour.
- Missing pivot colours, padded slots, and off-grid destinations produce all-false
  output channels.
- Overlapping stamps use OR semantics and cannot cancel one another.
- Bundled 265 cases are the authoritative adoption gate. Fresh checks are diagnostic;
  the inherited seed285/example80 neighbour-Gather OOB must not be described as a new
  candidate regression without incumbent A/B evidence.

## Build-Time and Runtime Rejection Rules

The builder prints the inferred size of every counted tensor, initializer count, total
cost, each Einsum operand count, and each Einsum prefix live-index estimate. It refuses
to emit a candidate when any of these holds:

- predicted total cost exceeds 14,511 for the primary candidate;
- a new counted intermediate exceeds 3,600 bytes unexpectedly;
- an Einsum has more than ten operands or an unbounded prefix product;
- shape inference is incomplete or an integer TopK input appears.

An emitted candidate is abandoned without retrying the same graph when a clean-process
first inference exceeds one second, session creation fails, or no first output appears
within ten seconds.

## Test and Adoption Strategy

Development follows RED-GREEN TDD inside `candidates/task285/scratch/tests/`:

1. Add a failing structural/cost test for the bounded renderer builder and its required
   output interface.
2. Add a failing numpy-oracle test for identity preservation, all four affine
   reflections, missing colours, overlaps, padding slots, and off-grid stamps.
3. Implement the smallest candidate that satisfies those tests.
4. Run `onnx.checker.check_model(full_check=True)`, strict shape inference, and one
   example in a fresh process.
5. Compare candidate and expected outputs on all 265 bundled cases in isolated
   evaluation, then run the focused task285 suite.
6. Print the scorer's exact memory/parameter inventory and run
   `uv run ng gate candidates/task285/<candidate>.onnx --task 285`.
7. Adopt only when bundled fail=0 and the candidate is strictly cheaper than deployed.
   A candidate below 16,038 but above 14,511 may be adopted as a real win, but research
   continues through the selected +0.1 route until its priced mechanism is consumed.
8. After adoption, synchronize `src/custom/task285.py` as a self-contained exact source
   and verify candidate/source/deployed SHA identity in fresh processes.

All candidates, generated ONNX files, profiles, and scratch tests remain under
`candidates/task285/`. No pack or submit is performed in this standalone session.

## Documentation and Reuse

Update `candidates/task285/DISCOVERY.md` with every built mechanism, runtime, exact
cost, correctness result, SHA, and rejection boundary. Update `state/tasks/task285.md`
through the adoption workflow. Register a new `state/insights.yaml` entry and run the
required 400-task rescan only if the terminal lowering is reusable beyond task285.
