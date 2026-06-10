# Writing a custom NeuroGolf network for one task

Target: an ONNX graph that passes **every** train/test/arc-gen example of the
task and minimizes `memory_bytes + params`. Score = `25 - ln(memory + params)`.
A typical hand-crafted solution costs 100–3000 → 17–20.4 pts. The memorizer
fallback already gives ~13.5–15, so a custom solution is only worth keeping if
it beats that (the pipeline's keep-best enforces this automatically).

## Hard constraints (rejected otherwise)

- opset 10, IR 10 (`builders._model` handles this), default domain only
- exactly one input `input` and one output `output`, both `[1,10,30,30]` float32
- static shapes everywhere; no Loop/Scan/NonZero/Unique/Compress/Sequence ops
- no functions/subgraphs; file ≤ 1.44MB
- Equal doesn't support float until opset 11 — Cast to int32 first
- Clip uses min/max **attributes** (not inputs) in opset 10

## Scoring model (what to optimize)

- `params` = total elements of all initializers + Constant-node tensors
  (a scalar counts 1; zeros count like any element — sparsity doesn't help,
  ORT doesn't accept sparse initializers)
- `memory` = sum over all intermediate tensors of `elements × dtype_size`
  (max of static shape and runtime-observed shape)
- tensors named exactly `input` / `output` are **FREE** — always compute the
  last op straight into `output`; a `[1,10,30,30]` float intermediate costs
  36,000, a bool one 9,000, so avoid canvas-sized intermediates
- ORT runs with ALL optimizations disabled — graphs are scored as written

## Exactness discipline (non-negotiable)

The checker is `result > 0.0` against a 0/1 one-hot target, evaluated in
float32. Keep every value an integer with |worst-case partial sum| < 2^24 and
float32 evaluation is exact regardless of summation order. Verify with
`harness.evaluate` (mirrors the official scorer bit-for-bit) before accepting.

## Data layout

`convert_to_numpy`: grid colors 0–9 → one-hot channels, grid top-left aligned
on the 30×30 canvas, rest all-zero ("no color"). Output must match the target
canvas exactly: 1-channel per colored cell, all-zero outside the output grid.
Grids larger than 30×30 are skipped by the scorer (already filtered by
`usable_examples`).

## Building blocks already available (src/builders.py)

- `conv_network(W, kh, kw, bias, groups)` — single Conv into output; any
  per-cell rule that's a linear threshold of the k×k one-hot neighborhood.
  `groups=10` = depthwise (channel-preserving rules), 10× fewer params.
- `memorizer_network(...)` — exact-match lookup (the fallback to beat)
- packing helpers `pack4_codes` / `pack6_codes` (base-11 cell codes via
  strided Conv; ≤ 6 cells per float stays exact)

## Idioms for common ARC patterns

- **per-cell color logic** → 1×1 or k×k Conv (try `solve_conv` first; it
  already searched the ladder, so if the task is still memorized, plain conv
  FAILED — you need something structurally different)
- **fixed spatial permutation** (all examples same grid size): flip/rotate/
  transpose/tile = row/col permutation matrices as MatMul on [30,30] axes
  (`[1,10,30,30] @ M` permutes columns; `M' @ x` rows; Transpose for H<->W).
  ~900–1800 params, tiny memory.
- **size-dependent permutation** (sizes vary): compute the grid width
  `w = ReduceSum(col-occupancy)` in-graph, then build the permutation matrix
  from `w` arithmetically, e.g. hflip is `M[i,j] = (i + j == w-1)`:
  store an i+j index matrix (900 params, int32), Equal against `w-1`, Cast,
  MatMul. Same trick for vflip/translation-to-edge ("gravity").
- **row/col aggregates** (occupancy, counts) → ReduceSum/ReduceMax along an
  axis, keepdims, broadcast back — these tensors are tiny ([1,10,30,1] = 1200B)
- **conditional select** (mask ? A : B) → out = mask*A + (1-mask)*B with Mul/
  Add/Sub; masks from Equal/Greater(Cast int32)
- **counting / thresholds** → Conv with all-ones kernel counts neighbors;
  Greater(int32) thresholds it

## Workflow for one task

```python
from src.harness import load_task, evaluate
from src.analyze import usable_examples
task = load_task(N)
exs = usable_examples(task)   # study ex["input"] / ex["output"] grids
# ... build model (onnx.helper / builders helpers) ...
res = evaluate(model, task)   # res["ok"], res["points"], res["fail"], ...
```

Put the final solution in `src/custom/taskNNN.py` exposing `build(task) ->
onnx.ModelProto`. The pipeline's `custom` method picks it up and keep-best
merges it. Iterate until `res["ok"]` and `res["points"]` beats the manifest
entry for the task (`reports/manifest.json`).
