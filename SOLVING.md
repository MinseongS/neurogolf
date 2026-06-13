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

## GENERALIZATION discipline (non-negotiable — learned the hard way)

Kaggle scores against **freshly generated arc-gen instances**, NOT the stored
examples we have locally. `harness.evaluate` only checks stored examples, so a
net that memorizes them (or a conv that overfits the patterns it saw) passes
locally but scores **0 on Kaggle**. (A submission of local 6505 scored 4374 on
the real LB because ~125 exact-match memorizer nets contributed ~0.)

Your custom net MUST implement the TRUE rule and pass fresh instances. Verify:
```python
from src.genverify import fresh_pass
ok, run = fresh_pass(N, n=50)   # generates 50 NEW instances, runs your net
assert ok == run, f"only {ok}/{run} fresh instances pass — does NOT generalize"
```
A net is only worth adopting if BOTH `evaluate(...)['ok']` (stored) AND
`fresh_pass` (ok==run) hold. If you can't make it pass fresh instances, the rule
isn't fully captured — keep working or report infeasible. Never ship a net that
only fits the stored examples.

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

## Lesson from pilot solves: MEMORY dominates, count your nodes

Both pilot custom nets (task198, task204) were logically correct but landed at
only ~13.9 pts because they used 20-40 ops, each producing a canvas-sized
intermediate: a `[1,1,30,30]` float intermediate costs 3,600 bytes, bool costs
900, int32 costs 3,600 — and they ADD UP (task198: 69,002 memory vs 1,906
params). The memorizer baseline is ~40,000-98,000 total, so a sloppy custom
net barely beats it.

Budget rule of thumb: every halving of `memory + params` = +0.69 pts.
- < 1,000 total → 18.1+ pts (gold: 1-3 ops, e.g. single Conv/MatMul into output)
- < 5,000 total → 16.5+ pts (good: ≤ a handful of canvas intermediates)
- < 20,000 total → 15.1+ pts (acceptable)
- \> 40,000 → you're barely beating the memorizer; redesign

Concrete tactics:
- fuse aggressively: combine masks/selects into ONE final Conv/MatMul that
  writes `output` directly (free tensor)
- prefer bool (Greater/And/Or, 900B/channel-canvas) over float intermediates
- keep aggregates 1-D ([1,1,30,1] = 120B float) instead of broadcasting early;
  broadcast at the last possible op
- a `[1,10,30,30]` float intermediate is 36,000B — almost never acceptable;
  slice to the channels you need first

## Infeasible class — bail fast (don't burn effort)

**Flood-fill / connectivity / BFS-reachability rules cannot beat the memorizer.**
Tasks whose rule is "color the open region reachable from a seed", "fill the
interior enclosed by a wall", maze/path connectivity, or any transitive-closure
over the grid require iterative frontier propagation whose depth (often 40–80
steps) must be unrolled (Loop/Scan are banned). Each step is a canvas-sized
Conv/MaxPool (3,600 B float) → tens to hundreds of thousands of bytes → scores
*below* the ~13.95 memorizer baseline. Confirmed dead on task338 and task286.
If `--gen` shows flood/connectivity, report infeasible immediately and stop.

**Output-grid-size not recoverable from input → also infeasible.** The scorer
checks the *whole* 30×30 canvas (incl. channel-0 background over the output
grid rectangle), so the net must reproduce the exact output H×W. If the grid
size isn't a computable function of the input — e.g. a small sprite sits in the
interior with no cell touching the grid border, so nothing signals the bounds —
then size can only be memorized, which is exactly what the memorizer already
does optimally. Quick check: do distinct inputs with the same salient features
map to different output sizes? (task358 had 57/137 feature-keys → multiple
sizes.) If size is ambiguous, report infeasible. (When the grid is full-canvas
or size = f(content) like s=Sqrt(count) or s=k·n, you're fine.)

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

## Ground-truth rules (don't guess!)

Every task is an ARC-AGI v1 public training task, and Google's ARC-GEN
generator source for it is available locally:

```
.venv/bin/python -m src.show N --gen     # prints the generator (exact rule)
```

The generator is the code that PRODUCED all the arc-gen examples — read it to
learn the exact transformation, parameter ranges (grid sizes, color choices),
and invariants (e.g. "boxes never overlap", "size is 10–20"). Knowing the real
parameter ranges often makes a compact network feasible (e.g. you only need to
handle widths the generator can actually produce). The repos live at
/tmp/arc-gen (generators, common.py helpers) and /tmp/arc_agi.

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
