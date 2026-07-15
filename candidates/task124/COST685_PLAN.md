# Task124 Corrected Cost686 Exact Composite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lower task124 from cost849 to the cheapest exact passing endpoint, targeting the schema-correct memory623 + params63 = cost686 while preserving the direct padded uint8 QLinearConv output.

**Architecture:** Build a local candidate ladder from the immutable cost849 semantic source: inverse-quantize the channel-0 crop, remove redundant row carriers, scalarize the period/shift route, then replace the bounded row fingerprints with six-tap negative-pad QLinearConv hashes. Verify every stage independently and adopt only the cheapest exact candidate that passes bundled, fresh, and off-grid gates.

**Tech Stack:** Python 3.12, NumPy, pytest, ONNX 1.21.0, ONNX Runtime 1.26.0, NeuroGolf `ng` CLI.

## Global Constraints

- Work only on task124 source, candidate artifacts, tests, and task124-specific records.
- Keep `onnx==1.21.0` and `onnxruntime==1.26.0` unchanged.
- The graph output remains uint8 and is produced directly by the padded final `QLinearConv`.
- Bundled gate must report fail=0 and cost strictly below the deployed task124 artifact.
- Fresh 2,000 must report incumbent fail=0, candidate fail=0, raw/sign divergence=0, and off-grid positives=0.
- Adoption must run `uv run ng gate` before `uv run ng adopt`; never copy into `submission/overfit_nets/` manually.
- After adoption, `src/custom/task124.py`, candidate, and deployment must serialize to the same SHA-256.
- Do not run `ng pack` or `ng submit`.
- Preserve unrelated changes in the shared dirty worktree and stage exact task124 paths only.

---

## File map

- Create `candidates/task124/build_cost686_composite.py`: one stage-aware candidate builder with `conservative`, `routing_u8`, `routing_i32`, `primary`, and `primary_i32` endpoints.
- Create `candidates/task124/test_cost686_composite.py`: structural, cost, quantized-semantics, generator-domain, bundled raw A/B, and off-grid tests.
- Create `candidates/task124/verify_fresh_cost686.py`: isolated ORT_DISABLE_ALL fresh 2,000 raw/sign A/B and off-grid verifier.
- Generate `candidates/task124/cost686_*.onnx`: ignored local candidate checkpoints; never commit these binaries.
- Modify `src/custom/task124.py`: semantic source owner for the adopted endpoint only.
- Modify historical `candidates/task124/build_*.py`: update fixed-point SHA constants after source advances so prior stage regressions remain runnable.
- Modify `candidates/task124/DISCOVERY.md`: measured endpoint, proof, fallback results, adoption, SHA, and no-pack/no-submit record.
- Modify task124-specific sections of `state/tasks/task124.md`, `state/insights.yaml`, `state/levers.yaml`, and `state/STATE.md` only after adoption.

---

### Task 1: Add the RED composite acceptance contract

**Files:**
- Create: `candidates/task124/test_cost686_composite.py`
- Read: `candidates/task124/COST685_DESIGN.md`
- Read: `candidates/task124/test_rank4_qlinear_hash.py`

**Interfaces:**
- Consumes: deployed `submission/overfit_nets/task124.onnx`, `load_task(124)`, and the current source SHA `4e4bafbb3d65046a1ec08a211de6c9951705b613777a2a3ece9f4c73f6041b25`.
- Produces: the contract `build(task=None, *, stage: str = "primary") -> onnx.ModelProto` and exact stage-cost assertions.

- [ ] **Step 1: Write the failing import and shared test helpers**

Create the test with these concrete interfaces:

```python
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper
from neurogolf.scoring import convert_to_numpy, evaluate, load_task
from build_cost686_composite import build

ROOT = Path(__file__).resolve().parents[2]
INCUMBENT = ROOT / "submission/overfit_nets/task124.onnx"

EXPECTED = {
    "conservative": (659, 69, 728),
    "routing_u8": (633, 68, 701),
    "routing_i32": (644, 68, 712),
    "primary": (623, 63, 686),
    "primary_i32": (634, 63, 697),
}

def initializer(model, name):
    item = next(value for value in model.graph.initializer if value.name == name)
    return numpy_helper.to_array(item)

def producer_map(model):
    return {output: node for node in model.graph.node for output in node.output}

def ort_session(model):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
```

- [ ] **Step 2: Write stage cost and structure tests**

Add one parameterized scorer test and primary structural assertions:

```python
@pytest.mark.parametrize("stage", EXPECTED)
def test_stage_is_checker_legal_and_has_exact_cost(stage, tmp_path):
    model = build(stage=stage)
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    path = tmp_path / f"{stage}.onnx"
    onnx.save(model, path)
    result = evaluate(path, load_task(124), keep_failures=True)
    memory, params, cost = EXPECTED[stage]
    assert (result["fail"], result["memory"], result["params"]) == (0, memory, params)
    assert result["memory"] + result["params"] == cost

def test_primary_deletes_all_priced_intermediates():
    model = build(stage="primary")
    nodes = producer_map(model)
    assert nodes["output"].op_type == "QLinearConv"
    assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.UINT8
    assert not {
        "fg5", "bottom_fg", "fg02_u8", "left_cols", "shift_kernel",
        "p3_rows_b", "bottom_start_0", "bottom_start_1", "bottom_start_2",
        "bottom_start_3", "bottom_start_4",
    }.intersection(nodes)
    assert sum(node.op_type == "QLinearConv" for node in model.graph.node) == 3
    assert not any(node.op_type == "QLinearMatMul" for node in model.graph.node)
```

The `bottom_start_*` names above refer only to deleted Reshape outputs; the rank-1 Split outputs
must use distinct names such as `bottom_start_vec_*` so the assertion is unambiguous.

- [ ] **Step 3: Write quantized polarity and generator-domain tests**

Assert exact stored/effective values and expand the existing 1,428-state enumeration:

```python
def test_primary_quantized_polarity_is_exact():
    model = build(stage="primary")
    assert float(initializer(model, "ng_half_scale")) == 0.5
    assert int(initializer(model, "ng_x_zero_point")) == 1
    assert np.array_equal(initializer(model, "ng_mask_base").reshape(-1), [2] + [1] * 9)
    assert np.array_equal(initializer(model, "ng_background8"), np.full((1, 1, 1, 8), 2, np.uint8))
    assert np.array_equal(initializer(model, "ng_zero_update"), np.zeros((1, 1, 1, 1), np.uint8))
    assert int(initializer(model, "ng_y_zero_point")) == 0
    for colour in range(1, 10):
        stored = np.asarray([2] + [1] * 9, dtype=np.int16)
        stored[colour] = 0
        effective = stored - 1
        assert effective[0] == 1
        assert effective[colour] == -1
        assert np.count_nonzero(effective) == 2

def test_quantize_maps_only_legal_crop_values():
    raw = np.asarray([0.0, 1.0], dtype=np.float32)
    stored = np.rint(raw / np.float32(0.5)).astype(np.uint8)
    assert np.array_equal(stored, np.asarray([0, 2], dtype=np.uint8))
```

Copy the legal sprite enumeration from `test_rank4_qlinear_hash.py`, change the centered row to
`2 - foreground_row`, use `weights=[1,2,4,8,16,32]`, and assert for every case:

```python
assert np.array_equal(row1, row4) == (hash1 == hash4)
assert hash1 <= 126 and hash4 <= 126
assert candidate_from_full_rows == candidate_from_hash
assert cases == 1_428
```

Add the bundled period-1 example from `task_53b68214.validate()` and assert candidate 0.

- [ ] **Step 4: Write bundled raw A/B and off-grid tests**

For every bundled train/test/arc-gen example, run incumbent and all retained stages with
ORT_DISABLE_ALL and require:

```python
assert np.array_equal(candidate_raw, incumbent_raw)
assert np.count_nonzero(candidate_raw[..., 10:, :]) == 0
assert np.count_nonzero(candidate_raw[..., :, 10:]) == 0
```

- [ ] **Step 5: Run the focused test and verify RED**

Run:

```bash
uv run pytest -q candidates/task124/test_cost686_composite.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'build_cost686_composite'`.

- [ ] **Step 6: Commit the RED contract**

```bash
git add -f candidates/task124/test_cost686_composite.py
git diff --cached --check
git commit -m "test(task124): specify cost686 exact composite"
```

---

### Task 2: Implement the conservative cost728 endpoint

**Files:**
- Create: `candidates/task124/build_cost686_composite.py`
- Test: `candidates/task124/test_cost686_composite.py`

**Interfaces:**
- Consumes: `src.custom.task124.build(task)` at the immutable cost849 SHA.
- Produces: `build(stage="conservative")`, a checker-legal cost728 model used by later stages.

- [ ] **Step 1: Add the stage-aware builder shell and immutable-baseline guard**

Use this public interface and reject unknown stages:

```python
BASELINE_SHA256 = "4e4bafbb3d65046a1ec08a211de6c9951705b613777a2a3ece9f4c73f6041b25"
STAGES = {"conservative", "routing_u8", "routing_i32", "primary", "primary_i32"}

def build(task=None, *, stage="primary"):
    if stage not in STAGES:
        raise ValueError(f"unknown task124 stage: {stage}")
    from src.custom.task124 import build as build_source
    model = build_source(task)
    assert hashlib.sha256(model.SerializeToString()).hexdigest() == BASELINE_SHA256
    model = _build_conservative(model)
    if stage != "conservative":
        model = _build_scalar_route(model, use_uint8=stage in {"routing_u8", "primary"})
    if stage in {"primary", "primary_i32"}:
        model = _build_qlinear_hash(model)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    return model
```

- [ ] **Step 2: Rewrite initializers for inverse polarity and separate rows**

In `_build_conservative`, replace the exact named initializers with:

```python
replacement = {
    "row02_idx": np.asarray([0], dtype=np.int64),
    "false8": np.full((1, 1, 1, 8), 2, dtype=np.uint8),
    "ng_mask_base": np.asarray([2] + [1] * 9, dtype=np.uint8).reshape(10, 1, 1, 1),
    "ng_two": np.zeros((1, 1, 1, 1), dtype=np.uint8),
}
rename = {
    "row02_idx": "row0_idx",
    "false8": "ng_background8",
    "ng_two": "ng_zero_update",
}
remove = {"shift_kernel"}
append = {
    "row2_idx": np.asarray([2], dtype=np.int64),
    "ng_half_scale": np.asarray(0.5, dtype=np.float32),
}
```

Keep the mandatory scalar `ng_y_zero_point=0` on every opset12 QLinear zero-point input. Only the
optional `QuantizeLinear` output zero point may use an empty input string. Keep
`ng_x_zero_point=1` for the final input and weight zero points.

- [ ] **Step 3: Rewrite the conservative nodes**

Replace the encoder/geometry prefix with these exact operations:

```python
QuantizeLinear(ch0_first5, ng_half_scale, "") -> centered5
Gather(centered5, row0_idx, axis=2) -> row0_4d
ArgMin(row0_4d, axis=3, keepdims=0) -> left0_i64
Gather(centered5, row2_idx, axis=2) -> row2_4d
ArgMin(row2_4d, axis=3, keepdims=0) -> left2_i64
Sub(left2_i64, left0_i64) -> shift
```

Keep `p3_rows_a`, `p3_rows_b`, and the two QLinearMatMul hashes with `ng_y_zero_point` in all three
mandatory zero-point positions. Preserve the current rank-4 `is_p3 -> Where -> candidate` route.

Delete the old row Split and build `fg_pad4d` from:

```python
Concat(ng_background8, row0_4d, ng_background8, p3_rows_a, row2_4d, axis=3)
    -> fg_pad4d
```

Replace the two-step final mask construction with:

```python
Concat(centered5, bottom_row_0, bottom_row_1, bottom_row_2,
       bottom_row_3, bottom_row_4, axis=2) -> ng_centered_mask
```

Change ScatterElements updates from `ng_two` to `ng_zero_update`. Retain explicit
`ng_x_zero_point` for final X/W and `ng_y_zero_point` for the final output.

- [ ] **Step 4: Run conservative GREEN tests**

Run:

```bash
uv run pytest -q candidates/task124/test_cost686_composite.py -k 'conservative or polarity or bundled'
```

Expected: conservative scorer row is fail0, memory659, params69, cost728; bundled raw A/B and
off-grid tests pass.

- [ ] **Step 5: Commit the conservative endpoint**

```bash
git add -f candidates/task124/build_cost686_composite.py candidates/task124/test_cost686_composite.py
git diff --cached --check
git commit -m "feat(task124): add exact cost728 inverse route"
```

---

### Task 3: Implement scalar routing and its int32 fallback

**Files:**
- Modify: `candidates/task124/build_cost686_composite.py`
- Modify: `candidates/task124/test_cost686_composite.py`

**Interfaces:**
- Consumes: the conservative model's `left0_i64`, `left2_i64`, `is_p3`, and source-offset table.
- Produces: exact `routing_u8` cost701 and `routing_i32` cost712 candidate models.

- [ ] **Step 1: Add uint8 controller-node regression assertions**

Assert inferred scalar shapes/types for `shift_scalar`, `is_p3_scalar`, `candidate_u8`, and
`candidate_i32`; assert `source_offset` is int32 `[5]`; assert every `bottom_start_vec_i` is int32
`[1]`; assert no start Reshape exists.

- [ ] **Step 2: Implement `_build_scalar_route(model, use_uint8=True)`**

For the primary route, replace `shift` and `candidate` with:

```python
Cast(left0_i64, to=UINT8) -> left0_u8
Cast(left2_i64, to=UINT8) -> left2_u8
Sub(left2_u8, left0_u8) -> shift_u8
Squeeze(shift_u8, axes=[0,1,2]) -> shift_scalar
Squeeze(is_p3, axes=[0,1,2,3]) -> is_p3_scalar
Where(is_p3_scalar, three_u8, shift_scalar) -> candidate_u8
Cast(candidate_u8, to=INT32) -> candidate_i32
```

Change `three_i64` to scalar uint8 `three_u8=3`. Reshape `source_offsets` to int32 `[4,5]`, remove
`slice_start_shape`, Gather with scalar `candidate_i32`, Split axis0 into five rank-1 outputs named
`bottom_start_vec_0` through `bottom_start_vec_4`, and feed them directly to the five Add/Slice
chains.

- [ ] **Step 3: Implement the exact int32 fallback**

When `use_uint8=False`, use:

```python
Cast(left0_i64, to=INT32) -> left0_i32
Cast(left2_i64, to=INT32) -> left2_i32
Sub(left2_i32, left0_i32) -> shift_i32
Squeeze(shift_i32, axes=[0,1,2]) -> shift_scalar
Squeeze(is_p3, axes=[0,1,2,3]) -> is_p3_scalar
Where(is_p3_scalar, three_i32, shift_scalar) -> candidate_i32
```

Use scalar int32 `three_i32=3`; the source table and rank-1 Split route are otherwise identical.

- [ ] **Step 4: Run pinned ORT and exact-cost tests**

Run:

```bash
uv run pytest -q candidates/task124/test_cost686_composite.py -k 'routing_u8 or routing_i32 or bundled or generator'
```

Expected: ORT accepts uint8 Sub; routing_u8 is fail0/cost701 and routing_i32 is fail0/cost712;
both are bundled raw-identical and the full controller enumeration has zero mismatch.

- [ ] **Step 5: Commit scalar routing**

```bash
git add -f candidates/task124/build_cost686_composite.py candidates/task124/test_cost686_composite.py
git diff --cached --check
git commit -m "feat(task124): scalarize exact period routing"
```

---

### Task 4: Implement the six-tap direct-row4 QLinear hash

**Files:**
- Modify: `candidates/task124/build_cost686_composite.py`
- Modify: `candidates/task124/test_cost686_composite.py`

**Interfaces:**
- Consumes: routing_u8/routing_i32 models with `centered5` and `p3_rows_a`.
- Produces: primary cost686 and primary_i32 cost697 candidates, or a measured schema rejection that leaves the routing endpoints intact.

- [ ] **Step 1: Add the primary hash structural assertions**

Require initializer `hash6` to equal `[1,2,4,8,16,32]` with shape `[1,1,1,6]`; require two
hash QLinearConv nodes with output shape `[1,1,1,1]`; require the row-1 pads
`[0,0,0,-4]` and row-4 pads `[-4,0,0,-4]`; require `p3_rows_b`, `p3b_idx`, `hash_code`, and both
QLinearMatMul nodes to be absent.

- [ ] **Step 2: Implement `_build_qlinear_hash(model)`**

Remove `p3b_idx` and `hash_code`, append:

```python
hash6 = np.asarray([1, 2, 4, 8, 16, 32], dtype=np.uint8).reshape(1, 1, 1, 6)
```

Replace the row-4 Gather and two QLinearMatMul nodes with:

```python
helper.make_node(
    "QLinearConv",
    ["p3_rows_a", "ng_scale", "ng_y_zero_point", "hash6", "ng_scale", "ng_y_zero_point", "ng_scale", "ng_y_zero_point"],
    ["p3_hash_a"], pads=[0, 0, 0, -4],
)
helper.make_node(
    "QLinearConv",
    ["centered5", "ng_scale", "ng_y_zero_point", "hash6", "ng_scale", "ng_y_zero_point", "ng_scale", "ng_y_zero_point"],
    ["p3_hash_b"], pads=[-4, 0, 0, -4],
)
helper.make_node("Equal", ["p3_hash_a", "p3_hash_b"], ["is_p3"])
```

Add explicit uint8 value information `[1,1,1,1]` for both hash outputs before strict inference.

- [ ] **Step 3: Run primary exactness and cost tests**

Run:

```bash
uv run pytest -q candidates/task124/test_cost686_composite.py
```

Expected: all tests pass; primary is fail0/memory623/params63/cost686; primary_i32 is
fail0/memory634/params63/cost697; every stage is raw-identical on 267 bundled examples.

- [ ] **Step 4: Serialize every checkpoint without touching deployment**

```bash
uv run python - <<'PY'
from pathlib import Path
import onnx
from candidates.task124.build_cost686_composite import build
root = Path('candidates/task124')
for stage in ('conservative', 'routing_u8', 'routing_i32', 'primary', 'primary_i32'):
    onnx.save(build(stage=stage), root / f'cost686_{stage}.onnx')
    print(stage, root / f'cost686_{stage}.onnx')
PY
```

Expected: five candidate paths print; deployed SHA remains the cost849 SHA.

- [ ] **Step 5: Commit the primary hash**

```bash
git add -f candidates/task124/build_cost686_composite.py candidates/task124/test_cost686_composite.py
git diff --cached --check
git commit -m "feat(task124): add exact cost686 qlinear composite"
```

---

### Task 5: Run fresh diagnostics, select one endpoint, gate, and adopt

**Files:**
- Create: `candidates/task124/verify_fresh_cost686.py`
- Generate: `candidates/task124/cost686_*.onnx`
- Modify through CLI on success: `submission/overfit_nets/task124.onnx`, `state/manifest.json`, `state/tasks/task124.md`

**Interfaces:**
- Consumes: all locally passing stage models and the immutable cost849 incumbent.
- Produces: one cheapest exact adopted endpoint, or no deployment change.

- [ ] **Step 1: Write the isolated fresh raw/sign/off-grid verifier**

The script must import `tasks.task_53b68214`, build ORT_DISABLE_ALL incumbent/candidate sessions,
generate until exactly 2,000 valid examples run, and maintain these counters:

```python
counters = {
    "runs": 0,
    "incumbent_fail": 0,
    "candidate_fail": 0,
    "raw_divergence": 0,
    "sign_divergence": 0,
    "off_grid_positives": 0,
}
```

For each raw candidate output, zero a copy's `[..., :10, :10]` region and add
`np.count_nonzero(off_grid > 0)`. Exit nonzero unless runs=2000 and every other counter is zero.
Accept `--stage` so each retained endpoint can be checked independently.

- [ ] **Step 2: Run focused and historical regressions**

```bash
uv run pytest -q candidates/task124/test_cost686_composite.py
uv run pytest -q candidates/task124/test_*.py
```

Expected: composite tests pass and the complete historical task124 suite remains green.

- [ ] **Step 3: Run fresh 2,000 on candidates from cheapest to most conservative**

Run primary first, then only the necessary fallback if a cheaper stage fails schema/runtime or
exactness:

```bash
uv run python candidates/task124/verify_fresh_cost686.py --stage primary --n 2000
```

Expected for the selected stage: runs2000, incumbent_fail0, candidate_fail0, raw_divergence0,
sign_divergence0, off_grid_positives0.

- [ ] **Step 4: Independently score and select the cheapest passing endpoint**

Run `evaluate` in a fresh process for all built candidates and print stage/memory/params/cost/fail.
Select the minimum-cost stage with checker, ORT, bundled, generator-domain, and fresh results all
passing. Do not select a fallback merely because it was tested first.

- [ ] **Step 5: Run the mandatory official gate**

For the selected file:

```bash
uv run ng gate candidates/task124/cost686_primary.onnx --task 124
```

Use the actual selected suffix when a fallback wins. Expected: PASS, bundled267/267, fail0,
selected cost lower than 849. On rejection, stop without adopt and preserve deployment.

- [ ] **Step 6: Adopt only after gate PASS**

```bash
uv run ng adopt candidates/task124/cost686_primary.onnx --task 124 --note "inverse quantized mask, scalar routing, six-tap direct QLinear row hash"
```

Use the actual selected suffix and a truthful note. Confirm `uv run ng score 124` reports fail0
and the selected measured cost. Do not run pack or submit.

- [ ] **Step 7: Commit the fresh verifier but not generated ONNX files**

```bash
git add -f candidates/task124/verify_fresh_cost686.py
git diff --cached --check
git commit -m "test(task124): verify cost686 fresh exactness"
```

---

### Task 6: Synchronize semantic source, records, and fixed-point builders

**Files:**
- Modify: `src/custom/task124.py`
- Modify: `candidates/task124/build_cost686_composite.py`
- Modify: `candidates/task124/build_dynamic_qlinear_tail.py`
- Modify: `candidates/task124/build_centered_qlinear_tail.py`
- Modify: `candidates/task124/build_scatter_weight.py`
- Modify: `candidates/task124/build_centered_slice_geometry.py`
- Modify: `candidates/task124/build_shared_initializers.py`
- Modify: `candidates/task124/build_rank4_qlinear_hash.py`
- Modify: `candidates/task124/DISCOVERY.md`
- Modify: `state/tasks/task124.md`
- Modify task124-specific entries: `state/insights.yaml`, `state/levers.yaml`, `state/STATE.md`

**Interfaces:**
- Consumes: adopted candidate stage, measured score, adoption timestamp, and final SHA.
- Produces: byte-identical source ownership and durable task124 handoff.

- [ ] **Step 1: Rewrite `src/custom/task124.py` semantically**

Construct the adopted stage directly rather than calling a candidate surgery helper. Preserve the
docstring's exact stored/effective mask and weight tables. Build in a fresh process and assert:

```python
source_sha = hashlib.sha256(build_source(load_task(124)).SerializeToString()).hexdigest()
deployed_sha = hashlib.sha256(Path("submission/overfit_nets/task124.onnx").read_bytes()).hexdigest()
assert source_sha == deployed_sha == FINAL_SHA256
```

- [ ] **Step 2: Make the composite builder a post-adoption fixed point**

Add `ADOPTED_SHA256`. If source already equals it, return the source build for the adopted stage;
otherwise require the immutable cost849 baseline before applying transformations. Update every
historical task124 builder's final/live SHA constant so it uses its pinned stage artifact when the
semantic source has advanced.

- [ ] **Step 3: Synchronize task124 discovery and state**

Record measured, not predicted, values for the complete cost849-to-final delta; all stage outcomes;
mandatory QLinear zero-point legality; uint8 underflow proof or fallback reason; six-column minimal-prefix
enumeration; bundled/fresh/off-grid results; gate/adopt evidence; final SHA; and the next `+0.1`
threshold. State explicitly that pack/submit did not run.

Append a new task124 lever ledger record with `date`, `ran`, `verdict`, `reopen`, and
`falsification_history`. Update the reusable insight only for mechanisms actually adopted. Replace
the task124 section in `state/STATE.md`; do not append a duplicate section.

- [ ] **Step 4: Run a read-only 400-graph signature rescan**

Scan deployed models for these exact/broad signatures without editing any task:

- `Less -> Where` over exact binary float crops that can invert through QuantizeLinear;
- redundant intermediate Concat feeding another Concat on the same axis;
- row Gather used only before a bounded six-prefix equality hash;
- non-scalar Gather indices that force shape-only Reshapes before Slice.

Record hit counts, task ids, and load errors in `DISCOVERY.md` and the adopted insight entry. The
legacy recursive inventory scripts are absent, so do not claim that unavailable queue was run.

- [ ] **Step 5: Run final verification-before-completion commands**

```bash
uv run ng score 124
uv run pytest -q candidates/task124/test_*.py
uv run python candidates/task124/verify_fresh_cost686.py --stage primary --n 2000
git diff --check -- src/custom/task124.py candidates/task124 state/tasks/task124.md state/insights.yaml state/levers.yaml state/STATE.md
```

Use the adopted stage name in the fresh command. Require scorer fail0, full focused suite green,
fresh counters all zero, and candidate/source/deployed SHA equality.

- [ ] **Step 6: Commit exact task124 scope**

Stage only the semantic source, task124 builders/tests/verifier/design/plan/discovery, task ledger,
and task124-specific state records. Inspect `git diff --cached --name-only` before commit:

```bash
git commit -m "opt(task124): adopt exact cost686 qlinear composite" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

Do not push, pack, or submit.
