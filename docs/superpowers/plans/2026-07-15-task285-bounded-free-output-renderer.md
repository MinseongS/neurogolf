# task285 Bounded FREE-Output Renderer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace task285's counted sparse Scatter/label-decoder suffix with a bounded analytical renderer costing at most 14,511 while preserving bundled 265/265 correctness.

**Architecture:** Reuse the deployed graph only through the exact three-pivot orientation tensors `a`, `acol`, `e`, and `f`. A 12-operand fp32 Einsum reconstructs a counted 300-byte `[1,3,5,5]` same-colour source mask. A 19-operand terminal Einsum reads the FREE input, applies four affine quadrant relations, and emits the FREE output directly; its two branches are identity and background-only stamp replacement.

**Tech Stack:** Python 3.12, NumPy, ONNX 1.21.0, ONNX Runtime 1.26.0, pytest, NeuroGolf `ng gate`/`ng adopt`.

## Global Constraints

- All candidate ONNX files, builders, profiles, and scratch tests stay under `candidates/task285/`.
- Every TopK input remains fp16 or fp32; integer TopK is forbidden.
- The primary candidate must have predicted cost <=14,511, counted intermediates <=3,600 bytes, source-shape Einsum arity <=12, terminal Einsum arity <=20, and estimated largest contraction intermediate <=1,000,000 elements.
- Abandon an unchanged candidate after session creation failure, first inference >1 second, or no output within 10 seconds.
- Adoption is only `uv run ng gate ... --task 285` followed by `uv run ng adopt ... --task 285`; never edit `submission/overfit_nets/task285.onnx` directly.
- Bundled 265/265 fail=0 is authoritative. Fresh seed285/example80 has an inherited neighbour-Gather OOB and is diagnostic only.
- Do not run `ng pack` or `ng submit` in this standalone task session.

---

### Task 1: Lock the affine and polynomial contracts with RED tests

**Files:**
- Create: `candidates/task285/scratch/tests/test_bounded_free_output_renderer.py`
- Reference: `candidates/task285/build_affine_reflection.py`
- Reference: `candidates/task285/build_sparse_renderer_candidate.py`

**Interfaces:**
- Consumes: deployed task285 input/output semantics and the period-31 `relation_core()` identity.
- Produces: failing acceptance tests for `build_bounded_free_output_renderer.py` and `bounded_free_output_renderer.onnx`.

- [ ] **Step 1: Write the failing builder/artifact test**

```python
ROOT = pathlib.Path(__file__).resolve().parents[4]
BUILDER = ROOT / "candidates/task285/build_bounded_free_output_renderer.py"
MODEL = ROOT / "candidates/task285/bounded_free_output_renderer.onnx"


def test_bounded_free_output_builder_exists():
    assert BUILDER.exists(), "bounded FREE-output renderer builder is not implemented"


def test_bounded_free_output_model_exists():
    assert MODEL.exists(), "bounded FREE-output renderer artifact is not built"
```

- [ ] **Step 2: Write the exact destination-colour polynomial test**

```python
def test_background_stamp_delta_replaces_only_the_destination_colour():
    classes = np.arange(10, dtype=np.float32)
    baseline = 1.0 - classes**2
    for colour in range(10):
        delta = colour * (2.0 * classes - colour)
        actual = baseline + delta
        assert np.array_equal(actual > 0, classes == colour)
```

- [ ] **Step 3: Write structural, cost, and TopK safety tests**

```python
def test_bounded_model_has_bounded_two_stage_contractions():
    inferred = onnx.shape_inference.infer_shapes(onnx.load(MODEL), strict_mode=True)
    einsums = [node for node in inferred.graph.node if node.op_type == "Einsum"]
    assert [len(node.input) for node in einsums[-2:]] == [12, 19]
    shape = value_shape(inferred, "source_shape")
    assert shape == [1, 3, 5, 5]
    assert largest_counted_bytes(inferred) <= 3_600
    assert estimate_largest_intermediate(inferred) <= 1_000_000


def test_bounded_model_reaches_next_point_one_budget():
    model = onnx.load(MODEL)
    result = profile(MODEL)
    assert result is not None
    params = calculate_params(model)
    assert params is not None
    assert int(result["memory_static"]) + params <= 14_511


def test_all_topk_inputs_are_float16_or_float32():
    inferred = onnx.shape_inference.infer_shapes(onnx.load(MODEL), strict_mode=True)
    types = value_types(inferred)
    for node in inferred.graph.node:
        if node.op_type == "TopK":
            assert types[node.input[0]] in {TensorProto.FLOAT16, TensorProto.FLOAT}
```

- [ ] **Step 4: Run RED and confirm the missing builder is the failure**

Run:

```bash
PYTHONPATH=src:. uv run pytest -q candidates/task285/scratch/tests/test_bounded_free_output_renderer.py
```

Expected: FAIL at `test_bounded_free_output_builder_exists` because the builder does not exist; the polynomial unit test passes.

- [ ] **Step 5: Commit the RED contract**

```bash
git add candidates/task285/scratch/tests/test_bounded_free_output_renderer.py
git commit -m "test: specify task285 bounded output renderer"
```

### Task 2: Build the bounded source-shape and terminal contractions

**Files:**
- Create: `candidates/task285/build_bounded_free_output_renderer.py`
- Generate: `candidates/task285/bounded_free_output_renderer.onnx`
- Modify: `candidates/task285/scratch/tests/test_bounded_free_output_renderer.py`

**Interfaces:**
- Consumes: immutable seed `candidates/task285/sparse_renderer_candidate.onnx` with SHA-256 `b201dd621c751a7a9a9352cf7fe6fbc6ac9d35c5c701857ee06953c037b148d4`.
- Produces: `build_model() -> onnx.ModelProto`, `materialize_relation_delta() -> np.ndarray`, `estimate_largest_intermediate(model) -> int`, and the candidate ONNX.

- [ ] **Step 1: Implement Fourier features and the exact affine relation**

Reuse the period-31 formulas from `build_affine_reflection.py` without loading any external array. `trig_embedding(30)`, `trig_embedding(5)`, and `relation_core()` must satisfy:

```python
delta = np.einsum("xmp,bmq,imr,smpqr->sbix", e30, e30, e5, relation_core())
expected = np.zeros((2, 30, 5, 30), dtype=np.float32)
for side in range(2):
    for pivot in range(30):
        for offset in range(5):
            target = pivot - offset if side == 0 else pivot + 1 + offset
            if 0 <= target < 30:
                expected[side, pivot, offset, target] = 1.0
assert np.max(np.abs(delta - expected)) < 2e-5
```

- [ ] **Step 2: Reuse the incumbent prefix and derive compact pivot features**

Clone nodes through output `f` (node 44), then add:

```python
Less(e, f0) -> e_negative
Less(f, f0) -> f_negative
Cast(e_negative, INT32) -> e_bit
Cast(f_negative, INT32) -> f_bit
pivot_flat = a - e_bit - 30 * f_bit
pivot_row = Div(pivot_flat, 30)
pivot_col = Mod(pivot_flat, 30)
pivot_row_feature = Gather(e30, pivot_row)
pivot_col_feature = Gather(e30, pivot_col)
orientation_row = OneHot(f_bit, depth=2, values=[0.0, 1.0])
orientation_col = OneHot(e_bit, depth=2, values=[0.0, 1.0])
valid = Cast(Reshape(av3, [3]), FLOAT)
root_colour = OneHot(acol, depth=10, values=[0.0, 1.0])
```

Build destination-label features from `Gather(g, pivot_flat + [0,1,30,31])`. Cast labels to fp32, unsqueeze them, and concatenate a `[3,4,1]` initializer of ones to form `destination_colour_feature[3,4,2] = [1,h]`.

- [ ] **Step 3: Emit the 300-byte source-shape Einsum**

Use this exact 12-input equation:

```python
equation = (
    "pds,nsEF,pu,pv,pma,imb,Emc,umabc,"
    "pMd,jMe,FMf,vMdef->npij"
)
```

Inputs in order are `root_colour`, FREE `input`, `orientation_row`, `orientation_col`, `pivot_row_feature`, `e5`, `e30`, `k_source`, `pivot_col_feature`, `e5`, `e30`, `k_source`. Declare `source_shape` as fp32 `[1,3,5,5]`.

- [ ] **Step 4: Emit the 19-input FREE-output Einsum**

Use this exact equation:

```python
equation = (
    "npij,nkyx,tzk,p,tp,tij,pqb,pqr,tzbr,qU,qV,"
    "pma,imb,ymc,tUmabc,pMd,jMe,xMf,tVMdef->nzyx"
)
```

The static factors are:

```python
input_mixer = np.zeros((2, 10, 10), np.float32)
input_mixer[0] = 2 * np.eye(10, dtype=np.float32) - 1
input_mixer[1, :, 0] = 1

colour_core = np.zeros((2, 10, 2, 2), np.float32)
colour_core[0, :, 0, 0] = 1
colour_core[1, :, 1, 0] = 2 * np.arange(10)
colour_core[1, :, 1, 1] = -1

local_gate = np.ones((2, 5, 5), np.float32)
local_gate[0] = 0
local_gate[0, 0, 0] = 1
local_gate[1, 0, 0] = 0

pivot_gate = np.ones((2, 3), np.float32)
pivot_gate[0, 1:] = 0

quadrant_row = np.array([[1,0],[1,0],[0,1],[0,1]], np.float32)
quadrant_col = np.array([[1,0],[0,1],[1,0],[0,1]], np.float32)

k_output = np.zeros((2, *relation_core().shape), np.float32)
k_output[0, 0, 0, 0, 0, 0] = 1
k_output[1] = relation_core()
```

The terminal output is the model's existing fp32 `[1,10,30,30]` graph output. Remove every unused suffix node, initializer, and value-info entry.

- [ ] **Step 5: Add build-time rejection and inventory output**

`build_model()` must run `onnx.checker.check_model(full_check=True)` and strict shape inference, reject non-float TopK feeds, assert the two final Einsum arities `[12,19]`, assert every counted non-output tensor is <=3,600 bytes, and assert estimated largest contraction intermediate <=1,000,000 elements. `main()` saves the model, prints each counted tensor, memory, parameters, total cost, Einsum arities, and the contraction estimate, then raises if total cost exceeds 14,511.

- [ ] **Step 6: Build and run GREEN structural tests**

Run:

```bash
PYTHONPATH=src:. uv run python candidates/task285/build_bounded_free_output_renderer.py
PYTHONPATH=src:. uv run pytest -q candidates/task285/scratch/tests/test_bounded_free_output_renderer.py
```

Expected: builder exits 0; tests pass; printed cost <=14,511; arities are `[12,19]`.

- [ ] **Step 7: Commit the bounded compiler**

```bash
git add candidates/task285/build_bounded_free_output_renderer.py candidates/task285/bounded_free_output_renderer.onnx candidates/task285/scratch/tests/test_bounded_free_output_renderer.py
git commit -m "feat: build task285 bounded free-output renderer"
```

### Task 3: Validate runtime and bundled correctness before gate

**Files:**
- Modify: `candidates/task285/scratch/tests/test_bounded_free_output_renderer.py`
- Create if needed: `candidates/task285/scratch/runtime_bounded_free_output.py`

**Interfaces:**
- Consumes: built candidate and `data/task285.json` through existing NeuroGolf loaders.
- Produces: fresh-process first-output timing, exact bundled comparison, and a reusable regression test.

- [ ] **Step 1: Add a clean-process one-example runtime test**

The subprocess creates an ORT session with graph optimizations disabled, runs stored example 0 once, prints `load_seconds`, `first_inference_seconds`, and the output shape, and enforces a 10-second parent timeout. Assert session creation succeeds, first inference is <1.0 second, and output shape is `[1,10,30,30]`.

- [ ] **Step 2: Run checker and isolated runtime**

Run:

```bash
PYTHONPATH=src:. uv run python candidates/task285/scratch/runtime_bounded_free_output.py
```

Expected: load succeeds, first output appears within one second, output shape `[1,10,30,30]`.

- [ ] **Step 3: Add and run the bundled 265-case test**

Use the same loader/evaluator as the existing task285 tests. Assert every thresholded candidate output equals the stored expected output and report mismatching example indices rather than weakening the oracle.

Run:

```bash
PYTHONPATH=src:. uv run pytest -q candidates/task285/scratch/tests/test_bounded_free_output_renderer.py
```

Expected: all tests pass, bundled 265/265.

- [ ] **Step 4: Run the complete focused regression suite**

Run:

```bash
PYTHONPATH=src:. uv run pytest -q \
  candidates/task285/scratch/tests/test_bounded_free_output_renderer.py \
  candidates/task285/scratch/tests/test_sparse_renderer_candidate.py \
  candidates/task285/scratch/tests/test_runtime_candidate.py \
  candidates/task285/scratch/tests/test_affine_reflection.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit runtime and bundled evidence**

```bash
git add candidates/task285/scratch/tests/test_bounded_free_output_renderer.py candidates/task285/scratch/runtime_bounded_free_output.py
git commit -m "test: verify task285 bounded renderer runtime"
```

### Task 4: Mandatory gate, adoption, and exact-source synchronization

**Files:**
- Modify: `src/custom/task285.py`
- Modify: `candidates/task285/DISCOVERY.md`
- Generated by adoption: `state/tasks/task285.md`, `state/manifest.json`, `state/SCOREBOARD.md`, and related gate state.

**Interfaces:**
- Consumes: verified `bounded_free_output_renderer.onnx`.
- Produces: adopted deployment and a self-contained Python source rebuilding identical bytes.

- [ ] **Step 1: Reconfirm incumbent and candidate immediately before gate**

Run:

```bash
shasum -a 256 submission/overfit_nets/task285.onnx candidates/task285/bounded_free_output_renderer.onnx
uv run ng score 285
```

Expected incumbent: bundled fail=0, deployed cost 16,038 unless another authorized task285 adoption occurred. If the deployed SHA changed, stop and reconcile instead of overwriting it.

- [ ] **Step 2: Run the mandatory gate**

Run:

```bash
uv run ng gate candidates/task285/bounded_free_output_renderer.onnx --task 285
```

Expected: PASS, 265/265, fail=0, cost <=14,511 and strictly below deployed.

- [ ] **Step 3: Adopt only the passing candidate**

Run:

```bash
uv run ng adopt candidates/task285/bounded_free_output_renderer.onnx --task 285 --note "bounded sparse-pivot Fourier renderer: 300B source-shape contraction plus polynomial FREE-output stamp decoder"
```

Expected: adoption reports the old and new costs and fail=0.

- [ ] **Step 4: Synchronize the canonical exact source**

Regenerate `src/custom/task285.py` with the repository's exact-source workflow so it embeds every initializer and node without reading `submission/`, `candidates/`, `.onnx`, or `.npy` files. Build it in a fresh process and assert source-built, candidate, and deployed serialized bytes have the same SHA-256.

- [ ] **Step 5: Run fresh post-adopt verification**

Run:

```bash
PYTHONPATH=src:. uv run pytest -q \
  candidates/task285/scratch/tests/test_bounded_free_output_renderer.py \
  candidates/source_sync_285_295/test_task285_295_source_sync.py
uv run ng score 285
```

Expected: all tests pass; score reports 265/265, fail=0, and the adopted cost.

- [ ] **Step 6: Document and commit the adopted endpoint**

Update `candidates/task285/DISCOVERY.md` with cost, memory, params, score delta, runtime, gate/adopt output, SHA, exact mechanism, and the inherited fresh limitation. Stage only task285-owned files plus adoption-generated state rows and commit:

```bash
git commit -m "optimize task285 bounded free-output renderer"
```

### Task 5: Execute the strict-cheaper fallback only if the analytical renderer is rejected

**Files:**
- Create: `candidates/task285/build_onehot_output_candidate.py`
- Create: `candidates/task285/onehot_output_candidate.onnx`
- Modify: `candidates/task285/scratch/tests/test_bounded_free_output_renderer.py`
- Modify: `candidates/task285/DISCOVERY.md`

**Interfaces:**
- Consumes: immutable sparse-renderer seed SHA `b201dd621c751a7a9a9352cf7fe6fbc6ac9d35c5c701857ee06953c037b148d4`.
- Produces: a strict-cheaper fallback that replaces `Reshape(newg)->Equal(ar10)` with `Reshape(newg,[1,30,30])->OneHot(axis=1)`.

- [ ] **Step 1: Write and verify a RED terminal-OneHot test**

Assert the candidate exists, has a BOOL `[1,10,30,30]` OneHot graph output, preserves fp16 TopK feeds, passes strict shape inference, and has cost <16,038.

- [ ] **Step 2: Implement the minimal fallback builder**

Remove initializer `ar10` and the old `sh2d`; add int64 depth `10`, BOOL values `[False, True]`, and shape `[1,30,30]`. Replace nodes 94/95 with:

```python
helper.make_node("Reshape", ["newg", "shape_1x30x30"], ["out3d"])
helper.make_node("OneHot", ["out3d", "depth10", "onehot_values"], ["output"], axis=1)
```

- [ ] **Step 3: Run bundled validation and mandatory gate/adopt**

Run the focused tests, `ng gate`, then `ng adopt` only if bundled fail=0 and the measured cost is strictly below the then-current deployment. Continue to document the analytical rejection boundary; do not claim the fallback reaches +0.1.

- [ ] **Step 4: Synchronize source and commit fallback evidence**

Rebuild `src/custom/task285.py`, verify candidate/source/deployed SHA identity, update `DISCOVERY.md`, and commit only scoped files.

### Task 6: Reusable insight and board-safe handoff

**Files:**
- Modify only if reusable: `state/insights.yaml`
- Modify: `state/STATE.md` by replacement, not append
- Modify: `state/levers.yaml` only through its four-field ledger if a lever is exhausted

**Interfaces:**
- Consumes: final adopted or rejected experimental evidence.
- Produces: current handoff and optional cross-task insight/rescan.

- [ ] **Step 1: Register the mechanism only if it generalized**

If the bounded source-shape plus polynomial label-delta renderer is reusable, add one dated insight with task285 evidence and run its prescribed 400-task structural rescan. Otherwise leave `state/insights.yaml` unchanged.

- [ ] **Step 2: Replace the live STATE handoff**

Record the final task285 cost/SHA, gate evidence, source equality, runtime, fresh limitation, and exact next pivot. Preserve concurrent sessions' current sections and replace the file rather than appending.

- [ ] **Step 3: Run final scoped verification**

Run:

```bash
uv run ng score 285
PYTHONPATH=src:. uv run pytest -q \
  candidates/task285/scratch/tests/test_bounded_free_output_renderer.py \
  candidates/task285/scratch/tests/test_sparse_renderer_candidate.py \
  candidates/task285/scratch/tests/test_runtime_candidate.py \
  candidates/task285/scratch/tests/test_affine_reflection.py \
  candidates/source_sync_285_295/test_task285_295_source_sync.py
git diff --check
```

Expected: task285 fail=0; all scoped tests pass; no whitespace errors. Report no pack/submit action.
