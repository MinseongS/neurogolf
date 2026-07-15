# Task271 Bitwise/Where Decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace task271's Mod/compare/Cast/BitShift payload decoder with the pinned-ORT-compatible UINT16 BitwiseAnd and UINT8 Where decoder, reducing cost366 to cost350.

**Architecture:** Keep the score Conv and value-only MaxPool byte-identical. Decode the winning scalar's low nine bits by broadcasting UINT16 masks, casting nonzero masks to BOOL, and selecting centered UINT8 values 0/2 in one Where node; the existing zero-point-centered ConvInteger remains the FREE-output renderer.

**Tech Stack:** Python 3.13, ONNX 1.21.0, ONNX Runtime 1.26.0, NumPy, pytest, `uv run ng`.

## Global Constraints

- Modify only task271 artifacts and shared records whose edited clauses are explicitly about task271.
- Work in the user-approved shared `main` checkout; do not create a worktree or stage unrelated dirty files.
- Keep onnx==1.21.0 and onnxruntime==1.26.0.
- Candidate and scratch artifacts stay under `candidates/task271/`.
- Require cost350, bundled fail0, 49/49 full-graph position equivalence, 512/512 decoder equivalence, fresh divergence0, two-pass fixed point, and exact candidate/deployed/network/source SHA equality.
- Use `ng gate` before adoption and only `ng adopt` to replace the deployed graph.
- Do not restore dual ReduceMax/ArgMax, `cyan_patch/crop8/Pad`, full-canvas Cast, or the rejected INT8 Where route.
- Treat cost350 as a safe below-threshold win. The next `+0.1` threshold after adoption is cost<=316.

---

### Task 1: Pin the cost350 graph contract with a RED test

**Files:**
- Create: `candidates/task271/test_bitwise_where_decoder.py`
- Read: `candidates/task271/encoded_score_payload.onnx`

**Interfaces:**
- Consumes: pinned cost366 artifact SHA `9f7303ea8e393536079fba14c87fec7b5e0908cf7b95917af2b3eb0b35f2e37d`.
- Produces: a regression contract for `build_bitwise_where_decoder.py::build_candidate(incumbent: Path, output: Path) -> None`.

- [ ] **Step 1: Write the failing test**

Create the test with these pinned constants and helper values:

```python
BASELINE = Path(__file__).with_name("encoded_score_payload.onnx")
BASELINE_SHA256 = "9f7303ea8e393536079fba14c87fec7b5e0908cf7b95917af2b3eb0b35f2e37d"
EXPECTED_OPS = [
    "Conv", "MaxPool", "Cast", "BitwiseAnd", "Cast", "Where", "ConvInteger",
]

def payload_masks() -> np.ndarray:
    return (1 << np.arange(9, dtype=np.uint16)).reshape(1, 1, 3, 3)

def signed_render_weight() -> np.ndarray:
    weight = np.zeros((10, 1, 1, 1), dtype=np.int8)
    weight[1, 0, 0, 0] = 1
    weight[8, 0, 0, 0] = -1
    return weight
```

The test imports the not-yet-existing builder, generates a temporary candidate, and asserts:

```python
assert default_opset == 18
assert [node.op_type for node in model.graph.node] == EXPECTED_OPS
assert list(model.graph.node[1].output) == ["max_score"]
assert list(model.graph.node[3].input) == ["encoded_u16", "payload_masks"]
assert helper.get_attribute_value(model.graph.node[4].attribute[0]) == TensorProto.BOOL
assert list(model.graph.node[5].input) == ["payload_bits", "two_u8", "zero_u8"]
assert list(model.graph.node[6].input) == [
    "blue2", "render_weight_signed", "one_u8",
]
assert not {"Mod", "GreaterOrEqual", "BitShift"} & {
    node.op_type for node in model.graph.node
}
```

Assert exact initializer names and values:

```python
assert set(initializers) == {
    "score_payload_kernel", "payload_masks", "two_u8", "zero_u8",
    "one_u8", "render_weight_signed",
}
np.testing.assert_array_equal(initializers["payload_masks"], payload_masks())
np.testing.assert_array_equal(initializers["two_u8"], [2])
np.testing.assert_array_equal(initializers["zero_u8"], [0])
np.testing.assert_array_equal(initializers["one_u8"], [1])
np.testing.assert_array_equal(
    initializers["render_weight_signed"], signed_render_weight()
)
```

Create an ORT session with `ORT_DISABLE_ALL`. Exercise the complete candidate with the pinned
patch code `0b101011001` at every `flat_index in range(49)` and require exact blue/cyan output.
Build a small decoder-only in-memory model using the candidate's BitwiseAnd/Cast/Where/
ConvInteger nodes, run `np.arange(512, dtype=np.uint16)`, and require all 512 decoded patches
plus zero off-grid positives.

Finish with the price, bundled, persistent, and fixed-point assertions:

```python
assert static_cost(candidate) == 350
result = eval_isolated(candidate, 271)
assert result["ok"] and result["pass"] == 267 and result["fail"] == 0
assert result["memory"] == 238 and result["params"] == 112
assert candidate.read_bytes() == persistent.read_bytes()
module.build_candidate(candidate, second_pass)
assert second_pass.read_bytes() == candidate.read_bytes()
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
PYTHONPATH=. uv run pytest -q candidates/task271/test_bitwise_where_decoder.py
```

Expected: one failure stating that `candidates/task271/build_bitwise_where_decoder.py` is missing.

- [ ] **Step 3: Commit the RED contract**

```bash
git add candidates/task271/test_bitwise_where_decoder.py
git commit -m "test(task271): specify bitwise where decoder"
```

Expected: a commit containing only the new task271 test.

---

### Task 2: Implement the fixed-point builder and make the test GREEN

**Files:**
- Create: `candidates/task271/build_bitwise_where_decoder.py`
- Create: `candidates/task271/bitwise_where_decoder.onnx`
- Test: `candidates/task271/test_bitwise_where_decoder.py`

**Interfaces:**
- Consumes: the pinned cost366 artifact or an already-valid cost350 artifact.
- Produces: `build_candidate(incumbent: Path, output: Path) -> None` and a persistent ONNX candidate.

- [ ] **Step 1: Define the immutable graph contract**

Use these constants and constructors:

```python
DEFAULT_INCUMBENT = Path(__file__).with_name("encoded_score_payload.onnx")
DEFAULT_OUTPUT = Path(__file__).with_name("bitwise_where_decoder.onnx")
BASELINE_SHA256 = "9f7303ea8e393536079fba14c87fec7b5e0908cf7b95917af2b3eb0b35f2e37d"
BASELINE_OPS = [
    "Conv", "MaxPool", "Cast", "Mod", "GreaterOrEqual", "Cast",
    "BitShift", "ConvInteger",
]
FOLDED_OPS = [
    "Conv", "MaxPool", "Cast", "BitwiseAnd", "Cast", "Where", "ConvInteger",
]

def _payload_masks() -> np.ndarray:
    return (1 << np.arange(9, dtype=np.uint16)).reshape(1, 1, 3, 3)

def _centered_values() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([2], dtype=np.uint8),
        np.array([0], dtype=np.uint8),
        np.array([1], dtype=np.uint8),
    )
```

Reuse the exact `_score_payload_kernel()` and `_signed_render_weight()` formulas from the
cost366 builder so validation rejects any semantic drift.

- [ ] **Step 2: Implement strict validation and fixed-point input handling**

`_validate_folded(model)` must require default opset18, `FOLDED_OPS`, graph name
`task271_bitwise_where_decoder`, the unchanged score Conv/MaxPool geometry, Cast targets
UINT16 and BOOL, exact Where and ConvInteger wiring, exact initializer set/values, INT32
output, strict shape inference, and full checker.

At the start of `build_candidate`:

```python
model = onnx.load(incumbent)
ops = [node.op_type for node in model.graph.node]
if ops == FOLDED_OPS:
    _validate_folded(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    return
if hashlib.sha256(incumbent.read_bytes()).hexdigest() != BASELINE_SHA256:
    raise ValueError("task271 input is not the pinned cost366 baseline")
if ops != BASELINE_OPS or _default_opset(model) != 12:
    raise ValueError("unexpected task271 cost366 baseline")
```

- [ ] **Step 3: Rewrite only the decoder**

Replace all initializers with the exact score kernel, masks, three UINT8 scalars, and signed
renderer weight. Set the default opset to 18 and replace the graph nodes with:

```python
helper.make_node(
    "Conv", ["input", "score_payload_kernel"], ["scores"],
    kernel_shape=[3, 3], pads=[0, 0, -21, -21],
)
helper.make_node("MaxPool", ["scores"], ["max_score"], kernel_shape=[7, 7])
helper.make_node("Cast", ["max_score"], ["encoded_u16"], to=TensorProto.UINT16)
helper.make_node(
    "BitwiseAnd", ["encoded_u16", "payload_masks"], ["masked_u16"],
)
helper.make_node("Cast", ["masked_u16"], ["payload_bits"], to=TensorProto.BOOL)
helper.make_node(
    "Where", ["payload_bits", "two_u8", "zero_u8"], ["blue2"],
)
helper.make_node(
    "ConvInteger", ["blue2", "render_weight_signed", "one_u8"], ["output"],
    kernel_shape=[1, 1], pads=[0, 0, 27, 27],
)
```

Declare exact value infos for `scores FLOAT[1,1,7,7]`, `max_score FLOAT[1,1,1,1]`,
`encoded_u16 UINT16[1,1,1,1]`, `masked_u16 UINT16[1,1,3,3]`,
`payload_bits BOOL[1,1,3,3]`, and `blue2 UINT8[1,1,3,3]`. Validate, create the output
directory, and save.

- [ ] **Step 4: Generate the persistent artifact**

```bash
PYTHONPATH=. uv run python candidates/task271/build_bitwise_where_decoder.py
```

Expected: prints `candidates/task271/bitwise_where_decoder.onnx`.

- [ ] **Step 5: Run the focused test and verify GREEN**

```bash
PYTHONPATH=. uv run pytest -q candidates/task271/test_bitwise_where_decoder.py
```

Expected: `1 passed` with cost350, 267/267, 49/49, and 512/512 assertions executed.

- [ ] **Step 6: Commit the implementation**

```bash
git add candidates/task271/build_bitwise_where_decoder.py \
  candidates/task271/bitwise_where_decoder.onnx
git commit -m "feat(task271): fold payload decoder with bitwise where"
```

Expected: a commit containing only the task271 builder and persistent candidate.

---

### Task 3: Run pre-adoption gates and create exact source

**Files:**
- Test: all six `candidates/task271/test_*.py` files
- Candidate: `candidates/task271/bitwise_where_decoder.onnx`
- Modify: `src/custom/task271.py`
- Generate: `candidates/task271/source_net/task271.onnx`
- Generate: `networks/task271.onnx`

**Interfaces:**
- Consumes: fixed-point cost350 candidate.
- Produces: gate, fresh, exact-source, and SHA evidence authorizing adoption.

- [ ] **Step 1: Run all task271 focused tests**

```bash
PYTHONPATH=. uv run pytest -q \
  candidates/task271/test_free_input_window.py \
  candidates/task271/test_maxpool_convinteger_fold.py \
  candidates/task271/test_direct_conv_score.py \
  candidates/task271/test_int32_signed_renderer.py \
  candidates/task271/test_encoded_score_payload.py \
  candidates/task271/test_bitwise_where_decoder.py
```

Expected: seven tests pass.

- [ ] **Step 2: Run the official gate**

```bash
uv run ng gate candidates/task271/bitwise_where_decoder.onnx --task 271
```

Expected: PASS with pass267, fail0, memory238, params112, and cost350.

- [ ] **Step 3: Run fresh comparison**

```bash
PYTHONPATH=. uv run python candidates/fresh_compare_onnx.py \
  --task 271 \
  --candidate candidates/task271/bitwise_where_decoder.onnx \
  --n 2000
```

Expected: incumbent_fail0, candidate_fail0, divergence0, generator_errors0, and
generator_timeouts0.

- [ ] **Step 4: Generate exact source from the candidate and rebuild**

```bash
PYTHONPATH=. uv run python candidates/task271/build_bitwise_where_decoder.py \
  --incumbent candidates/task271/bitwise_where_decoder.onnx \
  --output candidates/task271/source_net/task271.onnx
PYTHONPATH=. uv run python tools/live_to_exact_source.py 271 \
  --net-dir candidates/task271/source_net --write-src
PYTHONPATH=. uv run python tools/rebuild_networks_from_source.py --tasks 271
sha256sum candidates/task271/bitwise_where_decoder.onnx \
  candidates/task271/source_net/task271.onnx networks/task271.onnx
```

Expected: all three SHA256 values are identical before adoption.

- [ ] **Step 5: Commit exact source if the SHA check passes**

```bash
git add src/custom/task271.py
git commit -m "feat(task271): sync exact cost350 source"
```

Expected: a commit containing only the task271 exact source.

---

### Task 4: Adopt through the mandatory gate and close the fixed point

**Files:**
- Modify via CLI: `submission/overfit_nets/task271.onnx`
- Modify via CLI: `state/manifest.json`, `state/safe_manifest.json`, `state/tasks/task271.md`
- Modify: `candidates/task271/DISCOVERY.md`
- Modify: `state/insights.yaml`
- Modify: `state/STATE.md`
- Verify: `src/custom/task271.py`, `networks/task271.onnx`

**Interfaces:**
- Consumes: gate-approved, fresh-clean, source-reproducible cost350 candidate.
- Produces: adopted task271 deployment and current handoff records.

- [ ] **Step 1: Adopt only through `ng adopt`**

```bash
uv run ng adopt candidates/task271/bitwise_where_decoder.onnx --task 271 \
  --note "decode score payload with UINT16 BitwiseAnd and centered UINT8 Where"
```

Expected: cost366 ->350, bundled267/267, fail0, and a new adoption timestamp.

- [ ] **Step 2: Update task271 records without overwriting unrelated changes**

In `state/tasks/task271.md` and `candidates/task271/DISCOVERY.md`, record the cost350 graph,
memory238, params112, bundled267/267, 49/49, 512/512, fresh2000 divergence0, adoption
timestamp, final SHA, the rejected INT8 Where kernel, and the next threshold cost<=316.

Extend the task271 payload-carry entry in `state/insights.yaml` with the opset18 BitwiseAnd/
UINT8 Where decoder and its transformer path. Replace only the task271 clauses in
`state/STATE.md`; preserve every unrelated concurrent edit. If shared-file edits cannot be
isolated safely, leave them unstaged and report that condition rather than reverting them.

- [ ] **Step 3: Run completion verification from the adopted graph**

```bash
uv run ng score 271
PYTHONPATH=. uv run pytest -q candidates/task271/test_*.py
PYTHONPATH=. uv run python candidates/fresh_compare_onnx.py \
  --task 271 \
  --candidate candidates/task271/bitwise_where_decoder.onnx \
  --n 2000
PYTHONPATH=. uv run python tools/rebuild_networks_from_source.py --tasks 271
sha256sum candidates/task271/bitwise_where_decoder.onnx \
  submission/overfit_nets/task271.onnx networks/task271.onnx \
  candidates/task271/source_net/task271.onnx
uv run python -m py_compile \
  candidates/task271/build_bitwise_where_decoder.py \
  candidates/task271/test_bitwise_where_decoder.py \
  src/custom/task271.py
uv run python - <<'PY'
import yaml
for path in ['state/levers.yaml', 'state/insights.yaml']:
    with open(path) as handle:
        yaml.safe_load(handle)
print('yaml=PASS')
PY
git diff --check -- candidates/task271 src/custom/task271.py \
  state/tasks/task271.md state/insights.yaml state/STATE.md
```

Expected: score cost350/pass267/fail0, seven focused tests pass, fresh divergence0, all four
SHA values identical, Python compile succeeds, YAML parses, and scoped diff check is clean.

- [ ] **Step 4: Commit only isolatable task271 records**

Stage only task271-owned files and shared-file hunks that can be proven task271-specific. Do not
stage unrelated dirty state. Use:

```bash
git add candidates/task271/DISCOVERY.md state/tasks/task271.md
git commit -m "docs(task271): record cost350 decoder adoption"
```

If `DISCOVERY.md` is ignored or shared state cannot be safely isolated, commit only the tracked
task271 ledger and report the remaining uncommitted handoff files.
