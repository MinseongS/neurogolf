# Task271 Encoded-Score Payload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace task271's MaxPool-index/dynamic-Slice tail with an exact payload-carrying score and reach cost366.

**Architecture:** The direct FLOAT32 Conv stores blue count in score bits 9+ and the selected 3x3 blue pattern in bits 0..8. MaxPool returns only the winning scalar; UINT16 Mod/GreaterOrEqual decodes its payload and the existing signed ConvInteger renderer writes the FREE output.

**Tech Stack:** Python 3.13, ONNX 1.21.0, ONNX Runtime 1.26.0, NumPy, pytest, `uv run ng`.

## Global Constraints

- Modify only task271 artifacts and shared task271 handoff/insight records.
- Keep onnx==1.21.0 and onnxruntime==1.26.0.
- Candidate and scratch artifacts stay under `candidates/task271/`.
- Require cost<=371, bundled fail0, correct 49-position selection/payload decoding, fresh divergence0, two-pass fixed point, and exact source/candidate/deployed SHA equality.
- Use `ng gate` before adoption and only `ng adopt` to replace the deployed graph.
- Do not restore dual ReduceMax/ArgMax, `cyan_patch/crop8/Pad`, or a full-canvas Cast.

---

### Task 1: Pin the encoded-payload graph contract with a RED test

**Files:**
- Create: `candidates/task271/test_encoded_score_payload.py`
- Read: `candidates/task271/int32_signed_renderer.onnx`

**Interfaces:**
- Consumes: pinned cost411 artifact SHA `81e11815e677745cf8bdec7ba39f873278f67966d0d1ed3799dc5687698a90f3`.
- Produces: a regression contract for `build_encoded_score_payload.py::build_candidate(incumbent: Path, output: Path) -> None`.

- [ ] **Step 1: Write the failing regression**

Create a pytest that imports the not-yet-existing builder and then asserts the generated graph contract:

```python
BASELINE_SHA256 = "81e11815e677745cf8bdec7ba39f873278f67966d0d1ed3799dc5687698a90f3"
EXPECTED_OPS = [
    "Conv", "MaxPool", "Cast", "Mod", "GreaterOrEqual", "Cast",
    "BitShift", "ConvInteger",
]

def encoded_kernel() -> np.ndarray:
    weight = np.zeros((1, 10, 3, 3), dtype=np.float32)
    weight[0, 0] = -4096.0
    for k in range(9):
        weight[0, 1].reshape(-1)[k] = 512.0 + (1 << k)
    return weight
```

The test must assert opset12, MaxPool output `['max_score']`, exact kernel values,
UINT16 `moduli=[2,4,...,512]`, UINT16 `thresholds=[1,2,...,256]`, shared UINT8
`one_u8=1`, signed renderer weights, absence of the old coordinate/Slice initializers,
cost366, memory247, params119, bundled267/267, persistent byte equality, and second-pass
fixed point. It must also run 49 synthetic single-box inputs through pinned ORT and check
the 3x3 payload at every legal top-left.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
PYTHONPATH=. uv run pytest -q candidates/task271/test_encoded_score_payload.py
```

Expected: FAIL because `candidates/task271/build_encoded_score_payload.py` does not exist.

---

### Task 2: Implement the minimal fixed-point builder and make the test GREEN

**Files:**
- Create: `candidates/task271/build_encoded_score_payload.py`
- Create: `candidates/task271/encoded_score_payload.onnx`
- Test: `candidates/task271/test_encoded_score_payload.py`

**Interfaces:**
- Consumes: `int32_signed_renderer.onnx` and its pinned SHA.
- Produces: `build_candidate(incumbent: Path, output: Path) -> None`, accepting either the pinned baseline or an already-folded artifact.

- [ ] **Step 1: Implement validation constants and fixed-point handling**

Define:

```python
DEFAULT_INCUMBENT = Path(__file__).with_name("int32_signed_renderer.onnx")
DEFAULT_OUTPUT = Path(__file__).with_name("encoded_score_payload.onnx")
BASELINE_SHA256 = "81e11815e677745cf8bdec7ba39f873278f67966d0d1ed3799dc5687698a90f3"
FOLDED_OPS = [
    "Conv", "MaxPool", "Cast", "Mod", "GreaterOrEqual", "Cast",
    "BitShift", "ConvInteger",
]
```

`_validate_folded` must check opset12, node sequence, score geometry, kernel, UINT16
decoder tensors, `one_u8`, signed renderer weights, single MaxPool output, output INT32,
and full ONNX checker. If the input already validates as folded, save it unchanged.

- [ ] **Step 2: Rewrite only the score/selection/patch tail**

Retain `one_u8` and `render_weight_signed`; delete every other initializer. Add the
encoded FLOAT32 kernel and these tensors:

```python
moduli = (2 ** np.arange(1, 10, dtype=np.uint16)).reshape(1, 1, 3, 3)
thresholds = (2 ** np.arange(0, 9, dtype=np.uint16)).reshape(1, 1, 3, 3)
```

Replace the graph nodes with:

```python
helper.make_node("Conv", ["input", "score_payload_kernel"], ["scores"],
                 kernel_shape=[3, 3], pads=[0, 0, -21, -21])
helper.make_node("MaxPool", ["scores"], ["max_score"], kernel_shape=[7, 7])
helper.make_node("Cast", ["max_score"], ["encoded_u16"], to=TensorProto.UINT16)
helper.make_node("Mod", ["encoded_u16", "payload_moduli"], ["remainders_u16"])
helper.make_node("GreaterOrEqual", ["remainders_u16", "payload_thresholds"], ["payload_bits"])
helper.make_node("Cast", ["payload_bits"], ["blue_patch"], to=TensorProto.UINT8)
helper.make_node("BitShift", ["blue_patch", "one_u8"], ["blue2"], direction="LEFT")
helper.make_node("ConvInteger", ["blue2", "render_weight_signed", "one_u8"], ["output"],
                 kernel_shape=[1, 1], pads=[0, 0, 27, 27])
```

Declare every non-output tensor with its exact static shape and dtype, set graph name
`task271_encoded_score_payload`, run strict shape inference/full checker, and save.

- [ ] **Step 3: Generate the persistent artifact**

Run:

```bash
PYTHONPATH=. uv run python candidates/task271/build_encoded_score_payload.py
```

Expected: prints `candidates/task271/encoded_score_payload.onnx`.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
PYTHONPATH=. uv run pytest -q candidates/task271/test_encoded_score_payload.py
```

Expected: `1 passed`.

---

### Task 3: Run task271 regression and mandatory gates

**Files:**
- Test: all `candidates/task271/test_*.py`
- Candidate: `candidates/task271/encoded_score_payload.onnx`

**Interfaces:**
- Consumes: cost366 persistent candidate.
- Produces: bundled/fresh/49-position evidence authorizing adoption.

- [ ] **Step 1: Run all five task271 test files**

```bash
PYTHONPATH=. uv run pytest -q \
  candidates/task271/test_free_input_window.py \
  candidates/task271/test_maxpool_convinteger_fold.py \
  candidates/task271/test_direct_conv_score.py \
  candidates/task271/test_int32_signed_renderer.py \
  candidates/task271/test_encoded_score_payload.py
```

Expected: six tests pass: the four older files contain five tests and the new file one.

- [ ] **Step 2: Run official gate**

```bash
uv run ng gate candidates/task271/encoded_score_payload.onnx --task 271
```

Expected: PASS with cost366, memory247, params119, pass267, fail0.

- [ ] **Step 3: Run fresh comparison**

```bash
PYTHONPATH=. uv run python candidates/fresh_compare_onnx.py \
  --task 271 --candidate candidates/task271/encoded_score_payload.onnx --n 2000
```

Expected: candidate_fail0, candidate_vs_incumbent0, generator_errors0, generator_timeouts0.

- [ ] **Step 4: Build exact source before adoption**

Generate `candidates/task271/source_net/task271.onnx` with the candidate builder, run
`tools/live_to_exact_source.py 271 --net-dir candidates/task271/source_net --write-src`,
then `tools/rebuild_networks_from_source.py --tasks 271`. Require candidate/source-net/
network SHA equality.

---

### Task 4: Adopt and close the fixed point

**Files:**
- Modify via CLI: `submission/overfit_nets/task271.onnx`
- Modify via CLI: `state/manifest.json`, `state/safe_manifest.json`, `state/tasks/task271.md`
- Modify: `candidates/task271/DISCOVERY.md`
- Modify: `state/insights.yaml`
- Modify: `state/STATE.md`
- Modify: `src/custom/task271.py`

**Interfaces:**
- Consumes: gate-approved cost366 candidate and exact source.
- Produces: adopted task271 deployment with reproducible handoff.

- [ ] **Step 1: Adopt only through the mandatory CLI**

```bash
uv run ng adopt candidates/task271/encoded_score_payload.onnx --task 271 \
  --note "encode winning 3x3 payload in score low bits; remove MaxPool indices and dynamic Slice"
```

Expected: cost411 ->366, fail0, new SHA recorded.

- [ ] **Step 2: Update task271 handoff and reusable insight**

Update DISCOVERY current truth, add the completed encoded-payload hypothesis, set the next
`+0.1` bar to `floor(366/e^0.1)=331`, and record the failed UINT16 BitShift/Div capability
paths. Extend `unique_winner_maxpool_convinteger_free_output_renderer` with the payload-
carrying reduction and add the new builder as an additional transformer. Update only the
task251-275 task271 endpoint/test count in STATE while preserving concurrent edits.

- [ ] **Step 3: Run fresh completion verification**

Re-run all six focused tests, `uv run ng score 271`, fresh2000, exact source rebuild,
candidate/deployed/network SHA comparison, Python compile, YAML parse, and scoped
`git diff --check`. Require cost366, bundled267/267 fail0, fresh divergence0, and all
three artifact SHAs identical before claiming completion.

- [ ] **Step 4: Commit only task271 tracked changes if isolation is safe**

Stage the task271 source, task ledger, DISCOVERY if tracked, insight, STATE, manifest rows,
and this plan. Do not stage unrelated shared-worktree changes. If shared files contain
inseparable concurrent edits, leave implementation uncommitted and report that condition.
