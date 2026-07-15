# Task019 Min-Basis Whole Renderer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, verify, and adopt an exact task019 H/W-conditioned whole renderer with predicted and measured cost at most 1750, targeting cost 1653.

**Architecture:** A compact two-slice background detector recovers H and W, `Min` creates exact five-element state bases, and one terminal float32 `Einsum` fuses point placement, clipped diagonal halo placement, and channel mixing. Source/output rectangular bases and a joint point/halo core are shared across the row and column contractions.

**Tech Stack:** Python 3.12, NumPy, ONNX 1.21.0, ONNX Runtime 1.26.0, pytest, `uv run ng`.

## Global Constraints

- Work only on task019.
- Keep ONNX 1.21.0 and ONNX Runtime 1.26.0 pinned; do not upgrade either dependency.
- Build scratch artifacts only under `candidates/task019/`.
- Do not build the ONNX unless static accounting is at most 1750.
- Do not repeat M/D/R, categorical E/D, literal `E=R-5M`, or the 1200-element dense halo attempt.
- Every QLinearConv padding tuple and stored bias must be audited; this design requires no QLinearConv nodes and therefore empty tuples.
- Success requires checker and strict inference, bundled 267/267, measured cost at most 1750, `ng gate` PASS, and fresh A/B divergence zero.
- Adopt only through `uv run ng adopt`; never copy into `submission/overfit_nets/`.
- Do not pack or submit.

---

### Task 1: Lock the analytical cost and exactness contract with a failing test

**Files:**
- Create: `candidates/task019/test_min_basis_whole_renderer.py`
- Test: `candidates/task019/test_min_basis_whole_renderer.py`

**Interfaces:**
- Consumes: deployed task019 path, task019 bundled dataset, and the future builder path.
- Produces: the required public builder symbols `analysis() -> dict[str, object]`, `build(task=None) -> onnx.ModelProto`, and constants `CANDIDATE`, `REPORT`, `TARGET_COST`.

- [ ] **Step 1: Write the failing test**

Create a test that imports standard libraries plus NumPy, ONNX, ONNX Runtime,
pytest, and `neurogolf.scoring`. It must first require the builder path:

```python
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPT = HERE / "build_min_basis_whole_renderer.py"
DEPLOYED = ROOT / "submission/overfit_nets/task019.onnx"

def test_min_basis_whole_renderer_is_exact_priced_and_reproducible():
    assert SCRIPT.exists(), "implement build_min_basis_whole_renderer.py after observing RED"
```

After the initial RED is observed, extend the same test to execute the script,
load `whole_renderer_prices.json`, and assert all of the following exact values:

```python
assert report["predicted"] == {"memory": 176, "params": 1477, "cost": 1653}
assert report["target_cost"] == 1750
assert report["build_allowed"] is True
assert report["selector"]["states_checked"] == 5
assert report["selector"]["all_25_hw_exact"] is True
assert report["detector"]["stored_exact"] == 267
assert report["bias_audit"] == {
    "qlinearconv_nodes": 0,
    "implicit_padding_tuples": [],
    "stored_biases": [],
    "all_stored_biases_non_positive": True,
}
```

Load the candidate, run checker and strict shape inference, require no
QLinearConv nodes, require deterministic SHA after two script executions, and
require the measured result:

```python
result = evaluate(CANDIDATE, load_task(19))
assert (result["pass"], result["fail"]) == (267, 0)
assert result["memory"] + result["params"] <= 1750
assert (result["memory"], result["params"]) == (176, 1477)
```

Finally compare thresholded outputs from candidate and deployed sessions on
every bundled example and require equality.

- [ ] **Step 2: Run the focused test to verify RED**

Run:

```bash
uv run pytest -q candidates/task019/test_min_basis_whole_renderer.py
```

Expected result:

```text
FAILED ... AssertionError: implement build_min_basis_whole_renderer.py after observing RED
1 failed
```

---

### Task 2: Implement exact factors, static pricing, and the terminal renderer

**Files:**
- Create: `candidates/task019/build_min_basis_whole_renderer.py`
- Create at runtime: `candidates/task019/min_basis_whole_renderer.onnx`
- Create at runtime: `candidates/task019/whole_renderer_prices.json`
- Test: `candidates/task019/test_min_basis_whole_renderer.py`

**Interfaces:**
- Consumes: `neurogolf.scoring.load_task(19)` and fixed generator constraints H,W in 2..6.
- Produces: `selector_factors()`, `channel_mixer()`, `analysis()`, `build(task=None)`, and deterministic candidate/report files.

- [ ] **Step 1: Implement factor constructors without emitting ONNX**

Define the Min basis and its exact inverse:

```python
CAPS = np.arange(2, 7, dtype=np.float32)
MIN_INVERSE = np.array(
    [
        [1.5, -1, 0, 0, 0],
        [-1, 2, -1, 0, 0],
        [0, -1, 2, -1, 0],
        [0, 0, -1, 2, -1],
        [0, 0, 0, -1, 1],
    ],
    dtype=np.float32,
)
```

Construct `T[h,r,g,R]` by adding point coefficients at `r` and `r+h`, then
adding clipped ±1 halo coefficients. Return:

```python
source = np.zeros((30, 6), np.float32)
source[:6] = np.eye(6, dtype=np.float32)
output = np.zeros((12, 30), np.float32)
output[:, :12] = np.eye(12, dtype=np.float32)
core = np.einsum("ph,hagq->apgq", MIN_INVERSE, T[:, :6, :, :12])
```

Assert `min(H,CAPS) × core × source × output` reconstructs all five one-axis
states exactly with `np.array_equal`.

- [ ] **Step 2: Implement the channel mixer and price gate**

Build `mixer[g,v,V]` with base identity, cyan point suppression, background
halo suppression, and cyan halo insertion:

```python
active = [1, 2, 3, 4, 5, 6, 7, 9]
mixer = np.zeros((2, 10, 10), np.float32)
mixer[0] = np.eye(10, dtype=np.float32)
mixer[0, active, 8] -= 5
mixer[1, active, 0] -= 1
mixer[1, active, 8] += 1
```

Use these exact accounting dictionaries:

```python
params = {
    "slice_starts": 4,
    "height_slice_ends": 4,
    "width_slice_ends": 4,
    "caps": 5,
    "source_basis": 180,
    "joint_core": 720,
    "output_basis": 360,
    "channel_mixer": 200,
}
memory = {
    "height_probe": 48,
    "height_row_max": 24,
    "height": 4,
    "width_probe": 48,
    "width_row_sum": 8,
    "width": 4,
    "height_min_basis": 20,
    "width_min_basis": 20,
}
```

Assert total params 1477, memory 176, cost 1653, and cost <=1750 before
calling any ONNX save function.

- [ ] **Step 3: Build the ONNX graph only after the price assertion**

Use input/output shapes `[1,10,30,30]`, float32 dtype, IR version 10, and
opset 12. Add:

```text
Slice(input, starts, h_ends) -> h_probe [1,1,6,2]
ReduceMax(h_probe, axes=[3], keepdims=1) -> h_row_max [1,1,6,1]
ReduceSum(h_row_max, axes=[0,1,2,3], keepdims=0) -> H scalar
Slice(input, starts, w_ends) -> w_probe [1,1,2,6]
ReduceSum(w_probe, axes=[3], keepdims=1) -> w_row_sum [1,1,2,1]
ReduceMax(w_row_sum, axes=[0,1,2,3], keepdims=0) -> W scalar
Min(H, caps) -> h_basis [5]
Min(W, caps) -> w_basis [5]
Einsum(...) -> output [1,10,30,30]
```

Slice the background channel by using four-element starts/ends and omitting
the optional axes/steps inputs. Use exactly this Einsum equation:

```text
bvrc,p,ra,apgq,qR,w,cA,AwgQ,QC,gvV->bVRC
```

The scalar reductions make both Min outputs rank-one, so the terminal equation
consumes them directly. Do not add Squeeze or Reshape carriers.

- [ ] **Step 4: Validate before saving**

Run:

```python
onnx.checker.check_model(model, full_check=True)
onnx.shape_inference.infer_shapes(model, strict_mode=True)
```

Create an ORT session with `ORT_DISABLE_ALL`, run every bundled input, and
require threshold equality with the bundled target. Audit graph nodes and
assert no `QLinearConv` exists. Save only after these assertions pass.

- [ ] **Step 5: Run the focused test to verify GREEN**

Run:

```bash
uv run pytest -q candidates/task019/test_min_basis_whole_renderer.py
```

Expected result:

```text
1 passed
```

---

### Task 3: Run official local gates and fresh A/B

**Files:**
- Read: `candidates/task019/min_basis_whole_renderer.onnx`
- Read: `submission/overfit_nets/task019.onnx`
- Read: `src/custom/task019.py`

**Interfaces:**
- Consumes: deterministic candidate emitted by Task 2.
- Produces: evidence that the candidate satisfies every pre-adoption gate.

- [ ] **Step 1: Run independent checker, strict inference, and scoring**

Run a fresh Python process that loads the candidate, runs checker and strict
shape inference, confirms zero QLinearConv nodes, and prints its SHA. Then run:

```bash
uv run python -m neurogolf.scoring candidates/task019/min_basis_whole_renderer.onnx 19
```

Expected: pass267/fail0 and cost <=1750.

- [ ] **Step 2: Run the official adoption gate**

Run:

```bash
uv run ng gate candidates/task019/min_basis_whole_renderer.onnx --task 019
```

Expected: `PASS`, bundled fail=0, candidate strictly cheaper than deployed.

- [ ] **Step 3: Run fresh A/B from the candidate builder**

Run:

```bash
uv run python -c "from neurogolf.scans.fresh import fresh_check; print(fresh_check(19, 'candidates/task019/build_min_basis_whole_renderer.py', n=6000))"
```

Expected: incumbent fail=0, candidate fail=0, candidate/incumbent divergence=0,
and `(6000, 6000)`.

- [ ] **Step 4: Re-run the focused regression after all external gates**

Run:

```bash
uv run pytest -q candidates/task019/test_residual_lowerings.py candidates/task019/test_min_basis_whole_renderer.py
```

Expected: both tests pass.

---

### Task 4: Adopt and restore source ownership

**Files:**
- Modify through CLI: `submission/overfit_nets/task019.onnx`
- Modify through CLI: `state/manifest.json`
- Modify through CLI: `state/tasks/task019.md`
- Modify through exact-source tool: `src/custom/task019.py`

**Interfaces:**
- Consumes: gate- and fresh-approved candidate from Task 3.
- Produces: adopted deployment and byte-identical source reconstruction.

- [ ] **Step 1: Reconfirm deployed SHA/cost immediately before adoption**

Run `uv run ng score 019` and SHA-256 the deployed artifact. If it is no longer
cost1934/SHA `f4ad4596f413dc603b6fae06e5665d73b2439af69c2c220eebb622e361f3bb33`,
stop and investigate the concurrent change.

- [ ] **Step 2: Adopt through the mandatory CLI**

Run:

```bash
uv run ng adopt candidates/task019/min_basis_whole_renderer.onnx --task 019 --note "Min-basis H/W whole renderer: fused point, diagonal halo, and channel mixer"
```

Expected: re-gate PASS, deployed cost <=1750, task ledger automatically stamped.

- [ ] **Step 3: Regenerate source ownership and verify SHA parity**

Run:

```bash
uv run python tools/live_to_exact_source.py --write-src 19
```

Build `src.custom.task019.build(load_task(19))` in memory and require its
serialized SHA to equal both candidate and deployed SHA. Require source-built
evaluation pass267/fail0 with the adopted measured cost.

- [ ] **Step 4: Run post-adoption score and gate checks**

Run:

```bash
uv run ng score 019
uv run ng gate candidates/task019/min_basis_whole_renderer.onnx --task 019
```

The score must remain pass267/fail0/cost<=1750. The second gate may report that
the identical candidate is no longer strictly cheaper; that is acceptable only
after SHA equality proves it is the adopted artifact.

- [ ] **Step 5: Commit only task019 tracked changes**

Stage `src/custom/task019.py`, `state/manifest.json`, and `state/tasks/task019.md`
plus any task019-specific tracked report changed by the workflow. Do not stage
unrelated dirty files. Commit with:

```bash
git commit -m "optimize task019 whole renderer" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

If any Task 2 or Task 3 gate fails, skip this entire adoption task and record
the four-field dated negative result in the relevant `state/levers.yaml` ledger
instead.
