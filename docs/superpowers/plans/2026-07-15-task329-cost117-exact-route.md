# Task329 Cost-117 Exact Route Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace task329's cost134 quadratic-colour route with a proof-first exact repeated-affine-colour-root route at predicted cost117.

**Architecture:** Preserve the dynamic cell-count width code and FREE-output terminal Einsum. Replace the shared quadratic colour basis with `[1,x]` plus two affine-root cores, and change the cell-count reduction to `keepdims=1` so `Unsqueeze` and its axes initializer disappear.

**Tech Stack:** Python 3.13, NumPy float32 proofs, ONNX 1.21.0, ONNX Runtime 1.26.0, pytest, `uv run ng`.

## Global Constraints

- Modify only task329 artifacts and task329-specific state records.
- Do not repeat the rank2 Adam/hinge search.
- Prove all 12000 logical states with `desired > 0`, `wrong <= 0`, and margin at least 0.008 before writing ONNX.
- Do not build if predicted cost exceeds 121.
- Adoption must follow `uv run ng gate` then `uv run ng adopt`; never copy into deployment manually.
- Fresh1500 is diagnostic but must be recorded.
- Preserve `onnx==1.21.0` and `onnxruntime==1.26.0`.
- Keep the historical rank4 cost270 fallback.

---

### Task 1: Proof-first acceptance test

**Files:**
- Create: `candidates/task329/test_repeated_colour_root_route.py`
- Test: `candidates/task329/test_repeated_colour_root_route.py`

**Interfaces:**
- Consumes: the command-line builder contract used by existing task329 tests.
- Produces: proof-only acceptance criteria and post-adoption source/builder parity checks.

- [ ] **Step 1: Write the failing test**

Create three tests matching the current dynamic-route suite.  The first runs
`build_repeated_colour_root_route.py --prove-only --json` and asserts:

```python
assert proof["logical_states"] == 4 * 30 * 10 * 10
assert proof["desired_min"] >= 0.008
assert proof["wrong_max"] <= 0.0
assert proof["margin"] >= 0.008
assert proof["predicted_cost"] <= 121
```

The second compares `src.custom.task329.build(None).SerializeToString()` with deployed
task329 bytes.  The third runs the builder normally and compares candidate bytes with deployed
bytes after adoption.

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
PYTHONPATH=. uv run pytest -q candidates/task329/test_repeated_colour_root_route.py::test_prove_only_covers_all_states_before_build
```

Expected: FAIL because `build_repeated_colour_root_route.py` does not exist.

---

### Task 2: Exact proof and candidate builder

**Files:**
- Create: `candidates/task329/build_repeated_colour_root_route.py`
- Create after proof: `candidates/task329/repeated_colour_root_route.onnx`
- Test: `candidates/task329/test_repeated_colour_root_route.py`

**Interfaces:**
- Consumes: deployed cost134 task329 graph and its exact spatial code.
- Produces: `logical_proof() -> dict[str, float | int]` and `build() -> onnx.ModelProto`.

- [ ] **Step 1: Implement exact float32 colour roots**

Use the exact arrays:

```python
COLOUR_FEATURES = np.stack(
    [np.ones(10, dtype=np.float32), np.arange(10, dtype=np.float32)], axis=1
)
COLOUR_LEFT = np.array(
    [[[0.5, 1.0], [-1.0, 0.0]], [[10.0, -20.0], [0.0, 0.0]]],
    dtype=np.float32,
)
COLOUR_RIGHT = np.array(
    [[[0.5, -1.0], [1.0, 0.0]], [[10.0, 20.0], [0.0, 0.0]]],
    dtype=np.float32,
)
left = np.einsum("ka,tad,cd->tkc", COLOUR_FEATURES, COLOUR_LEFT, COLOUR_FEATURES, dtype=np.float32)
right = np.einsum("ke,tef,cf->tkc", COLOUR_FEATURES, COLOUR_RIGHT, COLOUR_FEATURES, dtype=np.float32)
colour = left * right
```

Assert branch zero equals `0.25-(k-c)**2` and branch one equals
`400*(0.25-c**2)` bit-exactly.  Reuse the cost134 spatial proof, enumerate all 12000 states,
and return `predicted_cost=117` only after all sign/margin assertions pass.

- [ ] **Step 2: Implement the three-node ONNX graph**

Use:

```python
reduce_sum = helper.make_node("ReduceSum", ["input"], ["cell_count_keep"], keepdims=1)
concat = helper.make_node("Concat", ["cell_count_keep", "one_state"], ["state_vec"], axis=1)
equation = "bkhw,ka,tad,cd,ke,tef,cf,tpq,bqij,pw,tur,brlm,uw->bchw"
```

The terminal inputs are `input`, four reuses of `colour_features`, `colour_left`,
`colour_right`, two reuses each of `root_core`, `state_vec`, and `route_r`.  Initializer order is
`colour_features`, `colour_left`, `colour_right`, `one_state`, `root_core`, `route_r`.
Declare `cell_count_keep FLOAT [1,1,1,1]` and `state_vec FLOAT [1,2,1,1]`, then run full
checker and strict shape inference.

- [ ] **Step 3: Verify GREEN proof before ONNX build**

Run the focused pytest and then:

```bash
uv run python candidates/task329/build_repeated_colour_root_route.py --prove-only --json
```

Expected: 12000 states, margin at least0.008, predicted cost117, and no ONNX write.

- [ ] **Step 4: Build only after proof**

Run:

```bash
uv run python candidates/task329/build_repeated_colour_root_route.py --json
```

Expected: the proof prints first, then the cost117 candidate is written.

---

### Task 3: Official evaluation and adoption

**Files:**
- Read: `candidates/task329/repeated_colour_root_route.onnx`
- Modify through CLI only: `submission/overfit_nets/task329.onnx`
- Modify through CLI only: `state/tasks/task329.md`

**Interfaces:**
- Consumes: proof-approved candidate.
- Produces: adopted cost117 deployment if every gate succeeds.

- [ ] **Step 1: Run official bundled gate**

```bash
uv run ng gate candidates/task329/repeated_colour_root_route.onnx --task 329
```

Expected: PASS, bundled fail0, cost117, strictly cheaper than cost134.

- [ ] **Step 2: Run fresh1500 diagnostic**

Invoke the repository's `fresh_check(329, candidate=..., n=1500)` helper.  Record incumbent and
candidate failures and divergence without changing deployment.

- [ ] **Step 3: Adopt through the mandatory path**

```bash
uv run ng adopt candidates/task329/repeated_colour_root_route.onnx --task 329 --note "exact repeated affine colour roots with keepdims dynamic state; all 12000 logical states margin>=0.008; bundled gate and fresh1500 checked; 134->117"
```

Expected: adopt re-gate PASS and an auto-stamped task329 record.

---

### Task 4: Exact source ownership and descendant idempotence

**Files:**
- Modify: `src/custom/task329.py`
- Modify: `candidates/task329/build_repeated_colour_root_route.py`
- Modify: `candidates/task329/build_dynamic_count_code_route.py`
- Modify: `candidates/task329/build_repeated_root_route.py`
- Modify: `candidates/task329/build_shared_bilinear_route.py`
- Test: the four task329 route test files.

**Interfaces:**
- Consumes: adopted cost117 protobuf.
- Produces: byte-identical source and historical builders that accept the newer deployment.

- [ ] **Step 1: Observe post-adoption RED**

Run the new three-test file.  Proof must pass; source parity and idempotence must fail because
they still describe cost134.

- [ ] **Step 2: Replace `src/custom/task329.py` with the exact graph**

Use the arrays, initializer order, node order, names, attributes, value infos, opset18, IR10,
and graph name defined in Task 2.  Require serialized equality with deployed bytes.

- [ ] **Step 3: Add descendant branches to all four builders**

Recognize node types `ReduceSum,Concat,Einsum` and the new 13 terminal inputs.  Import
`src.custom.task329`, assert serialized equality, run full checker, and return the incumbent.

- [ ] **Step 4: Run all route tests GREEN**

```bash
PYTHONPATH=. uv run pytest -q candidates/task329/test_shared_bilinear_route.py candidates/task329/test_repeated_root_route.py candidates/task329/test_dynamic_count_code_route.py candidates/task329/test_repeated_colour_root_route.py
```

Expected: 12 tests pass.

---

### Task 5: Records and final verification

**Files:**
- Modify: `candidates/task329/DISCOVERY.md`
- Modify: `state/STATE.md`
- Modify: `state/levers.yaml`
- Read/auto-modified: `state/tasks/task329.md`

**Interfaces:**
- Consumes: verified adopted endpoint.
- Produces: current handoff, four-field ledger entry, and next threshold.

- [ ] **Step 1: Update task-local records**

Record cost134 ->117, memory12, params105, gain `0.1356658652`, proof extrema,
bundled/fresh results, deployed SHA, and next threshold `floor(117/e^0.1)=105`.  Mark the
mechanism task329-local because the user prohibited cross-task work.

- [ ] **Step 2: Append the four-field lever entry**

Record concrete commands/scope/date, measured verdict, reopen trigger requiring a pre-priced
cost at most105 under a new premise, and the falsification history that quadratic colour storage
was not minimal because repeated affine roots share `[1,x]`.

- [ ] **Step 3: Run fresh final verification**

Run proof-only JSON, all four test files, isolated `uv run ng score 329`, byte equality among
candidate/source/deployed, YAML parse, and `git diff --check`.  Expected: proof passes, 12 tests
pass, isolated score is cost117/fail0, all bytes match, and records parse cleanly.
