# Task329 Cost-105 Shared-Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace task329's deployed cost117 route with the approved proof-first shared colour/spatial core construction at predicted cost105.

**Architecture:** Keep the three-node `ReduceSum -> Concat -> Einsum` graph and the dynamic cell-count state. Reuse one rational `shared_core[2,2,2]` for both colour products and both spatial roots, and reuse one `swap[2,2]` for sign reflection plus colour-to-spatial branch exchange, leaving only 93 initializer elements and 12 bytes of counted intermediates.

**Tech Stack:** Python 3.13, NumPy float32 proofs, ONNX 1.21.0, ONNX Runtime 1.26.0, pytest, `uv run ng`.

## Global Constraints

- Modify only task329 artifacts and task329-specific state records.
- Do not repeat rank2 Adam/hinge fitting; this implementation is an exact rational construction.
- Prove all `4*30*10*10=12000` logical states with `desired > 0`, `wrong <= 0`, and float32 margin at least `0.008` before writing ONNX.
- Do not write ONNX unless every score is finite and predicted cost is at most105.
- Adoption must run `uv run ng gate` and then `uv run ng adopt`; never copy into deployment manually.
- Run fresh1500 as a diagnostic and record its incumbent/candidate failures and divergence.
- Preserve `onnx==1.21.0`, `onnxruntime==1.26.0`, and the historical rank4 cost270 fallback.
- Do not run global scans, modify another task, pack, or submit.

---

### Task 1: Proof-first acceptance test

**Files:**
- Create: `candidates/task329/test_shared_colour_spatial_core_route.py`
- Test: `candidates/task329/test_shared_colour_spatial_core_route.py`

**Interfaces:**
- Consumes: a command-line builder at `candidates/task329/build_shared_colour_spatial_core_route.py`.
- Produces: assertions over `--prove-only --json`, deployed/source protobuf equality, and post-adoption builder idempotence.

- [ ] **Step 1: Write the missing-builder RED test**

Create `test_prove_only_covers_all_states_before_build()` using `subprocess.run` and assert:

```python
assert proof["logical_states"] == 4 * 30 * 10 * 10
assert proof["desired_min"] > 0.0
assert proof["wrong_max"] <= 0.0
assert proof["margin"] >= 0.008
assert proof["max_abs_target_root"] <= 3.0e-8
assert proof["min_abs_nontarget_root"] > 0.0
assert proof["all_finite"] is True
assert proof["predicted_cost"] <= 105
```

Also assert that a sentinel output path supplied with `--output` does not exist after a prove-only run.

- [ ] **Step 2: Run the focused test and observe RED**

Run:

```bash
PYTHONPATH=. uv run pytest -q candidates/task329/test_shared_colour_spatial_core_route.py::test_prove_only_covers_all_states_before_build
```

Expected: FAIL because `build_shared_colour_spatial_core_route.py` does not exist.

- [ ] **Step 3: Add parity tests without running them yet**

Add `test_source_rebuild_is_byte_identical_to_deployed_model()` and `test_builder_is_idempotent_after_adoption()` following the existing task329 route suites. They compare SHA-256 hashes and run `onnx.checker.check_model(..., full_check=True)`.

---

### Task 2: Exact proof and candidate builder

**Files:**
- Create: `candidates/task329/build_shared_colour_spatial_core_route.py`
- Create only after proof: `candidates/task329/shared_colour_spatial_core_route.onnx`
- Test: `candidates/task329/test_shared_colour_spatial_core_route.py`

**Interfaces:**
- Consumes: deployed cost117 task329 graph, `F(x)`, `J`, the shared core, and dynamic-count route from the approved design.
- Produces: `logical_proof() -> dict[str, float | int | bool]` and `build() -> onnx.ModelProto`.

- [ ] **Step 1: Implement the exact float32 proof**

Define:

```python
Q = np.float32(32.0)
COLOUR_FEATURES = np.stack([Q * (colours + 0.5), 0.5 - colours], axis=1).astype(np.float32)
SWAP = np.array([[0.0, 1.0 / 32.0], [32.0, 0.0]], dtype=np.float32)
SHARED_CORE = np.array(
    [
        [[1.0 / 1024.0, -1.0 / 32.0], [3.0 / 32.0, 1.0]],
        [[0.0, 1.25], [0.0, 40.0]],
    ],
    dtype=np.float32,
)
```

For each count `x` in `9,25,49,81`, store

```python
z = (3.0 * x - 32.0) / (-1280.0 * (x - 32.0))
route0 = 1024.0 * z
route1 = -1.0 / 20.0 - route0 / 32.0
```

in columns1..4; use `z=1/5` in every other column. Assert `F(x) @ SWAP == F(-x)`, exact `4E`/`4P` colour products, constant spatial branch2, bounded target residual, nonzero non-target roots, finite scores, and all 12000 sign conditions. Return measured extrema and `predicted_cost=105`.

- [ ] **Step 2: Verify proof GREEN before any ONNX write**

Run the focused test and:

```bash
uv run python candidates/task329/build_shared_colour_spatial_core_route.py --prove-only --json
```

Expected: 12000 states, margin at least0.008, finite values, target residual at most`3e-8`, predicted cost105, and no ONNX output.

- [ ] **Step 3: Implement the three-node graph**

Construct `ReduceSum(input, keepdims=1)`, `Concat(cell_count_keep, one_state, axis=1)`, and a FREE-output terminal Einsum with equation:

```text
bkhw,ka,tad,cd,ke,eg,tgi,cf,fi,tu,upq,bqmn,pw,urs,bsxy,rw->bchw
```

Use the exact 16-input order from the design. Initializers, in order, are `colour_features[10,2]`, `shared_core[2,2,2]`, `swap[2,2]`, `one_state[1,1,1,1]`, and `route_r[2,30]`. Run full checker and strict shape inference.

- [ ] **Step 4: Build after proof and measure isolated cost**

Run:

```bash
uv run python candidates/task329/build_shared_colour_spatial_core_route.py --json
uv run ng score candidates/task329/shared_colour_spatial_core_route.onnx --task 329
```

Expected: memory12, params93, cost105. If ORT rejects the 16-operand contraction or measured cost exceeds105, stop and retain cost117.

---

### Task 3: Official gate, fresh diagnostic, and adoption

**Files:**
- Read: `candidates/task329/shared_colour_spatial_core_route.onnx`
- Modify through CLI only: `submission/overfit_nets/task329.onnx`
- Modify through CLI only: `state/tasks/task329.md`

**Interfaces:**
- Consumes: the proof-approved cost105 candidate.
- Produces: an adopted deployment only if official evaluation passes.

- [ ] **Step 1: Run the official bundled gate**

```bash
uv run ng gate candidates/task329/shared_colour_spatial_core_route.onnx --task 329
```

Expected: PASS, bundled fail0, cost105, strictly cheaper than cost117.

- [ ] **Step 2: Run fresh1500 A/B**

Call the existing repository `fresh_check(329, candidate=..., n=1500)` helper in a fresh Python process and record incumbent fail, candidate fail, and divergence.

- [ ] **Step 3: Adopt through the mandatory CLI path**

```bash
uv run ng adopt candidates/task329/shared_colour_spatial_core_route.onnx --task 329 --note "exact shared colour/spatial core and shared swap; all 12000 logical states margin>=0.008; bundled gate and fresh1500 checked; 117->105"
```

Expected: adopt re-gate PASS and task329 ledger auto-stamp.

---

### Task 4: Source ownership and historical-builder idempotence

**Files:**
- Modify: `src/custom/task329.py`
- Modify: `candidates/task329/build_repeated_colour_root_route.py`
- Modify: `candidates/task329/build_dynamic_count_code_route.py`
- Modify: `candidates/task329/build_repeated_root_route.py`
- Modify: `candidates/task329/build_shared_bilinear_route.py`
- Test: all five task329 route test files.

**Interfaces:**
- Consumes: the adopted cost105 protobuf.
- Produces: a byte-identical source builder and historical builders that treat the newer graph as an idempotent descendant.

- [ ] **Step 1: Observe post-adoption parity RED**

Run the new full three-test file. The proof test must pass while source parity and idempotence fail because source still describes cost117.

- [ ] **Step 2: Replace exact source graph**

Update `src/custom/task329.py` with the builder's arrays, initializer order, node order, graph/value-info names, equation, opset18, and IR10. Assert its serialized protobuf equals deployment byte-for-byte.

- [ ] **Step 3: Add descendant recognition to historical builders**

Recognize the three-node graph plus the new 16 terminal inputs. Import `src.custom.task329`, assert deployed/source byte equality, run the full checker, and return the incumbent without reconstructing an older graph.

- [ ] **Step 4: Run all task329 route suites GREEN**

```bash
PYTHONPATH=. uv run pytest -q \
  candidates/task329/test_shared_bilinear_route.py \
  candidates/task329/test_repeated_root_route.py \
  candidates/task329/test_dynamic_count_code_route.py \
  candidates/task329/test_repeated_colour_root_route.py \
  candidates/task329/test_shared_colour_spatial_core_route.py
```

Expected: 15 tests pass.

---

### Task 5: Task-local records and final verification

**Files:**
- Modify: `candidates/task329/DISCOVERY.md`
- Replace current task329 section in: `state/STATE.md`
- Append one four-field entry in: `state/levers.yaml`
- Read/auto-modified: `state/tasks/task329.md`

**Interfaces:**
- Consumes: verified adopted cost105 endpoint.
- Produces: task-local handoff, authoritative lever entry, deployed hash, and next target cost95.

- [ ] **Step 1: Update task-local records**

Record cost117 ->105, memory12, params93, gain `ln(117/105)=0.108213584640`, proof extrema, bundled/fresh counts, SHA-256, and next threshold `floor(105/e^0.1)=95`. Mark the mechanism task329-local and state explicitly that no 400-task scan ran because of user scope.

- [ ] **Step 2: Append the mandatory four-field lever entry**

Include date/tool/scope in `ran`, measured `verdict`, a `reopen` requiring a pre-priced cost at most95 under a new premise, and `falsification_history` noting that separate colour/spatial cores were not minimal while prior floor claims have failed.

- [ ] **Step 3: Run fresh final verification**

Run proof-only JSON, all 15 focused tests, isolated deployed scoring, candidate/source/deployed SHA equality, ONNX full checker plus strict inference, YAML parse, and `git diff --check` over task329 files. Only report success from this fresh output.
