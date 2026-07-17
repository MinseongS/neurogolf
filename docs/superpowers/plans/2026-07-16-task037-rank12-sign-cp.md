# Task037 Rank-12 Sign-CP Stretch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, prove, gate, and adopt a task037 opset-20 rank-12 sign-preserving CP renderer at predicted cost1142, with an exact rank14 cost1188 fallback.

**Architecture:** Pack the six descriptor lanes in INT8 before one FLOAT Cast, replace the quadratic row basis with two reused affine row bases, and freeze a balanced rank-12 CP factorization in the existing single FREE-output six-slot Einsum.  Preserve the incumbent prefix, slot count, valid-grid masking, non-overlap product proof, and output polarity.

**Tech Stack:** Python 3.13, NumPy, PyTorch CPU for candidate-only factor search, ONNX 1.21.0, ONNX Runtime 1.26.0, pytest, `uv run ng`.

## Global Constraints

- Modify only task037 artifacts plus the required global STATE/insight/lever records.
- Keep opset20 and IR10; do not change `onnx==1.21.0` or `onnxruntime==1.26.0`.
- Keep six selected slots, the exact valid 10x10 support, and the existing non-overlap polarity proof.
- Reject any predicate sign flip, zero tie, non-finite output, raw off-grid nonzero, bundled failure, fresh divergence, or runtime above2x.
- Adoption must use `uv run ng gate` then `uv run ng adopt`; never copy into deployment manually.
- Work in `candidates/task037/`; do not pack or submit.
- The current shared checkout contains the uncommitted adopted cost1284 truth source, so do not create a worktree from stale HEAD and do not stage unrelated dirty files.

---

### Task 1: Proof-first RED test

**Files:**
- Create: `candidates/task037/test_rank12_sign_cp.py`
- Test: `candidates/task037/test_rank12_sign_cp.py`

**Interfaces:**
- Consumes: the missing `candidates.task037.build_rank12_sign_cp` module.
- Produces: acceptance assertions for `factor_proof() -> dict[str, int | float]` and `build(task) -> onnx.ModelProto`.

- [ ] **Step 1: Write the failing proof test**

Create a test that imports `factor_proof` and requires:

```python
proof = factor_proof()
assert proof["states"] == 5139
assert proof["evaluations"] == 5_139_000
assert proof["sign_flips"] == 0
assert proof["zero_ties"] == 0
assert proof["minimum_signed_margin"] > 0.15
assert proof["rank"] == 12
assert proof["predicted_cost"] <= 1142
```

Add a structural test requiring opset20, one output-producing terminal Einsum, no Scatter,
strict shape inference, cost at most1142, bundled fail0, classes0..9, finite logits, and exact
zero outside `[:, :, :10, :10]`.

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONPATH=. uv run pytest -q candidates/task037/test_rank12_sign_cp.py
```

Expected: collection fails because `build_rank12_sign_cp.py` does not exist.

---

### Task 2: Deterministic rank-12 factor search and freeze

**Files:**
- Create: `candidates/task037/fit_rank12_sign_cp.py`
- Create: `candidates/task037/build_rank12_sign_cp.py`
- Test: `candidates/task037/test_rank12_sign_cp.py`

**Interfaces:**
- Consumes: `src.custom.task037._TERMS` and the six-mode tensor mapping `r^0=(0,0)`, `r^1=(1,0)`, `r^2=(1,1)`.
- Produces: six balanced float32 arrays returned by `cp_factors()` with shapes `(12,6)`, `(12,6)`, `(12,2)`, `(12,2)`, `(12,3)`, `(12,4)`.

- [ ] **Step 1: Implement the deterministic search utility**

Use CPU float64, `torch.set_num_threads(1)`, seeds1200..1207, Adam at0.03 for4000 steps,
then LBFGS at0.5 for at most700 iterations.  Select the lowest relative tensor error.  For each
component, compute all six column norms, replace every column by its unit vector, and distribute
the signed product of norms evenly across the six factors before float32 conversion.  Print the
factor bytes as base64 plus the proof metrics; do not write outside `candidates/task037/`.

- [ ] **Step 2: Freeze factors in the builder**

Paste the emitted base64 blob into `build_rank12_sign_cp.py`.  Decode exactly 276 float32 values
and split them in the declared shape order.  `factor_proof()` reconstructs the six-mode tensor,
enumerates every valid segment whose two endpoints remain in0..9 for signs-1/+1 and colours1..9,
adds nine absent descriptors `(1,20,0,0,0,colour)`, and evaluates every row/column/class0..9.

- [ ] **Step 3: Run the proof test GREEN before building ONNX**

```bash
PYTHONPATH=. uv run pytest -q candidates/task037/test_rank12_sign_cp.py::test_factor_proof_covers_conservative_domain
```

Expected: PASS with 5,139 states, zero flips/ties, and margin above0.15.

---

### Task 3: Build the opset-20 candidate

**Files:**
- Modify: `candidates/task037/build_rank12_sign_cp.py`
- Create by builder: `candidates/task037/rank12_sign_cp.onnx`
- Test: `candidates/task037/test_rank12_sign_cp.py`

**Interfaces:**
- Consumes: `src.custom.task037.build(task)` as the exact cost1284 prefix and frozen CP factors.
- Produces: `build(task) -> onnx.ModelProto` and `rank12_sign_cp.onnx`.

- [ ] **Step 1: Implement INT8 descriptor packing**

Retain nodes through `selected_kvec`.  Add INT8 Casts for `selected_lo_safe`, `selected_n`, and
`selected_kvec`; concatenate `descriptor_one_i8`, those three outputs, `selected_base`, and
`selected_sgn` on axis2; then Cast the `[1,1,6,6]` descriptor once to FLOAT.

- [ ] **Step 2: Implement the repeated-row terminal Einsum**

Use row basis `[valid, r*valid]`; keep the three-row column basis and four-row channel basis.
For each slot allocate eight private labels: rank, descriptor-left, descriptor-right, slot,
row-left, row-right, column, and channel.  Label the descriptor `NNas`, fold component weights
into the left factor, and end with `->NKRC`.  Assert exactly52 distinct ASCII labels or fewer.

- [ ] **Step 3: Build only after proof**

`main()` calls `factor_proof()` first, asserts the cost estimate, builds, runs full checker and
strict shape inference, saves `rank12_sign_cp.onnx`, and evaluates it with the official scorer.
It exits nonzero and does not save if proof or predicted-cost assertions fail.

- [ ] **Step 4: Run structural and bundled tests GREEN**

```bash
PYTHONPATH=. uv run pytest -q candidates/task037/test_rank12_sign_cp.py
```

Expected: all tests pass, actual cost at most1142, bundled fail0, all ten classes observed, and
off-grid nonzero count0.

---

### Task 4: Fresh and runtime diagnostics

**Files:**
- Read: `candidates/task037/rank12_sign_cp.onnx`
- Read: `submission/overfit_nets/task037.onnx`
- Modify only if needed for diagnostics: `candidates/task037/test_rank12_sign_cp.py`

**Interfaces:**
- Consumes: structurally valid bundled-pass candidate.
- Produces: fresh semantic and fresh-process runtime evidence for the adoption decision.

- [ ] **Step 1: Run fresh differential**

Use the repository fresh generator for at least500 deterministic examples.  Require incumbent
fail0, candidate fail0, candidate/incumbent boolean-output divergence0, finite logits, and raw
off-grid nonzero0.  Extend to2000 if the first500 complete within the session budget.

- [ ] **Step 2: Run balanced fresh-process runtime**

Alternate incumbent/candidate process order over representative examples, discard first-load
time, and compare medians.  Require candidate median no more than2x incumbent.

- [ ] **Step 3: Reject or promote**

If either diagnostic fails, do not gate rank12 and build the exact rank14 packed fallback.
Otherwise record the measured metrics and proceed.

---

### Task 5: Mandatory gate, adopt, and source ownership

**Files:**
- Modify through CLI only: `submission/overfit_nets/task037.onnx`
- Modify through CLI only: `state/tasks/task037.md`
- Modify after adoption: `src/custom/task037.py`
- Test: `candidates/task037/test_rank12_sign_cp.py`

**Interfaces:**
- Consumes: proof/bundled/fresh/runtime-approved candidate.
- Produces: adopted deployment and byte-identical canonical source.

- [ ] **Step 1: Run mandatory gate**

```bash
uv run ng gate candidates/task037/rank12_sign_cp.onnx --task 37
```

Expected: PASS, bundled266/266, fail0, cost at most1142, cheaper than1284.

- [ ] **Step 2: Adopt through the only allowed path**

```bash
uv run ng adopt candidates/task037/rank12_sign_cp.onnx --task 37 --note "rank12 sign-preserving six-mode CP with int8 descriptor pack and repeated affine row basis; conservative 5,139-state predicate audit, bundled/fresh/off-grid/runtime gates passed"
```

Expected: re-gate PASS and task037 ledger update.

- [ ] **Step 3: Synchronize exact source**

Move the frozen factors and graph construction into `src/custom/task037.py` without importing
candidate files.  Preserve initializer order, node order, graph name, IR10, opset20, and optional
protobuf fields so `build(load_task(37)).SerializeToString()` equals deployed bytes exactly.

- [ ] **Step 4: Verify candidate/deployed/source parity**

Run the focused test, isolated score, and SHA comparison.  Expected: candidate, deployed, and
source bytes match; bundled fail0; cost unchanged from gate.

---

### Task 6: Records and final verification

**Files:**
- Modify: `state/STATE.md`
- Modify: `state/levers.yaml`
- Modify: `state/insights.yaml`
- Modify: `candidates/task037/DISCOVERY.md` if present, otherwise create it.

**Interfaces:**
- Consumes: verified adopted endpoint or measured rejection.
- Produces: current handoff, four-field lever ledger, reusable mechanism, and next threshold.

- [ ] **Step 1: Replace the task037 STATE section**

Record old/new memory, params, cost, score gain, rank, proof domain/margin, bundled/fresh/runtime,
off-grid result, SHA, and the next +0.1 threshold.  Preserve unrelated concurrent STATE content.

- [ ] **Step 2: Record the lever and insight**

Add a dated four-field ledger result under `free-output-einsum-regime-crack`.  Register the
reusable mechanism only if rank12 is adopted; otherwise record the exact falsification and reopen
condition without declaring a floor.

- [ ] **Step 3: Run final verification from fresh commands**

Run full checker/strict inference, proof test, focused pytest, isolated `ng score 37`, source SHA
parity, YAML parse, and `git diff --check`.  Do not claim success from earlier cached output.

