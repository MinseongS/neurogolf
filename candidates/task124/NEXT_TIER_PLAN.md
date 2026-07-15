# Task124 Exact Cost849 QLinear Fold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lower task124 from cost939 to cost849 by keeping the source-row bank rank 4 and replacing the ten-cell p3 equality reduction with an exact uint8 QLinear row fingerprint.

**Architecture:** Transform the immutable cost939 task124 graph in one task-local builder. Preserve the centered uint8 mask, runtime ScatterElements colour weights, and padded uint8 QLinearConv direct output; change only p3 detection and row-bank geometry. After official adoption, express the same graph semantically in `src/custom/task124.py` and synchronize task-local discovery plus live state.

**Tech Stack:** Python 3.12, NumPy, ONNX 1.21.0, ONNX Runtime 1.26.0, pytest, NeuroGolf `ng` CLI.

## Global Constraints

- Modify only task124 source, candidate artifacts, and task124-related state/discovery records.
- Keep `onnx==1.21.0` and `onnxruntime==1.26.0` unchanged.
- Final graph output must be uint8 and produced directly by padded `QLinearConv`.
- `w_zero_point=1` semantics remain background `-1`, selected `+1`, others `0`.
- Acceptance target is memory779, params70, cost849; bundled gate fail=0; fresh2000 candidate fail=0 and divergence=0; off-grid positives=0.
- Run `ng adopt` only after an official `ng gate` PASS; do not run `ng pack` or `ng submit`.
- Preserve unrelated concurrent changes and stage exact task124 paths only.

---

### Task 1: Add the failing cost849 regression

**Files:**
- Create: `candidates/task124/test_rank4_qlinear_hash.py`
- Test: `candidates/task124/test_rank4_qlinear_hash.py`

**Interfaces:**
- Consumes: deployed `submission/overfit_nets/task124.onnx`; generator rule in `arc-gen/tasks/task_53b68214.py`.
- Produces: acceptance contract for `build_rank4_qlinear_hash.build(task=None) -> onnx.ModelProto` and `rank4_qlinear_hash.onnx`.

- [ ] **Step 1: Write the failing import and structural/cost test**

Create a real-model test that imports `build`, runs checker plus strict inference, and asserts:

```python
from build_rank4_qlinear_hash import build

candidate = build()
nodes = {output: node for node in candidate.graph.node for output in node.output}
assert [node.op_type for node in candidate.graph.node].count("QLinearMatMul") == 2
assert nodes["fg_pad4d"].op_type == "Concat"
assert nodes["bottom_fg"].op_type == "Concat"
assert nodes["output"].op_type == "QLinearConv"
assert not {
    "row0_flat", "row1_flat", "row2_flat", "bottom_flat",
    "p3_equal", "p3_equal_u8", "p3_all_equal",
}.intersection(nodes)
result = evaluate(CANDIDATE, load_task(124), keep_failures=True)
assert (result["fail"], result["memory"], result["params"]) == (0, 779, 70)
assert result["memory"] + result["params"] == 849
```

Also inspect inferred types/shapes for `p3_hash_a`, `p3_hash_b`, `is_p3`, `fg_pad4d`, all five bottom rows, `bottom_fg`, and `output`.

- [ ] **Step 2: Add raw A/B, off-grid, quantized-tail, and exhaustive-hash tests**

Use actual ORT sessions for incumbent and candidate. For every bundled example, require byte-identical raw outputs and zero nonzero logits outside rows/columns 0..9. Inspect initializers to assert centered states `0/2`, shared x/w zero point `1`, weight base `[0,1,...,1]`, and hash code `[1,2,4,8,16,32,0,0,0,0]^T`.

Enumerate all legal `(tall, wide, sprite subset, offset, diag)` combinations:

```python
weights = np.asarray([1, 2, 4, 8, 16, 32, 0, 0, 0, 0], dtype=np.int64)
cases = 0
for tall in (2, 3):
    for wide in (1, 2, 3):
        cells = [(row, col) for row in range(tall) for col in range(wide)]
        for bits in range(1, 1 << len(cells)):
            sprite = [cells[index] for index in range(len(cells)) if bits >> index & 1]
            if {row for row, _ in sprite} != set(range(tall)):
                continue
            if {col for _, col in sprite} != set(range(wide)):
                continue
            for offset in range(4):
                for diag in ((0,) if tall == 3 else (0, 1)):
                    compared = []
                    for target_row in (1, 4):
                        mask = np.zeros(10, dtype=np.uint8)
                        for repeat in range(10):
                            for sprite_row, sprite_col in sprite:
                                row = repeat * tall + sprite_row
                                col = repeat * (wide - 1) * diag + sprite_col + offset
                                if row == target_row and 0 <= col < 10:
                                    mask[col] = 2
                        compared.append(mask)
                    row1, row4 = compared
                    row1_hash = int(row1.astype(np.int64) @ weights)
                    row4_hash = int(row4.astype(np.int64) @ weights)
                    assert np.array_equal(row1, row4) == (
                        row1_hash == row4_hash
                    )
                    assert max(row1_hash, row4_hash) <= 126
                    cases += 1
assert cases == 1_428
```

Require 1,428 cases, zero mismatches, and every fingerprint <=126.

- [ ] **Step 3: Run the focused test and verify RED**

Run:

```bash
uv run pytest -q candidates/task124/test_rank4_qlinear_hash.py
```

Expected: collection ERROR with `ModuleNotFoundError: No module named 'build_rank4_qlinear_hash'`. This proves the new regression is exercising a missing implementation.

- [ ] **Step 4: Commit the RED test**

Stage only the new ignored candidate test with `git add -f`, verify the cached diff contains no unrelated path, and commit:

```bash
git commit -m "test(task124): specify exact cost849 qlinear fold"
```

### Task 2: Implement the minimal rank-4/hash candidate

**Files:**
- Create: `candidates/task124/build_rank4_qlinear_hash.py`
- Create: `candidates/task124/rank4_qlinear_hash.onnx` (generated scratch, not committed)
- Test: `candidates/task124/test_rank4_qlinear_hash.py`

**Interfaces:**
- Consumes: exact current source-build SHA `49ececad3443d478c5f9b3e335f8ced4df82aa25648f89e85bacd985b8737632`.
- Produces: `build(task=None) -> onnx.ModelProto` and serialized `rank4_qlinear_hash.onnx`.

- [ ] **Step 1: Implement initializer rewrites**

Build the current semantic source graph, assert its SHA, and rewrite only:

```python
remove = {"row_flat_shape"}
p3b_idx = np.asarray([4], dtype=np.int64)
false8 = np.zeros((1, 1, 1, 8), dtype=np.uint8)
slice_axes = np.asarray([3], dtype=np.int32)
hash_code = np.asarray([1, 2, 4, 8, 16, 32, 0, 0, 0, 0], dtype=np.uint8).reshape(10, 1)
```

Preserve every other initializer byte/value and append `hash_code` in a deterministic position.

- [ ] **Step 2: Implement p3 hash and rank-4 row-bank node rewrites**

Replace the cellwise equality chain with two `QLinearMatMul` nodes reusing `ng_scale` and `ng_y_zero_point`, then scalar-element `Equal`. Keep `is_p3` rank 4, change the `source_offset` Split axis to 4, remove the three row Reshapes, concatenate the bank on axis 3, slice on axis 3, and concatenate bottom rows on axis 2:

```python
QLinearMatMul(row, ng_scale, ng_y_zero_point,
              hash_code, ng_scale, ng_y_zero_point,
              ng_scale, ng_y_zero_point) -> p3_hash_*
Concat(false8, row0_4d, false8, p3_rows_a, row2_4d, axis=3) -> fg_pad4d
Slice(fg_pad4d, ..., slice_axes=[3]) -> bottom_row_i
Concat(bottom_row_0, ..., bottom_row_4, axis=2) -> bottom_fg
```

Provide explicit uint8 value information for runtime Slice outputs and bottom tensors before strict inference.

- [ ] **Step 3: Run GREEN focused verification**

Run:

```bash
uv run pytest -q candidates/task124/test_rank4_qlinear_hash.py
```

Expected: all tests PASS; scorer reports exactly fail0, memory779, params70, cost849; raw bundled A/B and off-grid assertions pass.

- [ ] **Step 4: Run the previous task124 regression suite**

Run all task124 tests:

```bash
uv run pytest -q candidates/task124/test_*.py
```

Expected: all tests PASS. If historical builders intentionally pin previous stages, their old exact cost assertions must remain unchanged.

- [ ] **Step 5: Commit builder and GREEN regression**

Force-add only the builder and updated test, exclude generated ONNX, inspect cached diff, then commit:

```bash
git commit -m "feat(task124): add exact cost849 qlinear fold"
```

### Task 3: Gate and adopt the measured candidate

**Files:**
- Generate: `candidates/task124/rank4_qlinear_hash.onnx`
- Modify through CLI only on success: `submission/overfit_nets/task124.onnx`, manifest/task ledger files stamped by `ng adopt`.

**Interfaces:**
- Consumes: passing cost849 candidate.
- Produces: officially adopted task124 deployment or leaves cost939 deployment unchanged.

- [ ] **Step 1: Serialize and independently score the candidate**

Run the builder in a fresh process, checker/strict inference, `uv run ng score 124` for the incumbent, and isolated candidate evaluation. Confirm deployed SHA is still the expected cost939 baseline before gate.

- [ ] **Step 2: Run the mandatory gate**

```bash
uv run ng gate candidates/task124/rank4_qlinear_hash.onnx --task 124
```

Expected: PASS, bundled 267/267, fail0, cost849 < 939. On any rejection, stop without adopt and record the dated four-field negative ledger entry.

- [ ] **Step 3: Adopt only after PASS**

```bash
uv run ng adopt candidates/task124/rank4_qlinear_hash.onnx --task 124 --note "rank-4 row bank plus exact uint8 QLinear row hash"
```

Expected: re-gate PASS and deployed cost849. Do not run pack or submit.

### Task 4: Synchronize semantic source and durable records

**Files:**
- Modify: `src/custom/task124.py`
- Modify: `candidates/task124/build_rank4_qlinear_hash.py`
- Modify: `candidates/task124/build_dynamic_qlinear_tail.py`
- Modify: `candidates/task124/build_centered_qlinear_tail.py`
- Modify: `candidates/task124/build_scatter_weight.py`
- Modify: `candidates/task124/build_centered_slice_geometry.py`
- Modify: `candidates/task124/build_shared_initializers.py`
- Modify: `candidates/task124/DISCOVERY.md`
- Modify: `state/tasks/task124.md`
- Modify: `state/insights.yaml`
- Modify: `state/levers.yaml`
- Replace relevant live summary in: `state/STATE.md`

**Interfaces:**
- Consumes: adopted candidate SHA and measured cost.
- Produces: byte-identical semantic source build and synchronized task124 knowledge records.

- [ ] **Step 1: Update semantic source to construct the adopted graph directly**

Apply the same initializer/node/value-info changes from Task 2 to `src/custom/task124.py`. Keep the direct QLinearConv output and explanatory zero-point docstring. Build in a fresh process and require source-built SHA equals deployed SHA.

- [ ] **Step 2: Make the candidate builder a post-adoption fixed point**

Fill its adopted SHA constant. When the live deployed/source graph already has that SHA, return the source build unchanged and assert SHA equality; otherwise transform only the exact cost939 baseline.

Update every historical task124 stage builder's `FINAL_SHA256` to the new adopted SHA so it continues to load its own pinned ONNX stage after deployment advances. Add the same pinned-artifact fallback to `build_shared_initializers.py`, whose current cost939 endpoint had not previously needed one. This keeps all earlier exact-cost regressions runnable without rebuilding old stages from the new incumbent.

- [ ] **Step 3: Update discovery and state**

Record cost `939 -> 849`, exact graph mechanism, generator-domain proof, gate/adopt evidence, fresh/off-grid evidence, final SHA, next `+0.1` boundary `cost<=768`, and no pack/submit. Extend the reusable insight with rank-preserving dynamic row banks and collision-free bounded QLinear fingerprints. Update the existing live lever entry with a new four-field ledger record rather than overwriting history. Replace the task124 section in `STATE.md`; do not append a second stale section.

### Task 5: Fresh verification, scoped propagation scan, and final commit

**Files:**
- Verify: all Task 4 files and deployed artifact.
- May update: `candidates/task124/DISCOVERY.md`, `state/insights.yaml`, `state/levers.yaml`, `state/STATE.md` with measured evidence only.

**Interfaces:**
- Consumes: adopted/source-synchronized cost849 endpoint.
- Produces: final evidence-backed task124 handoff and task-scoped commit.

- [ ] **Step 1: Run fixed-point and focused regressions**

In fresh processes, require candidate/deployed/source SHA equality, checker/strict inference, `uv run ng score 124` fail0/cost849, and all `candidates/task124/test_*.py` passing.

- [ ] **Step 2: Run fresh2000 A/B and off-grid diagnostics**

Generate 2,000 valid task124 instances. Compare candidate and pre-adoption incumbent raw outputs, targets, and off-grid logits. Require candidate fail0, incumbent fail0, divergence0, and off-grid positives0.

- [ ] **Step 3: Run scoped post-insight scan**

Scan all 400 deployed graphs read-only for the specific signatures: rank-flattened short row banks feeding fixed-length dynamic Slices, and cellwise `Equal -> Cast -> ReduceMin` over bounded uint8 rows. Record exact/broad hits and errors. Do not modify another task.

- [ ] **Step 4: Run final verification-before-completion commands**

Re-read the design/plan requirements, run `git diff --check` on task124 paths, inspect exact diffs, confirm no pack/submit artifact was created by this session, and rerun the full focused verification command immediately before claiming success.

- [ ] **Step 5: Commit exact task124 scope**

Stage only the semantic source, builder/test/design/plan/discovery, task ledger, insight, lever, and replaced STATE files. Inspect `git diff --cached --name-only` for unrelated paths, then commit with the project trailer:

```bash
git commit -m "opt(task124): reach exact cost849 qlinear renderer" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

Do not push, pack, or submit.
