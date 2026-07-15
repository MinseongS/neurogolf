# Task271 Hash-CHD Payload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace task271's 7x7 score grid and MaxPool with a bundled-perfect rank-1 FREE-input hash, CHD perfect lookup, and packed payload decoder at expected cost283.

**Architecture:** A rank-1 FLOAT Einsum maps each bundled input to an exact positive INT32 hash. A deterministic 32-bucket/353-slot CHD table maps the 267 hashes to unique slots; seven 9-bit payloads share each signed INT64 word. INT32 index arithmetic and INT64 division recover the selected 9-bit patch, then the adopted BitwiseAnd/Where/ConvInteger tail renders the FREE output.

**Tech Stack:** Python 3.13, NumPy, SciPy 1.18 for the recorded compact-selector falsification, ONNX 1.21.0, ONNX Runtime 1.26.0, pytest, NeuroGolf `ng` CLI.

## Global Constraints

- Work only on task271; candidate and scratch files stay under `candidates/task271/`.
- Keep `onnx==1.21.0` and `onnxruntime==1.26.0` unchanged.
- Baseline SHA is `a9a0d70fe35d4f9360883188b62fa8244b473242502739f27d696391ba5ad51b`, cost350, bundled267/267.
- Hard adoption gate is bundled fail0 plus strictly cheaper; fresh comparison is diagnostic and must be recorded exactly.
- Do not edit `submission/overfit_nets/task271.onnx` directly; use `ng gate` then `ng adopt`.
- Preserve the 10-channel 30x30 thresholded output and run the existing 49-position MaxPool regression even though the new graph removes MaxPool.
- Do not retry dual ReduceMax/ArgMax, `cyan_patch/crop8/Pad`, full-canvas Cast/QLinearConv, INT8 Where, sparse Conv, or dense CP dominance lowering.
- Execute inline on the user-authorized shared main checkout; stage only task271-specific files and exact shared clauses.

---

### Task 1: Specify the deterministic hash/CHD artifact

**Files:**
- Create: `candidates/task271/test_hash_chd_payload.py`
- Read: `candidates/task271/bitwise_where_decoder.onnx`
- Read: `data/task271.json`

**Interfaces:**
- Consumes: the deployed-equivalent cost350 candidate and the 267 bundled examples.
- Produces: a failing behavioral contract for `build_hash_chd_payload.py::build_candidate(incumbent: Path, output: Path) -> None` plus its public constants.

- [ ] **Step 1: Write the failing test**

Create `candidates/task271/test_hash_chd_payload.py`. The test must import the missing builder and require these exact facts:

```python
BASELINE_SHA256 = "a9a0d70fe35d4f9360883188b62fa8244b473242502739f27d696391ba5ad51b"
EXPECTED_COST = 283
EXPECTED_OPS = [
    "Einsum", "Cast", "Mod", "Div", "Gather", "Add", "Mod",
    "Div", "Mod", "Gather", "Gather", "Div", "Cast",
    "BitwiseAnd", "Cast", "Where", "ConvInteger",
]

def test_hash_chd_payload_is_bundled_perfect_and_cost283(tmp_path):
    builder = Path(__file__).with_name("build_hash_chd_payload.py")
    assert builder.exists(), "task271 hash/CHD builder is missing"
    module = load_module(builder)
    assert module.BUCKET_COUNT == 32
    assert module.SLOT_COUNT == 353
    assert module.HASH_CHANNEL.tolist() == [22, 17, 8, 23, 25, 10, 10, 32, 3, 22]
    assert module.HASH_ROW9.tolist() == [14, 16, 27, 31, 22, 29, 24, 21, 17]
    assert module.HASH_COL9.tolist() == [24, 13, 12, 6, 9, 23, 25, 13, 31]
    candidate = tmp_path / "hash_chd_payload.onnx"
    module.build_candidate(BASELINE, candidate)
    model = onnx.load(candidate)
    assert [node.op_type for node in model.graph.node] == EXPECTED_OPS
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    assert static_cost(candidate) == EXPECTED_COST
    result = eval_isolated(candidate, 271)
    assert result["ok"] and result["pass"] == 267 and result["fail"] == 0, result
    assert result["memory"] == 98 and result["params"] == 185, result
```

The same test must call `module.bundled_hash_evidence()` and assert:

```python
evidence = module.bundled_hash_evidence()
assert evidence["examples"] == 267
assert evidence["unique_hashes"] == 267
assert evidence["unique_slots"] == 267
assert evidence["hash_min"] == 435196
assert evidence["hash_max"] == 585189
assert evidence["max_abs_float_integer"] < 2**24
assert evidence["decoded_payloads"] == 267
```

Also require opset18, FLOAT hash factors `[10]`, `[30]`, `[30]`, INT32 displacement `[32]`, INT64 packed words `[51]`, INT64 lane divisors `[7]`, UINT16 masks `[1,1,3,3]`, candidate/persistent byte equality, and second-pass byte equality.

- [ ] **Step 2: Run the focused test and observe RED**

Run:

```bash
PYTHONPATH=. uv run pytest -q candidates/task271/test_hash_chd_payload.py
```

Expected: FAIL only because `candidates/task271/build_hash_chd_payload.py` does not exist.

- [ ] **Step 3: Commit the RED contract**

```bash
git add candidates/task271/test_hash_chd_payload.py
git commit -m "test(task271): specify hash CHD payload rewrite"
```

Expected: one commit containing only the new task271 test.

---

### Task 2: Build the cost283 graph

**Files:**
- Create: `candidates/task271/build_hash_chd_payload.py`
- Create: `candidates/task271/hash_chd_payload.onnx`
- Test: `candidates/task271/test_hash_chd_payload.py`

**Interfaces:**
- Consumes: `build_candidate(incumbent, output)` contract from Task 1 and `data/task271.json` for deterministic table generation.
- Produces: fixed-point candidate `candidates/task271/hash_chd_payload.onnx` and `bundled_hash_evidence() -> dict[str, int]`.

- [ ] **Step 1: Define pinned constants and bundled extraction**

Implement these exact constants:

```python
BASELINE_SHA256 = "a9a0d70fe35d4f9360883188b62fa8244b473242502739f27d696391ba5ad51b"
HASH_CHANNEL = np.array([22, 17, 8, 23, 25, 10, 10, 32, 3, 22], np.float32)
HASH_ROW9 = np.array([14, 16, 27, 31, 22, 29, 24, 21, 17], np.float32)
HASH_COL9 = np.array([24, 13, 12, 6, 9, 23, 25, 13, 31], np.float32)
BUCKET_COUNT = 32
SLOT_COUNT = 353
EXPECTED_DISPLACEMENT = np.array(
    [82, 104, 45, 1, 17, 323, 25, 11, 36, 61, 100, 8, 9, 240, 107,
     289, 3, 153, 41, 310, 57, 20, 0, 85, 52, 104, 0, 4, 6, 34, 243, 110],
    dtype=np.int32,
)
```

Pad the row and column factors with 21 trailing FLOAT zeros. Load the bundled examples in
`train + test + arc-gen` order. Compute each hash with integer arithmetic
`sum(C[color] * R[row] * S[col])` over the 9x9 grid and compute each payload in row-major order
as `sum((cell == 1) << bit)`.

- [ ] **Step 2: Implement deterministic CHD and seven-lane packing**

Implement `_build_tables()` with buckets `hash % 32`, secondary key `hash // 32`, buckets processed by descending size and then bucket id, and displacement candidates `0..352`. A candidate displacement is valid when all `(secondary + displacement) % 353` slots are distinct and unused. Assign singleton buckets to remaining free slots in descending free-slot order. Assert the result equals `EXPECTED_DISPLACEMENT`.

Create a length353 UINT9 logical table, place each payload at its assigned slot, then pack slots
`7*w .. 7*w+6` into one signed INT64:

```python
word = sum(int(code) << (9 * lane) for lane, code in enumerate(chunk))
assert 0 <= word <= np.iinfo(np.int64).max
```

The resulting initializer has exactly 51 elements. Verify offline that
`(packed[slot // 7] // (512 ** (slot % 7))) & 511` equals every bundled payload.

- [ ] **Step 3: Construct the pinned ONNX graph**

Build opset18/IR10 with the exact data flow:

```text
input,C,R,S --Einsum('bcrs,c,r,s->b')--> hash_f32
hash_f32 --Cast(INT32)--> hash_i32
hash_i32 % 32 --> bucket
hash_i32 / 32 --> secondary
Gather(displacement,bucket) --> displacement_i32
(secondary + displacement_i32) % 353 --> slot
slot / 7 --> word_index
slot % 7 --> lane
Gather(lane_divisors,lane) --> lane_divisor_i64
Gather(packed_words,word_index) --> packed_word_i64
packed_word_i64 / lane_divisor_i64 --> quotient_i64
Cast(quotient_i64,UINT16) --> encoded_u16
encoded_u16 & payload_masks --> masked_u16
Cast(masked_u16,BOOL) --> payload_bits
Where(payload_bits,two_u8,zero_u8) --> blue2
ConvInteger(blue2,render_weight_signed,one_u8,pads=[0,0,27,27]) --> output
```

Declare every intermediate with its fixed shape and dtype so strict shape inference and the official memory counter see exactly memory98. Validate incumbent SHA/topology on the first pass and validate the complete candidate on the second pass before saving.

- [ ] **Step 4: Build the persistent candidate**

```bash
PYTHONPATH=. uv run python candidates/task271/build_hash_chd_payload.py
```

Expected: prints `candidates/task271/hash_chd_payload.onnx`.

- [ ] **Step 5: Run GREEN verification**

```bash
PYTHONPATH=. uv run pytest -q candidates/task271/test_hash_chd_payload.py
```

Expected: PASS with measured cost283 = memory98 + params185 and bundled267/267.

- [ ] **Step 6: Commit builder and candidate**

```bash
git add candidates/task271/build_hash_chd_payload.py candidates/task271/hash_chd_payload.onnx
git commit -m "feat(task271): replace score grid with packed hash lookup"
```

Expected: one commit containing only the new builder and persistent candidate.

---

### Task 3: Run full task271 gates and fresh diagnostics

**Files:**
- Test: all `candidates/task271/test_*.py`
- Candidate: `candidates/task271/hash_chd_payload.onnx`

**Interfaces:**
- Consumes: persistent cost283 candidate.
- Produces: focused regression evidence, official gate result, and explicit fresh risk measurement.

- [ ] **Step 1: Run all task271 tests**

```bash
PYTHONPATH=. uv run pytest -q \
  candidates/task271/test_free_input_window.py \
  candidates/task271/test_maxpool_convinteger_fold.py \
  candidates/task271/test_direct_conv_score.py \
  candidates/task271/test_int32_signed_renderer.py \
  candidates/task271/test_encoded_score_payload.py \
  candidates/task271/test_bitwise_where_decoder.py \
  candidates/task271/test_hash_chd_payload.py
```

Expected: all eight task271 tests pass, including the retained 49-position geometry and 512-payload checks.

- [ ] **Step 2: Run isolated score and official gate**

```bash
PYTHONPATH=. uv run python -m neurogolf.scoring \
  candidates/task271/hash_chd_payload.onnx 271
uv run ng gate candidates/task271/hash_chd_payload.onnx --task 271
```

Expected: candidate cost283, bundled267/267, fail0, gate PASS against deployed cost350.

- [ ] **Step 3: Run fresh diagnostic**

```bash
PYTHONPATH=. uv run python candidates/fresh_compare_onnx.py \
  --task 271 --candidate candidates/task271/hash_chd_payload.onnx --n 2000
```

Expected: command completes. Record candidate fail count, incumbent fail count, raw divergence, sign divergence, and off-grid positives verbatim. A fresh regression is allowed by the approved bundled-overfit mode.

- [ ] **Step 4: Stop on any hard-gate mismatch**

Do not adopt if measured cost is at least350, bundled fail is nonzero, a focused test fails, the builder is not byte-stable, or `ng gate` rejects. Preserve the failing evidence and add a dated four-field lever entry instead.

---

### Task 4: Adopt, reconstruct exact source, and record the mechanism

**Files:**
- Modify via CLI: `submission/overfit_nets/task271.onnx`
- Modify via CLI: `state/manifest.json`, `state/safe_manifest.json`, `state/tasks/task271.md`
- Modify: `src/custom/task271.py`
- Modify: `candidates/task271/DISCOVERY.md`
- Modify exact task271 clauses only: `state/STATE.md`, `state/insights.yaml`
- Generate: `networks/task271.onnx`

**Interfaces:**
- Consumes: gate-passing cost283 candidate and fresh diagnostic evidence.
- Produces: adopted source-owned task271 fixed point.

- [ ] **Step 1: Adopt only through the CLI**

```bash
uv run ng adopt candidates/task271/hash_chd_payload.onnx --task 271 \
  --note "bundled rank1 hash + CHD perfect lookup + seven-lane INT64 payload packing"
```

Expected: adoption re-gates, backs up cost350, installs cost283, and stamps the task ledger.

- [ ] **Step 2: Generate exact source and rebuild**

Use a task-local source-net directory and the existing exact-source tool:

```bash
PYTHONPATH=. uv run python candidates/task271/build_hash_chd_payload.py \
  --incumbent candidates/task271/hash_chd_payload.onnx \
  --output candidates/task271/source_net/task271.onnx
PYTHONPATH=. uv run python tools/live_to_exact_source.py 271 \
  --net-dir candidates/task271/source_net --write-src
PYTHONPATH=. uv run python -m src.build --tasks 271
```

Require SHA equality among candidate, deployed net, `networks/task271.onnx`, and source rebuild.

- [ ] **Step 3: Update task271 handoff and insight**

Record cost350→283, memory238→98, params112→185, point gain `ln(350/283)`, bundled267/267,
fresh diagnostic counts, hash range435196..585189, 267 unique hashes/slots, B32/N353, packed51,
focused test count, adopt timestamp, and final SHA. Register the reusable mechanism as a bundled-only
rank-1 FREE-input hash plus CHD/bit-packed lookup; explicitly state that it memorizes the bundled corpus.

- [ ] **Step 4: Run fresh final verification**

```bash
uv run ng score 271
PYTHONPATH=. uv run pytest -q candidates/task271/test_hash_chd_payload.py
sha256sum candidates/task271/hash_chd_payload.onnx \
  submission/overfit_nets/task271.onnx networks/task271.onnx
git diff --check
```

Expected: deployed cost283/fail0, focused test PASS, all three SHA values equal, and no whitespace errors in task271 edits.

- [ ] **Step 5: Commit only isolated task271 changes**

Stage the task271 builder/test/candidate/source/task ledger/discovery plus exact task271 clauses in shared records. Inspect `git diff --cached --name-only` before committing; never include unrelated concurrent edits.

```bash
git commit -m "optimize task271 with packed hash lookup"
```

Do not run `ng pack` or submit.

## Plan self-review

- Spec coverage: bundled-only risk, cost precondition, pinned runtime, deterministic build, official gate/adopt, fresh diagnostic, source ownership, and no-repeat routes are covered.
- Type consistency: hash/index arithmetic is INT32; packed words/divisors/quotient are INT64; payload mask input is UINT16; renderer input is UINT8.
- Cost consistency: memory98 + params185 = cost283. Parameters are factors70 + displacement32 + packed51 + integer scalars3 + lane divisors7 + masks9 + renderer scalars3 + render weights10. Memory is hash4 + eight INT32 scalars32 + three INT64 scalars24 + encoded2 + masked18 + bits9 + blue2 9.
- The compact-selector LP rejection is dated exploratory evidence, not a floor claim. The implementation plan contains no unpriced nonlinear selector branch.
