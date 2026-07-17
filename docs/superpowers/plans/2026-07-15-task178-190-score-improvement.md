# Task178 and Task190 Score Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land every currently executable score improvement in task178 and task190 without bypassing bundled gates.

**Architecture:** Task178 first probes a uint8 TopK lane, then pivots to a legal first-inclusive Min/Max orientation collapse when the package gate rejects integer TopK. Task190 source-owns the archive polynomial renderer and precontracts its constant `poly_diff_coeff × poly_col_features` pair into one `[2,2,30]` bank, reducing the terminal Einsum from 17 to 13 operands.

**Observed final pivot:** The precontracted backbone proved that operand order, not bank size, was
the runtime blocker. Restoring the smaller factorized banks in the same backbone order produced a
17-operand cost1367 endpoint, so the final task190 deployment is factorized rather than precontracted.

**Tech Stack:** Python 3.13, ONNX 1.21.0, ONNX Runtime 1.26.0, NumPy, `uv run ng`.

## Global Constraints

- Work only in `/Users/minseong/project/neurogolf`; candidates stay below `candidates/taskNNN/`.
- Keep `onnx==1.21.0` and `onnxruntime==1.26.0` unchanged.
- Every implementation follows RED → GREEN and full bundled `ng gate` before adoption.
- Adoption is only through `uv run ng adopt`; never copy into `submission/overfit_nets/`.
- The official scorer rejects node GRAPH attributes, so task178 must not use `If`, `Loop`, or subgraphs.
- Do not run unbounded task190 full scoring until a one-example fresh-process benchmark finishes under two seconds.
- Preserve unrelated dirty files from concurrent range sessions.

---

### Task 1: Task178 uint8 TopK lane

**Files:**
- Create: `candidates/task178/test_u8_topk.py`
- Create: `candidates/task178/build_u8_topk.py`
- Produce: `candidates/task178/u8_topk.onnx`
- Modify after adoption: `src/custom/task178.py`, `state/tasks/task178.md`, `state/manifest.json`

**Interfaces:**
- Consumes: deployed cost564 task178 graph with `selected_starts`, `desc_weights`, `selected_scores`, and `selected_top_values`.
- Produces: bundled-exact cost546 candidate using uint8 `Where` and uint8 `TopK` values while retaining int64 TopK indices.

- [x] **Step 1: Write the failing regression test**

```python
from pathlib import Path
from neurogolf.scoring import evaluate, load_task

candidate = Path(__file__).with_name("u8_topk.onnx")
assert candidate.exists(), f"missing candidate: {candidate}"
result = evaluate(candidate, load_task(178))
assert result["fail"] == 0, result
assert result["memory"] + result["params"] <= 546, result
```

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=. uv run python candidates/task178/test_u8_topk.py`

Expected: failure stating `missing candidate: .../u8_topk.onnx`.

- [x] **Step 3: Implement the minimal graph transform**

Load `submission/overfit_nets/task178.onnx`, replace `desc_weights` with the same integer values encoded as `uint8`, replace `zero_f16` with scalar uint8 zero, keep the `Where(selected_starts, desc_weights, zero)` and `TopK` nodes unchanged, remove unused initializers/value info, run full ONNX checker, and save `u8_topk.onnx`.

```python
weights = numpy_helper.to_array(by_name["desc_weights"])
assert np.array_equal(weights, weights.astype(np.uint8))
replace_initializer("desc_weights", weights.astype(np.uint8))
replace_initializer("zero_f16", np.array(0, dtype=np.uint8))
set_value_info("selected_scores", TensorProto.UINT8, [13])
set_value_info("selected_top_values", TensorProto.UINT8, [5])
```

- [x] **Step 4: Verify isolated GREEN and observe gate rejection**

Run:

```bash
PYTHONPATH=. uv run python candidates/task178/build_u8_topk.py
PYTHONPATH=. uv run python candidates/task178/test_u8_topk.py
uv run ng gate candidates/task178/u8_topk.onnx --task 178
```

Expected: test exit 0 and gate PASS with bundled 268/268, cost546.

- [x] **Step 5: Do not adopt after package gate rejection**

Run:

```bash
uv run ng adopt candidates/task178/u8_topk.onnx --task 178 --note "dtype-lowering: integer run-priority scores and TopK values fp16->uint8"
uv run python tools/live_to_exact_source.py --write-src 178
uv run ng score 178
```

Expected: adopted cost546, fail0, source regenerated.

**Observed pivot:** The candidate was exact 268/268 at cost546, but `ng gate` rejected integer
TopK as a package-killing unsupported dtype. Do not adopt it. Execute Task 1B instead.

### Task 1B: Task178 min/max orientation predicate

**Files:**
- Create: `candidates/task178/test_minmax_orientation.py`
- Create: `candidates/task178/build_minmax_orientation.py`
- Produce: `candidates/task178/minmax_orientation.onnx`

**Interfaces:**
- Consumes: current row first-four colour values and existing `row_change_valid`.
- Produces: the identical scalar `is_row_output` through zero-sanitization and Min/Max equality.

- [x] Write and run a failing test that requires candidate existence, bundled fail0, and cost<=561.
- [x] Replace invalid tail values with the first colour using `Where(valid,curr,prev)`.
- [x] Compute scalar `row_change_min`, `row_change_max`, `row_change_all_same`, then `Not` to
  produce `is_row_output`; remove the old Equal/Not/And/Cast/ReduceMax chain.
- [x] Run full test and `uv run ng gate --task 178 candidates/task178/minmax_orientation.onnx`.
- [x] On PASS, adopt only through `ng adopt`, regenerate exact source, and re-score task178.

---

### Task 2: Task190 constant-precontract fast archive

**Files:**
- Create: `candidates/task190/test_archive_fast.py`
- Create: `candidates/task190/build_archive_fast.py`
- Produce: `candidates/task190/archive_fast_dynamic.onnx`
- Produce: `candidates/task190/archive_fast_paired.onnx`
- Produce: `candidates/task190/archive_fast_backbone.onnx`
- Modify after adoption: `src/custom/task190.py`, `state/tasks/task190.md`, `state/manifest.json`

**Interfaces:**
- Consumes: `candidates/task190/archive_hybrid.onnx`, whose terminal equation is the 17-operand polynomial output renderer.
- Produces: mathematically identical 13-operand variants using `poly_diff_col[t,d,w] = Σp poly_diff_coeff[t,d,p] * poly_col_features[p,w]`, each initially cost1429 and cost1419 after deleting unused `C_u8`.

- [x] **Step 1: Write the failing bounded regression test**

The test evaluates only the first official example so a slow candidate cannot consume an unbounded gate run.

```python
import time
from pathlib import Path
from neurogolf.scoring import evaluate, load_task

candidate = Path(__file__).with_name("archive_fast_backbone.onnx")
assert candidate.exists(), f"missing candidate: {candidate}"
task = load_task(190)
mini = {"train": [task["train"][0]], "test": [], "arc-gen": []}
started = time.perf_counter()
result = evaluate(candidate, mini)
elapsed = time.perf_counter() - started
assert result["fail"] == 0, result
assert result["memory"] + result["params"] <= 1419, result
assert elapsed < 2.0, {"elapsed": elapsed, "result": result}
```

- [x] **Step 2: Verify RED**

Run: `PYTHONPATH=. uv run python candidates/task190/test_archive_fast.py`

Expected: failure stating `missing candidate: .../archive_fast_paired.onnx`.

- [x] **Step 3: Implement source-owned precontraction and operand orders**

Use this exact constant contraction:

```python
diff_col = np.einsum("tdp,pw->tdw", diff_coeff, col_features).astype(np.float32)
```

Delete `poly_diff_coeff` and `poly_col_features`, add `poly_diff_col[2,2,30]`, and replace the terminal algebra with:

```text
bshw,vkxy,rd,re,ra,rf,tdw,tew,taw,tfw,tv,rh,vks->bkhw
```

Write three files whose input/subscript pairs are reordered together:

- dynamic: dynamic features before the four banks;
- paired: each feature immediately followed by its corresponding bank;
- backbone: input/color/row/route/term before polynomial pairs.

Keep every upstream node and dtype unchanged. Run strict inference and full checker for all variants.

- [x] **Step 4: Benchmark variants in fresh bounded processes**

Run each candidate via `timeout 10` on one official example, three times. Select the lowest median that finishes every run. Expected static cost is memory750 + params679 =1429.

- [x] **Step 5: Verify GREEN on the measured-fastest backbone variant**

Use the measured-fastest `archive_fast_backbone.onnx` as the deployment candidate. The measured
one-example medians were dynamic ~1.52s, paired ~0.62s, and backbone ~0.089s. Run:

```bash
PYTHONPATH=. uv run python candidates/task190/test_archive_fast.py
```

Expected: one-example exact, cost<=1419, elapsed<2s.

- [x] **Step 6: Run the full bundled gate only after bounded GREEN**

Run: `uv run ng gate --task 190 candidates/task190/archive_fast_backbone.onnx`

Expected: 266/266, cost1419, PASS.

- [x] **Step 7: Adopt and synchronize source**

Run:

```bash
uv run ng adopt candidates/task190/archive_fast_backbone.onnx --task 190 --note "runtime-safe output-fold: precontract constant polynomial coefficient/column banks; 17->13 terminal Einsum operands"
uv run python tools/live_to_exact_source.py --write-src 190
uv run ng score 190
```

Expected: adopted cost1419, fail0.

### Task 2B: Task190 parameter-cheap factorized backbone

- [x] Add a RED bounded test for `archive_reordered_backbone.onnx` with cost<=1367 and elapsed<2s.
- [x] Reorder the original 17 input/subscript pairs around input/row/route/color/term while retaining
  `poly_diff_coeff` and `poly_col_features`; delete unused `C_u8`.
- [x] Benchmark three fresh processes: median ~0.389s, memory750, params617, cost1367.
- [x] Run full bundled gate: 266/266 PASS.
- [x] Adopt 1419->1367 through `ng adopt`, regenerate `src/custom/task190.py`, and rebuild cost1367.

---

### Task 3: Recursive rescan, documentation, and final verification

**Files:**
- Modify: `candidates/task178/DISCOVERY.md`
- Modify: `candidates/task190/DISCOVERY.md`
- Modify: `state/insights.yaml` if the task190 precontract lands
- Replace/merge current truth into: `state/STATE.md`

**Interfaces:**
- Consumes: adopted task178/task190 artifacts and measured gate outputs.
- Produces: current handoff, reusable constant-precontract insight, and a cross-board rescan result.

- [x] **Step 1: Update discovery documents with actual results**

Record task178's scorer-level If prohibition and uint8 TopK result. For task190 record every variant's one-example median, cost, full gate result, and the exact winning equation.

- [x] **Step 2: Register and rescan a landed task190 mechanism**

If adopted, update `state/insights.yaml` with the condition “initializer-only repeated bilinear operands inside a terminal variadic Einsum”, the cost equation, runtime evidence, rejection conditions, and transformer path. Scan all 400 terminal Einsums for repeated initializer pairs matching `A[x,y],B[y,z]` that can be prefolded without increasing params beyond the measured saving.

- [x] **Step 3: Run fresh final verification**

Run:

```bash
PYTHONPATH=. uv run python candidates/task178/test_u8_topk.py
uv run ng score 178
uv run ng score 184
uv run ng score 190
uv run python -m py_compile candidates/task178/build_u8_topk.py candidates/task178/test_u8_topk.py candidates/task190/build_archive_fast.py candidates/task190/test_archive_fast.py
uv run python -c "import yaml; yaml.safe_load(open('state/insights.yaml')); yaml.safe_load(open('state/levers.yaml'))"
uv run ng status
```

Expected: all changed task scores fail0, Python compilation succeeds, YAML parses, and 400/400 nets remain deployed.

- [x] **Step 4: Replace the session handoff and leave uncommitted because shared files overlap concurrent work**

Update `state/STATE.md` in place with only current facts. Stage only files owned by this task; do not stage shared unrelated changes. If shared-file overlap prevents a coherent selective commit, leave the verified working tree uncommitted and report the overlap rather than bundling other sessions' work.
