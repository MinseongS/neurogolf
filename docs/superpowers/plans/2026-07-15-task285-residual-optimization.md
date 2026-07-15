# Task285 Residual Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adopt two exact task285 sparse-graph reductions from cost 15,982 to 15,648, then reprice and record the remaining large-carrier lane.

**Architecture:** First remove the redundant four-tensor target-sentinel suffix and its one-use scalar initializer. Then replace the shared rank-2 pivot-index adapter with two smaller direct gathers plus rank restoration. Each stage has its own RED/GREEN test cycle, official gate, and adoption; source synchronization happens after the final adoption.

**Tech Stack:** Python 3.13, ONNX 1.21.0, ONNX Runtime 1.26.0, NumPy, pytest, `uv run ng`.

## Global Constraints

- Work only under `candidates/task285/`, task285 source/tests, and task285 state records.
- Keep `onnx==1.21.0` and `onnxruntime==1.26.0` unchanged.
- Every TopK input must remain FLOAT16 or FLOAT32.
- Every adoption must run `uv run ng gate` before `uv run ng adopt`; never copy into `submission/overfit_nets/`.
- Bundled success is exactly 265/265 with fail 0 and strictly lower deployed cost.
- Fresh evaluation is diagnostic; do not weaken the reconstructed generator to hide the known K=31 neighbor-Gather failure.
- Do not run `ng pack` or `ng submit` in this plan.
- Preserve unrelated dirty files and concurrent shared-state changes.

---

## File Structure

- `candidates/task285/build_target_sentinel_fold.py`: source-owned graph surgery for the 325-cost suffix deletion.
- `candidates/task285/target_sentinel_fold_candidate.onnx`: persistent stage-1 gate artifact.
- `candidates/task285/scratch/tests/test_target_sentinel_fold.py`: stage-1 structure, dtype, cost, and bundled contract.
- `candidates/task285/build_selected_index_fold.py`: stage-2 graph surgery for the 9-byte rank-adapter reduction.
- `candidates/task285/selected_index_fold_candidate.onnx`: persistent final gate artifact.
- `candidates/task285/scratch/tests/test_selected_index_fold.py`: stage-2 structure, cost, and bundled contract.
- `candidates/task285/rescan_redundant_target_sentinel.py`: structural 400-graph transfer scan.
- `candidates/task285/DISCOVERY.md`: measured implementation and residual-lane handoff.
- `src/custom/task285.py`: exact final deployed source, regenerated only after adoption.
- `candidates/source_sync_285_295/test_task285_295_source_sync.py`: final cost and source/deployment fixed-point contract.
- `state/tasks/task285.md`: adoption blocks written by `ng adopt`.
- `state/insights.yaml`, `state/levers.yaml`, `state/STATE.md`: reusable mechanism, scoped negative result, and live handoff; preserve concurrent edits.

---

### Task 1: Target-sentinel suffix RED/GREEN

**Files:**
- Create: `candidates/task285/scratch/tests/test_target_sentinel_fold.py`
- Create: `candidates/task285/build_target_sentinel_fold.py`
- Create: `candidates/task285/target_sentinel_fold_candidate.onnx`

**Interfaces:**
- Consumes: deployed task285 SHA `dc509d96955eea40d6984648934e9b345293057aa61b6f2bf0d575106fa09120` at cost 15,982.
- Produces: `build_model() -> onnx.ModelProto` and a cost-15,657 candidate whose `ScatterElements` consumes `v81` directly.

- [ ] **Step 1: Write the failing acceptance test**

Create the test with these contracts:

```python
from __future__ import annotations

import pathlib
import onnx
from onnx import TensorProto

from neurogolf.scans.public_autopsy import profile
from neurogolf.scoring import calculate_params, evaluate, load_task

ROOT = pathlib.Path(__file__).resolve().parents[4]
BUILDER = ROOT / "candidates/task285/build_target_sentinel_fold.py"
MODEL = ROOT / "candidates/task285/target_sentinel_fold_candidate.onnx"


def test_target_sentinel_builder_exists():
    assert BUILDER.exists(), "target-sentinel fold builder is not implemented"


def test_target_sentinel_fold_structure_and_cost():
    inferred = onnx.shape_inference.infer_shapes(onnx.load(MODEL), strict_mode=True)
    by_output = {output: node for node in inferred.graph.node for output in node.output}
    assert not {"target_base", "target10", "target_cap", "v81_safe"} & set(by_output)
    assert list(by_output["newg"].input) == ["g", "tidx", "v81"]
    assert "i89" not in {item.name for item in inferred.graph.initializer}
    types = {
        value.name: value.type.tensor_type.elem_type
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    for node in inferred.graph.node:
        if node.op_type == "TopK":
            assert types[node.input[0]] in {TensorProto.FLOAT16, TensorProto.FLOAT}
    result = profile(MODEL)
    params = calculate_params(inferred)
    assert result is not None and params is not None
    assert int(result["memory_static"]) == 15_424
    assert params == 233
    assert int(result["memory_static"]) + params == 15_657


def test_target_sentinel_fold_passes_bundled():
    result = evaluate(MODEL, load_task(285), keep_failures=True)
    assert result["ok"], result
    assert result["pass"] == 265
    assert result["fail"] == 0
    assert result["memory"] + result["params"] == 15_657
```

- [ ] **Step 2: Run the RED test**

Run:

```bash
PYTHONPATH=. uv run pytest -q candidates/task285/scratch/tests/test_target_sentinel_fold.py::test_target_sentinel_builder_exists
```

Expected: FAIL with `target-sentinel fold builder is not implemented`.

- [ ] **Step 3: Implement the minimal builder**

Create the builder with this complete transform:

```python
#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import pathlib

import onnx

from neurogolf.scans.public_autopsy import profile
from neurogolf.scoring import calculate_params

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "submission/overfit_nets/task285.onnx"
OUTPUT = pathlib.Path(__file__).resolve().parent / "target_sentinel_fold_candidate.onnx"
SOURCE_SHA256 = "dc509d96955eea40d6984648934e9b345293057aa61b6f2bf0d575106fa09120"
EXPECTED_COST = 15_657
REMOVED = {"target_base", "target10", "target_cap", "v81_safe"}


def clone_node(node: onnx.NodeProto) -> onnx.NodeProto:
    copied = onnx.NodeProto()
    copied.CopyFrom(node)
    return copied


def build_model() -> onnx.ModelProto:
    source_bytes = SOURCE.read_bytes()
    actual_sha = hashlib.sha256(source_bytes).hexdigest()
    if actual_sha != SOURCE_SHA256:
        raise ValueError(f"unrecognized task285 source sha256={actual_sha}")
    model = onnx.load_model_from_string(source_bytes)
    nodes = []
    for node in model.graph.node:
        if set(node.output) & REMOVED:
            continue
        copied = clone_node(node)
        if "newg" in copied.output:
            for index, name in enumerate(copied.input):
                if name == "v81_safe":
                    copied.input[index] = "v81"
        nodes.append(copied)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    used = {name for node in model.graph.node for name in node.input}
    initializers = []
    for initializer in model.graph.initializer:
        if initializer.name in used:
            copied = onnx.TensorProto()
            copied.CopyFrom(initializer)
            initializers.append(copied)
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    del model.graph.value_info[:]
    model.graph.name = "task285_redundant_target_sentinel_fold"
    model.producer_name = "task285-target-sentinel-fold"
    onnx.checker.check_model(model, full_check=True)
    return onnx.shape_inference.infer_shapes(model, strict_mode=True)


def main() -> None:
    model = build_model()
    onnx.save(model, OUTPUT)
    result = profile(OUTPUT)
    params = calculate_params(model)
    assert result is not None and params is not None
    cost = int(result["memory_static"]) + params
    print(f"saved {OUTPUT}")
    print(f"memory={result['memory_static']} params={params} cost={cost}")
    if cost != EXPECTED_COST:
        raise AssertionError(f"unexpected cost {cost} != {EXPECTED_COST}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Build and run the GREEN tests**

Run:

```bash
uv run python candidates/task285/build_target_sentinel_fold.py
PYTHONPATH=. uv run pytest -q candidates/task285/scratch/tests/test_target_sentinel_fold.py
```

Expected: builder prints `memory=15424 params=233 cost=15657`; all 3 tests PASS.

- [ ] **Step 5: Commit the isolated stage**

```bash
git add -f candidates/task285/build_target_sentinel_fold.py \
  candidates/task285/target_sentinel_fold_candidate.onnx \
  candidates/task285/scratch/tests/test_target_sentinel_fold.py
git commit -m "optimize task285 target sentinel suffix" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Gate and adopt the 15,657 stage

**Files:**
- Modify: `submission/overfit_nets/task285.onnx` through `ng adopt` only.
- Modify: `state/manifest.json` through `ng adopt` only.
- Modify: `state/tasks/task285.md` through `ng adopt` only.

**Interfaces:**
- Consumes: `candidates/task285/target_sentinel_fold_candidate.onnx`.
- Produces: deployed task285 cost 15,657 with an auto-stamped adoption record.

- [ ] **Step 1: Reconfirm the incumbent**

```bash
uv run ng score 285
shasum -a 256 submission/overfit_nets/task285.onnx
```

Expected: cost 15,982, pass 265, fail 0, SHA beginning `dc509d96955e`.

- [ ] **Step 2: Run the mandatory gate**

```bash
uv run ng gate candidates/task285/target_sentinel_fold_candidate.onnx --task 285
```

Expected: PASS, 265/265, fail 0, cost 15,657.

- [ ] **Step 3: Adopt through the mandatory command**

```bash
uv run ng adopt candidates/task285/target_sentinel_fold_candidate.onnx --task 285 \
  --note "remove redundant target sentinel after invalid sparse update gate"
```

Expected: JSON with `"ok": true`, `"fail": 0`, and `"cost": 15657`.

- [ ] **Step 4: Verify the live endpoint**

```bash
uv run ng score 285
```

Expected: pass 265, fail 0, cost 15,657.

---

### Task 3: Selected-index rank-adapter RED/GREEN and adoption

**Files:**
- Create: `candidates/task285/scratch/tests/test_selected_index_fold.py`
- Create: `candidates/task285/build_selected_index_fold.py`
- Create: `candidates/task285/selected_index_fold_candidate.onnx`

**Interfaces:**
- Consumes: immutable stage-1 candidate cost 15,657.
- Produces: cost-15,648 candidate with direct `Gather(t, si)` and `Gather(c, si)` followed by small rank restoration.

- [ ] **Step 1: Write the failing acceptance test**

```python
from __future__ import annotations

import pathlib
import onnx

from neurogolf.scans.public_autopsy import profile
from neurogolf.scoring import calculate_params, evaluate, load_task

ROOT = pathlib.Path(__file__).resolve().parents[4]
BUILDER = ROOT / "candidates/task285/build_selected_index_fold.py"
MODEL = ROOT / "candidates/task285/selected_index_fold_candidate.onnx"


def test_selected_index_builder_exists():
    assert BUILDER.exists(), "selected-index fold builder is not implemented"


def test_selected_index_fold_structure_cost_and_bundled():
    inferred = onnx.shape_inference.infer_shapes(onnx.load(MODEL), strict_mode=True)
    by_output = {output: node for node in inferred.graph.node for output in node.output}
    assert "si2" not in by_output
    assert list(by_output["a_flat"].input) == ["t", "si"]
    assert list(by_output["acol_flat"].input) == ["c", "si"]
    assert by_output["a"].op_type == "Unsqueeze"
    assert by_output["acol"].op_type == "Unsqueeze"
    result = profile(MODEL)
    params = calculate_params(inferred)
    assert result is not None and params is not None
    assert int(result["memory_static"]) == 15_415
    assert params == 233
    assert int(result["memory_static"]) + params == 15_648
    evaluated = evaluate(MODEL, load_task(285), keep_failures=True)
    assert evaluated["pass"] == 265
    assert evaluated["fail"] == 0
```

- [ ] **Step 2: Run the RED test**

```bash
PYTHONPATH=. uv run pytest -q candidates/task285/scratch/tests/test_selected_index_fold.py::test_selected_index_builder_exists
```

Expected: FAIL with `selected-index fold builder is not implemented`.

- [ ] **Step 3: Implement the minimal second builder**

Create a builder that loads `target_sentinel_fold_candidate.onnx`, asserts its
cost is 15,657, omits `si2`, and replaces only the `a`/`acol` producers:

```python
#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import onnx
from onnx import helper

from neurogolf.scans.public_autopsy import profile
from neurogolf.scoring import calculate_params

HERE = pathlib.Path(__file__).resolve().parent
SOURCE = HERE / "target_sentinel_fold_candidate.onnx"
OUTPUT = HERE / "selected_index_fold_candidate.onnx"
EXPECTED_SOURCE_COST = 15_657
EXPECTED_COST = 15_648


def clone_node(node: onnx.NodeProto) -> onnx.NodeProto:
    copied = onnx.NodeProto()
    copied.CopyFrom(node)
    return copied


def build_model() -> onnx.ModelProto:
    model = onnx.load(SOURCE)
    source_profile = profile(SOURCE)
    source_params = calculate_params(model)
    assert source_profile is not None and source_params is not None
    assert int(source_profile["memory_static"]) + source_params == EXPECTED_SOURCE_COST
    nodes = []
    for node in model.graph.node:
        outputs = set(node.output)
        if "si2" in outputs:
            continue
        if "a" in outputs:
            nodes.extend([
                helper.make_node("Gather", ["t", "si"], ["a_flat"]),
                helper.make_node("Unsqueeze", ["a_flat", "axs1"], ["a"]),
            ])
        elif "acol" in outputs:
            nodes.extend([
                helper.make_node("Gather", ["c", "si"], ["acol_flat"]),
                helper.make_node("Unsqueeze", ["acol_flat", "axs1"], ["acol"]),
            ])
        else:
            nodes.append(clone_node(node))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.value_info[:]
    model.graph.name = "task285_selected_index_rank_adapter_fold"
    model.producer_name = "task285-selected-index-fold"
    onnx.checker.check_model(model, full_check=True)
    return onnx.shape_inference.infer_shapes(model, strict_mode=True)


def main() -> None:
    model = build_model()
    onnx.save(model, OUTPUT)
    result = profile(OUTPUT)
    params = calculate_params(model)
    assert result is not None and params is not None
    cost = int(result["memory_static"]) + params
    print(f"saved {OUTPUT}")
    print(f"memory={result['memory_static']} params={params} cost={cost}")
    assert cost == EXPECTED_COST


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Build and run GREEN tests**

```bash
uv run python candidates/task285/build_selected_index_fold.py
PYTHONPATH=. uv run pytest -q candidates/task285/scratch/tests/test_selected_index_fold.py
```

Expected: `memory=15415 params=233 cost=15648`; both tests PASS.

- [ ] **Step 5: Gate and adopt the second stage**

```bash
uv run ng gate candidates/task285/selected_index_fold_candidate.onnx --task 285
uv run ng adopt candidates/task285/selected_index_fold_candidate.onnx --task 285 \
  --note "replace shared selected-index rank adapter with direct compact gathers"
uv run ng score 285
```

Expected: gate/adopt PASS 265/265, fail 0; final score reports cost 15,648.

- [ ] **Step 6: Commit the isolated second stage**

```bash
git add -f candidates/task285/build_selected_index_fold.py \
  candidates/task285/selected_index_fold_candidate.onnx \
  candidates/task285/scratch/tests/test_selected_index_fold.py
git commit -m "optimize task285 selected pivot indices" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Exact source synchronization

**Files:**
- Modify: `src/custom/task285.py`
- Modify: `candidates/source_sync_285_295/test_task285_295_source_sync.py`
- Modify: `state/tasks/task285.md` through the two completed adoptions.

**Interfaces:**
- Consumes: deployed final cost-15,648 graph.
- Produces: byte-identical self-contained Python source and updated fixed-point tests.

- [ ] **Step 1: Regenerate exact source**

```bash
uv run python tools/live_to_exact_source.py 285 --write-src
```

Expected: `src/custom/task285.py` is rewritten from the deployed artifact.

- [ ] **Step 2: Update the source-sync expected cost**

Change only the task285 tuple:

```python
CASES = ((285, 15648, 265), (295, 343, 268))
```

- [ ] **Step 3: Verify byte identity and focused behavior**

```bash
PYTHONPATH=. uv run pytest -q \
  candidates/task285/scratch/tests/test_target_sentinel_fold.py \
  candidates/task285/scratch/tests/test_selected_index_fold.py \
  candidates/task285/scratch/tests/test_connectivity_fold_candidate.py \
  candidates/task285/scratch/tests/test_sparse_renderer_candidate.py \
  candidates/source_sync_285_295/test_task285_295_source_sync.py
```

Expected: all tests PASS; source-sync evaluates task285 at cost 15,648 and 265/265.

- [ ] **Step 4: Verify candidate/deployment/source SHA equality**

```bash
uv run python - <<'PY'
import hashlib
from pathlib import Path
from src.custom import task285
from neurogolf.scoring import load_task

candidate = Path("candidates/task285/selected_index_fold_candidate.onnx").read_bytes()
deployed = Path("submission/overfit_nets/task285.onnx").read_bytes()
rebuilt = task285.build(load_task(285)).SerializeToString()
assert candidate == deployed == rebuilt
print(hashlib.sha256(deployed).hexdigest())
PY
```

Expected: one SHA is printed and no assertion fails.

- [ ] **Step 5: Commit source ownership and task ledger**

```bash
git add src/custom/task285.py \
  candidates/source_sync_285_295/test_task285_295_source_sync.py \
  state/tasks/task285.md
git commit -m "adopt task285 residual exact folds" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

Do not stage shared `STATE.md`, `levers.yaml`, `insights.yaml`, or another
session's manifest changes in this commit.

---

### Task 5: Reusable scan and residual large-lane decision

**Files:**
- Create: `candidates/task285/rescan_redundant_target_sentinel.py`
- Modify: `candidates/task285/DISCOVERY.md`
- Modify: `state/insights.yaml`
- Modify: `state/levers.yaml`
- Modify: `state/STATE.md`

**Interfaces:**
- Consumes: final cost 15,648 and the old/final structural delta.
- Produces: a 400-graph structural candidate list, reusable insight, four-field scoped decision, and current handoff.

- [ ] **Step 1: Add a structural scan**

Create this scanner. It reports a candidate when a `ScatterElements` update
comes from `Min(raw, Add(Mul(Gather(label, index), scalar), scalar))`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib

import onnx

ROOT = pathlib.Path(__file__).resolve().parents[2]
NETS = ROOT / "submission/overfit_nets"


def scan_model(path: pathlib.Path) -> list[dict[str, object]]:
    model = onnx.load(path)
    producers = {output: node for node in model.graph.node for output in node.output}
    hits = []
    for scatter in model.graph.node:
        if scatter.op_type != "ScatterElements" or len(scatter.input) < 3:
            continue
        update = producers.get(scatter.input[2])
        if update is None or update.op_type != "Min":
            continue
        for cap_name in update.input:
            add = producers.get(cap_name)
            if add is None or add.op_type != "Add":
                continue
            for mul_name in add.input:
                mul = producers.get(mul_name)
                if mul is None or mul.op_type != "Mul":
                    continue
                for gather_name in mul.input:
                    gather = producers.get(gather_name)
                    if gather is not None and gather.op_type == "Gather":
                        hits.append({
                            "task": int(path.stem[-3:]),
                            "scatter": scatter.output[0],
                            "update": update.output[0],
                            "target_gather": gather.output[0],
                        })
    return hits


def main() -> None:
    paths = sorted(NETS.glob("task*.onnx"))
    candidates = []
    errors = []
    for path in paths:
        try:
            candidates.extend(scan_model(path))
        except Exception as error:
            errors.append({"path": str(path), "error": str(error)})
    print(json.dumps({"scanned": len(paths), "errors": errors,
                      "candidates": candidates}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the 400-graph scan**

```bash
uv run python candidates/task285/rescan_redundant_target_sentinel.py
```

Expected after final adoption: `scanned` is 400, `errors` is empty, and
`candidates` is empty. The pre-adoption self-check found task285 as the sole
board hit.

- [ ] **Step 3: Reprice the large lane**

Run:

```bash
uv run python - <<'PY'
import math
from pathlib import Path
from neurogolf.scans.public_autopsy import profile

current = 15648
threshold = math.floor(current / math.exp(0.1))
control = profile(Path("candidates/task285/bounded_free_output_renderer.onnx"))
assert control is not None
control_cost = int(control["memory_static"]) + int(control["initializer_elems"])
print({"current": current, "next_plus_point_one": threshold,
       "rejected_control": control_cost, "extra_saving_needed": control_cost - threshold})
assert threshold == 14158
assert control_cost == 14415
assert control_cost - threshold == 257
PY
```

Expected: the existing control is 257 above the new +0.1 threshold before its
known margin and 2.503682-second runtime failures. Do not rebuild or rotate it.

- [ ] **Step 4: Update durable records**

Append this measured handoff to `candidates/task285/DISCOVERY.md`:

```markdown
## 2026-07-15 residual sparse exact folds — ADOPTED

- The target-sentinel chain was redundant on all 265 stored examples: its two
  negative target lookups already had invalid `-1` updates. Removing four 81B
  tensors plus scalar `i89` changed 15982 -> 15657.
- Replacing `si2[3,1]` with direct `Gather(t,si)`/`Gather(c,si)` followed by
  compact rank restoration changed 15657 -> 15648. Both stages passed official
  gate/adopt at 265/265, fail0; final memory/params are 15415/233.
- The post-adopt 400-graph target-sentinel scan found no remaining structural
  hit. The next +0.1 threshold is 14158. The old 14415 FREE-output control is
  257 above that threshold and retains its 1078-vs-900 sign failure and
  2.503682-second first inference, so it was not retried.
```

Register this insight in `state/insights.yaml`:

```yaml
- id: invalid_sparse_update_subsumes_target_sentinel
  title: Remove a target-bound sentinel when the sparse invalid-update gate already emits the same sentinel
  status: active
  source_tasks: [285]
  rationale: "Task285's sparse renderer clamped invalid pivot/member updates to -1 before a second Gather/Mul/Add/Min chain clamped off-grid targets to -1. Across all 265 stored examples, the second chain never changed an update; its two negative target reads already had invalid -1 values. Deleting four 81B tensors and scalar i89 changed cost15982->15657, bundled265/265."
  applies_when:
    all_ops: [Gather, Mul, Add, Min, ScatterElements]
    any_tags: [sparse_scatter, invalid_update_sentinel, target_bound_sentinel]
    notes: "Prove raw sparse updates already equal the sentinel whenever the target lookup is negative; otherwise this deletion can write into padded cells."
  reject_when:
    any_tags: [valid_update_can_target_padding, different_invalid_and_padding_sentinels]
  expected:
    gain_type: memory_and_parameter_tail
    risk: "low only after exhaustive authoritative-corpus comparison; fresh remains diagnostic in overfit mode"
    verification: "compare raw/final updates, checker+strict inference, mandatory gate/adopt, then exact-source SHA sync"
  rescan_candidates: []
  rescan_2026_07_15:
    scanned: "All 400 deployed graphs after task285 adoption; the exact structural fingerprint had no remaining hit. Pre-adopt self-check found task285 as the sole hit."
    direct_candidates: []
  transformer: candidates/task285/build_target_sentinel_fold.py
```

Replace the existing task285 paragraph in `state/STATE.md` with the final
cost/SHA/gate facts rather than appending a second task285 section. Add exactly
one four-field `runtime-timeout-spend` ledger entry:

```yaml
- date: 2026-07-15
  ran: "task285 post-microfold static reprice: final cost15648; next +0.1 threshold14158; existing bounded FREE-output control cost14415; retained prior margin and fresh-process runtime evidence"
  verdict: "This dated control remains non-buildable: it is 257 cost above the new threshold, its sign baseline produced 1078 rather than 900 positives, and first inference was 2.503682s. The two exact sparse microfolds were adopted separately; this is not a task floor, and earlier task285 floor claims were repeatedly falsified."
  reopen: "A different margin-correct colour basis and bounded contraction with static cost<=14158, strict wrong-class negative margin, FLOAT-only TopK, and first inference<1s; or another exact sparse fold that is independently strictly cheaper."
```

- [ ] **Step 5: Commit source-controlled scan and discovery**

```bash
git add -f candidates/task285/rescan_redundant_target_sentinel.py
git add candidates/task285/DISCOVERY.md
git commit -m "docs: record task285 residual optimization" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

Preserve shared-state edits in the working tree if they contain concurrent
changes that cannot be staged independently.

---

### Task 6: Final verification

**Files:**
- Verify all task285 files from Tasks 1-5.

**Interfaces:**
- Consumes: final deployment, source, tests, YAML, and rescan.
- Produces: evidence-backed completion with no pack/submit.

- [ ] **Step 1: Run isolated final score and focused suite**

```bash
uv run ng score 285
PYTHONPATH=. uv run pytest -q \
  candidates/task285/scratch/tests/test_target_sentinel_fold.py \
  candidates/task285/scratch/tests/test_selected_index_fold.py \
  candidates/task285/scratch/tests/test_connectivity_fold_candidate.py \
  candidates/task285/scratch/tests/test_sparse_renderer_candidate.py \
  candidates/task285/scratch/tests/test_bounded_free_output_renderer.py \
  candidates/task285/scratch/tests/test_runtime_candidate.py \
  candidates/task285/scratch/tests/test_affine_reflection.py \
  candidates/source_sync_285_295/test_task285_295_source_sync.py
```

Expected: score cost 15,648, pass 265, fail 0; all focused tests PASS.

- [ ] **Step 2: Validate YAML, scan, SHA, and whitespace**

```bash
uv run python - <<'PY'
import yaml
for path in ("state/insights.yaml", "state/levers.yaml"):
    with open(path) as handle:
        yaml.safe_load(handle)
print("YAML-OK")
PY
uv run python candidates/task285/rescan_redundant_target_sentinel.py
git diff --check
git status --short -- candidates/task285 src/custom/task285.py \
  candidates/source_sync_285_295 state/tasks/task285.md state/STATE.md \
  state/levers.yaml state/insights.yaml state/manifest.json
```

Expected: `YAML-OK`, scan `scanned=400` with no errors, no diff-check output,
and only deliberately preserved shared/concurrent changes remain unstaged.

- [ ] **Step 3: Report the result**

Report both adoption endpoints, final score delta from 15,982, gate results,
final SHA, focused test count, 400-scan result, the corrected next threshold
14,158, the known fresh limitation, and that no pack/submit was run.
