# NeuroGolf 7455 Campaign Inventory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, current-endpoint inventory that ranks all 400 deployed NeuroGolf models by concrete single-carrier and single-parameter-bank point-saving opportunities.

**Architecture:** A pure ONNX analyzer reads `submission/overfit_nets/` and `state/manifest.json`, shape-infers each graph, inventories counted intermediate tensors and parameter banks using the scorer's units, and computes optimistic point deltas. A CLI writes JSON and Markdown reports under `candidates/campaign_7455/`; existing NeuroGolf scanners are then refreshed and their measured hits are merged into the review queue without changing deployed artifacts.

**Tech Stack:** Python 3.12, ONNX 1.21.0, ONNX Runtime 1.26.0, pytest, `uv`, existing `src.neurogolf.scoring` conventions.

## Global Constraints

- Baseline is the current working manifest snapshot 7455.0891; recovery of older score does not count.
- Preserve all unrelated dirty files and concurrent-session work.
- Candidate and report artifacts stay under `candidates/` until a source-owned mechanism is selected.
- Never edit `submission/overfit_nets/` except through `uv run ng adopt`.
- Every adoption must pass bundled fail=0 and be cheaper through `uv run ng gate` then `uv run ng adopt`.
- Keep `onnx==1.21.0` and `onnxruntime==1.26.0` unchanged.
- Protect the repaired task118, task131, and task209 deployed hashes.

---

## File Structure

- Create `tools/campaign_inventory.py`: pure ONNX inventory functions, ranking logic, Markdown rendering, and CLI.
- Create `tests/test_campaign_inventory.py`: toy-model tests for memory units, parameter units, gain math, risk flags, and ordering.
- Generate `candidates/campaign_7455/inventory.json`: machine-readable current-board snapshot; keep gitignored.
- Generate `candidates/campaign_7455/queue.md`: human review queue; keep gitignored.
- Refresh `candidates/worklists/*.json` only by existing `uv run ng scan ...` commands; keep gitignored.

### Task 1: Pure ONNX Cost-Opportunity Analyzer

**Files:**
- Create: `tools/campaign_inventory.py`
- Create: `tests/test_campaign_inventory.py`

**Interfaces:**
- Consumes: `onnx.ModelProto`, one manifest row containing `cost`, `points`, and `sha256`.
- Produces: `analyze_model(task: int, model: onnx.ModelProto, manifest_row: dict) -> dict`.
- Produces: `point_gain(cost: int, saving: int) -> float` using `ln(cost / (cost-saving))`, clamped when saving is not in `1..cost-1`.

- [ ] **Step 1: Write failing tests for scorer-unit accounting**

```python
import math

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tools.campaign_inventory import analyze_model, point_gain


def toy_model():
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 4, 4])
    w = numpy_helper.from_array(np.ones((3, 1, 1, 1), dtype=np.float32), name="W")
    conv = helper.make_node("Conv", ["input", "W"], ["plane"], name="conv")
    reduce = helper.make_node("ReduceSum", ["plane"], ["output"], name="reduce", axes=[1], keepdims=1)
    graph = helper.make_graph([conv, reduce], "toy", [x], [y], [w])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def test_point_gain_uses_neurogolf_log_formula():
    assert point_gain(1000, 100) == pytest.approx(math.log(1000 / 900))


def test_analyzer_counts_intermediate_bytes_but_initializer_elements():
    row = analyze_model(1, toy_model(), {"cost": 1000, "points": 18.0, "sha256": "abc"})
    assert row["largest_tensor"]["name"] == "plane"
    assert row["largest_tensor"]["saving"] == 3 * 4 * 4 * 4
    assert row["largest_param_bank"]["name"] == "W"
    assert row["largest_param_bank"]["saving"] == 3
```

- [ ] **Step 2: Run the focused tests and confirm the missing-module failure**

Run: `uv run pytest tests/test_campaign_inventory.py -q`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'tools.campaign_inventory'`.

- [ ] **Step 3: Implement the analyzer with exact scorer units**

```python
from __future__ import annotations

import math
from typing import Any

import numpy as np
import onnx


def point_gain(cost: int, saving: int) -> float:
    if cost <= 1 or saving <= 0 or saving >= cost:
        return 0.0
    return math.log(cost / (cost - saving))


def _shape_elements(value_info: onnx.ValueInfoProto) -> int | None:
    tensor = value_info.type.tensor_type
    if not tensor.HasField("shape"):
        return None
    dims = [d.dim_value for d in tensor.shape.dim]
    if not dims or any(d <= 0 for d in dims):
        return None
    return math.prod(dims)


def analyze_model(task: int, model: onnx.ModelProto, manifest_row: dict[str, Any]) -> dict[str, Any]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    initializers = {item.name for item in inferred.graph.initializer}
    producers = {out: node.op_type for node in inferred.graph.node for out in node.output if out}
    tensors = []
    for item in inferred.graph.value_info:
        if item.name in initializers or item.name in {"input", "output"}:
            continue
        elements = _shape_elements(item)
        if elements is None:
            continue
        dtype = item.type.tensor_type.elem_type
        saving = elements * np.dtype(onnx.helper.tensor_dtype_to_np_dtype(dtype)).itemsize
        tensors.append({"name": item.name, "saving": saving, "producer": producers.get(item.name, "")})
    banks = [
        {"name": item.name, "saving": math.prod(item.dims)}
        for item in inferred.graph.initializer
        if item.dims and all(d > 0 for d in item.dims)
    ]
    cost = int(manifest_row["cost"])
    tensors.sort(key=lambda row: (-row["saving"], row["name"]))
    banks.sort(key=lambda row: (-row["saving"], row["name"]))
    largest_tensor = tensors[0] if tensors else {"name": "", "saving": 0, "producer": ""}
    largest_bank = banks[0] if banks else {"name": "", "saving": 0}
    return {
        "task": task,
        "cost": cost,
        "points": float(manifest_row["points"]),
        "sha256": str(manifest_row["sha256"]),
        "largest_tensor": {**largest_tensor, "expected_gain": point_gain(cost, largest_tensor["saving"])},
        "largest_param_bank": {**largest_bank, "expected_gain": point_gain(cost, largest_bank["saving"])},
    }
```

- [ ] **Step 4: Run the focused tests**

Run: `uv run pytest tests/test_campaign_inventory.py -q`

Expected: PASS for both tests.

- [ ] **Step 5: Commit the analyzer unit**

```bash
git add tools/campaign_inventory.py tests/test_campaign_inventory.py
git commit -m "feat: inventory neurogolf score opportunities"
```

### Task 2: Risk Flags, Stable Ranking, and Reports

**Files:**
- Modify: `tools/campaign_inventory.py`
- Modify: `tests/test_campaign_inventory.py`

**Interfaces:**
- Consumes: rows from `analyze_model` plus deployed model node signatures.
- Produces: `rank_rows(rows: list[dict]) -> list[dict]` ordered by `optimistic_gain`, then task number.
- Produces: `render_markdown(rows: list[dict], baseline: float) -> str`.
- CLI: `uv run python tools/campaign_inventory.py --manifest state/manifest.json --nets submission/overfit_nets --out candidates/campaign_7455`.

- [ ] **Step 1: Write failing tests for ordering and protected/runtime risk flags**

```python
from tools.campaign_inventory import rank_rows, risk_flags


def test_rank_rows_uses_best_single_removal_gain_then_task_number():
    rows = [
        {"task": 2, "optimistic_gain": 0.4},
        {"task": 3, "optimistic_gain": 0.8},
        {"task": 1, "optimistic_gain": 0.8},
    ]
    assert [row["task"] for row in rank_rows(rows)] == [1, 3, 2]


def test_risk_flags_protect_public_zero_repairs_and_large_einsums():
    model = toy_model()
    model.graph.node.append(helper.make_node("Einsum", ["plane"] * 13, ["extra"], equation="ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab->ab"))
    assert risk_flags(118, model) == ["protected-public-zero-repair", "runtime-heavy-einsum"]
```

- [ ] **Step 2: Run the focused tests and confirm undefined interfaces**

Run: `uv run pytest tests/test_campaign_inventory.py -q`

Expected: FAIL importing `rank_rows` and `risk_flags`.

- [ ] **Step 3: Add risk flags, ranking, JSON/Markdown rendering, and argparse CLI**

```python
PROTECTED_TASKS = {118, 131, 209}


def risk_flags(task: int, model: onnx.ModelProto) -> list[str]:
    flags = []
    if task in PROTECTED_TASKS:
        flags.append("protected-public-zero-repair")
    if any(node.op_type == "Einsum" and len(node.input) >= 12 for node in model.graph.node):
        flags.append("runtime-heavy-einsum")
    return flags


def rank_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: (-row["optimistic_gain"], row["task"]))
```

The CLI must load all 400 manifest rows, verify each deployed file SHA-256 against its manifest row, abort on any mismatch, set `optimistic_gain` to the larger of the largest tensor and parameter-bank gains, attach `risk_flags`, and write:

```json
{
  "baseline": 7455.0891,
  "task_count": 400,
  "rows": []
}
```

Here `baseline` is computed as the sum of the loaded manifest points; 7455.0891 is the expected value for the approved snapshot. The Markdown table columns must be `rank`, `task`, `cost`, `points`, `optimistic_gain`, `largest_tensor`, `largest_param_bank`, and `risk_flags`.

- [ ] **Step 4: Run unit tests and CLI help**

Run: `uv run pytest tests/test_campaign_inventory.py -q && uv run python tools/campaign_inventory.py --help`

Expected: all tests PASS; help lists `--manifest`, `--nets`, and `--out`.

- [ ] **Step 5: Commit reporting support**

```bash
git add tools/campaign_inventory.py tests/test_campaign_inventory.py
git commit -m "feat: rank 7455 campaign candidates"
```

### Task 3: Generate the Current 400-Task Queue

**Files:**
- Generate: `candidates/campaign_7455/inventory.json`
- Generate: `candidates/campaign_7455/queue.md`
- Generate: `candidates/worklists/mask_dominance.json`
- Generate: `candidates/worklists/kernel_collapse.json`
- Generate: `candidates/worklists/fold.json`
- Generate: `candidates/worklists/dtype_overpay.json`
- Generate: `candidates/worklists/public_autopsy.json`
- Generate: `candidates/worklists/qlinear_recast.json`

**Interfaces:**
- Consumes: the committed inventory CLI and existing registered NeuroGolf scanners.
- Produces: an evidence packet for selecting task-specific deep dives; no deployed or state file changes.

- [ ] **Step 1: Snapshot and verify the current endpoint**

Run: `uv run ng status`

Expected: `nets: 400/400` and manifest total at least 7455.0891. If the total changed, record the exact new baseline in the generated inventory rather than silently using 7455.0891.

- [ ] **Step 2: Generate the static opportunity inventory**

Run: `uv run python tools/campaign_inventory.py --manifest state/manifest.json --nets submission/overfit_nets --out candidates/campaign_7455`

Expected: `400 models inventoried; 0 SHA mismatches` and both report files exist.

- [ ] **Step 3: Refresh all low-cost registered scans**

```bash
uv run ng scan mask_dominance
uv run ng scan kernel_collapse
uv run ng scan fold
uv run ng scan dtype_overpay
uv run ng scan public_autopsy
uv run ng scan qlinear_recast
```

Expected: each command writes its named JSON worklist without modifying `state/manifest.json` or `submission/overfit_nets/`.

- [ ] **Step 4: Run the expensive canvas scan only on its current top candidates**

Run: `uv run ng scan canvas_crop_shrink --tasks 381 378 397 390 387 382 379`

Expected: a refreshed `candidates/worklists/canvas_crop_shrink.json`; failures are recorded per task rather than aborting the campaign.

- [ ] **Step 5: Review the top 30 rows against task ledgers and reopen triggers**

Run: `sed -n '1,80p' candidates/campaign_7455/queue.md && uv run ng queue`

Expected: select three unblocked deep-dive targets with a concrete removable carrier/bank, predicted gain, and no repeated ledger-negative family. Create a separate task-specific design/plan for each selected target before editing its builder.

- [ ] **Step 6: Verify the inventory implementation and preserve the scratch reports**

Run: `uv run pytest tests/test_campaign_inventory.py tests/test_scans_registry.py -q`

Expected: all tests PASS. Do not commit `candidates/` outputs.
