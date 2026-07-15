# Task333 Cost-1586 Lowering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, verify, and adopt an exact task333 candidate whose cost is 1586 instead of 1786.

**Architecture:** Preserve the incumbent's four directional MaxPools and two Min planes. Fold
the gated horizontal/vertical merge into two nested Where nodes, then broadcast one BitwiseAnd
with masks `[255, 27]` to produce both decoder channels without a separate bit plane or Concat.

**Tech Stack:** Python 3.13 via `uv`, ONNX 1.21.0, ONNX Runtime 1.26.0, NumPy, pytest, NeuroGolf `ng` CLI.

## Global Constraints

- Work only on task333.
- Do not build the rejected uint8 CumSum graph or the 2000-byte fp16 CumSum core.
- Do not reuse cached public task333 graphs; their measured minimum cost is 2401.
- Do not change the ONNX 1.21.0 or ONNX Runtime 1.26.0 pins.
- Candidate cost must be at most 1616.
- Success requires bundled fail=0, fresh1500 candidate fail=0, and fresh1500 divergence=0.
- Candidate artifacts stay under `candidates/task333/`; deployment changes only through `ng adopt`.

---

### Task 1: Exact Fold Builder

**Files:**
- Create: `candidates/task333/test_fused_select_broadcast_hull.py`
- Create: `candidates/task333/build_fused_select_broadcast_hull.py`
- Create: `candidates/task333/fused_select_broadcast_hull.onnx` (generated)

**Interfaces:**
- Consumes: deployed `submission/overfit_nets/task333.onnx` at cost 1786.
- Produces: `build(task=None) -> onnx.ModelProto` and a generated cost-1586 candidate.

- [ ] **Step 1: Write the failing structural and price test**

```python
import math

import numpy as np
import onnx
from onnx import numpy_helper

from build_fused_select_broadcast_hull import build


def tensor_bytes(vi: onnx.ValueInfoProto) -> int:
    tensor_type = vi.type.tensor_type
    shape = [dim.dim_value for dim in tensor_type.shape.dim]
    dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
    return math.prod(shape) * np.dtype(dtype).itemsize


def test_build_fuses_select_and_hull_feature_planes():
    model = onnx.shape_inference.infer_shapes(build(), strict_mode=True)
    onnx.checker.check_model(model, full_check=True)

    producers = {out: node for node in model.graph.node for out in node.output}
    assert "Hf" not in producers
    assert "Vf" not in producers
    assert "merged_bits" not in producers
    assert producers["hmerge"].op_type == "Where"
    assert list(producers["hmerge"].input) == ["rbox", "Hmin", "bar"]
    assert producers["merged"].op_type == "Where"
    assert list(producers["merged"].input) == ["cbox", "Vmin", "hmerge"]
    assert producers["hull_features"].op_type == "BitwiseAnd"
    assert list(producers["hull_features"].input) == ["merged", "hull_masks"]

    initializers = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    assert "z" not in initializers
    np.testing.assert_array_equal(
        initializers["hull_masks"],
        np.array([255, 27], dtype=np.uint8).reshape(1, 2, 1, 1),
    )

    memory = sum(tensor_bytes(vi) for vi in model.graph.value_info)
    params = sum(array.size for array in initializers.values())
    assert memory == 1540
    assert params == 46
    assert memory + params == 1586
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
uv run pytest -q candidates/task333/test_fused_select_broadcast_hull.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'build_fused_select_broadcast_hull'`.

- [ ] **Step 3: Implement the minimal builder**

```python
"""Build task333 by folding two select and two hull-feature planes."""

import copy
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "submission" / "overfit_nets" / "task333.onnx"
OUTPUT = Path(__file__).with_name("fused_select_broadcast_hull.onnx")


def build(task=None) -> onnx.ModelProto:
    model = onnx.load(SOURCE)
    graph = model.graph
    producers = {out: node for node in graph.node for out in node.output}
    expected = {
        "Hf": "Where",
        "Vf": "Where",
        "merged": "Max",
        "merged_bits": "BitwiseAnd",
        "hull_features": "Concat",
        "output": "QLinearConv",
    }
    for output, op_type in expected.items():
        actual = producers.get(output)
        if actual is None or actual.op_type != op_type:
            raise ValueError(f"unexpected incumbent producer for {output}: {actual}")

    qlinear = copy.deepcopy(producers["output"])
    replacement_outputs = {"Hf", "Vf", "merged", "merged_bits", "hull_features", "output"}
    kept_nodes = [
        node for node in graph.node
        if not replacement_outputs.intersection(node.output)
    ]
    kept_nodes.extend(
        [
            helper.make_node("Where", ["rbox", "Hmin", "bar"], ["hmerge"]),
            helper.make_node("Where", ["cbox", "Vmin", "hmerge"], ["merged"]),
            helper.make_node("BitwiseAnd", ["merged", "hull_masks"], ["hull_features"]),
            qlinear,
        ]
    )
    del graph.node[:]
    graph.node.extend(kept_nodes)

    kept_initializers = [
        init for init in graph.initializer
        if init.name not in {"z", "bitmask27", "hull_masks"}
    ]
    kept_initializers.append(
        numpy_helper.from_array(
            np.array([255, 27], dtype=np.uint8).reshape(1, 2, 1, 1),
            "hull_masks",
        )
    )
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)
    del graph.value_info[:]

    onnx.checker.check_model(model)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(inferred, full_check=True)
    return inferred


if __name__ == "__main__":
    onnx.save(build(), OUTPUT)
    print(OUTPUT)
```

- [ ] **Step 4: Run the structural test and verify GREEN**

Run:

```bash
uv run pytest -q candidates/task333/test_fused_select_broadcast_hull.py
```

Expected: `1 passed`.

- [ ] **Step 5: Generate the candidate and run an isolated ORT smoke check**

Run:

```bash
uv run python candidates/task333/build_fused_select_broadcast_hull.py
uv run python - <<'PY'
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession(
    "candidates/task333/fused_select_broadcast_hull.onnx",
    providers=["CPUExecutionProvider"],
)
output = session.run(None, {"input": np.zeros((1, 10, 30, 30), np.float32)})[0]
assert output.shape == (1, 10, 30, 30)
print(output.shape, output.dtype)
PY
```

Expected: `(1, 10, 30, 30) uint8`.

---

### Task 2: Bundled and Fresh Exactness Gates

**Files:**
- Create: `candidates/task333/verify_fresh_fused_select_broadcast_hull.py`
- Modify after adoption: `src/custom/task333.py`
- Modify after adoption: `candidates/task333/DISCOVERY.md`
- Modified by `ng adopt`: `submission/overfit_nets/task333.onnx`, `state/manifest.json`, `state/tasks/task333.md`

**Interfaces:**
- Consumes: generated `candidates/task333/fused_select_broadcast_hull.onnx`.
- Produces: bundled gate result and deterministic process exit for fresh1500 exactness.

- [ ] **Step 1: Run the bundled gate before the expensive fresh check**

Run:

```bash
uv run ng gate candidates/task333/fused_select_broadcast_hull.onnx --task 333
```

Expected: `PASS`, bundled fail=0, memory=1540, params=46, cost=1586.

- [ ] **Step 2: Add the fresh1500 verifier**

```python
"""Verify task333 incumbent/candidate/oracle agreement on fresh examples."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neurogolf.paths import STATE
from neurogolf.scoring import convert_to_numpy


INCUMBENT = ROOT / "submission" / "overfit_nets" / "task333.onnx"
CANDIDATE = Path(__file__).with_name("fused_select_broadcast_hull.onnx")


def make_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def predict(session: ort.InferenceSession, input_array: np.ndarray) -> np.ndarray:
    return (session.run(["output"], {"input": input_array})[0] > 0).astype(np.float32)


def main(target: int = 1500) -> None:
    mapping = json.load(open(STATE / "arc_mapping.json"))
    arc = mapping["333"]["arc_id"]
    arcgen = str(ROOT / "arc-gen")
    if arcgen not in sys.path:
        sys.path.append(arcgen)
    generator = importlib.import_module(f"tasks.task_{arc}")

    incumbent = make_session(INCUMBENT)
    candidate = make_session(CANDIDATE)
    runs = incumbent_fail = candidate_fail = divergence = 0
    while runs < target:
        try:
            example = generator.generate()
        except Exception:
            continue
        arrays = convert_to_numpy(example)
        if arrays is None:
            continue
        incumbent_output = predict(incumbent, arrays["input"])
        candidate_output = predict(candidate, arrays["input"])
        expected = arrays["output"]
        runs += 1
        incumbent_fail += not np.array_equal(incumbent_output, expected)
        candidate_fail += not np.array_equal(candidate_output, expected)
        divergence += not np.array_equal(candidate_output, incumbent_output)

    print(
        f"fresh runs={runs} incumbent_fail={incumbent_fail} "
        f"candidate_fail={candidate_fail} divergence={divergence}"
    )
    if candidate_fail or divergence:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run fresh1500 in a fresh process**

Run:

```bash
uv run python candidates/task333/verify_fresh_fused_select_broadcast_hull.py
```

Expected: `fresh runs=1500 incumbent_fail=0 candidate_fail=0 divergence=0` and exit 0.

- [ ] **Step 4: Adopt through the mandatory gate**

Run:

```bash
uv run ng adopt candidates/task333/fused_select_broadcast_hull.onnx --task 333 --note "nested directional select + broadcast bitwise hull feature fold"
```

Expected: re-gate passes and task333 changes from cost 1786 to cost 1586.

- [ ] **Step 5: Regenerate source ownership and verify byte identity**

Run:

```bash
uv run python tools/live_to_exact_source.py 333 --write-src
uv run python - <<'PY'
import hashlib
import onnx

from src.custom.task333 import build

built = build(None).SerializeToString()
deployed = onnx.load("submission/overfit_nets/task333.onnx").SerializeToString()
print(hashlib.sha256(built).hexdigest())
print(hashlib.sha256(deployed).hexdigest())
assert built == deployed
PY
```

Expected: both SHA-256 lines are identical and the assertion passes.

- [ ] **Step 6: Record the task-local mechanism in the discovery handoff**

Append this section to `candidates/task333/DISCOVERY.md`:

```markdown
## 8. Cost-1586 nested-select and broadcast-feature fold

- Replaced `Hf`, `Vf`, and `Max(bar,Hf,Vf)` by nested
  `Where(rbox,Hmin,bar)` / `Where(cbox,Vmin,hmerge)`, removing one counted 100B plane.
- Replaced scalar `BitwiseAnd` plus `Concat` by broadcast masks `[255,27]`, directly producing
  `[merged, merged&27]` and removing another counted 100B plane.
- Final price: memory1540 + params46 = cost1586, from incumbent1786.
- Verification: bundled fail=0 and fresh1500 candidate fail=0/divergence=0.
- Classification: task-local algebraic fold for this scoped session; no cross-task rescan was
  authorized.
```

- [ ] **Step 7: Run final isolated verification**

Run:

```bash
uv run ng score 333
uv run ng gate candidates/task333/fused_select_broadcast_hull.onnx --task 333
uv run python candidates/task333/verify_fresh_fused_select_broadcast_hull.py
```

Expected: deployed score reports cost1586/fail0; the candidate gate rejects only because it ties
the now-deployed candidate while still reporting bundled fail=0/cost1586; fresh1500 remains zero
for candidate_fail and divergence.

- [ ] **Step 8: Commit only task333 source and state files**

```bash
git add src/custom/task333.py submission/overfit_nets/task333.onnx state/manifest.json state/tasks/task333.md
git commit -m "optimize task333 select and decoder planes" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

Expected: commit succeeds without staging unrelated concurrent-session changes.
