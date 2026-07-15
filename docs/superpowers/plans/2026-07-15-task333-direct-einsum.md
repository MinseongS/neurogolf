# Task333 Direct-Einsum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, verify, and adopt an exact task333 single-Einsum candidate at cost 1295 instead of 1586.

**Architecture:** Read the FREE one-hot input three times inside one 26-operand fp32 `Einsum`:
one source pixel plus an ordered pair of green pixels. Reused crop, relation, mode, and channel
factors express preserve and four strict directional rays; the only node writes directly to the
FREE output, so memory is zero and all 1295 cost comes from initializer elements.

**Tech Stack:** Python 3.13 via `uv`, NumPy, pytest, ONNX 1.21.0, ONNX Runtime 1.26.0, NeuroGolf `ng` CLI.

## Global Constraints

- Work only on task333.
- Incumbent cost is 1586; the required `+0.1` ceiling is cost 1435; target cost is exactly 1295.
- Do not build the rejected uint8 CumSum graph, the cost-2000 fp16 CumSum core, or cached public task333 graphs.
- Do not change the ONNX 1.21.0 or ONNX Runtime 1.26.0 pins.
- Success requires bundled fail=0, fresh1500 candidate fail=0, and fresh1500 divergence=0.
- Candidate artifacts stay under `candidates/task333/`; deployment changes only through `ng adopt`.
- `candidates/` is intentionally gitignored. Do not force-add candidate scripts or ONNX files.
- No pack or submit is authorized.

---

### Task 1: One-Node Builder and Static Price Test

**Files:**
- Create: `candidates/task333/test_direct_einsum.py`
- Create: `candidates/task333/build_direct_einsum.py`
- Generate: `candidates/task333/direct_einsum.onnx`

**Interfaces:**
- Consumes: FREE input tensor `input: float32[1,10,30,30]`.
- Produces: `deterministic_orders() -> list[tuple[str, tuple[int, ...]]]` and
  `build(order_index: int = 0) -> onnx.ModelProto`.
- Produces: FREE output tensor `output: float32[1,10,30,30]` from one `Einsum` node.

- [ ] **Step 1: Write the failing structural and price test**

Create `candidates/task333/test_direct_einsum.py`:

```python
import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

from build_direct_einsum import BASE_TERMS, build, deterministic_orders


def shape(value_info: onnx.ValueInfoProto) -> list[int]:
    return [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]


def test_build_is_one_free_output_einsum_at_cost_1295():
    model = build()
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(inferred, full_check=True)

    assert len(model.graph.node) == 1
    node = model.graph.node[0]
    assert node.op_type == "Einsum"
    assert list(node.output) == ["output"]
    assert len(node.input) == len(BASE_TERMS) == 26
    equation = next(attr.s.decode() for attr in node.attribute if attr.name == "equation")
    assert equation.endswith("->noHW")

    assert model.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert shape(model.graph.input[0]) == [1, 10, 30, 30]
    assert shape(model.graph.output[0]) == [1, 10, 30, 30]
    assert list(model.graph.value_info) == []

    arrays = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    assert sum(array.size for array in arrays.values()) == 1295
    assert arrays["E"].shape == (10, 30)
    assert arrays["A"].shape == (3, 10, 10)
    assert arrays["B"].shape == (4, 10, 10)
    assert arrays["channel"].shape == (2, 10, 10)
    np.testing.assert_array_equal(arrays["green"], np.eye(10, dtype=np.float32)[3])
    np.testing.assert_array_equal(arrays["mode_weight"], [1 / 16, 1, 1, 1, 1])

    orders = deterministic_orders()
    assert len(orders) == 12
    assert orders[0][0] == "semantic"
    for _, order in orders:
        assert sorted(order) == list(range(26))
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
uv run pytest -q candidates/task333/test_direct_einsum.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'build_direct_einsum'`.

- [ ] **Step 3: Implement the complete builder**

Create `candidates/task333/build_direct_einsum.py`:

```python
"""Build task333 as one FREE-output relational Einsum."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


OUTPUT = Path(__file__).with_name("direct_einsum.onnx")

BASE_INPUTS = (
    "input", "E", "E",
    "input", "E", "E", "green",
    "input", "E", "E", "green",
    "A", "row_a", "B", "row_b", "B",
    "A", "col_a", "B", "col_b", "B",
    "channel", "mode_channel", "mode_weight", "E", "E",
)
BASE_TERMS = (
    "ncRS", "rR", "sS",
    "ngUV", "uU", "vV", "g",
    "ndPQ", "pP", "qQ", "d",
    "arh", "ma", "bhu", "mb", "bup",
    "isw", "mi", "jwv", "mj", "jvq",
    "kco", "mk", "m", "hH", "wW",
)


def deterministic_orders() -> list[tuple[str, tuple[int, ...]]]:
    base = tuple(range(26))
    orders = [
        ("semantic", base),
        ("reverse", tuple(reversed(base))),
        ("small_first", (23, 22, 12, 14, 17, 19, 6, 10, 21, 11, 13, 15, 16, 18, 20, 1, 2, 4, 5, 8, 9, 24, 25, 0, 3, 7)),
        ("relations_first", (12, 11, 14, 13, 15, 17, 16, 19, 18, 20, 23, 22, 21, 6, 10, 1, 2, 4, 5, 8, 9, 24, 25, 0, 3, 7)),
        ("channel_first", (23, 22, 21, 12, 11, 14, 13, 15, 17, 16, 19, 18, 20, 6, 10, 1, 2, 4, 5, 8, 9, 24, 25, 0, 3, 7)),
        ("row_first", (12, 11, 14, 13, 15, 1, 4, 8, 24, 17, 16, 19, 18, 20, 2, 5, 9, 25, 23, 22, 21, 6, 10, 0, 3, 7)),
        ("col_first", (17, 16, 19, 18, 20, 2, 5, 9, 25, 12, 11, 14, 13, 15, 1, 4, 8, 24, 23, 22, 21, 6, 10, 0, 3, 7)),
        ("green_first", (6, 10, 4, 5, 8, 9, 3, 7, 12, 11, 14, 13, 15, 17, 16, 19, 18, 20, 23, 22, 21, 1, 2, 24, 25, 0)),
        ("rotate4", base[4:] + base[:4]),
        ("rotate8", base[8:] + base[:8]),
        ("rotate12", base[12:] + base[:12]),
        ("rotate16", base[16:] + base[:16]),
    ]
    for _, order in orders:
        assert sorted(order) == list(base)
    return orders


def one_hot(rows: int, cols: int, indices: list[int]) -> np.ndarray:
    out = np.zeros((rows, cols), dtype=np.float32)
    out[np.arange(rows), np.asarray(indices)] = 1
    return out


def initializers() -> list[onnx.TensorProto]:
    identity = np.eye(10, dtype=np.float32)
    lower = np.triu(np.ones((10, 10), dtype=np.float32), 1)
    reverse = lower.T.copy()
    ones = np.ones((10, 10), dtype=np.float32)

    embed = np.zeros((10, 30), dtype=np.float32)
    embed[:, :10] = identity
    relation_a = np.stack([identity, lower, reverse])
    relation_b = np.stack([identity, ones, lower, reverse])

    row_a = one_hot(5, 3, [0, 0, 0, 1, 2])
    row_b = one_hot(5, 4, [1, 0, 0, 2, 3])
    col_a = one_hot(5, 3, [0, 1, 2, 0, 0])
    col_b = one_hot(5, 4, [1, 2, 3, 0, 0])

    channel = np.zeros((2, 10, 10), dtype=np.float32)
    channel[0] = identity
    for color in (1, 2, 4, 5, 6, 7, 8, 9):
        channel[1, color, color] = 1
        channel[1, color, 0] = -1

    mode_channel = one_hot(5, 2, [0, 1, 1, 1, 1])
    mode_weight = np.array([1 / 16, 1, 1, 1, 1], dtype=np.float32)
    green = np.eye(10, dtype=np.float32)[3]

    values = {
        "E": embed,
        "A": relation_a,
        "B": relation_b,
        "row_a": row_a,
        "row_b": row_b,
        "col_a": col_a,
        "col_b": col_b,
        "mode_weight": mode_weight,
        "channel": channel,
        "mode_channel": mode_channel,
        "green": green,
    }
    assert sum(value.size for value in values.values()) == 1295
    return [numpy_helper.from_array(value, name) for name, value in values.items()]


def build(order_index: int = 0) -> onnx.ModelProto:
    orders = deterministic_orders()
    if not 0 <= order_index < len(orders):
        raise ValueError(f"order_index must be 0..{len(orders) - 1}")
    _, order = orders[order_index]
    equation = ",".join(BASE_TERMS[index] for index in order) + "->noHW"
    inputs = [BASE_INPUTS[index] for index in order]

    node = helper.make_node("Einsum", inputs, ["output"], name="output", equation=equation)
    graph = helper.make_graph(
        [node],
        "task333_direct_einsum",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializers(),
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(inferred, full_check=True)
    return inferred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--order-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    onnx.save(build(args.order_index), args.output)
    print(args.output)
```

- [ ] **Step 4: Run the static test and verify GREEN**

Run:

```bash
uv run pytest -q candidates/task333/test_direct_einsum.py
uv run python candidates/task333/build_direct_einsum.py
```

Expected: `1 passed`, then `candidates/task333/direct_einsum.onnx` is printed.

- [ ] **Step 5: Record the local checkpoint without staging ignored scratch**

Run:

```bash
git status --short candidates/task333
```

Expected: no candidate paths appear because `/candidates/` is intentionally ignored. Do not use
`git add -f`; the independently testable deliverable is the green structural test and generated
cost-1295 graph.

---

### Task 2: Formula Oracle and Bounded ORT Planner Harness

**Files:**
- Modify: `candidates/task333/test_direct_einsum.py`
- Create: `candidates/task333/verify_direct_einsum.py`
- Regenerate: `candidates/task333/direct_einsum.onnx`

**Interfaces:**
- Consumes: `build(order_index)` and `deterministic_orders()` from Task 1.
- Produces: `direct_formula(input_array: np.ndarray) -> np.ndarray`.
- Produces CLI modes `--search`, `--smoke MODEL`, `--bundled`, and `--fresh N`.

- [ ] **Step 1: Append a failing four-direction formula test**

Append to `candidates/task333/test_direct_einsum.py`:

```python
from verify_direct_einsum import direct_formula


def make_four_ray_example() -> tuple[np.ndarray, np.ndarray]:
    value = np.zeros((1, 10, 30, 30), dtype=np.float32)
    value[0, 0, :10, :10] = 1

    def set_color(color: int, row: int, col: int) -> None:
        value[0, 0, row, col] = 0
        value[0, color, row, col] = 1

    for row, col in ((4, 4), (4, 5), (5, 4), (5, 5)):
        set_color(3, row, col)
    for color, row, col in ((1, 4, 1), (2, 5, 8), (4, 1, 4), (5, 8, 5)):
        set_color(color, row, col)

    expected = value.copy()
    for color, cells in (
        (1, ((4, 2), (4, 3))),
        (2, ((5, 7), (5, 6))),
        (4, ((2, 4), (3, 4))),
        (5, ((7, 5), (6, 5))),
    ):
        for row, col in cells:
            expected[0, 0, row, col] = 0
            expected[0, color, row, col] = 1
    return value, expected


def test_direct_formula_renders_all_four_strict_rays():
    value, expected = make_four_ray_example()
    actual = direct_formula(value)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual.sum(axis=1)[:, :10, :10], 1)
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
uv run pytest -q candidates/task333/test_direct_einsum.py::test_direct_formula_renders_all_four_strict_rays
```

Expected: collection fails with `ModuleNotFoundError: No module named 'verify_direct_einsum'`.

- [ ] **Step 3: Implement the planner and exactness verifier**

Create `candidates/task333/verify_direct_einsum.py`:

```python
"""Bounded ORT-order search and exactness checks for task333 direct Einsum."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_direct_einsum import OUTPUT, build, deterministic_orders
from neurogolf.paths import STATE
from neurogolf.scoring import convert_to_numpy, load_task


INCUMBENT = ROOT / "submission/overfit_nets/task333.onnx"


def direct_formula(input_array: np.ndarray) -> np.ndarray:
    source = np.asarray(input_array, dtype=np.float32)
    output = source.copy()
    green_cells = np.argwhere(source[0, 3, :10, :10] > 0)
    assert green_cells.shape == (4, 2)
    for color in (1, 2, 4, 5, 6, 7, 8, 9):
        for row, col in np.argwhere(source[0, color, :10, :10] > 0):
            for green1_row, green1_col in green_cells:
                for green2_row, green2_col in green_cells:
                    for target_row in range(10):
                        for target_col in range(10):
                            horizontal_left = (
                                row == target_row == green1_row == green2_row
                                and col < target_col < green1_col < green2_col
                            )
                            horizontal_right = (
                                row == target_row == green1_row == green2_row
                                and col > target_col > green1_col > green2_col
                            )
                            vertical_up = (
                                col == target_col == green1_col == green2_col
                                and row < target_row < green1_row < green2_row
                            )
                            vertical_down = (
                                col == target_col == green1_col == green2_col
                                and row > target_row > green1_row > green2_row
                            )
                            if horizontal_left or horizontal_right or vertical_up or vertical_down:
                                output[0, 0, target_row, target_col] -= 1
                                output[0, color, target_row, target_col] += 1
    return output


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


def examples() -> list[dict]:
    task = load_task(333)
    return task.get("train", []) + task.get("test", []) + task.get("arc-gen", [])


def smoke(path: Path) -> dict[str, object]:
    started = time.perf_counter()
    session = make_session(path)
    arrays = convert_to_numpy(examples()[0])
    assert arrays is not None
    actual = predict(session, arrays["input"])
    equal = bool(np.array_equal(actual, arrays["output"]))
    result = {"seconds": time.perf_counter() - started, "equal": equal}
    print(json.dumps(result), flush=True)
    if not equal:
        raise SystemExit(2)
    return result


def search(timeout: float = 20.0) -> None:
    with tempfile.TemporaryDirectory() as directory:
        directory_path = Path(directory)
        for index, (name, _) in enumerate(deterministic_orders()):
            path = directory_path / f"order_{index:02d}.onnx"
            onnx.save(build(index), path)
            try:
                process = subprocess.run(
                    [sys.executable, __file__, "--smoke", str(path)],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                print(json.dumps({"index": index, "name": name, "status": "timeout"}), flush=True)
                continue
            if process.returncode == 2:
                raise SystemExit(f"semantic mismatch for order {index} {name}: {process.stdout}")
            if process.returncode:
                print(json.dumps({"index": index, "name": name, "status": "error", "stderr": process.stderr[-500:]}), flush=True)
                continue
            result = json.loads(process.stdout.strip().splitlines()[-1])
            onnx.save(build(index), OUTPUT)
            print(json.dumps({"index": index, "name": name, "status": "selected", **result}), flush=True)
            return
    raise SystemExit("no ORT-valid order among 12 bounded candidates")


def bundled() -> None:
    session = make_session(OUTPUT)
    candidate_fail = formula_fail = 0
    all_examples = examples()
    for example in all_examples:
        arrays = convert_to_numpy(example)
        if arrays is None:
            continue
        candidate_fail += not np.array_equal(predict(session, arrays["input"]), arrays["output"])
        formula_fail += not np.array_equal(direct_formula(arrays["input"]), arrays["output"])
    print(f"bundled runs={len(all_examples)} candidate_fail={candidate_fail} formula_fail={formula_fail}")
    if candidate_fail or formula_fail:
        raise SystemExit(1)


def fresh(target: int) -> None:
    mapping = json.load(open(STATE / "arc_mapping.json"))
    arc = mapping["333"]["arc_id"]
    arcgen = str(ROOT / "arc-gen")
    if arcgen not in sys.path:
        sys.path.append(arcgen)
    generator = importlib.import_module(f"tasks.task_{arc}")

    incumbent = make_session(INCUMBENT)
    candidate = make_session(OUTPUT)
    runs = incumbent_fail = candidate_fail = formula_fail = divergence = 0
    while runs < target:
        try:
            example = generator.generate()
        except Exception:
            continue
        arrays = convert_to_numpy(example)
        if arrays is None:
            continue
        expected = arrays["output"]
        incumbent_output = predict(incumbent, arrays["input"])
        candidate_output = predict(candidate, arrays["input"])
        formula_output = direct_formula(arrays["input"])
        runs += 1
        incumbent_fail += not np.array_equal(incumbent_output, expected)
        candidate_fail += not np.array_equal(candidate_output, expected)
        formula_fail += not np.array_equal(formula_output, expected)
        divergence += not np.array_equal(candidate_output, incumbent_output)
    print(
        f"fresh runs={runs} incumbent_fail={incumbent_fail} candidate_fail={candidate_fail} "
        f"formula_fail={formula_fail} divergence={divergence}"
    )
    if candidate_fail or formula_fail or divergence:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--search", action="store_true")
    mode.add_argument("--smoke", type=Path)
    mode.add_argument("--bundled", action="store_true")
    mode.add_argument("--fresh", type=int)
    args = parser.parse_args()
    if args.search:
        search()
    elif args.smoke is not None:
        smoke(args.smoke)
    elif args.bundled:
        bundled()
    else:
        fresh(args.fresh)
```

- [ ] **Step 4: Run the formula test and verify GREEN**

Run:

```bash
uv run pytest -q candidates/task333/test_direct_einsum.py
```

Expected: `2 passed`.

- [ ] **Step 5: Run the bounded planner search**

Run:

```bash
uv run python candidates/task333/verify_direct_einsum.py --search
```

Expected success: one JSON line has `"status": "selected"`, `"equal": true`, and
`"seconds" < 20.0`; the selected graph is saved as `direct_einsum.onnx`.

If all 12 orders time out or error, stop implementation. Under the existing
`free-output-einsum-regime-crack` lever in `state/levers.yaml`, add exactly this four-field entry
and set no global floor claim:

```yaml
      - date: 2026-07-15
        ran: "task333 direct relational single-Einsum: exact 1295-element factor build plus 12 deterministic ORT operand orders, each bounded to 20 seconds for session creation and one bundled inference"
        verdict: "No runnable order found by this bounded ORT 1.26 search on 2026-07-15; the NumPy relation formula remains exact on bundled265+fresh1500, but this concrete 26-operand lowering did not produce within the runtime bound. This is an ORT planner result, not a semantic or task floor."
        reopen: "A new operand contraction order with measured <20s first inference, a staged lowering whose counted cost remains <=1435, or an ORT planner/runtime change under a fully revalidated pin"
```

Do not gate, fresh-check, or adopt after this stop.

---

### Task 3: Bundled Gate and Fresh1500 Exactness

**Files:**
- Consume: `candidates/task333/direct_einsum.onnx`
- Consume: `candidates/task333/verify_direct_einsum.py`

**Interfaces:**
- Consumes the selected runnable graph from Task 2.
- Produces official bundled pricing/failure evidence and fresh candidate/incumbent/oracle evidence.

- [ ] **Step 1: Recheck the direct formula and emitted graph on all bundled examples**

Run:

```bash
uv run python candidates/task333/verify_direct_einsum.py --bundled
```

Expected: `bundled runs=265 candidate_fail=0 formula_fail=0`.

- [ ] **Step 2: Run the mandatory bundled gate before fresh generation**

Run:

```bash
uv run ng gate candidates/task333/direct_einsum.onnx --task 333
```

Expected: `PASS`, bundled fail=0, memory=0, params=1295, cost=1295. If cost exceeds 1435 or
fail is nonzero, stop without fresh verification or adoption.

- [ ] **Step 3: Run fresh1500 in a fresh process**

Run:

```bash
uv run python candidates/task333/verify_direct_einsum.py --fresh 1500
```

Expected:

```text
fresh runs=1500 incumbent_fail=0 candidate_fail=0 formula_fail=0 divergence=0
```

Any nonzero count is a hard stop; retain the cost-1586 incumbent.

- [ ] **Step 4: Re-run focused tests after runtime evidence**

Run:

```bash
uv run pytest -q candidates/task333/test_direct_einsum.py
```

Expected: `2 passed`.

---

### Task 4: Mandatory Adoption, Exact Source, and Task333 Handoff

**Files:**
- Modify via `ng adopt`: `submission/overfit_nets/task333.onnx` (ignored deployment artifact)
- Modify via `ng adopt`: `state/manifest.json`
- Modify via `ng adopt`: `state/tasks/task333.md`
- Modify via exact-source generator: `src/custom/task333.py`
- Modify: `candidates/task333/DISCOVERY.md` (ignored task-local handoff)
- Modify task333 section only: `state/STATE.md`

**Interfaces:**
- Consumes: cost-1295 candidate with bundled fail=0 and fresh1500 divergence=0.
- Produces: adopted task333, byte-identical exact source, and durable task333 records.

- [ ] **Step 1: Adopt only through the mandatory command**

Run:

```bash
uv run ng adopt candidates/task333/direct_einsum.onnx --task 333 --note "single free-output relational Einsum"
```

Expected: the internal re-gate passes and task333 changes from cost1586 to cost1295.

- [ ] **Step 2: Regenerate exact source ownership**

Run:

```bash
uv run python tools/live_to_exact_source.py 333 --write-src
```

Expected: `src/custom/task333.py` is printed and contains one `Einsum` node plus 11 initializers.

- [ ] **Step 3: Verify source/deployed byte identity and static cost**

Run:

```bash
uv run python - <<'PY'
import hashlib
import json
import tempfile
from pathlib import Path

import onnx
from src.custom.task333 import build
from neurogolf.scoring import evaluate, load_task

built = build(None)
deployed = onnx.load("submission/overfit_nets/task333.onnx")
built_bytes = built.SerializeToString()
deployed_bytes = deployed.SerializeToString()
print(hashlib.sha256(built_bytes).hexdigest())
print(hashlib.sha256(deployed_bytes).hexdigest())
assert built_bytes == deployed_bytes
result = evaluate(built, load_task(333))
print(json.dumps(result, indent=2))
assert result["fail"] == 0
assert result["memory"] == 0
assert result["params"] == 1295
PY
```

Expected: identical SHA-256 lines and result memory0/params1295/fail0.

- [ ] **Step 4: Record the successful mechanism in task-local handoffs**

Append this exact section to `candidates/task333/DISCOVERY.md`:

```markdown
## 9. Cost-1295 direct relational Einsum

- Replaced the complete cost1586 directional graph with one fp32 FREE-output `Einsum`.
- The input is reused as one source and two ordered green pixels. Shared relation banks encode
  preserve plus strict left/right/up/down chains without counted directional planes.
- Final price: memory0 + params1295 = cost1295; gain `ln(1586/1295)=+0.202704428`.
- Verification: bundled265 candidate/formula fail0; fresh1500 incumbent/candidate/formula fail0
  and candidate/incumbent divergence0.
- ORT order: record the selected JSON `name`, `index`, and measured first-inference seconds from
  `verify_direct_einsum.py --search` on this bullet before saving the file.
- No CumSum graph, public task333 graph, pack, or submission was used.
```

The selected-order bullet must contain the actual JSON values printed in Task 2, for example
`ORT order: semantic (index0), 0.123s`; do not copy the example numbers unless they are the
measured result.

In `state/tasks/task333.md`, change frontmatter to:

```yaml
---
deployed_cost: 1295
logged_costs_match: true
migrated: 2026-07-09
---
```

The `ng adopt` appended `ADOPTED` block remains the authoritative timestamped record.

In the existing task333 section of `state/STATE.md`, replace the cost1586 endpoint with cost1295,
score `25-ln(1295) = 17.833734`, bundled265/fresh1500 zeros, and the exact source/deployed SHA from
Step 3. Do not rewrite or stage unrelated concurrent-session sections.

- [ ] **Step 5: Run final task333 verification**

Run:

```bash
uv run ng score 333
uv run python candidates/task333/verify_direct_einsum.py --fresh 1500
uv run pytest -q candidates/task333/test_direct_einsum.py
```

Expected: deployed cost1295/fail0; fresh1500 all zero; `2 passed`.

- [ ] **Step 6: Inspect and stage only task333 tracked changes**

Run:

```bash
git diff -- src/custom/task333.py state/tasks/task333.md state/manifest.json state/STATE.md
git add src/custom/task333.py state/tasks/task333.md
git add -p state/manifest.json
git add -p state/STATE.md
git diff --cached --check
git diff --cached --stat
```

For each interactive patch, answer `y` only for the hunk containing task333 cost1295/SHA/result;
answer `n` for every unrelated task or shared-session hunk. Expected staged content is only
`src/custom/task333.py`, the task333 ledger, the manifest's `"333"` row, and the task333 handoff
hunk. If `state/STATE.md` cannot isolate the task333 hunk, leave it unstaged rather than including
another session's content.

- [ ] **Step 7: Commit the adopted source and task333 state**

Run:

```bash
git commit -m "optimize task333 with direct relational einsum" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

Expected: commit succeeds and contains no task other than task333.

---

## Plan Self-Review

- Spec coverage: one FREE-output node, exact 1295-element budget, five modes, strict green ordering,
  12-order/20-second planner bound, bundled gate, fresh1500, mandatory adoption, source SHA, and
  no-submit scope each have an execution step.
- Placeholder scan: the plan contains no unfinished implementation marker, deferred code, or
  unspecified error-handling step. The only runtime-substituted values are measured ORT order/time and SHA,
  which cannot exist before execution and are explicitly sourced from command output.
- Type consistency: builder input/output are float32 `[1,10,30,30]`; all 11 factors are float32;
  `build(order_index)` and `deterministic_orders()` names match the verifier and tests; the selected
  graph path is consistently `candidates/task333/direct_einsum.onnx`.
