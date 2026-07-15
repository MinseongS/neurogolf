# Task264 Cost1227 Support-Gated Renderer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, verify, gate, and adopt a task264 candidate at cost1227 by combining exact glyph CP factoring, support-gated reuse of the detector interpolation basis, Round-carrier removal, and deletion of the all-one `shift_0` operand.

**Architecture:** Transform the deployed/source-owned cost1361 ONNX under `candidates/task264/` while preserving the eight detector expressions except for algebraically deleting `shift_0` and routing their pre-Round scalars forward. Replace only the terminal spatial factors; keep the current rank-3 channel factor unchanged. Validate the representation algebraically and with bounded fresh-process runtime before the mandatory isolated gate/adopt path.

**Tech Stack:** Python 3.13, NumPy, ONNX 1.21.0, ONNX Runtime 1.26.0, pytest, `uv run ng`.

## Global Constraints

- Work only on task264; candidate and scratch artifacts stay under `candidates/task264/`.
- Keep `onnx==1.21.0` and `onnxruntime==1.26.0`; do not upgrade them.
- Never restore the removed Gather/Equal/Pad tail.
- Candidate static cost must be exactly1227 and at most the +0.1 threshold1231.
- Off-chart raw output must be exactly zero.
- Use `NG_EVAL_TIMEOUT_SECONDS=1800` for full evaluation, gate, and adopt.
- Adoption must go through `uv run ng gate` then `uv run ng adopt`; never copy into `submission/overfit_nets/`.
- Fresh/generated evaluation is diagnostic; bundled fail0 plus cheaper-than-deployed remains the adoption gate.

---

### Task 1: TDD the support-gated renderer transform

**Files:**
- Create: `candidates/task264/test_support_gated_renderer.py`
- Create: `candidates/task264/build_support_gated_renderer.py`
- Create: `candidates/task264/support_gated_renderer.onnx`
- Consume: `candidates/task264/shared_rank_initializer_factor.onnx`

**Interfaces:**
- Consumes: `build_candidate(incumbent: Path, output: Path)` convention from the existing task264 builders.
- Produces: a persistent ONNX whose fixed-point rebuild is byte-identical and whose static cost is1227.

- [ ] **Step 1: Write the failing structural and fixed-point tests**

Create `candidates/task264/test_support_gated_renderer.py` with these assertions:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from neurogolf.scans.minmerge import static_cost


HERE = Path(__file__).resolve().parent
BUILDER = HERE / "build_support_gated_renderer.py"
INCUMBENT = HERE / "shared_rank_initializer_factor.onnx"
CANDIDATE = HERE / "support_gated_renderer.onnx"


def _load_builder():
    assert BUILDER.exists(), "task264 support-gated builder has not been implemented"
    spec = importlib.util.spec_from_file_location("task264_support_gated_renderer", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}


def _equation(node: onnx.NodeProto) -> bytes | None:
    return next(
        (helper.get_attribute_value(attr) for attr in node.attribute if attr.name == "equation"),
        None,
    )


def test_support_gated_renderer_reaches_cost1227_and_preserves_sign_margin(tmp_path: Path) -> None:
    rebuilt = tmp_path / "candidate.onnx"
    _load_builder().build_candidate(INCUMBENT, rebuilt)
    assert rebuilt.read_bytes() == CANDIDATE.read_bytes()

    incumbent = onnx.load(INCUMBENT)
    candidate = onnx.load(rebuilt)
    old = _arrays(incumbent)
    new = _arrays(candidate)

    assert all(node.op_type != "Round" for node in candidate.graph.node)
    assert "shift_0" not in new
    assert all("shift_0" not in node.input for node in candidate.graph.node)
    concat = next(node for node in candidate.graph.node if node.name == "fold_color_augment")
    assert list(concat.input[:4]) == ["e0", "e1", "e2", "e3"]
    assert list(concat.input[5:9]) == ["e5", "e6", "e7", "e8"]
    assert new["glyph_row"].shape == (5, 3, 3)
    assert new["glyph_col"].shape == (5, 3, 3)
    assert new["glyph_term"].shape == (5, 2)
    assert new["block_shift"].shape == (3, 16)
    assert new["local_shift"].shape == (3, 16)
    assert new["chart_support"].shape == (30,)
    assert np.array_equal(new["channel_basis"], old["channel_basis"])
    assert np.array_equal(new["channel_core"], old["channel_core"])

    old_pattern = np.einsum(
        "uza,awpq->uwpqz", old["glyph_pattern_left"], old["glyph_pattern_right"]
    )
    new_pattern = np.einsum(
        "aup,awq,az->uwpqz", new["glyph_row"], new["glyph_col"], new["glyph_term"]
    )
    assert np.array_equal(new_pattern, old_pattern)

    embedding = np.einsum(
        "uj,pj,Rj,R->upR",
        new["block_shift"],
        new["local_shift"],
        new["shift_col"],
        new["chart_support"],
        optimize=False,
    )
    exact_embedding = np.einsum(
        "uR,pR->upR", old["glyph_block_embed"], old["glyph_local_embed"]
    )
    assert np.count_nonzero(embedding[..., 9:]) == 0
    assert np.max(np.abs(embedding[..., :9] - exact_embedding[..., :9])) <= 0.00055
    assert _load_builder().worst_signed_margin(candidate) >= 0.237

    assert static_cost(rebuilt) == 1227
    assert _equation(candidate.graph.node[-1]) == (
        b"nsuw,ntuw,vh,hstz,aup,awq,az,uj,pj,Rj,R,wk,qk,Ck,C->nvRC"
    )
    onnx.checker.check_model(candidate, full_check=True)
    onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)


def test_support_gated_renderer_is_a_byte_identical_fixed_point(tmp_path: Path) -> None:
    rebuilt = tmp_path / "rebuilt.onnx"
    _load_builder().build_candidate(CANDIDATE, rebuilt)
    assert rebuilt.read_bytes() == CANDIDATE.read_bytes()
```

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
PYTHONPATH=. uv run pytest -q candidates/task264/test_support_gated_renderer.py -x
```

Expected: FAIL with `task264 support-gated builder has not been implemented`.

- [ ] **Step 3: Implement the minimal graph transform**

Create `candidates/task264/build_support_gated_renderer.py`. It must:

```python
OLD_SPATIAL = {
    "glyph_pattern_left",
    "glyph_pattern_right",
    "glyph_block_embed",
    "glyph_local_embed",
}
NEW_INPUTS = [
    "glyph_color_aug", "glyph_color_aug", "channel_basis", "channel_core",
    "glyph_row", "glyph_col", "glyph_term",
    "block_shift", "local_shift", "shift_col", "chart_support",
    "block_shift", "local_shift", "shift_col", "chart_support",
]
NEW_EQUATION = b"nsuw,ntuw,vh,hstz,aup,awq,az,uj,pj,Rj,R,wk,qk,Ck,C->nvRC"
```

Reconstruct the current dense pattern, use the existing exact `rank=4` matrix-factor helper on
the coloured 9x9 slice, prepend one all-ones base term, and emit:

```python
glyph_row = np.concatenate([np.ones((9, 1), np.float32), left], axis=1).T.reshape(5, 3, 3)
glyph_col = np.concatenate([np.ones((9, 1), np.float32), right.T], axis=1).T.reshape(5, 3, 3)
glyph_term = np.zeros((5, 2), np.float32)
glyph_term[0, 0] = 1.0
glyph_term[1:, 1] = 1.0
block_shift = shift_row[[0, 3, 6]].copy()
local_shift = shift_row[:3].copy()
chart_support = np.zeros(30, np.float32)
chart_support[:9] = 1.0
```

For every detector Einsum except the terminal, split the equation at commas and remove exactly
the operand/subscript positions whose input name is `shift_0`. Delete all Round nodes, replace
`eN_rounded` with `eN` in `fold_color_augment`, remove rounded value-info entries, delete the old
spatial and `shift_0` initializers, and append the six new initializers. Replace the terminal with
the 15-operand equation above. Validate exact pattern reconstruction, support zeros, checker, and
strict shape inference before saving. If the input already has `NEW_EQUATION`, validate shapes and
save it unchanged to provide the fixed point.

Expose `worst_signed_margin(model: onnx.ModelProto) -> float`, implementing the separable worst-case
palette calculation from the design: reconstruct channel scores and spatial contributions for all
nine slots, ten colours, ten output channels, and every 9x9 chart cell; include detector deviation
endpoints at `integer ± 0.045`; return the minimum signed margin.

- [ ] **Step 4: Build the persistent candidate and verify GREEN**

Run:

```bash
PYTHONPATH=. uv run python candidates/task264/build_support_gated_renderer.py
PYTHONPATH=. uv run pytest -q candidates/task264/test_support_gated_renderer.py
```

Expected: candidate path printed; `2 passed`; static cost1227 and margin at least0.237.

---

### Task 2: Bounded fresh-process runtime and output diagnostics

**Files:**
- Create: `candidates/task264/diagnose_support_gated_renderer.py`
- Consume: cost1361 incumbent and cost1227 candidate.

**Interfaces:**
- Consumes: two ONNX paths and the first20 bundled examples.
- Produces: JSON containing load time, one-example timeout status, per-model median runtime, sign divergence, oracle failures, and off-chart nonzero count.

- [ ] **Step 1: Implement the isolated diagnostic**

Create a parent/worker script using `subprocess.run(..., timeout=20)` for the one-example worker.
Inside the worker, sanitize each model, set `ORT_DISABLE_ALL`, load the same20 task264 bundled
examples, and record:

```python
raw = session.run(["output"], {"input": x})[0]
sign = raw > 0.0
off_chart_nonzero = np.count_nonzero(
    np.concatenate([raw[:, :, 9:, :].ravel(), raw[:, :, :9, 9:].ravel()])
)
```

The parent must fail unless candidate/oracle sign divergence is zero, off-chart nonzero is zero,
the one-example worker finishes within20 seconds, and candidate median runtime is at most1.10 times
the incumbent median over the same20 examples.

- [ ] **Step 2: Run bounded diagnostics**

Run:

```bash
PYTHONPATH=. uv run python candidates/task264/diagnose_support_gated_renderer.py \
  candidates/task264/shared_rank_initializer_factor.onnx \
  candidates/task264/support_gated_renderer.onnx
```

Expected: exit0 JSON with zero divergence/fail/off-chart values and runtime ratio <=1.10. If it
fails, stop this representation and document the dated result in the four-field lever ledger.

---

### Task 3: Full bundled gate and adoption

**Files:**
- Consume: `candidates/task264/support_gated_renderer.onnx`
- Modify via `ng adopt`: `submission/overfit_nets/task264.onnx`, `state/manifest.json`, `state/tasks/task264.md`

**Interfaces:**
- Consumes: bounded diagnostic PASS and candidate cost1227.
- Produces: adopted task264 only if both mandatory gates pass.

- [ ] **Step 1: Run full isolated candidate evaluation**

Run:

```bash
NG_EVAL_TIMEOUT_SECONDS=1800 PYTHONPATH=. uv run python -c \
  'from pathlib import Path; from neurogolf.gate import eval_isolated; print(eval_isolated(Path("candidates/task264/support_gated_renderer.onnx"), 264))'
```

Expected: `ok=True`, `pass=265`, `fail=0`, `cost=1227`.

- [ ] **Step 2: Run the mandatory gate**

Run:

```bash
NG_EVAL_TIMEOUT_SECONDS=1800 uv run ng gate \
  candidates/task264/support_gated_renderer.onnx --task 264
```

Expected: PASS, bundled265/265, fail0, cost1227 below deployed1361.

- [ ] **Step 3: Adopt through the mandatory re-gate**

Run:

```bash
NG_EVAL_TIMEOUT_SECONDS=1800 uv run ng adopt \
  candidates/task264/support_gated_renderer.onnx --task 264 \
  --note "support-gated shift interpolation, exact glyph CP rank5, raw detector carrier fold"
```

Expected: JSON with `ok=true`, `fail=0`, `cost=1227`, and a new task264 SHA.

---

### Task 4: Exact-source synchronization and handoff

**Files:**
- Modify: `src/custom/task264.py`
- Modify: `networks/task264.onnx`
- Modify: `candidates/task264/DISCOVERY.md`
- Modify: `state/STATE.md`
- Modify: `state/insights.yaml`

**Interfaces:**
- Consumes: adopted task264 SHA and cost1227 candidate.
- Produces: source/candidate/deployed/network SHA equality plus current task264 handoff.

- [ ] **Step 1: Regenerate exact source**

Run:

```bash
uv run python tools/live_to_exact_source.py 264 --write-src
```

Expected: `src/custom/task264.py` written from the adopted graph.

- [ ] **Step 2: Rebuild and evaluate the source-owned network**

Run:

```bash
NG_EVAL_TIMEOUT_SECONDS=1800 PYTHONPATH=. uv run python \
  tools/rebuild_networks_from_source.py --tasks 264
```

Expected: `rebuilt=1 failed=0`, points for cost1227.

- [ ] **Step 3: Verify fixed points, YAML, SHA, and whitespace**

Run:

```bash
PYTHONPATH=. uv run pytest -q \
  candidates/task264/test_free_output_tail_fold.py \
  candidates/task264/test_shared_rank_initializer_factor.py \
  candidates/task264/test_support_gated_renderer.py -k fixed_point
PYTHONPATH=. uv run pytest -q candidates/task264/test_support_gated_renderer.py
git diff --check -- candidates/task264 src/custom/task264.py state/STATE.md \
  state/insights.yaml state/tasks/task264.md state/manifest.json
```

Then build `src.custom.task264` in memory and assert its serialized bytes equal the candidate,
deployed, and `networks/task264.onnx` bytes. Expected: all tests pass, diff check exit0, and one SHA.

- [ ] **Step 4: Replace current task264 handoff facts**

Update `candidates/task264/DISCOVERY.md`, the task251-275 paragraph in `state/STATE.md`, and the
existing `quadratic_palette_glyph_free_output_tail` entry in `state/insights.yaml` with cost1227,
measured score, runtime, bundled result, factor arithmetic, SHA, and the next +0.1 threshold. Do not
append stale state or record a win as a negative ledger entry.

- [ ] **Step 5: Commit only safely isolated tracked changes**

Inspect every tracked file for unrelated pre-existing hunks. Commit only files/hunks attributable
to task264; if shared files contain inseparable concurrent edits, leave them uncommitted and report
that constraint. Never use `git add -A`.
