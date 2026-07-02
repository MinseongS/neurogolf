# NeuroGolf ONNX Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Streamlit viewer for task-by-task NeuroGolf input/output/candidate ONNX comparison.

**Architecture:** A single Streamlit app calls the existing Python harness directly. Candidate models are loaded either from `networks/taskNNN.onnx`, from `src/custom/taskNNN.py`, or from a scratch `cand.py` path exposing `build(task)`.

**Tech Stack:** Python, Streamlit, ONNX, ONNX Runtime, NumPy, existing `src.harness`.

## Global Constraints

- App path: `tools/onnx_viewer.py`.
- Run command: `PYTHONPATH=. .venv/bin/streamlit run tools/onnx_viewer.py`.
- Do not modify `networks/`, `src/custom/`, `reports/manifest.json`, or submission artifacts.
- Default displayed examples: 30.
- Version 1 does not include a node-by-node ONNX graph editor.

---

### Task 1: Streamlit Viewer Core

**Files:**
- Create: `tools/onnx_viewer.py`

**Interfaces:**
- Consumes: `src.harness.load_task`, `convert_to_numpy`, `evaluate`, `run_network`.
- Produces: Streamlit app with task selector, candidate selector, metrics, and grid comparisons.

- [ ] **Step 1: Create the app file**

Write `tools/onnx_viewer.py` with helpers:

```python
def grid_from_onehot(arr: np.ndarray) -> np.ndarray
def render_grid(grid: np.ndarray, key: str, mismatches: np.ndarray | None = None) -> None
def load_candidate_model(task_num: int, task: dict, mode: str, cand_path: str) -> tuple[Any | None, str | None]
def run_examples(model_or_path: Any, examples: list[dict]) -> list[dict]
```

- [ ] **Step 2: Verify syntax**

Run: `PYTHONPATH=. .venv/bin/python -m py_compile tools/onnx_viewer.py`
Expected: exits 0.

### Task 2: Smoke Test With Current ONNX

**Files:**
- Modify: `tools/onnx_viewer.py` if smoke test exposes errors.

**Interfaces:**
- Consumes: `tools/onnx_viewer.py`.
- Produces: a runnable Streamlit app.

- [ ] **Step 1: Run non-UI smoke import**

Run:

```bash
PYTHONPATH=. .venv/bin/python - <<'PY'
from tools import onnx_viewer as v
from src.harness import load_task
t = load_task(187)
m, err = v.load_candidate_model(187, t, "deployed", "")
assert err is None, err
rows = v.run_examples(m, t["train"][:1])
assert len(rows) == 1
assert rows[0]["output_grid"].shape == (30, 30)
print("viewer smoke ok")
PY
```

Expected: prints `viewer smoke ok`.

- [ ] **Step 2: Start Streamlit**

Run: `PYTHONPATH=. .venv/bin/streamlit run tools/onnx_viewer.py --server.port 8501`
Expected: Streamlit serves the app at `http://localhost:8501`.
