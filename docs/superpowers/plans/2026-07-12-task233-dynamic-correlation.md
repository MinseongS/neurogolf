# Task233 Dynamic Correlation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a source-controlled task233 candidate below cost 22,026 by replacing hash-table placement lookup with runtime-weight uint8 correlation.

**Architecture:** A graph-surgery builder starts from `src.custom.task233.build(None)`, removes only the sprite-hash/inverse-table subgraph, inserts dynamic-weight `QLinearConv` plus global per-channel `MaxPool`, and preserves the deployed detector, crop, override, stamp, and output tail. Focused tests establish runtime-weight and index semantics before the full bundled gate.

**Tech Stack:** Python 3.13, ONNX 1.21.0, ONNX Runtime 1.26.0, NumPy, repository scoring harness.

## Global Constraints

- Candidate and scratch artifacts live only under `candidates/task233/`.
- Do not modify `submission/overfit_nets/` or `src/custom/task233.py` during discovery.
- Keep onnx==1.21.0 and onnxruntime==1.26.0 pinned.
- Success requires bundled fail=0 and measured cost below both 24,703 and 22,026.
- Adoption, if warranted, must use `ng gate` followed by `ng adopt`; no gate bypass.

---

### Task 1: Runtime-correlation regression test

**Files:**
- Create: `candidates/task233/test_dynamic_corr.py`
- Create: `candidates/task233/build_dynamic_corr.py`

**Interfaces:**
- Consumes: `src.custom.task233.build(task) -> onnx.ModelProto`
- Produces: `build(task=None) -> onnx.ModelProto` and `write_candidate(path: Path) -> Path`

- [ ] **Step 1: Write the failing structural and runtime test**

  The test imports `build_dynamic_corr.build`, asserts that the result contains one runtime-weight `QLinearConv`, a two-output `MaxPool`, no `MatMul`, and no 512-entry initializer. It constructs an ORT session and compares candidate and baseline outputs on all bundled task233 examples loaded from `NEUROGOLF_ROOT/data/task233.json`.

- [ ] **Step 2: Run the test to verify RED**

  Run: `NEUROGOLF_ROOT=/Users/minseong/project/neurogolf PYTHONPATH=. uv run pytest -q candidates/task233/test_dynamic_corr.py`

  Expected: FAIL because `candidates.task233.build_dynamic_corr` does not exist.

- [ ] **Step 3: Implement minimal graph surgery**

  In `build_dynamic_corr.py`, clone the exact-source model, remove nodes producing `redf`, `shash`, `holf`, `hsh`, `hshr`, `hshf`, `tbl`, `shi`, `posg`, and `pos0`, and insert:

  ```python
  Reshape(red, shape_5133) -> red4b
  Cast(red4b, UINT8) -> red4u
  QLinearConv(holm, one, zero, red4u, one, zero, one, zero) -> corr
  MaxPool(corr, kernel_shape=[18, 18]) -> corr_max, corr_idx
  Mod(corr_idx, 324) -> corr_pos64
  Cast(corr_pos64, FLOAT16) -> corr_posf
  Reshape(corr_posf, sh51) -> pos0
  ```

  Add exact value_info, prune now-unused initializers, run strict shape inference, and expose `write_candidate` for `candidates/task233/cand_dynamic_corr.onnx`.

- [ ] **Step 4: Run the focused test to verify GREEN**

  Run the command from Step 2.

  Expected: PASS with candidate output equal to baseline on every bundled example.

---

### Task 2: Official-style cost and correctness gate

**Files:**
- Modify: `candidates/task233/test_dynamic_corr.py`
- Create: `candidates/task233/cand_dynamic_corr.onnx`

**Interfaces:**
- Consumes: `build_dynamic_corr.write_candidate`
- Produces: a checked candidate ONNX and official-style evaluation evidence

- [ ] **Step 1: Add a failing cost assertion**

  Evaluate the candidate with `src.harness.evaluate`; assert `fail == 0`, `memory + params < 24703`, and `memory + params < 22026`.

- [ ] **Step 2: Run the focused test and verify the cost assertion fails if the design misses its target**

  Run: `NEUROGOLF_ROOT=/Users/minseong/project/neurogolf PYTHONPATH=. uv run pytest -q candidates/task233/test_dynamic_corr.py`

  Expected before final pruning: either a target-specific assertion failure with the measured cost, or PASS if the minimal surgery already clears 22,026.

- [ ] **Step 3: Apply only semantics-preserving cleanup needed to clear the target**

  Remove initializers and value_info made unreachable by the deleted hash path. Do not change detector, crop, override, stamp, public lanes, or output behavior.

- [ ] **Step 4: Write and evaluate the candidate**

  Run: `NEUROGOLF_ROOT=/Users/minseong/project/neurogolf PYTHONPATH=. uv run python candidates/task233/build_dynamic_corr.py`

  Run: `NEUROGOLF_ROOT=/Users/minseong/project/neurogolf PYTHONPATH=. uv run python -m src.harness candidates/task233/cand_dynamic_corr.onnx 233`

  Expected: `fail: 0`, `ok: true`, cost below 22,026.

---

### Task 3: Fresh A/B and integration decision

**Files:**
- Create or modify only if required: `candidates/task233/fresh_ab_dynamic_corr.py`
- Modify on success: `state/tasks/task233.md`
- Modify on reusable success: `state/insights.yaml`

**Interfaces:**
- Consumes: baseline exact-source model and `cand_dynamic_corr.onnx`
- Produces: isolated-process divergence counts and a ledgered verdict

- [ ] **Step 1: Run an initial isolated fresh A/B sample**

  Use the task233 generator from `/Users/minseong/project/neurogolf/arc-gen/tasks/task_97a05b5b.py`; compare baseline and candidate on identical draws in fresh ORT processes.

  Expected: no candidate-only regressions in the initial sample.

- [ ] **Step 2: Escalate to the project gate only if the initial sample is clean**

  Run: `NEUROGOLF_ROOT=/Users/minseong/project/neurogolf PYTHONPATH=. uv run ng gate candidates/task233/cand_dynamic_corr.onnx --task 233`

  Expected: bundled fail=0 and candidate cheaper than deployed.

- [ ] **Step 3: Record the result**

  If successful, document measured cost, points, bundled gate, fresh A/B, and the dynamic-correlation mechanism in `state/tasks/task233.md` and `state/insights.yaml`; then rescan related sprite-placement tasks before considering `ng adopt`. If unsuccessful, append a four-field dated negative verdict with a concrete reopen trigger and leave deployed files untouched.

- [ ] **Step 4: Run final verification**

  Run the focused pytest, official-style harness evaluation, `git diff --check`, and `git status --short` immediately before reporting the result.
