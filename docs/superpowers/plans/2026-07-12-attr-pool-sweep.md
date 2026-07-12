# Attribute-Only Pool Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exhaustively identify parameter-free rectangular MaxPool solutions across all 400 NeuroGolf tasks.

**Architecture:** Enumerate legal one-dimensional pool configurations, filter them independently against row and column presence projections, and exact-check only surviving 2D pairs. Materialize and officially score every novel hit.

**Tech Stack:** Python 3.13, NumPy, ONNX 1.21.0, ONNX Runtime 1.26.0, pytest, repository scoring harness.

## Global Constraints

- Scratch and candidates stay under `candidates/attr_pool_sweep/`.
- Search the full padded `[1,10,30,30]` channel tensor, including channel 0 and zero-hot out-of-grid cells.
- Do not modify deployed ONNX files during discovery.
- Adoption requires `ng gate` then `ng adopt`.

---

### Task 1: Pool semantics and configuration enumeration

**Files:**
- Create: `candidates/attr_pool_sweep/test_pool_sweep.py`
- Create: `candidates/attr_pool_sweep/pool_sweep.py`

**Interfaces:**
- Produces: `AxisConfig`, `axis_configs(size=30)`, `pool1d`, and `pool2d`.

- [ ] Write tests for identity, asymmetric dense dilation, spaced dilation, and recoloring rejection.
- [ ] Run pytest and confirm import failure.
- [ ] Implement the minimal axis enumeration and exact boolean pooling helpers.
- [ ] Run pytest and confirm the synthetic tests pass.

### Task 2: Projection-filtered task search

**Files:**
- Modify: `candidates/attr_pool_sweep/pool_sweep.py`
- Modify: `candidates/attr_pool_sweep/test_pool_sweep.py`

**Interfaces:**
- Produces: `find_pool_configs(examples) -> list[tuple[AxisConfig, AxisConfig]]`.

- [ ] Add a failing test that recovers a known synthetic row/column configuration from multiple examples.
- [ ] Implement row/column projection filtering and exact 2D verification.
- [ ] Verify the focused pytest passes.

### Task 3: Board sweep and hit validation

**Files:**
- Create on results: `candidates/attr_pool_sweep/report.json`
- Create on hit: `candidates/attr_pool_sweep/taskNNN.onnx`
- Modify on 0 hits: `state/levers.yaml`
- Modify on hits: `state/tasks/taskNNN.md` and `state/insights.yaml`

- [ ] Scan train+test for tasks 001 through 400.
- [ ] Verify survivors on all arc-gen examples.
- [ ] Build a one-node ONNX model for every novel survivor and run the official harness plus `ng gate`.
- [ ] Record either verified hits or a dated four-field 0-hit verdict with exact reopen triggers.
- [ ] Run focused tests, YAML parsing, `git diff --check`, and inspect repository status before reporting.
