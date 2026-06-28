# Public Teacher Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a public-model teacher workflow that preserves source ownership while using public ONNX files to discover reusable scoring mechanisms.

**Architecture:** Add two focused scripts: one scanner that compares public candidates against live/source models, and one extractor that produces non-mutating source/insight drafts. Reports are written under `reports/`, candidates live under `public_candidates/`, and recursive workflow documentation is updated to require teacher-to-source-to-insight conversion.

**Tech Stack:** Python 3, ONNX, existing `src.harness.evaluate`, existing `reports/manifest.json`, markdown/json reports.

## Global Constraints

- Do not blindly overwrite `networks/taskNNN.onnx` with public candidates.
- Public models are teacher artifacts until source ownership or an explicit submission-test exception exists.
- Lower-score public models still matter when they expose better reusable mechanisms.
- Scripts must work with an empty `public_candidates/` directory.

---

### Task 1: Public teacher scanner

**Files:**
- Create: `reports/scripts/public_teacher_scan.py`
- Create/update outputs: `reports/public_teacher_report.json`, `reports/public_teacher_report.md`

**Interfaces:**
- Consumes: public ONNX files whose path contains a task id.
- Produces: JSON rows with `task`, `candidate_path`, `public_eval`, `live_eval`, `source_eval`, `flags`, `op_delta`, and `recommendation`.

- [ ] Create a scanner that recursively finds `.onnx` files under `public_candidates/`.
- [ ] Infer the task id from filename/path patterns like `task017`, `017`, or `task_017`.
- [ ] Evaluate public, live, and source models with `src.harness.evaluate`.
- [ ] Compare score, memory, params, op counts, and source lag.
- [ ] Write JSON and markdown reports.
- [ ] Verify empty directory succeeds.

### Task 2: Public teacher extractor

**Files:**
- Create: `reports/scripts/public_teacher_extract.py`
- Create/update outputs: `reports/public_teacher_sources/taskNNN_public_exact.py`, `reports/public_teacher_insights/taskNNN_public_teacher.md`

**Interfaces:**
- Consumes: a task number and public candidate path.
- Produces: an exact source scaffold draft and a mechanism extraction note.

- [ ] Encode the public ONNX as gzip+base64 in a source draft.
- [ ] Ensure draft `build(task)` loads from embedded bytes, not from the original public path.
- [ ] Write a markdown note with live/public/source deltas and op-family changes.
- [ ] Do not mutate `src/custom`, `networks`, or `reports/manifest.json`.
- [ ] Verify extraction on an existing live ONNX path as a safe stand-in.

### Task 3: Workflow integration

**Files:**
- Modify: `/Users/minseong/.codex/skills/neurogolf-recursive-improvement/SKILL.md`
- Create: `public_candidates/.gitkeep`

**Interfaces:**
- Consumes: scanner/extractor commands.
- Produces: repeatable operator workflow for future sessions.

- [ ] Add the public teacher loop to the local NeuroGolf recursive improvement skill.
- [ ] Document that public candidates are teacher artifacts, not direct adoption artifacts.
- [ ] Commit repo-owned scripts, docs, reports, and `.gitkeep`.

## Self-review

- Spec coverage: scanner, extractor, non-mutating policy, and workflow docs are covered.
- Placeholder scan: no TBD/TODO placeholders.
- Type consistency: report field names are defined in Task 1 and reused by Task 2 documentation.
