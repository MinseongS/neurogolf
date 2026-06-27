# Aggressive NeuroGolf Task Mining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce non-public, generator-derived NeuroGolf score gains larger than tail-only +0.1 improvements.

**Architecture:** Work one high-headroom task at a time. For each task, inspect the generator, build or patch a candidate ONNX in `/tmp` or `src/custom/taskNNN.py`, verify local/fresh, submit probe zips freely, and adopt only positive Kaggle results.

**Tech Stack:** Python, ONNX helper APIs, `src.harness.evaluate`, `src.genverify`, Kaggle CLI.

## Global Constraints

- Do not overwrite unrelated dirty files: `src/custom/task064.py`, `src/custom/task110.py`, `src/custom/task198.py`, `src/custom/task370.py`.
- Avoid `python -m src.pipeline --pack`; package zips directly from `networks/task*.onnx`.
- Kaggle submissions are effectively not scarce: use clearly labeled probe zips.
- Prefer original generator-derived solver changes over public notebook copying.
- Commit only confirmed positive or strategically stabilizing changes.

---

### Task 1: Attack task209 as first high-headroom candidate

**Files:**
- Read: `networks/task209.onnx`
- Optionally create: `src/custom/task209.py`
- Optionally modify after positive probe: `networks/task209.onnx`, `reports/manifest.json`, `reports/lb_anchor.json`, `reports/submission_log.md`

**Interfaces:**
- Consumes: task generator via `PYTHONPATH=. .venv/bin/python -m src.show 209 --gen`
- Produces: a candidate ONNX path and probe submission result.

- [ ] Inspect task209 generator and current ONNX memory profile.
- [ ] Build the smallest candidate that can recover the full magnified sprite in the yellow box.
- [ ] Verify local stored examples and at least 200 fresh generated examples.
- [ ] If local score improves or hidden behavior is uncertain but plausible, submit a probe zip.
- [ ] Adopt/commit only if Kaggle beats current best 7170.36.

### Task 2: Fast-switch to geometry/fill tasks if task209 stalls

**Files:**
- Candidate tasks: `255`, `349`, `002`, `187`

**Interfaces:**
- Consumes: same harness and packaging commands as Task 1.
- Produces: adopted positive ONNX replacements or documented negative probes.

- [ ] For each candidate, classify whether the generator rule is closed-form geometry/fill.
- [ ] Prefer candidates with expected +0.5 or more local score gain.
- [ ] Submit probes for high-upside approximate candidates rather than waiting for proof.

### Task 3: Record every meaningful result

**Files:**
- Modify: `reports/lb_anchor.json`
- Modify: `reports/submission_log.md`
- Modify: `reports/RESUME.md`

**Interfaces:**
- Consumes: Kaggle submission refs and local score deltas.
- Produces: restart-safe record of positive and negative probes.

- [ ] Record positive changes with task number, local delta, zip SHA, submission ref, and public score.
- [ ] Record negative probes if they reveal hidden/private mismatch or a dead direction.
- [ ] Commit only intentional files.
