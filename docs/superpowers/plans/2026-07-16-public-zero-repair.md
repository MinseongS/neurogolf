# Public-Zero Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Kaggle-zero task118/task131 deployments with evidenced safe artifacts and synchronize their Python builders without weakening ordinary cheaper-only adoption.

**Architecture:** A source-controlled JSON registry binds each repair to the task number, zero-score Kaggle reference, current incumbent SHA, and approved candidate SHA. `ng gate` permits a cost increase only when every binding matches; `ng adopt` records the action as `REPAIRED`. Normal adoption and unsupported-TopK repair behavior remain unchanged.

**Tech Stack:** Python 3.12, pytest, ONNX 1.21.0, ONNX Runtime 1.26.0, `uv run ng`.

## Global Constraints

- Keep `onnx==1.21.0` and `onnxruntime==1.26.0`.
- Never copy into `submission/overfit_nets/` outside `ng adopt`.
- Candidate bundled evaluation must have fail=0 in an isolated process.
- Public-zero repair evidence must bind task, submission ref, incumbent SHA, and candidate SHA.
- Do not pack or submit until a submission-qualified 400/400 verification completes.
- Preserve unrelated dirty files.

---

### Task 1: Evidence-bound public-zero gate

**Files:**
- Create: `state/public_zero_repairs.json`
- Modify: `tests/test_gate.py`
- Modify: `src/neurogolf/gate.py`

**Interfaces:**
- Consumes: `gate(candidate: Path, task_num: int, *, repair_invalid=False, public_zero_ref: int | None=None)`.
- Produces: `GateResult.repairing_public_zero: bool` and an evidence check keyed by decimal submission ref.

- [ ] **Step 1: Write failing gate tests**

Add tests that write a temporary registry containing task/ref/current SHA/candidate SHA, assert the expensive candidate passes only with the matching ref, and assert wrong ref, wrong task, wrong current SHA, wrong candidate SHA, and ordinary mode reject.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_gate.py -q`

Expected: failures because `gate()` has no `public_zero_ref` parameter or evidence logic.

- [ ] **Step 3: Add the registry**

Create `state/public_zero_repairs.json` with these exact bindings:

```json
{
  "54732112": {
    "task": 118,
    "score": 0.0,
    "incumbent_sha256": "7c4afc9cda73b9027b0576dad0dc89e573df19738d679b64a264272dd0b40ca9",
    "candidate_sha256": "f3b7cfd77d55353ff522b7d172b7dd38a46ac5f3df585dcc1a837c8e96c37247",
    "reason": "fixed hash/edit lookup scored zero; restore Kaggle-positive semantic archive graft"
  },
  "54732114": {
    "task": 131,
    "score": 0.0,
    "incumbent_sha256": "f07341f0d75d91f78be2d07e3073fdead3c54888cf524a47d4e101acdd2d2f0d",
    "candidate_sha256": "79ff05f1116e16a8f7198a96b75fb7b9e84664a2e376311b852f51f20c5da67f",
    "reason": "119-position bundled support prune scored zero; restore full 155-position support"
  }
}
```

- [ ] **Step 4: Implement the minimal evidence check**

In `src/neurogolf/gate.py`, hash files with `hashlib.sha256`, load `STATE / "public_zero_repairs.json"`, require `score == 0.0`, and set `repairing_public_zero=True` only when all four bindings match. Permit `candidate_cost >= incumbent_cost` only for `repairing_invalid_topk` or `repairing_public_zero`.

- [ ] **Step 5: Verify GREEN**

Run: `uv run pytest tests/test_gate.py -q`

Expected: all tests pass.

### Task 2: CLI and adoption propagation

**Files:**
- Modify: `tests/test_adoption.py`
- Modify: `src/neurogolf/cli.py`
- Modify: `src/neurogolf/adoption.py`

**Interfaces:**
- Consumes: `public_zero_ref: int | None` passed to `gate()`.
- Produces: `ng gate ... --repair-public-zero REF`, `ng adopt ... --repair-public-zero REF`, and a `REPAIRED` task-log stamp containing the ref.

- [ ] **Step 1: Write failing adoption tests**

Test that `adopt(..., public_zero_ref=54732112)` forwards the ref, replaces and backs up the model after a successful public-zero gate, and stamps `REPAIRED` plus `public-zero-ref: 54732112`.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_adoption.py -q`

Expected: failure because the argument and stamp do not exist.

- [ ] **Step 3: Implement CLI and adoption plumbing**

Add `--repair-public-zero` as an integer argument to gate/adopt parsers, pass it to the gate, and make adoption choose `REPAIRED` when either repair mode succeeds. Include the ref in the stamp only for public-zero repairs.

- [ ] **Step 4: Verify GREEN and regression suite**

Run: `uv run pytest tests/test_gate.py tests/test_adoption.py -q`

Expected: all tests pass.

### Task 3: Repair task118 and task131 deployments

**Files:**
- Modify through CLI: `submission/overfit_nets/task118.onnx`
- Modify through CLI: `submission/overfit_nets/task131.onnx`
- Modify through CLI: `state/manifest.json`
- Modify through CLI: `state/tasks/task118.md`
- Modify through CLI: `state/tasks/task131.md`

**Interfaces:**
- Consumes: the two backup ONNX files and evidence refs.
- Produces: deployed task118 SHA `f3b7...7247`, task131 SHA `79ff...a67f`.

- [ ] **Step 1: Gate both candidates**

Run:

```bash
uv run ng gate submission/.backups/task118_20260715T110407Z.onnx --task 118 --repair-public-zero 54732112
uv run ng gate submission/.backups/task131_20260715T115906Z.onnx --task 131 --repair-public-zero 54732114
```

Expected: PASS, bundled 267/267 cost8699 and 266/266 cost3874.

- [ ] **Step 2: Adopt both repairs**

Run the matching `ng adopt` commands with notes identifying the zero submission and safe provenance. Expected: both return JSON rows and task logs contain `REPAIRED`.

- [ ] **Step 3: Score and hash deployed artifacts**

Run `uv run ng score 118 131` and `shasum -a 256` on both deployed files. Expected: fail0 and the approved candidate SHAs.

### Task 4: Synchronize Python ownership and verify

**Files:**
- Modify: `src/custom/task118.py`
- Modify: `src/custom/task131.py`
- Modify: `state/submissions.md`
- Replace at session end: `state/STATE.md`

**Interfaces:**
- Consumes: repaired deployed ONNX files.
- Produces: Python builders that reproduce deployed semantics/cost and durable repair records.

- [ ] **Step 1: Regenerate exact source builders**

Use `tools/live_to_exact_source.py --write-src` for tasks 118 and 131, following the tool's current CLI help. Do not modify other task builders.

- [ ] **Step 2: Rebuild in fresh processes**

Use the repository rebuild tool restricted to tasks 118 and 131, evaluate the rebuilt files, and compare cost, bundled pass count, and SHA or topology-equivalent output as supported by the tool.

- [ ] **Step 3: Run focused tests**

Run `uv run pytest tests/test_gate.py tests/test_adoption.py candidates/task118/test_hashmask_*.py candidates/task131/test_prune_probe_k9.py candidates/task131/test_dynamic_nonzero.py -q`. Historical hash/prune builder tests may continue testing rejected candidates; production-source assertions must point at the repaired deployment.

- [ ] **Step 4: Record the external evidence**

Append the two completed 0.00 refs and repair decision to `state/submissions.md`. Replace `state/STATE.md` with the current board truth, explicitly noting that task118/task131 are repaired but not yet resubmitted.

- [ ] **Step 5: Final verification**

Run the focused score/hash checks and the relevant test suite. Run full `uv run ng verify` only if the workspace is quiescent; do not pack or submit if it times out or the manifest is moving.

- [ ] **Step 6: Commit scoped files**

Stage only the gate/adoption/CLI tests and implementation, repair registry, task118/task131 sources and logs, manifest, state handoff, and submission log. Preserve every unrelated dirty file.
