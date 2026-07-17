# Task251-275 Score Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for every builder. The controller owns gate/adopt/state integration; workers never edit deployed nets, manifests, state files, or git state.

**Goal:** Finish every byte-proven score improvement from the task251-275 discovery pass, adopting only bundled-exact candidates that are strictly cheaper.

**Architecture:** Task251 and task264 are independent source-owned ONNX rewrites under their task candidate directories. Task267 and task271 are residual feasibility lanes: a candidate is built only after static pricing meets the current +0.1 thresholds. The controller independently evaluates, gates, adopts, regenerates exact source, records reusable insights, and rescans.

**Tech Stack:** Python 3.13, onnx==1.21.0, onnxruntime==1.26.0, NumPy, pytest, `uv run ng`.

**Execution result (2026-07-15):** Complete. Adopted endpoints are task251=1335,
task264=1507, task267=220, task271=586. The task267/task271 residual implementations
beat the planning thresholds with stronger algebraic variants. Focused regression,
exact-source SHA sync, reusable insight registration, rescan, and STATE handoff are done;
full-board pack/submit remains intentionally gated by unrelated concurrent hash drift.

## Global Constraints

- Goal 8000; default gate is bundled fail=0 and strictly cheaper than the manifest incumbent.
- All scratch and generated ONNX files stay under `candidates/taskNNN/`.
- Never copy into `submission/overfit_nets/`; adoption is `ng gate` then `ng adopt` only.
- FREE input/output may be used; counted intermediates and initializer parameters must be statically priced.
- Fresh evaluation is diagnostic and never replaces the bundled gate.
- Preserve onnx==1.21.0 and onnxruntime==1.26.0.
- Shared checkout: workers do not run git commands or edit `src/custom`, `state/`, `submission/`, or `networks/`; the controller integrates successful candidates.

---

### Task 1: Task251 channel-support collapse

**Files:**
- Create: `candidates/task251/test_channel_support_collapse.py`
- Create: `candidates/task251/build_channel_support_collapse.py`
- Create: `candidates/task251/channel_support_collapse.onnx`
- Read: `candidates/task251/DISCOVERY.md`, `src/custom/task251.py`, `submission/overfit_nets/task251.onnx`

**Interfaces:**
- `build_candidate(source_model: Path, output: Path) -> None`
- Candidate must replace dense `T[2,10,10]` with `E[2,10]` and `Tprime[2,10,2]`, while retaining the source's factor-seed representation.

- [x] Write a failing test that builds from `src.custom.task251.build()`, asserts dense `T` is absent, checks E/Tprime shapes and active channels 0/2, requires static cost 1335, and executes bundled task251 with fail=0.
- [x] Run the test before the builder exists and record the expected failure.
- [x] Implement the minimal ONNX rewrite: `E[0,0]=1`, `E[1,2]=1`, `Tprime=T[:,:,[0,2]]`; replace the final Einsum's T operand with E and Tprime and update the equation without adding a counted intermediate.
- [x] Run checker, strict shape inference, the focused pytest, isolated bundled evaluation, and a direct deployed/candidate threshold comparison.
- [x] If the high-arity equation fails to load, search only algebraically equivalent operand/equation orders; if any bundled input uses a channel outside 0/2, stop and report generalization-risk falsification.

### Task 2: Task264 free-output tail fold

**Files:**
- Create: `candidates/task264/test_free_output_tail_fold.py`
- Create: `candidates/task264/build_free_output_tail_fold.py`
- Create: `candidates/task264/live_exact_baseline.onnx`
- Create: `candidates/task264/free_output_tail_fold.onnx`
- Read: `candidates/task264/DISCOVERY.md`, `state/tasks/task264.md`, `submission/overfit_nets/task264.onnx`, `playbook/free-output-einsum.md`

**Interfaces:**
- `build_candidate(incumbent: Path, output: Path) -> None`
- Candidate preserves the eight live scalar color detectors and replaces only `bank -> Gather(tpl) -> cgrid -> Equal -> Pad` with a final FREE-output expression.

- [x] Write a failing test that identifies the live tail structurally, requires `bank/cgrid/oh` and terminal Gather/Equal/Pad to disappear, sets success cost <=1676, and executes bundled task264 with fail=0.
- [x] Run the test before implementation and record the expected failure.
- [x] Generate a candidate-local exact baseline from the deployed graph; do not write `src/custom` before a win.
- [x] Implement the recorded `nbvt,bpqt,br,bc,rpR,cqC->nvRC`-family fold using fixed row/column glyph factors and fp32 sign-positive channel output. Price all new operands; target cost1511.
- [x] Probe ORT load and 20 stored examples before the full test. If zero margins fail, adjust only fixed coefficients. If cost exceeds1676, compute exact shared row/column rank and continue only when it restores cost<=1676.
- [x] Run checker, strict shape inference, focused pytest, isolated bundled evaluation, and direct deployed/candidate comparison.

### Task 3: Task267 and task271 residual feasibility

**Files:**
- Create only if threshold is met: `candidates/task267/test_marker_inline.py`, `build_marker_inline.py`, `marker_inline.onnx`
- Create only if threshold is met: `candidates/task271/test_winner_output_fold.py`, `build_winner_output_fold.py`, `winner_output_fold.onnx`
- Read: both task DISCOVERY files and their current adopted builders/tests.

**Interfaces:**
- Task267 must statically price at cost<=244 before implementation.
- Task271 MaxPool-indices plus output fold must statically price at cost<=617 before implementation.

- [x] For task267, algebraically price direct marker reread/inlining into the final output contraction, including every fixed initializer. If >244, record the numeric rejection and do not build.
- [x] For task271, build a tiny in-memory ORT MaxPool-indices probe and verify flattened index semantics. Price deleted dual maxima/argmax and crop8 tail against all new integer/embedding parameters. If >617, record the numeric rejection and do not build.
- [x] If either threshold is met, follow a complete RED-GREEN builder test and bundled gate cycle identical to Tasks 1/2.

### Task 4: Controller integration and recursive rescan

**Files:**
- Modify successful task exact sources via `tools/live_to_exact_source.py --write-src` only after adoption.
- Modify: `state/insights.yaml` only for a genuinely reusable mechanism.
- Replace: `state/STATE.md` with the current shared-board handoff.

- [x] Independently inspect every worker file and rerun its focused tests.
- [x] Run `uv run ng gate` for every bundled-exact, cheaper candidate; never adopt a rejection.
- [x] Run `uv run ng adopt` for each PASS and then regenerate/rebuild exact source.
- [x] Run `ng score` for each adopted task and source/candidate/deployed hash or normalized-graph checks.
- [x] Rescan task251-275 and the full deployed set for the new structural signature; register reusable mechanisms.
- [x] Run focused regression tests and a non-updating hash/status audit. Run full `ng verify` only if concurrent evaluation load has quiesced.
- [x] Replace STATE with only current truth. Do not pack/submit until manifest/artifact drift is reconciled and complete 400/400 verification succeeds.

## Self-review

- Spec coverage: task251, task264, and residual task267/task271 are each mapped to a thresholded implementation lane; gate/adopt/source/rescan are covered by Task 4.
- Placeholder scan: no TBD/TODO or unspecified success threshold remains.
- Interface consistency: every builder consumes a Path and writes a candidate Path; controller alone mutates deployed/state/source paths.
