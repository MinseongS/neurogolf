# Tasks 032-048 Score Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and gate every concrete follow-up attack documented for task032, task035, task036, task041, task046, and task048, adopting every bundled-safe candidate that beats its current deployed cost.

**Architecture:** Each task owns a disjoint `candidates/taskNNN/` workspace. The task cycle is oracle/test RED, minimal builder GREEN, ONNX 1.21 strict inference/full checker, isolated `ng gate`, then root-owned `ng adopt`. Failed hypotheses remain task-local artifacts and receive a dated four-field lever ledger entry.

**Tech Stack:** Python 3 via `uv`, NumPy, ONNX 1.21.0, ONNX Runtime 1.26.0, NeuroGolf `ng` CLI.

## Global Constraints

- Default gate is bundled fail=0 and strictly cheaper; fresh evaluation is diagnostic.
- All candidate files remain under `candidates/taskNNN/`.
- Agents must not edit `submission/overfit_nets/`, manifest files, `STATE.md`, task ledgers, or `src/custom/`.
- Root performs every adoption through `uv run ng gate` followed by `uv run ng adopt`.
- Never feed uint8 or int8 to TopK; `src/neurogolf/topk.py` treats that as a Kaggle package-killer.
- Keep onnx==1.21.0 and onnxruntime==1.26.0.
- Preserve unrelated dirty files from concurrent sessions.

---

### Task 1: task032 sign-rank and CP collapse

**Files:**
- Create: `candidates/task032/test_rank2_oracle.py`
- Create: `candidates/task032/fit_rank2_sign.py`
- Create: `candidates/task032/build_rank2_or_cp.py`
- Read: `candidates/task032/DISCOVERY.md`

**Interfaces:**
- Consumes deployed cost588 rank-3 selector/channel/vertical factors.
- Produces `candidates/task032/rank2_sign.onnx` at cost452 or `cp_collapsed.onnx` at cost≤518.

- [ ] Write a test that asserts the rank-2 fitter returns bundled zero-error factors with positive on-logits and off-logits≤-0.05; initially fail because no fitter/result exists.
- [ ] Run `uv run python candidates/task032/test_rank2_oracle.py` and record the expected RED.
- [ ] Implement bounded multi-seed sign-loss fitting for selector `[6,30]`, channel basis `[10,10,2]`, and vertical `[2,6,6]`.
- [ ] If no rank-2 zero-error checkpoint exists, fit a CP representation whose parameter total is at most518.
- [ ] Build only a zero-error result, then run strict inference/full checker and `uv run ng gate <candidate> --task 32`.

### Task 2: task035 fp16 TopK perimeter packing

**Files:**
- Create: `candidates/task035/test_topk10_oracle.py`
- Create: `candidates/task035/build_topk10_fp16.py`
- Produce: `candidates/task035/topk10_fp16.onnx`

**Interfaces:**
- Consumes actual disk incumbent cost1891 and its 14-slot/28-write tail.
- Produces a 10-slot/20-write int64 ScatterND tail at cost≤1711.

- [ ] Write an oracle test proving bundled edit-count max is10 and a K=9 control fails exactly the count10 cases; initially fail on a missing K=10 implementation.
- [ ] Run the test and record RED.
- [ ] Cast `safe_name_67` to fp16, run TopK K=10, Gather raw row/column by the returned int64 indices, and mask zero ties to `(0,0)`.
- [ ] Assemble 20 writes and retain int64 ScatterND indices. Do not create uint8/int8 TopK or int32 ScatterND.
- [ ] Run the oracle GREEN, unsupported-TopK scan, checker/strict inference, actual isolated score, and `ng gate --task 35`. If cost exceeds1711, combine the documented joint-axis-basis parameter recovery.

### Task 3: task036 scalar coordinate decoding

**Files:**
- Create: `candidates/task036/test_power_coordinate_oracle.py`
- Create: `candidates/task036/build_power_coordinate.py`
- Produce: `candidates/task036/power_coordinate.onnx`

**Interfaces:**
- Consumes current cost940 graph and its selected blob channel.
- Produces exact min-row/min-column scalars with total candidate cost≤850.

- [ ] Write a bundled oracle comparing the current ArgMax coordinates against `sum(X*base^(29-position))` plus logarithmic decode; begin with a deliberately invalid equation so the test demonstrates RED.
- [ ] Implement base6, base8, and base16 NumPy decoders and require exact coordinates on all bundled examples.
- [ ] For an exact base, replace only yscore/xscore, ArgMax, and coordinate casts with FREE-input scalar Einsum plus Log/Div/Cast/Sub.
- [ ] Run GREEN, strict checker/inference, and `ng gate --task 36`.
- [ ] If every base is inexact, probe the shifted `[x,x²]` QLinearConv tail and build it only if combined cost≤850.

### Task 4: task041 restricted rank-9 interval classifier

**Files:**
- Create: `candidates/task041/test_rank9_truth.py`
- Create: `candidates/task041/fit_rank9.py`
- Create: `candidates/task041/build_rank9.py`
- Produce: `candidates/task041/rank9.onnx`

**Interfaces:**
- Consumes the 22 generator-supported endpoint pairs and current cost460 one-Einsum graph.
- Produces a rank-9 sign classifier at cost411, or a rank-9 plus shared-channel model at cost≤416.

- [ ] Enumerate the 22-pair×10-column truth table and assert a zero-error rank-9 result; initially fail because no checkpoint exists.
- [ ] Run RED, then fit rank9 using bounded multi-seed optimization with checkpoints.
- [ ] Build ONNX only from zero-error factors; never use simple truncation or the old 30×30 triangular candidate.
- [ ] If rank9 position fitting fails, test shared analytic channel code at rank10, then combine only if the final cost≤416.
- [ ] Run GREEN, checker/strict inference, time-bounded isolated gate, and diagnostic fresh comparison.

### Task 5: task046 two-feature decoder plus stable compaction

**Files:**
- Create: `candidates/task046/test_qlinear_truth.py`
- Create: `candidates/task046/probe_compaction_dtypes.py`
- Create: `candidates/task046/build_qlinear_compact.py`
- Produce: `candidates/task046/qlinear_compact.onnx`

**Interfaces:**
- Consumes current cost1846 polynomial decoder and TopK-based 20-column compaction.
- Produces an exact candidate at cost≤1670.

- [ ] Write an exhaustive label0..11 test requiring `[L,L²]` QLinearConv+bias output to be positive only for its decode label; initially fail without quantization parameters.
- [ ] Search scalar scales/zero-points and verify the truth table GREEN before editing ONNX.
- [ ] Probe ScatterElements/Gather index dtypes and build a prefix-rank stable-compaction oracle for all bundled width16 cases.
- [ ] Combine the two-feature decoder with compaction only when static pricing is≤1670; stop if an int64 20-vector removes the saving.
- [ ] Run checker/inference, bundled A/B, `ng gate --task 46`, width16 controls, and fresh3000 diagnostics.

### Task 6: task048 packed-flood tail and rank-1 entry

**Files:**
- Create: `candidates/task048/test_tail_depth.py`
- Create: `candidates/task048/build_tail_rewire.py`
- Create: `candidates/task048/probe_rank1_entry.py`
- Produce: `candidates/task048/tail_two_removed.onnx` or `rank1_entry.onnx`

**Interfaces:**
- Consumes current cost622 packed uint8 flood graph.
- Produces a bundled-safe candidate at cost≤562.

- [ ] Write a bundled oracle that compares final reach against the earlier `safe_name_72` and `safe_name_82` states; initially fail for at least one premature state.
- [ ] Build two explicit rewires and remove dead nodes/initializers. The two-block candidate is primary; one-block cost566 is below the score threshold.
- [ ] Run checker/inference and `ng gate --task 48`. If two-block removal passes at cost≤562, run deployed-vs-candidate fresh5000 and record added long-path failures.
- [ ] If it fails, micro-probe a single-op lowering of rank-1 `W[c,j]=C[c]P[j]`; reject any design that creates a counted 8×8 intermediate or 8×30 selector.
- [ ] If both fail, synthesize a lower-temporary uint8 Shift/Or/And transition against an exhaustive 8-bit-row oracle and gate only candidates priced≤562.

### Task 7: Root integration

**Files:**
- Modify only after PASS: `src/custom/taskNNN.py`, `state/tasks/taskNNN.md`, `state/manifest.json`, `state/safe_manifest.json`, `state/SCOREBOARD.md`
- Replace at session end: `state/STATE.md`
- Update after mechanism result: `state/insights.yaml` or `state/levers.yaml`

- [ ] Independently inspect every agent artifact and rerun its task test, checker/strict inference, and isolated gate.
- [ ] Adopt each PASS with `uv run ng adopt`; regenerate exact source with `tools/live_to_exact_source.py` and run the one-task pipeline.
- [ ] Record each bounded negative result as a four-field dated lever entry with a concrete reopen trigger and falsification history.
- [ ] Rerun the available board-wide insight scans; document that the recursive inventory scripts are absent if still unavailable.
- [ ] When concurrent sessions are quiescent, run complete verification, check Kaggle submissions, then `ng pack` and `ng submit` as one batch.
