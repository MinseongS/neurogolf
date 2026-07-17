# Task190 Route Sign Factor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace task190's dense route and repeated ray/colour factors with threshold-equivalent low-rank polynomials and adopt a bundled-exact cost1127 endpoint.

**Architecture:** Keep the adopted coordinate and spatial polynomial graph intact. Replace `poly_route[v,k,s]` by shared `F[x]=[1,x,x^2]` and `C[v,I,J]`, and replace the terminal `vks` operand with `kI,sJ,vIJ` while preserving backbone operand order.

**Tech Stack:** Python 3.13, NumPy, ONNX 1.21.0, ONNX Runtime 1.26.0, `uv run ng`.

## Global Constraints

- Candidate files stay under `candidates/task190/`.
- Preserve pinned ONNX/ORT versions.
- Use RED -> GREEN, bounded fresh-process runtime, full bundled `ng gate`, then `ng adopt`.
- Never copy into `submission/overfit_nets/` manually.
- Success requires fail0, cost<=1215, and candidate runtime<2s on one official example.

---

### Task 1: Algebra and candidate regression

**Files:**
- Create: `candidates/task190/test_route_sign_factor.py`
- Create: `candidates/task190/build_route_sign_factor.py`
- Produce: `candidates/task190/route_sign_factor.onnx`

**Interfaces:**
- Consumes: deployed cost1367 task190 graph and its terminal factorized Einsum.
- Produces: `route_features() -> np.ndarray[10,3]`, `route_core() -> np.ndarray[2,3,3]`, and a cost1215 candidate.

- [x] Write a failing test that imports the two factor functions, reconstructs route v0/v1,
  exhausts reachable `(paint,input,output,P)` signs, and requires the candidate file.
- [x] Run `PYTHONPATH=. uv run python candidates/task190/test_route_sign_factor.py`; expect
  `ModuleNotFoundError` or missing candidate.
- [x] Implement F/C and transform terminal `vks` into `kI,sJ,vIJ`; remove `poly_route`, run strict
  shape inference/checker, and save the candidate.
- [x] Run the test; expect algebra exact, one-example fail0, cost<=1215, elapsed<2s.

### Task 2: Runtime, gate, and adoption

**Files:**
- Modify after PASS: `src/custom/task190.py`, `state/manifest.json`, `state/tasks/task190.md`

- [x] Benchmark the candidate in three fresh processes with 10-second limits and record median.
- [x] Run `uv run ng gate --task 190 candidates/task190/route_sign_factor.onnx`; require 266/266 and cost1215.
- [x] Adopt only through `uv run ng adopt --task 190 ...`, regenerate exact source, rebuild task190,
  and re-score it.

### Task 3: Recursive record and verification

**Files:**
- Modify: `candidates/task190/DISCOVERY.md`
- Modify: `state/insights.yaml`
- Modify: `state/levers.yaml`
- Replace current truth in: `state/STATE.md`

- [x] Record the measured algebra/runtime/gate result and update the route-sign insight.
- [x] Rescan deployed terminal route tensors for dense `[V,10,10]` colour relations and list only
  byte-proven follow-ups.
- [x] Run regression, py_compile, YAML parse, exact-source rebuild, `ng score 178 184 190`, and
  `ng status`; then run scoped `git diff --check`.

### Task 4: Colour-branch sign factor

**Files:**
- Create: `candidates/task190/test_color_sign_factor.py`
- Create: `candidates/task190/build_color_sign_factor.py`
- Produce: `candidates/task190/color_sign_factor.onnx`

- [x] Write a failing test that exhausts `B+P*(A-B)` for paint1..9, input `{0,paint}`,
  output0..9, and all actual P values; require candidate cost<=1163.
- [x] Implement `A=0.5-(k-s)^2`, `B=(1/128)*(0.5-(k-c)^2)`, obtain colour features with
  `ArgMax(color_vec)->Gather(poly_route_features)`, and replace `vkxy` by `kK,bxyL,vKL`.
- [x] Run bounded three-process benchmark, bundled gate, and adopt only on fail0/cost<=1163.

### Task 5: Share the quadratic branch core

**Files:**
- Create: `candidates/task190/test_shared_branch_core.py`
- Create: `candidates/task190/build_shared_branch_core.py`
- Produce: `candidates/task190/shared_branch_core.onnx`

- [x] Write a failing test for shared bank `[E00,Q]`, swap selector, epsilon-scaled term branch,
  exhaustive output signs, and candidate cost<=1149.
- [x] Replace route/color cores36 params with one bank18 plus selector4; move epsilon into the
  v=1 column of `poly_term_branch`; update terminal operands together with subscripts.
- [x] Run bounded benchmark, full gate, and adopt only on fail0/cost<=1149.

### Task 6: Symmetric ray polynomial

**Files:**
- Create: `candidates/task190/test_symmetric_ray_factor.py`
- Create: `candidates/task190/build_symmetric_ray_factor.py`
- Produce: `candidates/task190/symmetric_ray_factor.onnx`

- [x] Write a failing test proving cap10 preserves every root on w0..9, reconstructing
  `w^2-(d+a)w+da`, and requiring candidate cost<=1127.
- [x] Replace target Cast/Pow nodes and 12 terminal operands by u8 Add/Mul/Concat, one fp32 Cast,
  shared spatial features, and two uses of a `[2,3,3]` symmetric core.
- [x] Run bounded benchmark, full gate, and adopt only on fail0/cost<=1127.
