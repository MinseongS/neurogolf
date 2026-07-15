# Task002/019/025 Residual Score Work Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exhaust every currently concrete strictly-cheaper rewrite for task002, task019, and task025, adopting every bundled-fail=0 win through `ng gate` and `ng adopt`.

**Architecture:** Run one isolated research lane per task under `candidates/taskNNN/`. Every lane starts with a static tensor-price oracle and a failing behavioral/cost regression test, then builds the smallest exact ONNX candidate. Main integrates only candidates that independently score below the live incumbent.

**Tech Stack:** Python 3.13, onnx==1.21.0, onnxruntime==1.26.0, NumPy, pytest, `uv run ng`.

## Global Constraints

- Never modify `submission/overfit_nets/` directly; adoption must be `ng gate` then `ng adopt`.
- Write scratch builders/tests/models only below `candidates/taskNNN/`.
- Bundled fail must be zero and cost must be strictly below the current deployed cost.
- Fresh evaluation is diagnostic; do not change the pinned ONNX/ORT versions.
- Do not touch unrelated dirty files from concurrent range sessions.
- A negative result must state the executed tool/date, exact verdict, reopen trigger, and falsification history in `state/levers.yaml` only.

---

### Task 1: Task002 FREE-output flood-tail fold

**Files:**
- Create: `candidates/task002/test_free_output_fold.py`
- Create: `candidates/task002/build_free_output_fold.py`
- Create: `candidates/task002/free_output_fold.onnx`
- Read: `state/tasks/task002.md`
- Read: `submission/overfit_nets/task002.onnx`

**Interfaces:**
- Consumes: current cost-5408 exact graph and its `W`, `t`, `mask30`, channel-3 input semantics.
- Produces: an exact candidate plus a price report; success requires cost below 5408, with the `+0.1` research target cost at most 4893.

- [ ] **Step 1: Write the failing test**

```python
def test_free_output_fold_is_exact_and_cheaper():
    candidate = build()
    assert bundled_differences(candidate, task=2) == []
    assert score(candidate, task=2)["cost"] < 5408
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest -q candidates/task002/test_free_output_fold.py`

Expected: FAIL because `build_free_output_fold.py` or the cheaper exact candidate does not yet exist.

- [ ] **Step 3: Implement the minimum exact fold**

Use the existing free fp32 input and terminal threshold contract to encode the three relevant channels directly: background score from the reached/fill state, green from input channel 3, and yellow from the unreached open mask. First price letter count and every counted operand. Do not build if the formula requires a counted replacement carrier of at least the deleted bytes.

- [ ] **Step 4: Verify GREEN or record the priced wall**

Run: `uv run pytest -q candidates/task002/test_free_output_fold.py`

Then, only if green: `uv run ng gate candidates/task002/free_output_fold.onnx --task 002`.

- [ ] **Step 5: Hand off exact measurements**

Report removed tensors, added parameters/intermediates, bundled failures, cost, and the precise reopen trigger if no candidate beats 5408.

---

### Task 2: Task019 two-feature and selector lowerings

**Files:**
- Create: `candidates/task019/test_residual_lowerings.py`
- Create: `candidates/task019/price_residual_lowerings.py`
- Create: `candidates/task019/build_two_feature_qconv.py`
- Create if price passes: `candidates/task019/build_hw_selector_einsum.py`
- Read: `candidates/task019/DISCOVERY.md`
- Read: `submission/overfit_nets/task019.onnx`

**Interfaces:**
- Consumes: current M/D/R runtime-weight QLinearConv graph at cost 1952.
- Produces: an exact candidate below 1952; the `+0.1` research target is cost at most 1766.

- [ ] **Step 1: Write the failing price/behavior tests**

```python
def test_two_feature_candidate_is_exact_and_cheaper():
    candidate = build_two_feature()
    assert bundled_differences(candidate, task=19) == []
    assert score(candidate, task=19)["cost"] < 1952

def test_hw_selector_price_can_reach_target():
    price = price_hw_selector()
    assert price.total_cost <= 1766
    assert price.selector_params <= 1500
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest -q candidates/task019/test_residual_lowerings.py`

Expected: FAIL because the new builders/price proof are absent or exceed their targets.

- [ ] **Step 3: Implement `E=R-5M` plus D control**

Encode `E` as a biased uint8 feature with x-zero-point 5. Use final channel scores `background=E-5D`, `cyan=7E+2D-8`, and `active_color=-E`, with the detected-colour runtime weights. Count the producer of E and the two-plane concat; retain only variants with total cost below 1952.

- [ ] **Step 4: Implement the H/W selector only after price proof**

Enumerate H,W in 2..6, compute the exact point-tiling selector rank/parameter count, and build a free-output N-ary Einsum only when the static total is at most 1766 and the terminal operand count stays within 52 letters and a locally runnable contraction plan.

- [ ] **Step 5: Verify candidates**

Run: `uv run pytest -q candidates/task019/test_residual_lowerings.py`.

For each green candidate run `uv run ng gate <candidate> --task 019`, followed by fresh 6000 diagnostics.

---

### Task 3: Task025 exact boolean fusion and residual capacity proof

**Files:**
- Create: `candidates/task025/test_line_position_fusion.py`
- Create: `candidates/task025/build_line_position_fusion.py`
- Create: `candidates/task025/price_k3_special_residual.py`
- Create only if proven: `candidates/task025/build_k3_special_residual.py`
- Read: `candidates/task025/DISCOVERY.md`
- Read: `submission/overfit_nets/task025.onnx`

**Interfaces:**
- Consumes: current shared-axis candidate at cost 8663.
- Produces: an exact micro candidate below 8663; a new `+0.1` candidate must cost at most 7838.

- [ ] **Step 1: Write the failing fusion test**

```python
def test_line_position_fusion_is_exact_and_cheaper():
    candidate = build()
    assert bundled_differences(candidate, task=25) == []
    assert score(candidate, task=25)["cost"] < 8663
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest -q candidates/task025/test_line_position_fusion.py`

Expected: FAIL because no fused candidate exists yet.

- [ ] **Step 3: Implement exact line-position fusion**

Replace `Cast(v_pos_b)+Cast(h_pos_b)` with a single mutually-exclusive boolean `Or` followed by one fp16 Cast. Derive the orientation scalar through the cheapest ORT-1.26-supported bool/u8 reduction. Keep `Where` data tensors fp16 because bool-data Where is unsupported.

- [ ] **Step 4: Prove or reject K3 special residual before building**

Inventory every residual position, colour, signed-side, adjacency, and factor tensor. A new +0.1 win requires the complete residual repair to add at most 176 cost over K3 cost 7662. Do not build the ONNX when the static lower bound exceeds 176.

- [ ] **Step 5: Verify and gate**

Run: `uv run pytest -q candidates/task025/test_line_position_fusion.py`.

If green: `uv run ng gate candidates/task025/line_position_fusion.onnx --task 025` and fresh threshold-output comparison.

---

### Task 4: Integrate, adopt, and generalize

**Files:**
- Modify on adoption: `src/custom/taskNNN.py` via `tools/live_to_exact_source.py --write-src`
- Auto-modify on adoption: `state/tasks/taskNNN.md`, `state/manifest.json`, `state/safe_manifest.json`, `state/SCOREBOARD.md`
- Modify for reusable wins: `state/insights.yaml`
- Modify at session end: `state/STATE.md`

**Interfaces:**
- Consumes: independently gated candidates from Tasks 1–3.
- Produces: adopted cheaper live nets, source parity, insight rescans, and a verified batch.

- [ ] **Step 1: Independently rerun every winning gate from main**

Run `uv run ng gate <candidate> --task NNN` and require pass/fail evidence in the current process.

- [ ] **Step 2: Adopt every PASS**

Run `uv run ng adopt <candidate> --task NNN --note "<exact mechanism>"`.

- [ ] **Step 3: Regenerate exact sources and rebuild**

Run `uv run python tools/live_to_exact_source.py --write-src NNN` and `PYTHONPATH=. uv run python tools/rebuild_networks_from_source.py --tasks NNN`.

- [ ] **Step 4: Register reusable mechanisms and rescan**

Update `state/insights.yaml`, run `uv run ng scan mask_dominance`, and run the available targeted syntactic scan across all 400 deployed models. The two recursive `reports/scripts` commands remain required when those files reappear.

- [ ] **Step 5: Verify and package safely**

Run the scoped pytest suite, isolated `ng score` for modified tasks, `uv run ng verify`, and `uv run ng verify --hash`. Pack/submit only after a complete 400/400 pass and after checking current Kaggle submissions.

- [ ] **Step 6: Commit owned files**

Stage only files owned by this work, then commit with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
