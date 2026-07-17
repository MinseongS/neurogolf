# Task 209 Winner-Preserving Selector Compression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. This is an explicitly solo task209 session; do not dispatch subagents. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and conditionally adopt a task209 selector replacement with cost at most 5509, bundled fail=0, and zero divergence from the cost6089 incumbent on a deterministic fresh 5000 holdout.

**Architecture:** Treat the adopted selector as a teacher and expose its `Bw`, `Lc`, `rowscore4`, `colscore4`, `ssel_i`, `kbest_i`, and `jbest_i` tensors. Search only lowering-aware one- and two-plane uint8 feature codes produced by broadcast `BitwiseAnd`, simulate the exact quotient-pool/QLinearConv arithmetic including saturation and first-tie behavior, and price the tensor graph before constructing the full candidate. If this family is infeasible, try a bounded modular classifier over existing scalar geometry; build no full ONNX artifact unless its teacher agreement and static price both pass.

**Tech Stack:** Python 3.13, NumPy, SciPy `milp`, ONNX 1.21.0, ONNX Runtime 1.26.0, pytest, and the repository `ng`/scoring tools.

## Global Constraints

- Work only on task209; do not spawn subagents.
- Keep candidate, generated corpus, search result, micro-model, and scratch files under `candidates/task209/`.
- Do not repeat QA/QS splitting, uint8 geometry/index recasting, full-input uint8 recasting, sparse initializers, or the completed terminal output fold.
- Keep `onnx==1.21.0` and `onnxruntime==1.26.0` unchanged.
- Qualifying cost is at most 5509, measured as memory plus parameters.
- Qualifying correctness is bundled fail=0 and fresh 5000 candidate-vs-incumbent divergence=0.
- Preserve first-index tie behavior for scale, row phase, and column phase.
- Never edit `submission/overfit_nets/task209.onnx` directly; adoption is `uv run ng gate` followed by `uv run ng adopt` only.
- Do not run `ng pack` or `ng submit`.
- Preserve unrelated dirty files and stage only task209-owned integration files.

## File Map

- Create `candidates/task209/selector_oracle.py`: teacher tensor extraction, deterministic corpus generation, exact NumPy selector simulation, and seeded candidate/incumbent comparison.
- Create `candidates/task209/test_selector_oracle.py`: tie, quotient, quantization, and deterministic split tests.
- Create `candidates/task209/search_selector_basis.py`: exhaustive lowering-aware rank-1/rank-2 mask search and JSON evidence.
- Create `candidates/task209/test_selector_basis.py`: search ordering and zero-violation acceptance tests.
- Create `candidates/task209/build_selector_compressed.py`: selector micro-model, static price ledger, and SHA-pinned full candidate rewrite.
- Create `candidates/task209/test_selector_compressed.py`: ORT arithmetic parity, price, bundled, and persistent-artifact tests.
- Conditionally create `candidates/task209/search_selector_classifier.py`: modular scalar fallback when the mask search cannot qualify.
- Conditionally create `candidates/task209/test_selector_classifier.py`: collision and exact modular-rule tests.
- Modify after a successful adoption only: `src/custom/task209.py`, `candidates/task209/DISCOVERY.md`, `state/insights.yaml`, and `state/STATE.md`.

---

### Task 1: Deterministic Teacher Corpus and Exact Selector Oracle

**Files:**
- Create: `candidates/task209/selector_oracle.py`
- Create: `candidates/task209/test_selector_oracle.py`
- Generate: `candidates/task209/selector_bundled.npz`
- Generate: `candidates/task209/selector_discovery.npz`

**Interfaces:**
- Consumes: deployed SHA `eae4bcd1fb864d2b22acaad721f48e6c4f04d13a29fa3a24ddc02021bd3c9a0d`, task209 JSON examples, and arc generator `task_8a004b2b`.
- Produces: `axis_color_components(box_profile: np.ndarray, sprite_profile: np.ndarray) -> np.ndarray`, `simulate_selector_scores(box: np.ndarray, sprite: np.ndarray, box_masks: tuple[int, ...], sprite_masks: tuple[int, ...], score_scale: int) -> tuple[np.ndarray, np.ndarray]`, `simulate_selector(box: np.ndarray, sprite: np.ndarray, box_masks: tuple[int, ...], sprite_masks: tuple[int, ...], score_scale: int) -> tuple[int, int, int]`, `winner_signature(row_scores: np.ndarray, col_scores: np.ndarray) -> np.ndarray`, `collect_split(split: str, count: int) -> dict[str, np.ndarray]`, and compressed arrays `box`, `sprite`, `teacher_row_components`, `teacher_col_components`, `teacher_rows`, `teacher_cols`, `teacher_signature`, and `teacher`.

- [ ] **Step 1: Write failing unit tests for tie order and quantized selector arithmetic**

  Add tests with these exact assertions:

  ```python
  import numpy as np

  from selector_oracle import (
      first_selector_winner,
      qlinear_u8,
      simulate_selector,
      winner_signature,
  )


  def test_first_selector_winner_preserves_first_ties():
      row = np.array([[5, 5, 1], [4, 4, 4], [3, 2, 1]], dtype=np.int32)
      col = np.array([[2, 2, 0], [3, 3, 3], [4, 1, 0]], dtype=np.int32)
      assert first_selector_winner(row, col) == (0, 0, 0)


  def test_qlinear_u8_uses_nearest_even_and_saturates():
      values = np.array([0, 1, 2, 3, 509, 5100], dtype=np.int32)
      assert np.array_equal(
          qlinear_u8(values, 2),
          np.array([0, 0, 1, 2, 254, 255], dtype=np.uint8),
      )


  def test_zero_grid_has_deterministic_zero_winner():
      grid = np.zeros((12, 12), dtype=np.uint8)
      sprite = np.zeros((3, 5), dtype=np.uint8)
      assert simulate_selector(grid, sprite, (1,), (1,), 1) == (0, 0, 0)


  def test_signature_records_every_phase_winner_before_scale():
      row = np.array([[5, 5, 1], [4, 1, 4], [3, 2, 3]], dtype=np.int32)
      col = np.array([[2, 2, 0], [3, 1, 3], [4, 1, 4]], dtype=np.int32)
      assert np.array_equal(
          winner_signature(row, col),
          np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.uint8),
      )
  ```

- [ ] **Step 2: Run the focused test and confirm RED**

  Run:

  ```bash
  PYTHONPATH=.:candidates/task209 uv run pytest candidates/task209/test_selector_oracle.py -q
  ```

  Expected: collection fails because `selector_oracle` does not exist.

- [ ] **Step 3: Implement the arithmetic oracle**

  Implement the public arithmetic with these formulas:

  ```python
  DIVISORS = (2, 3, 4)
  DISCOVERY_SEED = 209_000
  HOLDOUT_SEED = 209_001
  DISCOVERY_COUNT = 20_000
  HOLDOUT_COUNT = 5_000


  def qlinear_u8(accumulator: np.ndarray, output_scale: int) -> np.ndarray:
      rounded = np.rint(np.asarray(accumulator, dtype=np.float64) / output_scale)
      return np.clip(rounded, 0, 255).astype(np.uint8)


  def axis_scores(box_profile: np.ndarray, sprite_profile: np.ndarray,
                  score_scale: int) -> np.ndarray:
      scores = np.zeros((3, 3), dtype=np.uint8)
      for scale_index, divisor in enumerate(DIVISORS):
          bins = box_profile.reshape(
              box_profile.shape[0], 12 // divisor, divisor
          ).sum(axis=2, dtype=np.int32)
          for phase in range(3):
              width = min(bins.shape[1], 5 - phase)
              accumulator = np.sum(
                  bins[:, :width].astype(np.int32)
                  * sprite_profile[:, phase:phase + width].astype(np.int32),
                  dtype=np.int32,
              )
              scores[scale_index, phase] = qlinear_u8(accumulator, score_scale)
      return scores


  def axis_color_components(box_profile: np.ndarray,
                            sprite_profile: np.ndarray) -> np.ndarray:
      components = np.zeros((3, 3, 4), dtype=np.uint8)
      for scale_index, divisor in enumerate(DIVISORS):
          bins = box_profile.reshape(4, 12 // divisor, divisor).sum(axis=2)
          for phase in range(3):
              width = min(bins.shape[1], 5 - phase)
              components[scale_index, phase] = np.sum(
                  bins[:, :width] * sprite_profile[:, phase:phase + width],
                  axis=1,
              )
      return components


  def first_selector_winner(row_scores: np.ndarray,
                            col_scores: np.ndarray) -> tuple[int, int, int]:
      row_phase = np.argmax(row_scores, axis=1)
      col_phase = np.argmax(col_scores, axis=1)
      scales = np.arange(3)
      scale_scores = row_scores[scales, row_phase].astype(np.int32)
      scale_scores += col_scores[scales, col_phase].astype(np.int32)
      scale = int(np.argmax(scale_scores))
      return scale, int(row_phase[scale]), int(col_phase[scale])


  def winner_signature(row_scores: np.ndarray,
                       col_scores: np.ndarray) -> np.ndarray:
      row_phase = np.argmax(row_scores, axis=1).astype(np.uint8)
      col_phase = np.argmax(col_scores, axis=1).astype(np.uint8)
      scale, _, _ = first_selector_winner(row_scores, col_scores)
      return np.concatenate((np.array([scale], dtype=np.uint8), row_phase, col_phase))


  def simulate_selector_scores(box: np.ndarray, sprite: np.ndarray,
                               box_masks: tuple[int, ...],
                               sprite_masks: tuple[int, ...],
                               score_scale: int) -> tuple[np.ndarray, np.ndarray]:
      bm = np.bitwise_and(
          np.asarray(box, dtype=np.uint8)[None, :, :],
          np.asarray(box_masks, dtype=np.uint8)[:, None, None],
      )
      lm = np.bitwise_and(
          np.asarray(sprite, dtype=np.uint8)[None, :, :],
          np.asarray(sprite_masks, dtype=np.uint8)[:, None, None],
      )
      box_rows = bm.max(axis=2)
      box_cols = bm.max(axis=1)
      sprite_rows = np.pad(lm.max(axis=2), ((0, 0), (0, 2)))
      sprite_cols = lm.max(axis=1)
      return (
          axis_scores(box_rows, sprite_rows, score_scale),
          axis_scores(box_cols, sprite_cols, score_scale),
      )


  def simulate_selector(box: np.ndarray, sprite: np.ndarray,
                        box_masks: tuple[int, ...],
                        sprite_masks: tuple[int, ...],
                        score_scale: int) -> tuple[int, int, int]:
      return first_selector_winner(
          *simulate_selector_scores(
              box, sprite, box_masks, sprite_masks, score_scale
          )
      )
  ```

  Add an ORT debug-session helper that appends `Bw`, `Lc`, `rowscore4`, `colscore4`, `ssel_i`, `kbest_i`, and `jbest_i` as outputs after checking the deployed SHA. Derive four one-hot presence profiles from `Bw` and `Lc`, record `axis_color_components` for both axes, and assert their channel sums equal the incumbent score tensors. Reverse the phase axis of `rowscore4` and `colscore4` into normal phase order, compute `teacher_signature`, and assert its final-scale phases equal the three deployed winner outputs. For bundled data, iterate `train + test + arc-gen`. For discovery data, set both `random.seed(DISCOVERY_SEED)` and `np.random.seed(DISCOVERY_SEED)`, generate until exactly 20,000 valid one-hot examples have been collected, and save:

  ```python
  np.savez_compressed(
      output_path,
      box=np.stack(boxes).astype(np.uint8),
      sprite=np.stack(sprites).astype(np.uint8),
      teacher_row_components=np.stack(row_components).astype(np.uint8),
      teacher_col_components=np.stack(col_components).astype(np.uint8),
      teacher_rows=np.stack(row_scores).astype(np.uint8),
      teacher_cols=np.stack(col_scores).astype(np.uint8),
      teacher_signature=np.stack(signatures).astype(np.uint8),
      teacher=np.stack(winners).astype(np.uint8),
      seed=np.array(seed, dtype=np.int64),
      attempts=np.array(attempts, dtype=np.int64),
  )
  ```

- [ ] **Step 4: Run the unit tests and generate both corpora**

  Run:

  ```bash
  PYTHONPATH=.:candidates/task209 uv run pytest candidates/task209/test_selector_oracle.py -q
  PYTHONPATH=.:candidates/task209 uv run python candidates/task209/selector_oracle.py --split bundled --count 266 --output candidates/task209/selector_bundled.npz
  PYTHONPATH=.:candidates/task209 uv run python candidates/task209/selector_oracle.py --split discovery --count 20000 --output candidates/task209/selector_discovery.npz
  ```

  Expected: tests pass; bundled arrays have leading dimension 266; discovery arrays have leading dimension 20000; teacher and signature values stay in `[0, 2]`; every signature agrees with the deployed winner; metadata records seed 209000 for discovery.

- [ ] **Step 5: Checkpoint the ignored scratch state**

  Run `git status --short --ignored candidates/task209 | tail -20`. Expected: new files are ignored under the repository-wide `/candidates/` rule, and no unrelated file is staged.

---

### Task 2: Lowering-Aware Rank-1 and Rank-2 Mask Search

**Files:**
- Create: `candidates/task209/search_selector_basis.py`
- Create: `candidates/task209/test_selector_basis.py`
- Generate: `candidates/task209/selector_search.json`

**Interfaces:**
- Consumes: the two corpus NPZ files and `simulate_selector`.
- Produces: `MaskPlan(rank: int, box_masks: tuple[int, ...], sprite_masks: tuple[int, ...], score_scale: int, violations: int)`, `order_violations(scores: np.ndarray, winner: int) -> int`, and JSON with the first zero-violation plan or an explicit infeasible result.

- [ ] **Step 1: Write failing tests for search order and acceptance**

  ```python
  import numpy as np

  from search_selector_basis import (
      MaskPlan,
      candidate_plans,
      evaluate_plan,
      order_violations,
  )


  def test_search_visits_rank_one_before_rank_two():
      plans = candidate_plans()
      assert next(plans).rank == 1


  def test_evaluate_plan_counts_exact_tuple_mismatches():
      boxes = np.zeros((2, 12, 12), dtype=np.uint8)
      sprites = np.zeros((2, 3, 5), dtype=np.uint8)
      teacher = np.zeros((2, 7), dtype=np.uint8)
      plan = MaskPlan(1, (1,), (1,), 1, -1)
      assert evaluate_plan(plan, boxes, sprites, teacher) == 0


  def test_order_violations_distinguishes_earlier_and_later_ties():
      assert order_violations(np.array([4, 4, 3]), 0) == 0
      assert order_violations(np.array([4, 4, 3]), 1) == 1
  ```

- [ ] **Step 2: Run the focused test and confirm RED**

  Run:

  ```bash
  PYTHONPATH=.:candidates/task209 uv run pytest candidates/task209/test_selector_basis.py -q
  ```

  Expected: collection fails because `search_selector_basis` does not exist.

- [ ] **Step 3: Implement deterministic candidate enumeration**

  Use masks 1 through 15 and score scales `(1, 2, 4, 8, 16, 32, 64)`. Rank-1 plans are enumerated first. Rank-2 plans use ordered channel pairs but canonicalize simultaneous channel swaps:

  ```python
  from dataclasses import dataclass
  import itertools


  @dataclass(frozen=True)
  class MaskPlan:
      rank: int
      box_masks: tuple[int, ...]
      sprite_masks: tuple[int, ...]
      score_scale: int
      violations: int


  def candidate_plans():
      scales = (1, 2, 4, 8, 16, 32, 64)
      for box_mask in range(1, 16):
          for sprite_mask in range(1, 16):
              for scale in scales:
                  yield MaskPlan(1, (box_mask,), (sprite_mask,), scale, -1)
      for box_masks in itertools.combinations_with_replacement(range(1, 16), 2):
          for sprite_masks in itertools.product(range(1, 16), repeat=2):
              paired = tuple(zip(box_masks, sprite_masks))
              if paired > paired[::-1]:
                  continue
              for scale in scales:
                  yield MaskPlan(2, box_masks, sprite_masks, scale, -1)
  ```

  `order_violations` must enforce `winner > earlier competitor` and `winner >= later competitor`. For every example, apply it to all three row vectors, all three column vectors, and the three scale totals formed from their first-winning phases. This produces the seven-value signature and makes tie semantics explicit. Evaluate every plan on bundled first. Only bundled-zero plans advance to discovery, processed in 256-example batches with immediate exit on the first violation. Write JSON fields `status`, `rank`, `box_masks`, `sprite_masks`, `score_scale`, `bundled_violations`, `discovery_violations`, `ordered_constraints`, and `searched_plans`.

- [ ] **Step 4: Run tests and the exhaustive search**

  Run:

  ```bash
  PYTHONPATH=.:candidates/task209 uv run pytest candidates/task209/test_selector_basis.py -q
  PYTHONPATH=.:candidates/task209 uv run python candidates/task209/search_selector_basis.py --bundled candidates/task209/selector_bundled.npz --discovery candidates/task209/selector_discovery.npz --output candidates/task209/selector_search.json
  ```

  Expected: tests pass. A qualifying mask result has `status: exact`, both violation counts zero, and rank 1 or 2. If the result is `status: infeasible`, skip Task 3 and execute Task 4.

---

### Task 3: ORT Micro-Proof, Static Price Gate, and Full Candidate

**Files:**
- Create: `candidates/task209/build_selector_compressed.py`
- Create: `candidates/task209/test_selector_compressed.py`
- Generate: `candidates/task209/selector_micro.onnx`
- Generate after the price gate only: `candidates/task209/selector_compressed.onnx`

**Interfaces:**
- Consumes: an exact `selector_search.json`.
- Produces: `load_plan(path: Path) -> MaskPlan`, `build_selector_micro(plan: MaskPlan) -> onnx.ModelProto`, `estimate_full_cost(plan: MaskPlan) -> dict[str, int]`, and `build_candidate(plan: MaskPlan) -> onnx.ModelProto`.

- [ ] **Step 1: Write failing tests for static price and ORT parity**

  Tests must load the exact search plan and assert:

  ```python
  from pathlib import Path

  SEARCH_RESULT = Path("candidates/task209/selector_search.json")
  DISCOVERY_CORPUS = Path("candidates/task209/selector_discovery.npz")


  def test_static_price_reaches_strict_boundary():
      plan = load_plan(SEARCH_RESULT)
      ledger = estimate_full_cost(plan)
      assert ledger["incumbent_cost"] == 6089
      assert ledger["predicted_cost"] <= 5509


  def test_micro_model_matches_numpy_on_corpus_prefix():
      plan = load_plan(SEARCH_RESULT)
      session = ort.InferenceSession(
          build_selector_micro(plan).SerializeToString(),
          providers=["CPUExecutionProvider"],
      )
      corpus = np.load(DISCOVERY_CORPUS)
      for box, sprite in zip(corpus["box"][:512], corpus["sprite"][:512]):
          actual = tuple(int(v) for v in session.run(
              None, {"Bw": box[None, None], "Lc": sprite[None, None]}
          )[0])
          assert actual == simulate_selector(
              box, sprite, plan.box_masks, plan.sprite_masks, plan.score_scale
          )
  ```

- [ ] **Step 2: Run the focused test and confirm RED**

  Run:

  ```bash
  PYTHONPATH=.:candidates/task209 uv run pytest candidates/task209/test_selector_compressed.py -q
  ```

  Expected: collection fails because `build_selector_compressed` does not exist.

- [ ] **Step 3: Implement the selector micro-model and price ledger**

  Load only an exact zero-violation search result:

  ```python
  def load_plan(path: Path) -> MaskPlan:
      payload = json.loads(path.read_text())
      if payload["status"] != "exact":
          raise ValueError(f"selector plan is not exact: {payload['status']}")
      if payload["bundled_violations"] or payload["discovery_violations"]:
          raise ValueError("selector plan contains recorded violations")
      return MaskPlan(
          rank=int(payload["rank"]),
          box_masks=tuple(int(v) for v in payload["box_masks"]),
          sprite_masks=tuple(int(v) for v in payload["sprite_masks"]),
          score_scale=int(payload["score_scale"]),
          violations=0,
      )
  ```

  The micro-model replaces `Equal(Bw, colorvals)` and `Equal(Lc, colorvals)` with broadcast `BitwiseAnd` against initializers shaped `[1, rank, 1, 1]`. It then uses the incumbent reduction, divisor `(2,3,4)` quotient pools, dynamic correlation, and ArgMax order. Pool weights have shapes `[rank,1,divisor,1]` and `[rank,1,1,divisor]`; dynamic correlation output uses initializer `selector_score_scale` for its output scale.

  Price only these changed tensors and parameters before a full model exists:

  ```python
  OLD_TENSORS = {
      "BmCell", "BmRowHist_b", "BmColHist_b", "BmRowHist", "BmColHist",
      "LmCell", "LmRow_b", "LmCol_b", "LmRow3", "LmCol", "LmRow",
      "row_bins2", "col_bins2", "row_bins3", "col_bins3",
      "row_bins4", "col_bins4",
  }
  OLD_INITIALIZERS = {
      "colorvals", "row_pool_w2", "col_pool_w2", "row_pool_w3",
      "col_pool_w3", "row_pool_w4", "col_pool_w4",
  }
  ```

  Infer the micro-model shapes, sum every replacement tensor byte and initializer element, and return `6089 - old_memory - old_params + new_memory + new_params`. Reject the plan before full construction unless this value is at most 5509.

- [ ] **Step 4: Run micro-tests and enforce the static price gate**

  Run:

  ```bash
  PYTHONPATH=.:candidates/task209 uv run pytest candidates/task209/test_selector_compressed.py -q -k 'static_price or micro_model'
  PYTHONPATH=.:candidates/task209 uv run python candidates/task209/build_selector_compressed.py --plan candidates/task209/selector_search.json --ort-verify-all --bundled candidates/task209/selector_bundled.npz --discovery candidates/task209/selector_discovery.npz
  ```

  Expected: ORT matches the NumPy oracle on all 512 examples and predicted cost is at most 5509. Then run `build_selector_compressed.py --ort-verify-all` over all 266 bundled and 20,000 discovery states; it must report zero score, winner, padding, saturation, and requantization mismatches. If any assertion fails, do not build the full candidate; execute Task 4.

- [ ] **Step 5: Implement and build the full SHA-pinned rewrite**

  Pin the input SHA to `eae4bcd1fb864d2b22acaad721f48e6c4f04d13a29fa3a24ddc02021bd3c9a0d`. Copy the incumbent prefix through the `Bw`/`Lc` geometry, replace only the selector nodes and their obsolete initializers, then preserve the existing geometry-after-selector and terminal output fold byte-for-byte. End with checker plus strict shape inference:

  ```python
  onnx.checker.check_model(model, full_check=True)
  return onnx.shape_inference.infer_shapes(model, strict_mode=True)
  ```

  Build with:

  ```bash
  PYTHONPATH=.:candidates/task209 uv run python candidates/task209/build_selector_compressed.py --plan candidates/task209/selector_search.json --output candidates/task209/selector_compressed.onnx
  ```

- [ ] **Step 6: Measure the real artifact and complete focused tests**

  Run:

  ```bash
  PYTHONPATH=.:candidates/task209 uv run pytest candidates/task209/test_selector_compressed.py -q
  uv run python -m neurogolf.scoring candidates/task209/selector_compressed.onnx 209
  ```

  Expected: all focused tests pass, bundled fail=0, and actual memory plus parameters is at most 5509. A larger actual cost rejects this route and sends execution to Task 4 without gate or adoption.

---

### Task 4: Conditional Scalar Geometry Classifier Pivot

**Files:**
- Create only after a Task 2 or Task 3 rejection: `candidates/task209/search_selector_classifier.py`
- Create only after a Task 2 or Task 3 rejection: `candidates/task209/test_selector_classifier.py`
- Generate: `candidates/task209/selector_classifier_search.json`
- Reuse: `candidates/task209/build_selector_compressed.py`

**Interfaces:**
- Consumes: the same teacher corpus plus scalar debug outputs `Hm`, `Wm`, `cr0b`, and `cc0b` added by `selector_oracle.py`.
- Produces: `deduplicate_labeled_rows(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]`, `solve_modular(features: np.ndarray, labels: np.ndarray) -> tuple[int, np.ndarray] | None`, three modular affine rules `output = (bias + features @ weights) % 3`, or an explicit infeasible result.

- [ ] **Step 1: Write failing fallback tests**

  ```python
  import numpy as np
  import pytest

  from search_selector_classifier import deduplicate_labeled_rows, solve_modular


  def test_conflicting_duplicate_feature_rows_are_rejected():
      features = np.array([[1, 2], [1, 2]], dtype=np.int32)
      labels = np.array([0, 1], dtype=np.int32)
      with pytest.raises(ValueError, match="conflicting labels"):
          deduplicate_labeled_rows(features, labels)


  def test_modular_solver_recovers_identity_modulo_three():
      features = np.array([[1, 0], [1, 1], [1, 2]], dtype=np.int32)
      labels = np.array([0, 1, 2], dtype=np.int32)
      solution = solve_modular(features, labels)
      assert solution is not None
      bias, weights = solution
      assert np.array_equal((bias + features @ weights) % 3, labels)
  ```

  Run:

  ```bash
  PYTHONPATH=.:candidates/task209 uv run pytest candidates/task209/test_selector_classifier.py -q
  ```

  Expected: collection fails because `search_selector_classifier` does not exist.

- [ ] **Step 2: Extend corpus extraction with exact scalar features**

  Save columns in this exact order:

  ```python
  features = np.array(
      [1, Hm, Wm, cr0b, cc0b, Hm % 4, Wm % 4, cr0b % 4, cc0b % 4],
      dtype=np.int32,
  )
  ```

  Regenerate bundled and discovery corpora with the same seeds and assert that `box`, `sprite`, and `teacher` hashes are unchanged.

- [ ] **Step 3: Implement bounded integer feasibility with SciPy MILP**

  Deduplicate feature rows and reject immediately if one feature row has two teacher labels for the same output. For each of the three outputs, solve the exact equalities

  `features @ weights + bias - 3 * quotient = teacher`

  with integer weights and bias bounded to `[-8,8]`; quotient variables are unbounded integers. Accept only solver status 0 and then re-evaluate the rule over all 20,266 discovery examples. Write the three biases, weight vectors, violation counts, and solver status to JSON. Run the fallback test again and require both tests to pass.

- [ ] **Step 4: Price the modular ONNX lowering before construction**

  Lower the accepted rules using only existing scalar tensors, `Mod`, `Mul`, `Add`, and uint8 casts. Compute the same full-cost ledger and require predicted cost at most 5509. If no exact rule exists or the price is too high, terminate candidate construction and record the dated four-field negative lever entry with:

  - ran: rank-1/rank-2 mask search on 266 bundled plus 20,000 discovery, ORT micro-tests when applicable, and modular MILP fallback;
  - verdict: no qualifying cost5509 candidate found by these concrete families;
  - reopen: a new compact set-union primitive, a selector-free positioning formula, or an independent low-rank factor oracle;
  - falsification history: prior NeuroGolf floor claims have been disproved, so this is family-local evidence and not a permanent floor.

- [ ] **Step 5: Build only an exact and statically qualifying classifier**

  Reuse the candidate builder's prefix and unchanged downstream renderer. Run the same focused tests and cost command from Task 3. Continue only if actual cost is at most 5509 and bundled fail=0.

  The combined partial-fold route from the design remains closed unless this task first produces a proved selector replacement that misses 5509 only on price. Do not improvise a one-plane renderer. If that narrow condition occurs, report the exact shortfall and amend the design before changing the already fresh-proven terminal decoder.

---

### Task 5: Untouched Holdout, Official Gate, and Conditional Adoption

**Files:**
- Consume: `candidates/task209/selector_compressed.onnx`
- Modify through `ng adopt` only: `submission/overfit_nets/task209.onnx`, `state/manifest.json`, and `state/tasks/task209.md`

**Interfaces:**
- Consumes: an actual cost-at-most-5509 bundled-clean candidate.
- Produces: either an adopted task209 deployment or an unchanged deployment plus concrete failure evidence.

- [ ] **Step 1: Run the full task209 regression suite**

  ```bash
  PYTHONPATH=.:candidates/task209 uv run pytest candidates/task209/test_u8_qsplit.py candidates/task209/test_output_fold.py candidates/task209/test_selector_oracle.py candidates/task209/test_selector_basis.py candidates/task209/test_selector_compressed.py -q
  ```

  Expected: every test passes.

- [ ] **Step 2: Run deterministic fresh 5000 A/B in a new process**

  Extend `selector_oracle.py` with `--compare` so it seeds both RNGs with 209001, generates exactly 5000 valid examples, and emits JSON counts for incumbent failures, candidate failures, and candidate-vs-incumbent divergence. Run:

  ```bash
  PYTHONPATH=.:candidates/task209 uv run python candidates/task209/selector_oracle.py --compare candidates/task209/selector_compressed.onnx --count 5000 --seed 209001
  ```

  Expected: `runs=5000` and `candidate_vs_incumbent=0`. Inherited incumbent and candidate ground-truth failures may be nonzero but must be equal case-for-case.

- [ ] **Step 3: Run the mandatory official gate**

  ```bash
  uv run ng gate candidates/task209/selector_compressed.onnx --task 209
  ```

  Expected: PASS, bundled fail=0, and cost at most 5509. Do not continue after any other result.

- [ ] **Step 4: Adopt through the normal command**

  ```bash
  uv run ng adopt candidates/task209/selector_compressed.onnx --task 209 --note "replace four-color selector planes with winner-preserving compact feature code"
  ```

  Expected: adoption from cost6089 to cost at most5509 and automatic task-ledger stamping.

---

### Task 6: Exact Source, Discovery, Insight, and Full Rescan

**Files:**
- Modify: `src/custom/task209.py`
- Modify: `candidates/task209/DISCOVERY.md`
- Modify: `state/insights.yaml`
- Modify in place, never append: `state/STATE.md`
- Generate: `networks/task209.onnx`
- Refresh: `candidates/task209/rescan_adopted_insights.json`

**Interfaces:**
- Consumes: the adopted candidate SHA, measured cost, gate result, and fresh result.
- Produces: byte-identical source rebuild, synchronized documentation/insight, and a completed 400-task rescan.

- [ ] **Step 1: Port the accepted selector into exact source**

  Replace only the four-color selector block in `src/custom/task209.py`; preserve geometry and the terminal parabolic decoder. Encode the accepted masks, score scale, rank-sized pool weights, `BitwiseAnd` nodes, reductions, quotient pools, and first-tie ArgMax nodes exactly as in the adopted artifact.

- [ ] **Step 2: Rebuild source and prove byte identity**

  ```bash
  PYTHONPATH=. uv run python tools/rebuild_networks_from_source.py --tasks 209
  shasum -a 256 candidates/task209/selector_compressed.onnx submission/overfit_nets/task209.onnx networks/task209.onnx
  uv run ng score 209
  uv run python tools/per_tensor_cost.py 209
  ```

  Expected: all three SHA values are identical; isolated score reports bundled fail=0 and cost at most5509.

- [ ] **Step 3: Synchronize task discovery and reusable insight**

  Update `DISCOVERY.md` with the mask/classifier formula, exact tensor savings, bundled result, fresh seed/count/divergence, gate/adopt result, SHA, and next +0.1 threshold. Update `state/insights.yaml` with the reusable fingerprint: replace broadcast per-color equality planes by a lowering-aware bitwise quotient basis only when order-aware winner inequalities and ORT requantization are exhaustive.

- [ ] **Step 4: Run the required full rescan**

  Run the task209-specific adopted-insight scanner over every deployed graph:

  ```bash
  PYTHONPATH=. uv run python candidates/task209/rescan_adopted_insights.py
  ```

  Expected: 400 graphs scanned, zero scanner errors, and refreshed candidate counts written to `candidates/task209/rescan_adopted_insights.json`.

- [ ] **Step 5: Replace the live handoff with current truth**

  Re-read the concurrently dirty `state/STATE.md`, replace it in place with current totals, active veins, invariants, and next-session start instructions, and do not append history. Preserve unrelated concurrent session facts.

- [ ] **Step 6: Run final verification before claiming completion**

  ```bash
  PYTHONPATH=.:candidates/task209 uv run pytest candidates/task209/test_u8_qsplit.py candidates/task209/test_output_fold.py candidates/task209/test_selector_oracle.py candidates/task209/test_selector_basis.py candidates/task209/test_selector_compressed.py -q
  uv run ng score 209
  uv run ng status
  uv run ng gate candidates/task209/selector_compressed.onnx --task 209
  PYTHONPATH=.:candidates/task209 uv run python candidates/task209/selector_oracle.py --compare candidates/task209/selector_compressed.onnx --count 5000 --seed 209001
  git diff --check -- src/custom/task209.py state/insights.yaml state/STATE.md docs/superpowers/specs/2026-07-15-task209-selector-compression-design.md docs/superpowers/plans/2026-07-15-task209-selector-compression.md
  ```

  Expected: tests pass, score cost is at most5509 with fail0, status remains 400/400, gate PASS, fresh divergence0, and diff check exits 0.

- [ ] **Step 7: Commit only task209 integration files**

  Recheck the dirty worktree, stage only the exact source plus task209-owned documentation and safe non-conflicting insight/handoff hunks, then commit with:

  ```bash
  git commit -m "optimize task209 selector basis" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

  Do not stage unrelated task files, shared adoption changes from other sessions, `submission.zip`, or any pack/submit artifact.
