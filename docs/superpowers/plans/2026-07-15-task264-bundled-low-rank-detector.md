# Task 264 Bundled Low-Rank Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and adopt the cheapest task264 shared low-rank detector that preserves bundled 265/265, exact off-chart zeros, and acceptable runtime, targeting cost 1097 or lower.

**Architecture:** Preserve all eight deployed detector Einsum nodes byte-for-byte while replacing their four shared shift initializers with a bundled-trained rank-`r` factor. Decouple the renderer from the learned detector basis using exact block/local placement arrays, then sweep ranks 12, 10, 8, 6, and 4 and optionally apply the independent exact channel-root reduction.

**Tech Stack:** Python 3.13, NumPy, PyTorch 2.13, ONNX 1.21, ONNX Runtime 1.26, pytest, NeuroGolf CLI.

## Global Constraints

- Work only on task264; candidate and scratch artifacts belong under `candidates/task264/`.
- Keep `onnx==1.21.0` and `onnxruntime==1.26.0` pinned.
- Preserve the detector node protobufs byte-for-byte in the first implementation.
- Do not restore or reimplement the removed `Gather -> Equal -> Pad` tail.
- Gate is bundled fail=0 and cheaper than deployed; fresh is diagnostic only.
- Pivot on nonzero off-chart output or runtime regression.
- Use `NG_EVAL_TIMEOUT_SECONDS=1800` for full evaluation, gate, and adopt.
- Adoption must use `ng gate` followed by `ng adopt`; never edit the manifest or deployed ONNX by hand.
- Replace the task264 handoff in `state/STATE.md`; do not append a second task264 handoff.
- Record an exhausted/floor verdict only in the four ledger fields in `state/levers.yaml`; levers become dormant, never dead.

---

### Task 1: Specify and reproduce the detector math

**Files:**
- Create: `candidates/task264/test_bundled_low_rank_detector.py`
- Create: `candidates/task264/fit_bundled_low_rank_detector.py`
- Read: `submission/overfit_nets/task264.onnx`

**Interfaces:**
- Produces: `Probe(channel: str, row_offset: int, col_offset: int)`
- Produces: `parse_detector_probes(model: onnx.ModelProto) -> dict[str, tuple[Probe, ...]]`
- Produces: `shift_tensor(row, col, shift_1, shift_2) -> torch.Tensor`
- Produces: `detector_forward(inputs, row, col, shift_1, shift_2, probes) -> torch.Tensor`
- Produces: `load_bundled_examples() -> tuple[np.ndarray, list[dict]]`
- Produces: `load_teacher_targets(model, inputs) -> np.ndarray`

- [ ] **Step 1: Write the failing parser and forward-equivalence tests**

Add tests that load the deployed graph, identify `e0`, `e1`, `e2`, `e3`, `e5`, `e6`, `e7`, and `e8`, and require one `Bc` probe plus the exact `A`/`Cg` probes encoded by each node. Require the differentiable evaluator using the deployed rank-16 arrays to match ONNX Runtime teacher scalars on all 265 bundled examples within `2e-3`:

```python
DETECTOR_NAMES = ("e0", "e1", "e2", "e3", "e5", "e6", "e7", "e8")

def test_probe_parser_and_torch_forward_match_deployed_teacher():
    fitter = _load_fitter()
    model = onnx.load(INCUMBENT)
    probes = fitter.parse_detector_probes(model)
    assert tuple(probes) == DETECTOR_NAMES
    assert all(sum(p.channel == "Bc" for p in items) == 1 for items in probes.values())
    inputs, _ = fitter.load_bundled_examples()
    teacher = fitter.load_teacher_targets(model, inputs)
    arrays = _arrays(model)
    actual = fitter.detector_forward_numpy(inputs, arrays, probes)
    np.testing.assert_allclose(actual, teacher, atol=2e-3, rtol=0.0)
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run:

```bash
PYTHONPATH=. uv run pytest -q candidates/task264/test_bundled_low_rank_detector.py -k 'probe_parser or torch_forward'
```

Expected: FAIL because `fit_bundled_low_rank_detector.py` or its interfaces do not exist.

- [ ] **Step 3: Implement equation-driven probe parsing**

In `fit_bundled_low_rank_detector.py`, zip each detector's input names with the left-hand equation terms. For every `input` term, use its last two symbols to find the matching `shift_col` relation, the latent symbol, and an optional `shift_1` or `shift_2` operand:

```python
@dataclass(frozen=True)
class Probe:
    channel: str
    row_offset: int
    col_offset: int

def _relation_offset(names: list[str], terms: list[str], coordinate: str) -> int:
    latent = next(
        term[1]
        for name, term in zip(names, terms)
        if name == "shift_col" and len(term) == 2 and term[0] == coordinate
    )
    return next(
        (int(name[-1]) for name, term in zip(names, terms)
         if name in {"shift_1", "shift_2"} and term == latent),
        0,
    )
```

Split one detector equation at each `input` operand and construct a `Probe` from the immediately following channel selector plus the two spatial relations. Reject unknown selectors, offsets outside `0..2`, a detector without exactly one `Bc`, or any output other than the eight expected scalar names.

- [ ] **Step 4: Implement bundled loading, teacher export, and differentiable forward**

Load `train + test + arc-gen` using `load_task(264)` and `convert_to_numpy`. Deep-copy the deployed model, append the eight detector values to graph outputs, sanitize it, and collect teacher scalars with ORT optimizations disabled. Implement:

```python
def shift_tensor(row, col, shift_1, shift_2):
    states = torch.stack((torch.ones_like(shift_1), shift_1, shift_2))
    return torch.einsum("hr,dr,jr->dhj", row, states, col)

def _sample(image, relation_row, relation_col):
    return torch.einsum("nij,hi,wj->nhw", image, relation_row, relation_col)

def detector_forward(inputs, row, col, shift_1, shift_2, probes):
    relation = shift_tensor(row, col, shift_1, shift_2)
    maps = {
        "Bc": torch.einsum("ncij,c->nij", inputs, CHANNEL_BC),
        "A": torch.einsum("ncij,c->nij", inputs, CHANNEL_A),
        "Cg": torch.einsum("ncij,c->nij", inputs, CHANNEL_CG),
    }
    outputs = []
    for detector_probes in probes.values():
        sampled = [
            _sample(maps[p.channel], relation[p.row_offset], relation[p.col_offset])
            for p in detector_probes
        ]
        outputs.append(torch.stack(sampled).prod(dim=0).sum(dim=(1, 2)))
    return torch.stack(outputs, dim=1)
```

Read the three channel arrays from the deployed ONNX instead of duplicating their values in production code.

- [ ] **Step 5: Run equivalence tests and commit**

Run:

```bash
PYTHONPATH=. uv run pytest -q candidates/task264/test_bundled_low_rank_detector.py -k 'probe_parser or torch_forward'
```

Expected: PASS with all 265 teacher scalars reproduced.

Candidate files are gitignored; do not force-add them. The durable design and eventual exact source are committed separately.

---

### Task 2: Build the exact rank-parametric ONNX candidate

**Files:**
- Modify: `candidates/task264/test_bundled_low_rank_detector.py`
- Create: `candidates/task264/build_bundled_low_rank_detector.py`
- Create during tests/search: `candidates/task264/bundled_low_rank_r<rank>.onnx`

**Interfaces:**
- Consumes: checkpoint keys `rank`, `shift_row`, `shift_col`, `shift_1`, `shift_2`
- Produces: `exact_placement() -> tuple[np.ndarray, np.ndarray]`
- Produces: `build_candidate(incumbent: Path, checkpoint: Path, output: Path) -> None`

- [ ] **Step 1: Write failing rank-16 reconstruction and contract tests**

Create a temporary rank-16 checkpoint from the deployed initializers and require:

```python
def test_rank16_builder_preserves_detector_nodes_and_outputs(tmp_path):
    fitter = _load_fitter()
    builder = _load_builder()
    checkpoint = fitter.save_deployed_checkpoint(tmp_path / "rank16.npz")
    output = tmp_path / "rank16.onnx"
    builder.build_candidate(INCUMBENT, checkpoint, output)
    old = onnx.load(INCUMBENT)
    new = onnx.load(output)
    assert [n.SerializeToString() for n in new.graph.node[:8]] == [
        n.SerializeToString() for n in old.graph.node[:8]
    ]
    assert static_cost(output) == 545 + 46 * 16
    _assert_all_bundled_exact(output)
    _assert_off_chart_zero(output)
```

Also build twice from the same incumbent and checkpoint and require byte
identity.

- [ ] **Step 2: Run the builder tests and verify they fail**

Run:

```bash
PYTHONPATH=. uv run pytest -q candidates/task264/test_bundled_low_rank_detector.py -k 'rank16_builder or fixed_point'
```

Expected: FAIL because the builder does not exist.

- [ ] **Step 3: Implement exact placement and candidate construction**

Construct two `(3, 30)` float32 one-hot arrays:

```python
def exact_placement():
    block = np.zeros((3, 30), dtype=np.float32)
    local = np.zeros((3, 30), dtype=np.float32)
    for coordinate in range(9):
        outer, inner = divmod(coordinate, 3)
        block[outer, coordinate] = 1.0
        local[inner, coordinate] = 1.0
    return block, local
```

Replace only the four shift initializer protos plus `block_shift`,
`local_shift`, and `chart_support`. Add `glyph_block_embed` and
`glyph_local_embed`, and replace only the terminal renderer node with:

```python
helper.make_node(
    "Einsum",
    ["glyph_color_aug", "glyph_color_aug", "channel_basis", "channel_core",
     "glyph_row", "glyph_col", "glyph_term", "glyph_block_embed",
     "glyph_local_embed", "glyph_block_embed", "glyph_local_embed"],
    ["output"],
    equation="nsuw,ntuw,vh,hstz,aup,awq,az,uR,pR,wC,qC->nvRC",
    name="exact_placement_glyph_fold",
)
```

Validate checkpoint shapes exactly against `rank`, run ONNX full checking and
strict shape inference, and make repeated builds from the same incumbent and
checkpoint byte-identical. The CLI takes `checkpoint` and `output` positional
arguments and defaults the incumbent to `submission/overfit_nets/task264.onnx`.

- [ ] **Step 4: Run the builder contract tests**

Run:

```bash
PYTHONPATH=. uv run pytest -q candidates/task264/test_bundled_low_rank_detector.py -k 'rank16_builder or fixed_point'
```

Expected: PASS; rank-16 direct-placement cost is 1281 and bundled output signs are unchanged.

---

### Task 3: Implement the bounded deterministic rank fitter

**Files:**
- Modify: `candidates/task264/test_bundled_low_rank_detector.py`
- Modify: `candidates/task264/fit_bundled_low_rank_detector.py`
- Create during search: `candidates/task264/low_rank_r<rank>_best.npz`
- Create during search: `candidates/task264/low_rank_r<rank>_solution.npz`
- Create during search: `candidates/task264/low_rank_search.jsonl`

**Interfaces:**
- Produces: `float32_metrics(...) -> dict[str, float | int]`
- Produces: `initial_factors(rank: int, seed: int, teacher_tensor) -> tuple[torch.Tensor, ...]`
- Produces: CLI flags `--rank`, `--seeds`, `--steps-per-seed`, `--max-seconds`, `--learning-rate`, `--check-every`
- A solution checkpoint is written only when float32 detector colors round exactly on all 265 examples and `min_color_margin >= 0.25`.

- [ ] **Step 1: Write failing deterministic-metrics and smoke-fit tests**

Require deployed rank 16 to have zero violations, a deliberately perturbed factor to report violations, two identical seed initializations to be array-equal, and an invalid rank or nonpositive bound to be rejected. Add a short rank-12 smoke run that produces a best-attempt checkpoint without claiming it is a solution.

- [ ] **Step 2: Run tests and verify the fitter APIs fail**

Run:

```bash
PYTHONPATH=. uv run pytest -q candidates/task264/test_bundled_low_rank_detector.py -k 'metrics or initialization or smoke_fit'
```

Expected: FAIL on the missing fitter APIs.

- [ ] **Step 3: Implement teacher-tensor initialization and gauge normalization**

Build the deployed three-state tensor and initialize each rank by optimizing its tensor reconstruction from deterministic Gaussian projections. Normalize each column after every optimizer step so `shift_row[:, k]` has unit norm and absorb its scale into `shift_col[:, k]`; reject NaN, infinity, or a column norm below `1e-12`.

- [ ] **Step 4: Implement staged detector fitting**

Use float64 PyTorch training and float32 acceptance metrics. For every seed:

```python
tensor_error = functional.mse_loss(candidate_tensor, teacher_tensor)
color_error = prediction - teacher_colors
color_loss = functional.smooth_l1_loss(color_error, torch.zeros_like(color_error))
rounded_margin = 0.49 - color_error.abs()
worst = torch.topk(functional.relu(0.25 - rounded_margin).reshape(-1), k=256).values
loss = tensor_weight * tensor_error + color_loss + 4.0 * worst.square().mean()
```

Anneal `tensor_weight` from `1.0` to `0.01`, use Adam plus cosine learning-rate decay, clip gradient norm to 20, and evaluate the exported float32 arrays every `check_every` steps. Rank attempts are ordered by `(violations, -min_color_margin, max_abs_error)`.

Write `*_solution.npz` only when every teacher color rounds exactly and the minimum color margin is at least 0.25. Log seed, step, elapsed time, loss, violations, minimum margin, and maximum absolute error as sorted JSON lines. Return exit code 0 for a solution and 2 for a bounded non-solution.

- [ ] **Step 5: Run fitter unit tests**

Run:

```bash
PYTHONPATH=. uv run pytest -q candidates/task264/test_bundled_low_rank_detector.py -k 'metrics or initialization or smoke_fit'
```

Expected: PASS.

---

### Task 4: Sweep ranks and validate complete ONNX candidates

**Files:**
- Modify if required by observed numerical behavior: `candidates/task264/fit_bundled_low_rank_detector.py`
- Modify if required by export checks: `candidates/task264/build_bundled_low_rank_detector.py`
- Use: `candidates/task264/diagnose_support_gated_renderer.py`
- Update: `candidates/task264/DISCOVERY.md`

**Interfaces:**
- Consumes: solution checkpoints from Task 3
- Produces: passing `bundled_low_rank_r<rank>.onnx` candidates and a ranked result table in `DISCOVERY.md`

- [ ] **Step 1: Establish the rank-12 bar**

Run a bounded rank-12 search first:

```bash
PYTHONPATH=. uv run python candidates/task264/fit_bundled_low_rank_detector.py \
  --rank 12 --seeds 12 --steps-per-seed 4000 --max-seconds 900 \
  --learning-rate 0.003 --check-every 50
```

Expected: either a zero-violation solution checkpoint or a bounded exit code 2 with best-attempt metrics.

- [ ] **Step 2: Build and test every solution candidate**

For a solution at rank `r`, run:

```bash
PYTHONPATH=. uv run python candidates/task264/build_bundled_low_rank_detector.py \
  candidates/task264/low_rank_r${r}_solution.npz \
  candidates/task264/bundled_low_rank_r${r}.onnx
PYTHONPATH=. uv run pytest -q candidates/task264/test_bundled_low_rank_detector.py
NG_EVAL_TIMEOUT_SECONDS=1800 uv run ng gate --task 264 \
  candidates/task264/bundled_low_rank_r${r}.onnx
PYTHONPATH=. uv run python candidates/task264/diagnose_support_gated_renderer.py \
  submission/overfit_nets/task264.onnx \
  candidates/task264/bundled_low_rank_r${r}.onnx --limit 265
```

Expected: bundled 265/265, exact off-chart zero, cost `545 + 46r`, and runtime ratio at most 1.10.

- [ ] **Step 3: Continue below the first passing rank**

After rank 12, repeat the bounded search and validation for ranks 10, 8, 6, and 4. Warm-start only from a higher-rank solution through a deterministic low-rank projection; keep a fresh-seed fraction so projection bias cannot hide a better basin. Stop descending only after two consecutive lower ranks exhaust their declared bounds, or rank 4 passes.

- [ ] **Step 4: Apply the documented pivot rule**

If shared rank 12 exhausts its bound, change only the shift coefficients to detector-specific arrays and price/test the `545 + 60r` fallback. If it cannot beat 1110 at bundled 265/265, or if a candidate has nonzero off-chart output/runtime regression, stop factor work and record evidence for detector-core/public-graft research. Do not implement the old tail.

- [ ] **Step 5: Document the sweep**

Replace the current next-step section in `candidates/task264/DISCOVERY.md` with the attempted ranks, bounds, violations, margins, candidate costs, runtimes, and the selected winner or pivot verdict.

---

### Task 5: Optional exact channel factor and guarded adoption

**Files:**
- Modify only after a detector passes: `candidates/task264/build_bundled_low_rank_detector.py`
- Modify: `candidates/task264/test_bundled_low_rank_detector.py`
- Modify after adoption: `src/custom/task264.py`
- Modify after adoption: `state/tasks/task264.md`
- Modify after adoption: `state/STATE.md`
- Modify after adoption if the lever is exhausted: `state/levers.yaml`
- Modify after adoption if task264 mechanism changed materially: `state/insights.yaml`

**Interfaces:**
- Produces: the cheapest accepted task264 ONNX and byte-identical `src/custom/task264.py`

- [ ] **Step 1: Test the independent channel-root candidate**

Add an opt-in builder flag for the exact repeated-root channel representation only after a rank candidate passes. Require exact reconstruction or identical signed output with a documented minimum margin, and require static cost to fall by 32. If the proof/test fails, retain the detector-only candidate.

- [ ] **Step 2: Run final candidate verification**

Run:

```bash
PYTHONPATH=. uv run pytest -q candidates/task264/test_free_output_tail_fold.py -k fixed_point
PYTHONPATH=. uv run pytest -q candidates/task264/test_support_gated_renderer.py
PYTHONPATH=. uv run pytest -q candidates/task264/test_bundled_low_rank_detector.py
NG_EVAL_TIMEOUT_SECONDS=1800 uv run ng gate --task 264 <winner.onnx>
PYTHONPATH=. uv run python candidates/task264/diagnose_support_gated_renderer.py \
  submission/overfit_nets/task264.onnx <winner.onnx> --limit 265
```

Expected: all tests pass, bundled 265/265, exact off-chart zero, cost at most 1097, and no runtime regression.

- [ ] **Step 3: Gate and adopt without bypass**

Use the verified CLI syntax:

```bash
NG_EVAL_TIMEOUT_SECONDS=1800 uv run ng gate --task 264 <winner.onnx>
NG_EVAL_TIMEOUT_SECONDS=1800 uv run ng adopt --task 264 <winner.onnx>
NG_EVAL_TIMEOUT_SECONDS=1800 uv run ng score 264
```

Expected: gate reports bundled fail=0 and cheaper than cost 1227; adopt succeeds.

- [ ] **Step 4: Regenerate exact source and task264 state**

Use the repository's exact-source helper or the supported `ng adopt` source path, then require a rebuilt model to be byte-identical to the deployed SHA. Replace the existing task264 entries in `state/tasks/task264.md` and `state/STATE.md` with cost, score, SHA, bundled count, runtime, fresh diagnostic, and the next score bar. Touch no other task ledger.

- [ ] **Step 5: Run post-adoption verification and commit task264-only files**

Run:

```bash
uv run ng status
PYTHONPATH=. uv run pytest -q candidates/task264/test_bundled_low_rank_detector.py
git diff --check -- src/custom/task264.py state/tasks/task264.md state/STATE.md state/levers.yaml state/insights.yaml
```

Expected: status reports the adopted task264 cost/SHA and bundled 265/265. Commit only task264-owned source and non-conflicting task264 ledger hunks; preserve all unrelated dirty worktree changes.
