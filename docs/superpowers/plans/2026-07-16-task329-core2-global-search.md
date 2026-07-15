# Task329 Core2 Global Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Search a bounded, genuinely new family of core-free/core2 terminal contractions and deploy task329 at cost93 or95 if and only if one passes exhaustive float32 proof.

**Architecture:** Normalize every candidate with no coupling tensor larger than a 2-vector into a product of diagonal binary power-sums over the shared colour, state, and route features.  A deterministic SciPy differential-evolution driver searches a fixed, pre-priced topology catalog, freezes any feasible coefficients in JSON, and a generic proof-first builder translates that exact topology into one terminal ONNX Einsum.  If the bounded catalog has no feasible solution, the run records a solver-scoped negative result and writes no ONNX; the separate 2-by-2 selector lane then needs its own design update.

**Tech Stack:** Python 3.13, NumPy float32, SciPy 1.18 differential evolution, pytest, ONNX 1.21.0, ONNX Runtime 1.26.0, `uv run ng`.

## Global Constraints

- Modify only task329 candidates, tests, source, and task329-specific records.
- Do not repeat the rank2 Adam/hinge search, its 16 seeds, or its 8000-step loss.
- Search only exact diagonal-power topologies with terminal-Einsum operand count at most16.
- Prove all `4 * 30 * 10 * 10 = 12000` logical states before writing ONNX.
- Require `desired > 0`, `wrong <= 0`, all finite, and float32 margin at least0.008.
- Predicted and measured cost must be93 without `core2`, or95 with `core2`.
- Proof-only mode must never create or modify an ONNX file.
- Every adoption must use `uv run ng gate` followed by `uv run ng adopt`; never copy into deployment or edit the manifest manually.
- Fresh1500 is diagnostic and must be recorded before adoption.
- Preserve `onnx==1.21.0`, `onnxruntime==1.26.0`, the deployed cost105 model while searching, and the historical rank4 cost270 fallback.
- Do not use sparse initializers, pack, submit, scan other tasks, or change another task.
- Record any negative verdict only in `state/levers.yaml` with date/ran/verdict/reopen/falsification history and scope it to the exact catalog and solver budget.

---

## File structure

- `candidates/task329/core2_global_search.py`: topology grammar, exact logical evaluator, cost accounting, bounded global solver, and JSON CLI.
- `candidates/task329/test_core2_global_search.py`: TDD coverage for labels, normal-form evaluation, operand/cost accounting, deterministic report schema, and no-write behavior.
- `candidates/task329/core2_global_search.json`: generated complete search report and, only on success, the frozen float32 solution.
- `candidates/task329/build_core2_factor_route.py`: created only after a feasible report; generic proof-first ONNX lowering of the frozen topology.
- `candidates/task329/test_core2_factor_route.py`: created only after a feasible report; proof/build/source fixed-point acceptance tests.
- `candidates/task329/core2_factor_route.onnx`: created only after the builder re-proves all acceptance conditions.
- `src/custom/task329.py`: regenerated from the adopted graph only after gate/fresh/adopt succeed.
- The five historical `candidates/task329/build_*route.py` descendants: recognize the newer terminal node and return the live exact source model unchanged.
- `candidates/task329/DISCOVERY.md`: append the successful mechanism and exact measurements; a failed catalog is instead recorded only in the lever ledger.
- `state/levers.yaml`: receives a four-field, solver-scoped negative entry only if the bounded catalog returns no solution.

### Task 1: Exact normal-form evaluator

**Files:**
- Create: `candidates/task329/test_core2_global_search.py`
- Create: `candidates/task329/core2_global_search.py`

**Interfaces:**
- Consumes: legal cell counts `[9,25,49,81]` and target columns `[1,2,3,4]`.
- Produces: `Factor`, `Topology`, `logical_mask()`, `route_features()`, `evaluate()`, `proof_report()`, and `predicted_cost()`.

- [ ] **Step 1: Write failing normal-form tests**

```python
import numpy as np

from core2_global_search import (
    Factor,
    Topology,
    evaluate,
    logical_mask,
    predicted_cost,
    route_features,
)


def test_logical_mask_has_one_desired_channel_per_state() -> None:
    mask = logical_mask()
    assert mask.shape == (4, 30, 10, 10)
    assert int(mask.sum()) == 4 * 30 * 10
    assert mask[2, 3, 7, 7]
    assert mask[2, 2, 7, 0]
    assert not mask[2, 2, 7, 7]


def test_route_features_encode_exact_target_ratios() -> None:
    route = route_features(np.float32(0.0))
    state = np.stack(
        [np.ones(4, dtype=np.float32), np.array([9, 25, 49, 81], dtype=np.float32)],
        axis=1,
    )
    spatial = state[:, :, None] * route[None, :, :]
    assert np.array_equal(spatial[np.arange(4), 1, np.arange(1, 5)], np.ones(4, dtype=np.float32))


def test_one_factor_matches_manual_diagonal_sum() -> None:
    feature = np.stack(
        [np.ones(10, dtype=np.float32), np.arange(10, dtype=np.float32)], axis=1
    )
    core = np.array([2.0, -3.0], dtype=np.float32)
    topology = Topology("unit", (Factor(1, 1, 0, 1),))
    scores = evaluate(topology, feature, core, np.float32(0.0))
    expected = 2.0 - 3.0 * np.arange(10)[:, None] * np.arange(10)[None, :]
    assert np.array_equal(scores[0, 0], expected.astype(np.float32))


def test_cost_and_operand_accounting() -> None:
    without_core = Topology("plain", (Factor(1, 1, 1, 0), Factor(1, 2, 1, 0)))
    with_core = Topology("weighted", (Factor(1, 1, 1, 0), Factor(2, 1, 1, 1)))
    assert predicted_cost(without_core) == 93
    assert predicted_cost(with_core) == 95
    assert without_core.operand_count == 1 + (1 + 1 + 2) + (1 + 2 + 2)
    assert with_core.operand_count <= 16
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPATH=candidates/task329 uv run pytest -q \
  candidates/task329/test_core2_global_search.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'core2_global_search'`.

- [ ] **Step 3: Implement the exact grammar and evaluator**

```python
from dataclasses import dataclass
import numpy as np

COUNTS = np.array([9.0, 25.0, 49.0, 81.0], dtype=np.float32)


@dataclass(frozen=True)
class Factor:
    input_power: int
    output_power: int
    spatial_power: int
    core_power: int

    @property
    def operand_count(self) -> int:
        return self.input_power + self.output_power + 2 * self.spatial_power + self.core_power


@dataclass(frozen=True)
class Topology:
    name: str
    factors: tuple[Factor, ...]

    @property
    def operand_count(self) -> int:
        return 1 + sum(factor.operand_count for factor in self.factors)

    @property
    def uses_core(self) -> bool:
        return any(factor.core_power for factor in self.factors)


def logical_mask() -> np.ndarray:
    mask = np.zeros((4, 30, 10, 10), dtype=bool)
    for state in range(4):
        for column in range(30):
            for input_colour in range(10):
                output_colour = input_colour if column == state + 1 else 0
                mask[state, column, input_colour, output_colour] = True
    return mask


def route_features(default_route: np.float32) -> np.ndarray:
    route = np.ones((2, 30), dtype=np.float32)
    route[1] = default_route
    route[1, 1:5] = np.float32(1.0) / COUNTS
    return route


def _power(value: np.ndarray | np.float32, exponent: int):
    if exponent == 0:
        return np.ones_like(value, dtype=np.float32)
    result = np.asarray(value, dtype=np.float32)
    for _ in range(1, exponent):
        result = np.multiply(result, value, dtype=np.float32)
    return result


def evaluate(
    topology: Topology,
    feature: np.ndarray,
    core: np.ndarray,
    default_route: np.float32,
) -> np.ndarray:
    assert feature.shape == (10, 2)
    assert core.shape == (2,)
    assert topology.operand_count <= 16
    state = np.stack([np.ones(4, dtype=np.float32), COUNTS], axis=1)
    route = route_features(default_route)
    scores = np.ones((4, 30, 10, 10), dtype=np.float32)
    for factor in topology.factors:
        term = np.zeros_like(scores)
        for latent in range(2):
            value = _power(feature[:, latent][None, None, :, None], factor.input_power)
            value = np.multiply(
                value,
                _power(feature[:, latent][None, None, None, :], factor.output_power),
                dtype=np.float32,
            )
            value = np.multiply(
                value,
                _power(state[:, latent][:, None, None, None], factor.spatial_power),
                dtype=np.float32,
            )
            value = np.multiply(
                value,
                _power(route[latent][None, :, None, None], factor.spatial_power),
                dtype=np.float32,
            )
            value = np.multiply(value, _power(core[latent], factor.core_power), dtype=np.float32)
            term = np.add(term, value, dtype=np.float32)
        scores = np.multiply(scores, term, dtype=np.float32)
    return scores


def predicted_cost(topology: Topology) -> int:
    params = 20 + 60 + 1 + (2 if topology.uses_core else 0)
    memory = 4 + 8
    return params + memory
```

- [ ] **Step 4: Add exhaustive proof reporting**

```python
def proof_report(topology, feature, core, default_route):
    scores = evaluate(topology, feature, core, np.float32(default_route))
    mask = logical_mask()
    desired = scores[mask]
    wrong = scores[~mask]
    desired_min = float(np.min(desired))
    wrong_max = float(np.max(wrong))
    return {
        "logical_states": int(scores.size),
        "desired_min": desired_min,
        "wrong_max": wrong_max,
        "margin": min(desired_min, -wrong_max),
        "all_finite": bool(np.isfinite(scores).all()),
        "max_abs_score": float(np.max(np.abs(scores))),
        "operand_count": topology.operand_count,
        "predicted_cost": predicted_cost(topology),
    }
```

- [ ] **Step 5: Run tests and checkpoint the evaluator**

Run the focused pytest command from Step2.  Expected: `4 passed`.

The repository intentionally ignores `candidates/`, so do not force-add these scratch
files.  The green focused test run is the Task1 checkpoint.

### Task 2: Deterministic bounded global solver

**Files:**
- Modify: `candidates/task329/core2_global_search.py`
- Modify: `candidates/task329/test_core2_global_search.py`

**Interfaces:**
- Consumes: `Topology` and `evaluate()` from Task1.
- Produces: `TOPOLOGIES`, `decode_parameters()`, `objective()`, `search()`, CLI exit0 for feasible and exit2 for bounded-no-solution.

- [ ] **Step 1: Add failing catalog and report-schema tests**

```python
from core2_global_search import TOPOLOGIES, objective, search


def test_catalog_is_prepriced_and_uses_a_new_topology_family() -> None:
    assert len(TOPOLOGIES) == 14
    assert all(topology.operand_count <= 16 for topology in TOPOLOGIES)
    assert any(len(topology.factors) == 3 for topology in TOPOLOGIES)
    assert any(factor.spatial_power == 2 for topology in TOPOLOGIES for factor in topology.factors)


def test_objective_rejects_nonfinite_scores() -> None:
    topology = TOPOLOGIES[0]
    feature = np.full((10, 2), np.inf, dtype=np.float32)
    assert objective(topology, feature, np.ones(2, dtype=np.float32), np.float32(0.0)) >= 1.0e12


def test_zero_budget_search_has_stable_schema() -> None:
    report = search(maxiter=0, seeds=(329,), topologies=TOPOLOGIES[:1])
    assert report["solver"] == "scipy.optimize.differential_evolution"
    assert report["logical_states"] == 12000
    assert report["status"] in {"feasible", "bounded_no_solution"}
    assert len(report["runs"]) == 2
```

- [ ] **Step 2: Run the three new tests and verify RED**

Run:

```bash
PYTHONPATH=candidates/task329 uv run pytest -q \
  candidates/task329/test_core2_global_search.py \
  -k 'catalog or objective or zero_budget'
```

Expected: import errors for `TOPOLOGIES`, `objective`, and `search`.

- [ ] **Step 3: Add the fixed 14-topology catalog**

```python
TOPOLOGIES = (
    Topology("pair_11_21_plain", (Factor(1, 1, 0, 0), Factor(2, 1, 1, 0))),
    Topology("pair_11_12_plain", (Factor(1, 1, 0, 0), Factor(1, 2, 1, 0))),
    Topology("cross_12_21", (Factor(1, 2, 1, 0), Factor(2, 1, 1, 1))),
    Topology("cross_13_31", (Factor(1, 3, 1, 0), Factor(3, 1, 1, 1))),
    Topology("cross_12s2_21s1", (Factor(1, 2, 2, 0), Factor(2, 1, 1, 1))),
    Topology("cross_12s1_13s2", (Factor(1, 2, 1, 0), Factor(1, 3, 2, 1))),
    Topology("pair_11s1_22s2", (Factor(1, 1, 1, 0), Factor(2, 2, 2, 1))),
    Topology("pair_21s2_12s1", (Factor(2, 1, 2, 0), Factor(1, 2, 1, 1))),
    Topology("triple_linear_mixed", (Factor(1, 0, 1, 0), Factor(0, 1, 1, 1), Factor(1, 1, 0, 0))),
    Topology("triple_quadratic_mixed", (Factor(2, 0, 1, 0), Factor(0, 2, 1, 1), Factor(1, 1, 0, 0))),
    Topology("triple_11_12_21", (Factor(1, 1, 0, 0), Factor(1, 2, 1, 0), Factor(2, 1, 1, 1))),
    Topology("triple_spatial_split", (Factor(1, 1, 1, 0), Factor(1, 2, 0, 0), Factor(2, 1, 0, 1))),
    Topology("triple_11_13_31", (Factor(1, 1, 0, 0), Factor(1, 3, 1, 0), Factor(3, 1, 1, 1))),
    Topology("triple_asymmetric_s2", (Factor(1, 0, 1, 0), Factor(0, 1, 2, 1), Factor(1, 1, 0, 0))),
)
assert len({topology.name for topology in TOPOLOGIES}) == len(TOPOLOGIES)
assert all(topology.operand_count <= 16 for topology in TOPOLOGIES)
```

- [ ] **Step 4: Implement scale-stable decoding and objective**

Use affine parameters for the first pass and free 20-element colour features for the second pass.  Normalize each feature column by its maximum absolute value before scoring; this removes irrelevant scale directions.  The final solution is positively rescaled and re-proved without normalization.

```python
def _normalize_columns(feature):
    scale = np.maximum(np.max(np.abs(feature), axis=0, keepdims=True), np.float32(1.0e-4))
    return np.asarray(feature / scale, dtype=np.float32)


def decode_parameters(parameters, mode):
    values = np.asarray(parameters, dtype=np.float32)
    if mode == "affine":
        x = np.arange(10, dtype=np.float32)[:, None]
        feature = values[:2][None, :] + x * values[2:4][None, :]
        offset = 4
    else:
        feature = values[:20].reshape(10, 2)
        offset = 20
    core = values[offset : offset + 2]
    default_route = values[offset + 2]
    return _normalize_columns(feature), core, np.float32(default_route)


def objective(topology, feature, core, default_route, target_only=False):
    with np.errstate(over="ignore", invalid="ignore"):
        scores = evaluate(topology, feature, core, default_route)
    if target_only:
        scores = scores[np.arange(4), np.arange(1, 5)]
        mask = np.eye(10, dtype=bool)[None].repeat(4, axis=0)
    else:
        mask = logical_mask()
    if not np.isfinite(scores).all():
        return 1.0e12
    signed = np.where(mask, scores, -scores)
    scale = max(float(np.max(np.abs(scores))), 1.0e-12)
    normalized = signed / scale
    bad = int(np.count_nonzero(normalized <= 0.0))
    return float(2 * bad + 1.0 - np.clip(np.min(normalized), -1.0, 1.0))
```

- [ ] **Step 5: Implement the two-stage global search**

Stage1 runs target-only affine features for all 14 topologies with seeds329 and1329, `maxiter=240`, `popsize=6`.  Stage2 takes the four best topology names and runs free features against all12000 states with seeds2329 and3329, `maxiter=320`, `popsize=6`, `polish=True`, `workers=1`, and `updating="immediate"`.

```python
from scipy.optimize import differential_evolution


def _run_de(topology, mode, seed, maxiter, target_only):
    dimensions = 7 if mode == "affine" else 23
    result = differential_evolution(
        lambda p: objective(topology, *decode_parameters(p, mode), target_only=target_only),
        [(-8.0, 8.0)] * dimensions,
        seed=seed,
        maxiter=maxiter,
        popsize=6,
        polish=maxiter > 0,
        workers=1,
        updating="immediate",
        tol=1.0e-8,
    )
    feature, core, default_route = decode_parameters(result.x, mode)
    proof = proof_report(topology, feature, core, default_route)
    return {
        "topology": topology.name,
        "mode": mode,
        "seed": seed,
        "objective": float(result.fun),
        "feature": feature.tolist(),
        "core": core.tolist(),
        "default_route": float(default_route),
        "proof": proof,
    }


def _rescale_run(run, topology):
    scaled = dict(run)
    feature = np.asarray(run["feature"], dtype=np.float32)
    core = np.asarray(run["core"], dtype=np.float32)
    degree = sum(f.input_power + f.output_power for f in topology.factors)
    scale = max(1.0, (0.032 / run["proof"]["margin"]) ** (1.0 / degree))
    feature = np.asarray(feature * np.float32(scale), dtype=np.float32)
    proof = proof_report(topology, feature, core, np.float32(run["default_route"]))
    assert proof["all_finite"]
    assert proof["desired_min"] > 0.0
    assert proof["wrong_max"] <= 0.0
    assert proof["margin"] >= 0.008
    assert proof["predicted_cost"] <= 95
    scaled["feature"] = feature.tolist()
    scaled["proof"] = proof
    return scaled


def search(maxiter=None, seeds=None, topologies=TOPOLOGIES):
    stage1_iter = 240 if maxiter is None else maxiter
    stage2_iter = 320 if maxiter is None else maxiter
    stage1_seeds = (329, 1329) if seeds is None else tuple(seeds)
    stage2_seeds = (2329, 3329) if seeds is None else tuple(seeds)
    runs = [
        _run_de(topology, "affine", seed, stage1_iter, True)
        for topology in topologies
        for seed in stage1_seeds
    ]
    best_names = []
    for run in sorted(runs, key=lambda item: item["objective"]):
        if run["topology"] not in best_names:
            best_names.append(run["topology"])
        if len(best_names) == min(4, len(topologies)):
            break
    selected = [topology for topology in topologies if topology.name in best_names]
    runs.extend(
        _run_de(topology, "free", seed, stage2_iter, False)
        for topology in selected
        for seed in stage2_seeds
    )
    feasible = [
        run for run in runs
        if run["mode"] == "free"
        and run["proof"]["all_finite"]
        and run["proof"]["desired_min"] > 0.0
        and run["proof"]["wrong_max"] <= 0.0
    ]
    solution = max(feasible, key=lambda item: item["proof"]["margin"], default=None)
    if solution is not None:
        topology = next(item for item in topologies if item.name == solution["topology"])
        solution = _rescale_run(solution, topology)
    return {
        "status": "feasible" if solution is not None else "bounded_no_solution",
        "solver": "scipy.optimize.differential_evolution",
        "logical_states": 12000,
        "catalog_size": len(topologies),
        "runs": runs,
        "solution": solution,
    }
```

- [ ] **Step 6: Add JSON CLI and pass the tests**

The CLI writes atomically only after the complete report is assembled and returns exit2 for `bounded_no_solution`.

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("core2_global_search.json"))
    parser.add_argument("--maxiter", type=int)
    args = parser.parse_args()
    report = search(maxiter=args.maxiter)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({key: report[key] for key in ("status", "catalog_size", "logical_states")}, sort_keys=True))
    raise SystemExit(0 if report["status"] == "feasible" else 2)
```

Run the full Task1/Task2 test file.  Expected: all tests pass.

- [ ] **Step 7: Checkpoint the bounded solver**

The repository intentionally ignores `candidates/`, so do not force-add these scratch
search files.  Run `git status --ignored --short candidates/task329` and verify that only
the intended task329 artifacts were added by this plan.

### Task 3: Execute and freeze the bounded result

**Files:**
- Create: `candidates/task329/core2_global_search.json`
- Modify on bounded failure only: `state/levers.yaml`

**Interfaces:**
- Consumes: deterministic solver from Task2.
- Produces: a complete result report; `solution != null` is the sole authorization for Task4.

- [ ] **Step 1: Run the full bounded search**

```bash
uv run --extra research python candidates/task329/core2_global_search.py \
  --output candidates/task329/core2_global_search.json
```

Expected success path: exit0 and JSON summary with `"status": "feasible"`.

Expected bounded-negative path: exit2 and JSON summary with `"status": "bounded_no_solution"`.  Exit2 is a valid research outcome, not permission to build ONNX.

- [ ] **Step 2: Independently audit the report**

```bash
uv run --extra research python - <<'PY'
import json
from pathlib import Path
report = json.loads(Path("candidates/task329/core2_global_search.json").read_text())
assert report["logical_states"] == 12000
assert report["catalog_size"] == 14
assert len(report["runs"]) == 36
if report["status"] == "feasible":
    proof = report["solution"]["proof"]
    assert proof["all_finite"]
    assert proof["desired_min"] > 0.0
    assert proof["wrong_max"] <= 0.0
    assert proof["predicted_cost"] <= 95
else:
    assert report["solution"] is None
print(report["status"])
PY
```

Expected: prints exactly `feasible` or `bounded_no_solution` and exits0.

- [ ] **Step 3A: On feasible, audit the positive rescale**

Task2's `_rescale_run()` multiplies both feature columns by the positive scale

```python
degree = sum(f.input_power + f.output_power for f in topology.factors)
scale = max(1.0, (0.032 / proof["margin"]) ** (1.0 / degree))
feature = np.asarray(feature * np.float32(scale), dtype=np.float32)
```

Rerun Step2 and additionally assert `report["solution"]["proof"]["margin"] >= 0.008`.

- [ ] **Step 3B: On bounded-no-solution, record the scoped negative and stop this plan**

Append one entry under the existing task329/free-output-Einsum lever.  Its `ran` field must name the 14 topology names, two-stage DE budgets, seeds, and the report path; `verdict` must say only that this dated core-free/core2 diagonal-power catalog found no cost<=95 solution; `reopen` must name a coupled 2-by-2 selector, a topology outside the catalog, SMT/rational certificates, or a different operator family; `falsification_history` must state that successive task329 storage assumptions were previously falsified and this is not a task floor.

Run:

```bash
git diff --check -- state/levers.yaml
git add state/levers.yaml
git commit -m "docs(task329): record bounded core2 search" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

Do not create `build_core2_factor_route.py` or any ONNX in this path.  Return to the approved design's Lane B with a new plan.

### Task 4: Proof-first generic ONNX lowering

**Files:**
- Create: `candidates/task329/test_core2_factor_route.py`
- Create: `candidates/task329/build_core2_factor_route.py`
- Create after proof: `candidates/task329/core2_factor_route.onnx`

**Interfaces:**
- Consumes: feasible, rescaled `solution` in `core2_global_search.json`.
- Produces: `equation_and_inputs()`, `logical_proof()`, `build()`, and a cost93/95 ONNX candidate.

- [ ] **Step 1: Write the failing proof-only acceptance test**

```python
def test_prove_only_covers_all_states_without_writing(tmp_path):
    forbidden = tmp_path / "forbidden.onnx"
    result = subprocess.run(
        [sys.executable, str(BUILDER), "--prove-only", "--json", "--output", str(forbidden)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    proof = json.loads(result.stdout)
    assert proof["logical_states"] == 12000
    assert proof["desired_min"] > 0.0
    assert proof["wrong_max"] <= 0.0
    assert proof["margin"] >= 0.008
    assert proof["all_finite"] is True
    assert proof["operand_count"] <= 16
    assert proof["predicted_cost"] in {93, 95}
    assert not forbidden.exists()
```

- [ ] **Step 2: Run the test and verify RED**

```bash
PYTHONPATH=. uv run pytest -q \
  candidates/task329/test_core2_factor_route.py::test_prove_only_covers_all_states_without_writing
```

Expected: FAIL because `build_core2_factor_route.py` does not exist.

- [ ] **Step 3: Implement generic equation generation**

```python
LATENTS = "qrv"


def equation_and_inputs(topology):
    equations = ["bkhw"]
    inputs = ["input"]
    for latent, factor in zip(LATENTS, topology.factors, strict=True):
        for _ in range(factor.input_power):
            equations.append(f"k{latent}")
            inputs.append("colour_features")
        for _ in range(factor.output_power):
            equations.append(f"c{latent}")
            inputs.append("colour_features")
        for _ in range(factor.core_power):
            equations.append(latent)
            inputs.append("core2")
        for _ in range(factor.spatial_power):
            equations.extend((f"b{latent}mn", f"{latent}w"))
            inputs.extend(("state_vec", "route_r"))
    assert len(inputs) == topology.operand_count <= 16
    return ",".join(equations) + "->bchw", inputs
```

- [ ] **Step 4: Implement proof before serialization**

Load the frozen JSON solution, reconstruct the named topology, and call Task1's `proof_report()`.  Also assert the generated equation's operand count, `route_r[0] == 1`, target reciprocals, initializer element count, memory12, finite arrays, and margin.  `build()` begins with `logical_proof()` and raises before creating a model if any assertion fails.

```python
def logical_proof():
    topology, feature, core, default_route = load_solution()
    proof = proof_report(topology, feature, core, default_route)
    assert proof["logical_states"] == 12000
    assert proof["desired_min"] > 0.0
    assert proof["wrong_max"] <= 0.0
    assert proof["margin"] >= 0.008
    assert proof["all_finite"]
    assert proof["operand_count"] <= 16
    assert proof["predicted_cost"] in {93, 95}
    return proof
```

- [ ] **Step 5: Build the three-node graph**

Use `ReduceSum(input, keepdims=1) -> cell_count_keep`, then `Concat(one_state, cell_count_keep, axis=1) -> state_vec`, then the generated terminal Einsum named `core2_factor_route_output`.  Initializers are `colour_features[10,2]`, optional `core2[2]`, `one_state[1,1,1,1]`, and `route_r[2,30]`.  Declare only `cell_count_keep FLOAT[1,1,1,1]` and `state_vec FLOAT[1,2,1,1]` as counted value infos.  Run `onnx.checker.check_model(full_check=True)` and strict shape inference.

- [ ] **Step 6: Verify GREEN proof before build**

```bash
PYTHONPATH=. uv run pytest -q candidates/task329/test_core2_factor_route.py
uv run python candidates/task329/build_core2_factor_route.py --prove-only --json
test ! -e candidates/task329/core2_factor_route.onnx
```

Expected: tests pass; JSON reports margin>=0.008 and cost93 or95; no ONNX exists.

- [ ] **Step 7: Build only after proof and validate model structure**

```bash
uv run python candidates/task329/build_core2_factor_route.py --json
uv run python - <<'PY'
import onnx
model = onnx.load("candidates/task329/core2_factor_route.onnx")
assert [node.op_type for node in model.graph.node] == ["ReduceSum", "Concat", "Einsum"]
onnx.checker.check_model(model, full_check=True)
onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
print("checker+strict PASS")
PY
```

Expected: prints `checker+strict PASS`.

- [ ] **Step 8: Checkpoint proof and builder**

The repository intentionally ignores `candidates/`.  Do not force-add the builder,
tests, report, or ONNX; retain the green focused tests and checker output as the
checkpoint evidence.

### Task 5: Official gate, fresh diagnostic, and adoption

**Files:**
- Read: `candidates/task329/core2_factor_route.onnx`
- Modify through CLI only: `submission/overfit_nets/task329.onnx`
- Modify through CLI only: `state/manifest.json`, `state/tasks/task329.md`

**Interfaces:**
- Consumes: proof-approved cost93/95 candidate.
- Produces: adopted task329 deployment only after all official checks pass.

- [ ] **Step 1: Run official bundled gate**

```bash
uv run ng gate candidates/task329/core2_factor_route.onnx --task 329
```

Expected: PASS, bundled fail0, and measured cost exactly93 or95, strictly below105.

- [ ] **Step 2: Run fresh1500 diagnostic**

```bash
PYTHONPATH=. uv run python - <<'PY'
from neurogolf.scans.fresh import fresh_check
passes, runs = fresh_check(
    329,
    candidate="candidates/task329/build_core2_factor_route.py",
    n=1500,
)
assert (passes, runs) == (1500, 1500)
PY
```

Expected: incumbent fail0, candidate fail0, candidate/incumbent divergence0, and `(1500,1500)`.

- [ ] **Step 3: Adopt through the mandatory path**

```bash
COST=$(uv run python - <<'PY'
import json
print(json.load(open("candidates/task329/core2_global_search.json"))["solution"]["proof"]["predicted_cost"])
PY
)
uv run ng adopt candidates/task329/core2_factor_route.onnx --task 329 \
  --note "bounded global-solver diagonal-power contraction; all 12000 logical states margin>=0.008; bundled gate and fresh1500 checked; 105->$COST"
```

Expected: adopt re-gate PASS and task329 auto-stamp with the measured terminal cost.

### Task 6: Exact source ownership and final records

**Files:**
- Modify: `src/custom/task329.py`
- Modify: `candidates/task329/DISCOVERY.md` (ignored task-local research record)
- Modify: `candidates/task329/build_shared_bilinear_route.py`
- Modify: `candidates/task329/build_repeated_root_route.py`
- Modify: `candidates/task329/build_dynamic_count_code_route.py`
- Modify: `candidates/task329/build_repeated_colour_root_route.py`
- Modify: `candidates/task329/build_shared_colour_spatial_core_route.py`
- Modify through `ng adopt`: `state/tasks/task329.md`
- Test: all five task329 focused suites plus the new suites.

**Interfaces:**
- Consumes: adopted deployed protobuf.
- Produces: byte-identical source rebuild, idempotent builders, and a task-local mechanism record.

- [ ] **Step 1: Regenerate exact source from the adopted graph**

```bash
uv run python tools/live_to_exact_source.py --write-src 329
```

Expected: only `src/custom/task329.py` is rewritten and its `build(None)` serialization matches deployment.

- [ ] **Step 2: Make historical builders descendant-idempotent**

Immediately after each historical builder loads the incumbent and obtains `graph`, add this exact newer-descendant guard before any old-tail assertion:

```python
if graph.node[-1].name == "core2_factor_route_output":
    from src.custom import task329

    rebuilt = task329.build(None)
    assert model.SerializeToString() == rebuilt.SerializeToString()
    onnx.checker.check_model(model, full_check=True)
    return model
```

This guard is added independently to all five files listed above; it does not change their historical construction path.

- [ ] **Step 3: Add byte-identity and new-builder idempotence tests**

```python
def test_source_rebuild_is_byte_identical_to_deployed_model():
    deployed = (ROOT / "submission/overfit_nets/task329.onnx").read_bytes()
    rebuilt = task329.build(None).SerializeToString()
    assert hashlib.sha256(rebuilt).hexdigest() == hashlib.sha256(deployed).hexdigest()


def test_builder_is_idempotent_after_adoption():
    subprocess.run([sys.executable, str(BUILDER), "--json"], cwd=ROOT, check=True)
    assert hashlib.sha256(CANDIDATE.read_bytes()).hexdigest() == hashlib.sha256(
        (ROOT / "submission/overfit_nets/task329.onnx").read_bytes()
    ).hexdigest()
```

- [ ] **Step 4: Run focused regression and isolated score**

```bash
PYTHONPATH=. uv run pytest -q \
  candidates/task329/test_shared_bilinear_route.py \
  candidates/task329/test_repeated_root_route.py \
  candidates/task329/test_dynamic_count_code_route.py \
  candidates/task329/test_repeated_colour_root_route.py \
  candidates/task329/test_shared_colour_spatial_core_route.py \
  candidates/task329/test_core2_global_search.py \
  candidates/task329/test_core2_factor_route.py
uv run ng score 329
```

Expected: every focused test passes; isolated score reports the adopted cost93 or95.

- [ ] **Step 5: Record the task-local mechanism**

Update `candidates/task329/DISCOVERY.md` with the exact topology name, solver/date/budget, logical margin, operand count, memory, params, cost, bundled/fresh results, deployment SHA, and the explicit statement that no cross-task scan was run because the user restricted work to task329.

- [ ] **Step 6: Verify and commit only task329-owned files**

```bash
git diff --check -- src/custom/task329.py state/tasks/task329.md
git status --short
git add src/custom/task329.py state/tasks/task329.md
git commit -m "feat(task329): adopt exact core2 factor route" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

Do not stage unrelated shared-worktree changes.  Report the deployed cost, point gain, proof margin, bundled/fresh counts, SHA, and focused test count.
