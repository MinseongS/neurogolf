# NeuroGolf 7470 Direct-Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Discover, adopt, and submit enough source-owned ONNX improvements to raise the confirmed public leaderboard score from 7424.42 to at least 7470.00.

**Architecture:** The primary lane is a typed conjunctive-query synthesizer for zero-parameter, zero-intermediate self-Einsum models, followed by three independent semantic compiler deep dives on the most expensive unresolved task families. Every discovered mechanism is gated on one task, registered in the insight registry, rescanned across all 400 tasks, and only then composed into provenance-isolated submission batches.

**Tech Stack:** Python 3.13, NumPy, ONNX 1.21.0, ONNX Runtime 1.26.0, pytest, `uv`, the `ng` CLI, and Kaggle CLI.

## Global Constraints

- The success condition is a completed Kaggle public score of at least **7470.00**, not a local projection.
- Preserve submission 54654166 / public LB 7424.42 / local manifest 7424.2851 as the recovery baseline until a higher composed submission is confirmed.
- Keep `onnx==1.21.0` and `onnxruntime==1.26.0`; do not upgrade either dependency.
- Candidate models and experimental builders live only under `candidates/taskNNN/` or `candidates/direct_discovery/`.
- Every adoption must use `uv run ng gate` followed by `uv run ng adopt`; never copy into `submission/overfit_nets/` directly.
- Fresh evaluation is diagnostic; the mandatory local adoption gate is bundled fail=0 plus lower deployed cost.
- The direct-discovery allocation is 60% global mechanism search, 35% high-yield task deep dives, and 5% opportunistic public polling.
- Before every submission, inspect `kaggle competitions submissions -c neurogolf-2026`; batch changes and respect the 100/day limit.
- For a high-upside candidate whose Kaggle validity is locally ambiguous, the user explicitly authorizes a one-file zip containing only `taskNNN.onnx`; the primary agent submits that isolated probe and treats its returned task score, `0`, or `ERROR` as authoritative. Workers never submit directly.
- Existing floor verdicts are method-relative, not task impossibility claims: scan all 400 tasks, including ledgered floors, whenever a genuinely new primitive or representation appears, while never repeating the same exhausted method.
- A negative verdict is recorded only in `state/levers.yaml` with date, concrete run, scoped verdict, and reopen trigger. A lever becomes dormant, never dead.
- At session close, replace `state/STATE.md`; do not append to it.

---

### Task 1: Typed self-Einsum query core

**Files:**
- Create: `tools/self_einsum_search.py`
- Create: `tests/test_self_einsum_search.py`

**Interfaces:**
- Produces: `Query = tuple[tuple[str, str, str], ...]`, where each atom indexes one copy of the `[B,K,R,C]` input without the batch label.
- Produces: `canonicalize(query: Query) -> Query`.
- Produces: `to_equation(query: Query) -> str` returning an ONNX/NumPy equation ending in `->bkrc`.
- Produces: `evaluate_query(query: Query, input_onehot: np.ndarray) -> np.ndarray` returning a boolean `[1,10,30,30]` tensor.
- Produces: `build_model(query: Query) -> onnx.ModelProto` with one `Einsum` node, repeated `input` operands, no initializers, and graph output `output`.

- [ ] **Step 1: Write canonicalization and known-query tests**

```python
import numpy as np

from neurogolf.scoring import convert_to_numpy, load_task
from tools.self_einsum_search import (
    build_model,
    canonicalize,
    evaluate_query,
    to_equation,
)


def test_internal_variable_renaming_is_canonical():
    left = (("k", "r", "c"), ("u", "c", "x"))
    right = (("k", "r", "c"), ("v", "c", "y"))
    assert canonicalize(left) == canonicalize(right)


def test_task067_zero_cost_crop_query_matches_all_bundled_examples():
    query = (("k", "r", "c"), ("u", "c", "x"))
    assert to_equation(query) == "bkrc,bucx->bkrc"
    for example in load_task(67)["train"] + load_task(67)["test"] + load_task(67)["arc-gen"]:
        arrays = convert_to_numpy(example)
        if arrays is not None:
            assert np.array_equal(evaluate_query(query, arrays["input"]), arrays["output"])


def test_built_model_has_one_node_and_no_initializers():
    model = build_model((("k", "r", "c"), ("u", "c", "x")))
    assert [node.op_type for node in model.graph.node] == ["Einsum"]
    assert len(model.graph.initializer) == 0
    assert list(model.graph.node[0].input) == ["input", "input"]
```

- [ ] **Step 2: Run the focused tests and verify they fail because the module is absent**

Run: `uv run pytest tests/test_self_einsum_search.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'tools.self_einsum_search'`.

- [ ] **Step 3: Implement canonicalization, equation evaluation, and the one-node builder**

```python
from __future__ import annotations

from collections.abc import Iterable
from typing import TypeAlias

import numpy as np
import onnx
from onnx import TensorProto, helper

Query: TypeAlias = tuple[tuple[str, str, str], ...]


def canonicalize(query: Query) -> Query:
    color_map: dict[str, str] = {"k": "k"}
    space_map: dict[str, str] = {"r": "r", "c": "c"}
    # Keep typed label pools disjoint and reserve b/k/r/c for batch/output axes.
    color_names = iter("uvwlmnopqst")
    space_names = iter("xyzadefghij")
    out = []
    for color, row, col in query:
        if color not in color_map:
            color_map[color] = next(color_names)
        for value in (row, col):
            if value not in space_map:
                space_map[value] = next(space_names)
        out.append((color_map[color], space_map[row], space_map[col]))
    return tuple(sorted(out))


def to_equation(query: Query) -> str:
    operands = ["b" + "".join(atom) for atom in canonicalize(query)]
    return ",".join(operands) + "->bkrc"


def evaluate_query(query: Query, input_onehot: np.ndarray) -> np.ndarray:
    equation = to_equation(query)
    values = np.einsum(equation, *([input_onehot] * len(query)), optimize="greedy")
    return values > 0


def build_model(query: Query) -> onnx.ModelProto:
    equation = to_equation(query)
    node = helper.make_node(
        "Einsum",
        ["input"] * len(query),
        ["output"],
        name="output",
        equation=equation,
    )
    tensor = lambda name: helper.make_tensor_value_info(
        name, TensorProto.FLOAT, [1, 10, 30, 30]
    )
    graph = helper.make_graph([node], "self_einsum", [tensor("input")], [tensor("output")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model
```

- [ ] **Step 4: Run the focused tests**

Run: `uv run pytest tests/test_self_einsum_search.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit the query core**

```bash
git add tools/self_einsum_search.py tests/test_self_einsum_search.py
git commit -m "feat: add typed self-einsum query core" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 2: Beam enumeration and 400-task search

**Files:**
- Modify: `tools/self_einsum_search.py`
- Modify: `tests/test_self_einsum_search.py`
- Create at runtime: `candidates/direct_discovery/self_einsum_hits.json`

**Interfaces:**
- Consumes: `Query`, `canonicalize`, and `evaluate_query` from Task 1.
- Produces: `expand_query(query: Query) -> set[Query]` adding one connected typed input atom.
- Produces: `query_loss(query: Query, examples: list[tuple[np.ndarray, np.ndarray]]) -> int`.
- Produces: `search_task(task_num: int, max_atoms: int = 8, beam: int = 3000) -> list[dict]`.
- Produces: `Correction` with optional per-query-label channel mask `[10]`, internal row/column gates `[30]`, and source-to-output channel mixer `[10,10]`, all embedded as small operands in the final output Einsum.
- Produces: `fit_small_corrections(query, examples, max_params: int = 150) -> list[Correction]` for task017/task197-style completion. It inserts candidate operands on the query's internal labels before contraction; it must not try to infer internal gates from an already-contracted raw output.
- CLI: `uv run python tools/self_einsum_search.py --tasks 1-400 --max-atoms 8 --beam 3000 --output candidates/direct_discovery/self_einsum_hits.json`.

- [ ] **Step 1: Write expansion and search regression tests**

```python
import numpy as np

from tools.self_einsum_search import expand_query, fit_small_corrections, search_task


def test_expansion_is_connected_and_deduplicated():
    seed = (("k", "r", "c"),)
    expanded = expand_query(seed)
    assert expanded
    assert len(expanded) == len(set(expanded))
    for query in expanded:
        assert len(query) == 2
        assert set(query[0]) & set(query[1])


def test_search_rediscovers_task067_with_two_atoms():
    hits = search_task(67, max_atoms=2, beam=500)
    assert any(hit["fail"] == 0 and hit["cost"] == 0 for hit in hits)


def test_channel_mask_completes_a_low_cost_near_hit():
    raw_input = np.ones((1, 10, 2, 2), dtype=bool)
    target = raw_input.copy()
    target[:, 1] = False
    query = (("k", "r", "c"),)
    corrections = fit_small_corrections(query, [(raw_input, target)], max_params=10)
    assert any(c.channel_mask == (1, 0, 1, 1, 1, 1, 1, 1, 1, 1) for c in corrections)
```

Also add two deployed-mechanism controls: rebuild task017's five-input query plus one reused `[10]` mask at cost 10, and task197's three-input raw query plus one reused `[30]` internal gate and `[10,10]` channel mixer at cost 130. Each rebuilt model must contain one graph-output `Einsum`, match every bundled example, and have exactly the expected initializer element count. These are regression oracles for the correction compiler, not claims that the search beam must rediscover both immediately.

- [ ] **Step 2: Run the new tests and verify failure on missing interfaces**

Run: `uv run pytest tests/test_self_einsum_search.py -q`

Expected: import fails for `expand_query` or `search_task`.

- [ ] **Step 3: Implement the connected typed grammar and beam search**

The implementation must:

- seed every one-atom mapping whose color index is `k` or one internal color variable and whose two spatial indices are drawn from `r`, `c`, and up to two internal spatial variables;
- keep only queries where every atom shares at least one variable with the existing query;
- introduce at most one new color variable and at most one new spatial variable per expansion;
- canonicalize before deduplication;
- evaluate all bundled examples, rank by total wrong output cells, and retain exact hits plus the best `beam` candidates at every depth;
- test structured near-hits with channel keep/drop masks, channel permutation or signed mixers, and separable row/column gates;
- fit masks and gates against the uncontracted query variables across all bundled examples; in particular, a gate on internal label `x` changes the contraction and cannot be reconstructed from the final raw `[B,K,R,C]` tensor alone;
- when a channel mixer is present, alpha-rename the query's anchored source channel `k` to a fresh internal color label `u`, append mixer operand `ku`, and still emit final output `bkrc`; this lets the valid raw query `nkrj,naxc,nayj` compile to task197's deployed form `nurj,naxc,nayj,x,y,ku->nkrc` without weakening Task 1's query validation;
- accept a correction only when every bundled example is exact, its initializers contain at most 150 total elements, and it creates no counted intermediate tensor;
- compile each correction as extra operands of the same graph-output Einsum, following task017's cost-10 precedent rather than assuming that only cost-0 hits matter;
- reject equations longer than 52 distinct labels or models that fail ONNX checker/ORT construction;
- evaluate suspicious mem-0 hits in a fresh Python process before writing them to the report.

The report row is exact and stable:

```python
{
    "task": task_num,
    "atoms": len(query),
    "query": query,
    "equation": to_equation(query),
    "pass": passed_examples,
    "fail": failed_examples,
    "wrong_cells": wrong_cells,
    "correction": correction.as_json() if correction else None,
    "cost": correction.param_count if correction else 0,
    "projected_gain": round(
        max(1.0, 25.0 - math.log(max(1, correction.param_count if correction else 0)))
        - incumbent_points,
        6,
    ),
}
```

- [ ] **Step 4: Run the search tests, then a known-task smoke search**

Run: `uv run pytest tests/test_self_einsum_search.py -q`

Expected: all tests pass.

Run: `uv run python tools/self_einsum_search.py --tasks 67 179 241 --max-atoms 4 --beam 1000 --output candidates/direct_discovery/self_einsum_smoke.json`

Expected: task067 appears as an exact cost-0 hit; task179/task241 are allowed to remain Transpose-only controls.

- [ ] **Step 5: Run the full high-degree search**

Run: `uv run python tools/self_einsum_search.py --tasks 1-400 --max-atoms 8 --beam 3000 --output candidates/direct_discovery/self_einsum_hits.json`

Expected: JSON contains every exact hit and the best near-hit per task/depth; process exits 0 even when no new exact hit exists.

- [ ] **Step 6: Commit the search engine**

```bash
git add tools/self_einsum_search.py tests/test_self_einsum_search.py
git commit -m "feat: search high-degree self-einsum programs" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 3: Validate and adopt exact or cheaply corrected self-Einsum discoveries

**Files:**
- Create per hit: `candidates/taskNNN/build_self_einsum.py`
- Create per hit: `candidates/taskNNN/self_einsum.onnx`
- Modify after a win: `src/custom/taskNNN.py`
- Modify after a reusable win: `state/insights.yaml`
- Modified automatically by adoption: `state/manifest.json`, `state/tasks/taskNNN.md`, `submission/overfit_nets/taskNNN.onnx`

**Interfaces:**
- Consumes: exact or at-most-150-parameter corrected `query` rows from `candidates/direct_discovery/self_einsum_hits.json`.
- Produces: source-owned graph-output Einsum ONNX candidates and, for every official PASS, an adopted deployment.

- [ ] **Step 1: Rebuild each exact hit in its task candidate directory**

The builder imports `build_model` and serializes the exact query plus any reported low-cost correction; do not hand-edit the ONNX. A zero-cost example is:

```python
from pathlib import Path
import onnx
from tools.self_einsum_search import build_model

QUERY = (("k", "r", "c"), ("u", "c", "x"))
onnx.save(build_model(QUERY), Path(__file__).with_name("self_einsum.onnx"))
```

Replace `QUERY` with the concrete report value for that task. For a corrected hit, also pass the exact report `Correction` to the builder.

- [ ] **Step 2: Verify each hit in an isolated process**

Run: `uv run ng score NNN`

Run: `uv run ng gate candidates/taskNNN/self_einsum.onnx --task NNN`

Expected for an adoptable hit: bundled `fail=0`, candidate cost exactly equal to the report row, cheaper than deployed, and `PASS`.

- [ ] **Step 3: Adopt every official PASS**

Run: `uv run ng adopt candidates/taskNNN/self_einsum.onnx --task NNN --note "direct discovery: high-degree self-Einsum with minimal output correction"`

Expected: `state/manifest.json` gains the candidate row and `state/tasks/taskNNN.md` receives an `## ADOPTED` block.

- [ ] **Step 4: Register and rescan the mechanism**

Add one `state/insights.yaml` entry named `high_degree_self_einsum_conjunctive_query` containing the exact source task, equation, semantic interpretation, atom/variable limits, and the 400-task report path. Rerun the full search after every adoption because the report's incumbent-gain ranking changes.

- [ ] **Step 5: Commit tracked source and state changes**

```bash
git add src/custom/taskNNN.py state/insights.yaml state/manifest.json state/tasks/taskNNN.md
git commit -m "feat: adopt taskNNN high-degree self-einsum" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

If the full search has no new exact hit, record the concrete max-atoms/beam/task scope and dated result in the `fourth-25pt-hunt` ledger with reopen trigger `a larger canonical grammar, a non-conjunctive one-node operator family, or a disclosed equation from a scored 25-point task`; then proceed immediately to Task 4.

### Task 4: Parallel semantic compiler wave on tasks 366, 233, and 158

**Files:**
- Task 366 candidate files: `candidates/task366/build_streamed_exact_cover.py`, `candidates/task366/streamed_exact_cover.onnx`
- Task 233 candidate files: `candidates/task233/build_safe_dynamic_correlation.py`, `candidates/task233/safe_dynamic_correlation.onnx`
- Task 158 candidate files: `candidates/task158/build_orientation_factored.py`, `candidates/task158/orientation_factored.onnx`
- Read: `playbook/public-insight.md`, each task ledger, each deployed ONNX, each `src/custom/taskNNN.py`, and the generator from `state/arc_mapping.json`

**Interfaces:**
- Produces: candidates only; workers do not adopt, pack, submit, or edit shared deployment state.
- Task 366 acceptance target: exact bundled output and cost below 22211 by streaming candidate background/rectangle evidence through compact scalar or per-box state instead of materializing full-size planes per box.
- Task 233 acceptance target: preserve the dynamic signed-correlation simplification while replacing every unsigned/signed TopK feed with Kaggle-safe fp16 or float and beating deployed cost 24703.
- Task 158 acceptance target: factor the orientation dimension out of the dominant `sat`/match planes, preserve exact-cover confirmation, and beat deployed cost 18530.

- [ ] **Step 1: Dispatch one isolated worker per task**

Each worker prompt includes the exact recipe path, `state/tasks/taskNNN.md`, `submission/overfit_nets/taskNNN.onnx`, `src/custom/taskNNN.py`, the mapped generator, the named candidate paths above, and the prohibition on adoption/submission.

- [ ] **Step 2: Require byte accounting before each build**

Each worker reports incumbent dominant tensors, proposed deleted/added tensors, predicted old/new cost, correctness invariant, and the specific prior floor assumption being changed. A worker builds only when predicted cost is lower or when one small capability probe can falsify a global assumption.

- [ ] **Step 3: Review and gate every produced candidate centrally**

Run for each existing candidate:

```bash
uv run ng gate candidates/task366/streamed_exact_cover.onnx --task 366
uv run ng gate candidates/task233/safe_dynamic_correlation.onnx --task 233
uv run ng gate candidates/task158/orientation_factored.onnx --task 158
```

Expected: a winning candidate prints `PASS`; a losing candidate remains unadopted and receives a four-field scoped verdict in the relevant lever ledger.

- [ ] **Step 4: Adopt PASS candidates and register transferable mechanisms**

Use `uv run ng adopt` with a mechanism-specific note. For each win, rebuild or update `src/custom/taskNNN.py`, add/update `state/insights.yaml`, and rescan all 400 deployed models for the same op/tensor fingerprint before starting another unrelated task.

- [ ] **Step 5: Commit each accepted task independently**

Stage only that task's source, manifest/task ledger, insight entry, and any shared scanner needed to reproduce the fanout. Use one commit per accepted mechanism with the required co-author trailer.

### Task 5: New-physics pivot after a low-yield wave

**Files:**
- Create: `tools/single_node_family_search.py`
- Create: `tests/test_single_node_family_search.py`
- Create at runtime: `candidates/direct_discovery/single_node_family_hits.json`
- Modify on a measured verdict: `state/levers.yaml`

**Interfaces:**
- Runs only if Tasks 1-4 produce less than 0.5 aggregate adoptable points.
- Produces: a legality/cost/exactness census for same-shape single-node operators not already covered by the Pool, CumSum, ReverseSequence, Trilu, Hardmax, Pad/Slice, Transpose, and self-Einsum searches.

- [ ] **Step 1: Write a parameterized model-construction test**

The test constructs every proposed one-node family, checks `onnx.checker.check_model(..., full_check=True)`, creates an ORT 1.26 session with optimizations disabled, and asserts the candidate output shape is exactly `[1,10,30,30]`.

- [ ] **Step 2: Implement the family registry**

Include same-input variadic arithmetic/reduction operators, normalization operators whose attributes can change support, and every standard-domain operator accepted by the pinned ONNX schema with no required initializer and a same-shape tensor output. Explicitly exclude previously measured families and every scorer-excluded `Loop`, `Scan`, `Sequence*`, `NonZero`, `Unique`, `Script`, `Function`, and `Compress` operator.

- [ ] **Step 3: Evaluate legal candidates across all 400 bundled tasks**

Run: `uv run python tools/single_node_family_search.py --tasks 1-400 --output candidates/direct_discovery/single_node_family_hits.json`

Expected: exact hits include task179/task241 Transpose controls; new hits are rebuilt, gated, adopted, registered, and globally rescanned by the Task 3 procedure.

- [ ] **Step 4: Record a scoped negative if the family has no new hit**

Append a `fourth-25pt-hunt` ledger entry containing the exact operator registry, 400-task scope, date, no-new-hit result, prior false-floor history, and reopen trigger `new legal standard-domain operator or scored public disclosure of a missing 25-point task`.

### Task 6: Verify, probe ambiguous tasks, compose, and submit provenance-isolated batches

**Files:**
- Modify: `state/submissions.md`
- Modify at session close: `state/STATE.md`
- Generated: `submission.zip`

**Interfaces:**
- Consumes: high-upside locally ambiguous candidates for one-task probes, and only adopted candidates for composed 400-task submissions.
- Produces: authoritative isolated Kaggle verdicts for ambiguous high-upside candidates, then a completed composed submission and confirmed public score, repeating discovery waves until score >=7470.00.

- [ ] **Step 1: Run repository verification**

Run: `uv run ng verify --hash`

Expected: `400/400` and `HASH-OK`.

Run the repository's unsigned-TopK scanner or equivalent graph census and require zero unsigned/signed int8 TopK feeds before packing.

- [ ] **Step 2: Check concurrent submissions**

Run: `kaggle competitions submissions -c neurogolf-2026`

Expected: no newer overlapping submission from another session that would change the baseline or duplicate the batch.

- [ ] **Step 3: Submit a one-task probe when local validity remains ambiguous**

Only the primary agent may perform this user-authorized exception. Put the candidate in an otherwise empty directory, name it exactly `taskNNN.onnx`, and create a zip containing only that file:

```bash
mkdir -p candidates/taskNNN/lb_probe
cp candidates/taskNNN/<candidate>.onnx candidates/taskNNN/lb_probe/taskNNN.onnx
(cd candidates/taskNNN/lb_probe && zip -q -j taskNNN_probe.zip taskNNN.onnx)
kaggle competitions submissions -c neurogolf-2026
kaggle competitions submit -c neurogolf-2026 -f candidates/taskNNN/lb_probe/taskNNN_probe.zip -m "PARTIAL PROBE taskNNN only: <hypothesis>"
```

Poll until `COMPLETE` or `ERROR`. A completed score is the direct score of that task; `0` or `ERROR` rejects the candidate unless it creates a new, concrete hypothesis. Record the submission id, exact artifact hash, result, and interpretation in `state/submissions.md` and the task ledger. Immediately run `uv run ng pack` afterward to restore the canonical 400-file `submission.zip`. The normal adopted batch still uses `ng pack` followed by `ng submit`.

- [ ] **Step 4: Pack and submit one mechanism-provenance batch**

Run: `uv run ng pack`

Expected: `submission.zip` contains exactly 400 task ONNX files.

Run: `uv run ng submit -m "7424.42 base + <mechanism> direct-discovery batch; local <points>; 400/400 HASH-OK"`

- [ ] **Step 5: Poll until the submission reaches COMPLETE or ERROR**

Run: `kaggle competitions submissions -c neurogolf-2026`

If COMPLETE and higher, make it the new baseline. If a provenance-isolated batch zeros/errors, restore the prior deployed task through `ng adopt` on its saved backup, verify 400/400, and record the exact attribution in the task ledger and `state/submissions.md`.

- [ ] **Step 6: Continue waves until the leaderboard target is confirmed**

If score is below 7470, return to Task 2 with either a larger self-Einsum grammar justified by the prior report or to Task 4 with the next mechanism cluster from `state/insights.yaml`. Do not spend a wave on sub-0.1 tails unless they are free byproducts of a larger rewrite.

- [ ] **Step 7: Close the successful session**

Replace `state/STATE.md` with the confirmed score, submission id, exact adopted mechanisms, active veins, invariants, and next-session start procedure. Update `state/submissions.md`, run `uv run ng verify --hash` again, stage only the tracked state/source files touched, and commit with the required co-author trailer.
