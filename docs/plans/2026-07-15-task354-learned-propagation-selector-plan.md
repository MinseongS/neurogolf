# task354 learned propagation selector implementation plan

> Approved design: `docs/plans/2026-07-15-task354-learned-propagation-selector-design.md`

**Goal:** Replace task354's eight-plane horizontal propagation chain with the cheapest
bundled-exact learned integer selector, retain the complete padded division renderer, and adopt
only a cheaper bundled-fail-zero graph.

**Baseline:** `submission/overfit_nets/task354.onnx`, cost 1268, bundled 266/266.

## Task 1: Establish RED selector contracts and extract the exact training relation

**Files**

- Add: `candidates/task354/test_learned_propagation_selector.py`
- Add: `candidates/task354/extract_propagation_dataset.py`
- Generate: `candidates/task354/propagation_selector_dataset.npz`

**Steps**

1. Write tests requiring a candidate ONNX, exact `Fb` agreement with the incumbent on all bundled
   examples, final raw-output A/B, strict inference, ORT execution, cost below 1268, and the
   existing label/padding exhaustive contract.
2. Run the focused test and confirm it fails because the candidate/builder does not exist.
3. Extract exact `Uraw`, `x0`, packed `P`, and `Fb` tensors by exposing incumbent intermediates in
   an in-memory model. Save deterministic arrays and assert dataset shapes and SHA in the test.

Run:

```bash
uv run pytest -q candidates/task354/test_learned_propagation_selector.py
```

## Task 2: Decide and build the one-layer packed selector

**Files**

- Add: `candidates/task354/search_learned_propagation_selector.py`
- Add: `candidates/task354/build_learned_propagation_selector.py`
- Generate: `candidates/task354/learned_propagation_selector.onnx`
- Generate: `candidates/task354/learned_propagation_search.json`

**Steps**

1. Search odd horizontal receptive fields and legal QLinearConv zero-point/scale choices on
   `P = Uraw - x0`.
2. Require exact incumbent code at every bundled `Fb` cell after pinned-runtime quantization.
3. If feasible, splice `Sub(P)` and one selector into the deployed graph, deleting the seven
   post-`x0` propagation intermediates.
4. Run the focused test. Keep the candidate only if it is GREEN and measured cost is below 1268.

## Task 3: Escalate capacity only when the one-layer grammar is infeasible

**Files**

- Extend the search/build files above, or add narrowly named fallback builders under
  `candidates/task354/`.
- Record bounded results in `candidates/task354/learned_propagation_search.json`.

**Steps**

1. Try a two-channel latent quantized selector plus 1x1 decoder.
2. If still infeasible, try a three-band reachability ranker with exact colour selection.
3. Stop a grammar when it cannot produce exact bundled `Fb` or when its static lower-bound cost is
   not below 1268. Do not adopt a partial improvement before checking whether a composed larger
   deletion is available.

## Task 4: Verify, gate, and adopt the best real candidate

**Files**

- Existing: `candidates/task354/test_complete_division_qlinear.py`
- Existing/all focused: `candidates/task354/test_*.py`

**Steps**

1. Run all task354 focused tests.
2. Run ONNX checker, strict shape inference, isolated `ng score 354`, and direct raw-output A/B.
3. Run fresh generation as diagnostics when the repository helper is available.
4. Run `uv run ng gate 354 <candidate>` and inspect bundled fail/cost.
5. Only if gate is fail=0 and cheaper, run `uv run ng adopt 354 <candidate>`.

## Task 5: Restore source ownership and update truth sources

**Files**

- Modify: `src/custom/task354.py`
- Modify: `state/tasks/task354.md`
- Modify: `state/levers.yaml`
- Replace relevant live handoff text: `state/STATE.md`

**Steps**

1. Regenerate exact source from the adopted graph and verify rebuilt cost/fail=0.
2. Record the adoption in the task ledger and the learned mechanism or bounded negative in the
   four-field lever ledger.
3. Replace the task354 live handoff section and current snapshot facts; do not append history.
4. Re-run focused tests and isolated scoring, then report actual cost/score and whether adoption
   occurred. Do not pack or submit.

