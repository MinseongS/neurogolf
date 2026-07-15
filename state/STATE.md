# STATE - NeuroGolf live handoff (updated 2026-07-15 20:17 KST; working manifest 7433.8988)
> Replace this file at session end; do not append. History lives in git, `state/tasks/`,
> `state/submissions.md`, and `state/levers.yaml`.

## Confirmed state
- There are **400/400** deployed nets. The shared working manifest was **7433.8988** at this
  snapshot, but it changed repeatedly during concurrent range sessions and is **not** a completed
  full-board verification result.
- Last fully confirmed local deployment remains **7431.0339**, bundled fail=0. Best confirmed
  leaderboard score remains **7431.10**, submission **54719275** (submitted from local 7430.9666).
- The task276-300 audit isolated and rescored all 25 current deployed artifacts; all 25 had bundled
  fail=0. That session did not adopt, pack, or submit anything.

## task276-300 discovery handoff
- All 25 tasks were inspected. Only **task285** and **task295** met the concrete >=+0.1 discovery
  bar. Their full handoffs are `candidates/task285/DISCOVERY.md` and
  `candidates/task295/DISCOVERY.md`.
- task285: actual deployed artifact cost **18674**. The exact bounded-affine model has analytical
  cost **8921** (expected +0.7387) and its numpy/oracle components pass, but the monolithic output
  Einsum produces no output under pinned ORT. Next action is immediate Top-3 pivot sparsification
  and a bounded 3x5x5 renderer; target the stricter **cost <=16104**.
- task295: `candidates/task295/discovery.onnx` is an actual gate PASS: **393 -> 343**,
  **+0.136079**, bundled **268/268**, fresh candidate-vs-incumbent **0/1500** differences. It
  replaces B3=[1,x,x^2] with repeated B2=[1,x] factors and exact latent sharing. Do not adopt yet:
  the 268-example runtime median was slower than the incumbent and needs operand-order/full-board
  runtime work first.
- New reusable insight `repeated_power_basis_inside_free_output_einsum` is registered in
  `state/insights.yaml`. A full-400 strict-signature rescan found task246 as the only follow-up
  outside this assignment.
- Scoped negative experiments: task277 propagation depth3/4 failed 71/24 examples; task279
  erosion depth5/6/7 failed 184/86/24; task289's cost560 OneHot lowering has no ORT 1.26 CPU
  kernel; task293 rank-3 fit still had 6056 violations; task294 candidates cost612/642 lose to
  deployed601.

## Concurrent locally gated progress
- Concurrent task026-050 work gated/adopted task032 **910 -> 588**, task036 **1051 -> 940**,
  task046 **2075 -> 1846**, and task048 **744 -> 622** through `ng adopt`; the recorded combined
  local gain is +0.844376. Six discoveries live under
  `candidates/task032|035|036|041|046|048/DISCOVERY.md`.
- Other concurrent range sessions continued changing the manifest after that handoff. Treat the
  working total as a moving snapshot until sessions quiesce.

## Integrity and runtime warnings
- Manifest and deployed artifacts are known to disagree at least on task035, 050, 280, 281, 285,
  and 286. In the task276-300 range the actual/manifest costs were: task280 **4145/4142**,
  task281 **1329/1304**, task285 **18674/17798**, task286 **18271/18169**. `ng gate` compares
  against manifest cost, so reconcile only through the normal source/gate/adopt workflow; never
  copy artifacts manually.
- A duplicate full `ng verify` was stopped because another concurrent verify was already running;
  under that contention task047 exceeded the 600s isolated timeout. No `--update` was run. Wait
  for a stable board before the next full verify.
- Submission **54718839** lost roughly 17 leaderboard points; task080 remains the leading hidden
  failure/aggregate-timeout suspect. Runtime is a real package constraint for large variadic
  Einsum waves.
- Before any submission: wait for concurrent sessions, run a complete 400/400 verification,
  inspect current Kaggle submissions, then use `ng pack` -> `ng submit`.

## Invariants
- Goal remains 8000; default mode is 8000-overfit: bundled fail=0 + strictly cheaper. Fresh is
  diagnostic.
- Adoption must use `ng adopt`; submission must use `ng pack` then `ng submit`.
- Keep onnx==1.21.0 and onnxruntime==1.26.0 until a complete 400/400 revalidation authorizes change.

## task151-175 execution result
- All 25 live artifacts were independently scored and graph-audited. The three concrete wins in
  this range, **task153, task173, and task174**, are now adopted for a combined **+0.387640**.
  Full technical handoffs and execution evidence are
  `candidates/task173/DISCOVERY.md` and `candidates/task174/DISCOVERY.md`; the complete disposition
  is `candidates/range151_175/REVIEW.md` with reproducible `audit.py`/`audit.json` and
  `live_scores.json`.
- task153 completed a real adopted win: dtype lowering of presence/crop/color carriers plus
  initializer dedupe changed **685 -> 618**, **+0.102930**, bundled **265/265**, fail=0. The
  immutable-backup builder reproduces the deployed SHA, and `src/custom/task153.py` now rebuilds
  the same 56 nodes/15 initializers and measured cost618.
- task173 restored the shifted FREE off-grid code and removed complete-prototype no-op stamps from
  the ranked tail, making k=6 exact on bundled data. It was gated/adopted **11320 -> 10233**
  (**+0.100953**), bundled **266/266**. Fresh4000 is diagnostic-only and found one extra tail miss
  (candidate fail13 vs incumbent fail12); the risk is recorded in the task discovery/log.
- task174 reconstructed the lost bbox rewrite and improved it: first/last `ArgMax`, fp16 geometry,
  1-D window masks, nested `Where`, int8 binary reduction, fp16 factor `Pow`, and scalar
  selector `ArgMax->Gather` produced **3348 -> 2786** (**+0.183756**), bundled **266/266**, fresh
  **4000/4000 divergence0**. Both task173/174 exact sources rebuild to semantically identical
  topology and the same measured cost/fail as their adopted live graphs.
- A tempting task160/task169 constant-feature-to-QLinearConv-bias fold was implemented and
  falsified: task169 failed 266/266. The final Conv pads a 10x10 feature asymmetrically to 30x30,
  so a bias incorrectly writes the padded region. Do not retry without an equally cheap spatial
  support term.
- Registered `skip_semantic_noop_stamps_before_topk` and `unique_selector_argmax_gather` in
  `state/insights.yaml`. Full-400 rescans found no immediate additional +0.1 target: 12 non-task173
  TopK-to-stamp tasks lacked an exposed complete/no-op subset, and task054/133/324 selector-shaped
  hits were non-unique or too small.
- No pack or submit was run. A full `uv run ng verify` was attempted after the adoptions, but the
  existing task017 exceeded the project's default 600-second isolated timeout and the verifier
  aborted before a 400/400 summary. This is not a task173/174 failure: both were freshly verified
  individually at bundled fail0 and exact-source equivalence immediately before the full-board run.
