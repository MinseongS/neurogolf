# task285 discovery — bounded affine reflection compiler

**Attack types:** `mathematical_rewrite`, `oracle_first`, `output_folding`,
`graph_surgery`, `generalization_risk`

## 1. 현재 incumbent

- Authoritative file inspected: `submission/overfit_nets/task285.onnx`, SHA-256
  `bc3544388fab7276343dc6b4778fe8dabc7acc4cdf23ebe7b768d84027afb4bc`.
- Isolated `uv run ng score 285`: cost **18,674** = memory **18,256** + params
  **418**, score **15.1651125**, bundled **265/265**.
- There is a live-state mismatch that must be reconciled before any adoption:
  `state/manifest.json` and the newest task ledger entry claim cost **17,798**
  (15.2131586), but the deployed artifact above still hashes and scores as 18,674.
  The stricter +0.1 target is therefore 16,104 if the manifest artifact is restored;
  against the file actually on disk it is 16,896.
- Main graph stages:
  1. `Conv(input,Wc)->cf` makes a fp32 colour-label plane.
  2. `Cast/Reshape/Cast/TopK(k=31)` enumerates visible coloured cells.
  3. `[8,31]` neighbour gathers and equality tests locate up to three pivots.
  4. A small `MatMul` decodes shown orientation; `[3,45]` local masks plus two
     `MaxPool`s recover each connected creature, then `TopK(k=9)` enumerates it.
  5. A compact affine renderer emits 81 possible reflected cells into a 900-cell
     label grid with `ScatterElements`; a row/column sentinel tail and final
     `Equal` produce the free BOOL output.
- Cost-dominating intermediates (bytes): `cf` fp32 30x30 **3,600**;
  `gf` fp16[900] **1,800**; `nidx` int32[8,31] **992**; and five uint8
  900-cell aliases/carriers (`c2d`, `g`, `newg`, `out2d`, `fin`) at **900 each**.
  The remaining sparse tensors include `woff`/`widx` at **540** each and
  `tidx` int32[81] at **324**.
- Dominant parameters: `WLUT` **180**, `J1d` **45**, `I1d` **45**, `minit`
  **45**, `MT` **32**; all initializers total 418 elements.

## 2. 규칙 이해

- The input contains one to three connected creatures, each 4--8 cells inside a
  5x5 local frame. A creature is generated around a 2x2 pivot. The four pivot
  cells carry distinct destination-quadrant colours (one colour may be dropped
  to background), while only one reflected orientation of the non-root shape is
  visible.
- The output reflects every visible local offset across both pivot mid-axes,
  restoring all four legal copies. Each destination copy takes the colour of its
  pivot/quadrant anchor. Existing visible cells remain unchanged.
- A legal pivot is exactly a 2x2 window with four distinct labels on the generator
  distribution. This was checked on all **265** stored examples. Given pivot
  `(A,B)`, local offset `(i,j)`, and side bits, every source/destination coordinate
  is affine: `A-i` or `A+1+i`, and `B-j` or `B+1+j`.
- The incumbent implements the rule operationally: enumerate nonzero cells,
  recognize the pivot from neighbour equality, recover orientation/connectivity,
  enumerate up to nine shape cells, compute 81 affine destinations, and scatter.
- What is over-materialized is not the rule but the enumeration representation:
  a 3,600-byte colour plane, its 900/900/1,800-byte aliases, the `[8,31]` neighbour
  bank, and three separate 900-byte label-grid carriers. The affine rule itself can
  be expressed without any of those grids if the output contraction is scheduled
  sanely.

## 3. 예상 개선

- Candidate: `discovery_bounded_affine.onnx`, rebuilt by
  `build_discovery_candidate.py`.
- It replaces the incumbent pipeline with a bounded quadratic pivot certificate
  followed by a Fourier-factorized affine reflection directly into the free graph
  output.
- Counted memory: **18,256 -> 3,600 bytes**. The only counted tensor is the fp32
  `pivot_score[1,30,30]`; **14,656 bytes** of counted intermediates disappear.
- Parameters: **418 -> 5,321 elements** because the spatial affine relation is
  moved into Fourier/route factors. Net cost: **18,674 -> 8,921**, a **9,753-byte
  (52.23%)** reduction. Expected score: **15.9038367**, improvement **+0.7387241**.
  Even against the manifest's 17,798 baseline, the calculated gain is **+0.6906780**.
- The +0.1 thresholds are concrete: cost <=16,896 against the artifact, or <=16,104
  against the manifest. Thus a staged implementation may add roughly **7.2--8.0KB**
  beyond the 8,921 analytical graph and still qualify.
- Optimistic assumptions: ORT must avoid materializing the enormous implicit
  contraction product; Fourier delta error must stay sign-safe; multiple creatures
  must add without destructive overlap; and the four-distinct pivot certificate
  must remain generator-exact. The last point is supported by 5,000 generator
  checks recorded in the existing tests and 265/265 stored checks, but the current
  full ONNX still fails at execution scheduling before accuracy can be measured.

## 4. 기존 실패 재평가

- The old dense flip/shift rewrite was correct on 2,000 fresh examples but cost
  **99,952**. It proved only that a dense multi-pivot representation is bad; it did
  not disprove the affine identity.
- `affine_reflection.onnx` derived the exact coordinate formula, matched the numpy
  oracle on **5,000/5,000**, and priced at **9,225**, but its monolithic contraction
  induced an estimated **291.6TB** temporary. Submission 54688350 returned `ERROR`.
- `bounded_pivot_reflection.onnx` replaced the factorial colour-inequality pivot
  contraction with
  `2T^2 + N^2 + TN - 11T - 2N - 12E`, positive exactly for a legal pivot on the
  finite 2x2 domain. It lowers static cost to **8,921**. Its pivot-only probe runs
  (first inference **11.192s**), proving that portion is executable, but the full
  output contraction still terminates without output.
- This session rebuilt the same graph as `discovery_bounded_affine.onnx`. Isolated
  execution produced no output; `uv run ng gate ... --task 285` returned
  `REJECT`, `fail=None`, `cost=None`, `error="no output"`. This is an implementation/
  contraction-path failure, not a counterexample to the mathematical rewrite.
- `hashscatter.onnx` is a complete bundled lookup oracle: gate **265 pass / 0 fail**,
  but cost **20,877**, so it is useful only as a correctness/control artifact.
- Do not repeat the signed/unsigned INT8 TopK attempts. They can pass local ORT but
  have twice caused Kaggle submission-level `ERROR`; every production TopK feed
  must remain fp16/fp32.

## 5. 공격 가설

### 1. Four-distinct hash pivot + incumbent sparse renderer

- Mathematical idea: encode each colour as `2^colour`; the sum over a 2x2 window
  has popcount four iff all four labels are distinct. A 2x2 fp32 `Conv`, int32
  cast, and uint8 LUT therefore produce the exact pivot mask.
- ONNX/dtype: `Conv(fp32) -> Cast(int32) -> Gather(uint8 LUT)`; use fp16 for any
  production `TopK`, int32 indices until the existing sparse renderer, uint8 label
  carriers, BOOL/free final `Equal`.
- Measured component facts: `hash_pivot_probe.onnx` is runnable and matches the
  four-distinct oracle **265/265**. Component cost with its mask counted is about
  **9,658** (6,728 intermediate + 841 integrated mask + 2,089 params).
- Expected saving: the hash alone is not a win; the target is to delete the
  incumbent's neighbour/pivot/orientation subsystem while reusing the already
  bounded 81-slot renderer. Goal **<=15,500**, at least 3,174 below the artifact.
- Risks: the 2,049-entry LUT is parameter-heavy; a false pivot outside the
  generator distribution; and shape/orientation information may still require
  some of the old `[3,45]` path.
- Minimum first experiment: splice the hash mask only through pivot `TopK`, compare
  its three pivot indices with the incumbent on all 265 stored examples, and print
  the byte delta before reconnecting the renderer.

### 2. Top-3 sparsification before affine output rendering

- Mathematical idea: the task has at most three pivots. Convert the exact
  `pivot_score[30,30]` to three `(A,B)` slots first, then evaluate the affine
  relations only for `3 x 5 x 5`, instead of asking one Einsum to contract all
  900 possible pivots against all 900 outputs.
- ONNX/dtype: `Reshape(fp32)` (or an output-shape-preserving flatten alternative),
  fp16-safe `TopK(k=3)`, `Div/Mod` int32 coordinates, small `Gather`/`Einsum` factors,
  uint8 `ScatterElements`, free BOOL `Equal` output.
- Expected cost: the analytical 8,921 plus a counted 3,600-byte flatten and under
  3KB of slot/renderer tensors is **<15,600**, still +0.18 or better.
- Risks: the reshape is separately counted; zero-score padding slots need a sink;
  multiple creatures may overlap; Fourier sign margins can degrade after fp16
  slot conversion; ORT TopK integer feeds are forbidden.
- Minimum first experiment: build only `pivot_score -> fp16 TopK -> (row,col)` and
  validate exact pivot slots on 265 stored examples before adding any renderer.

### 3. Manual contraction tree with bounded intermediates

- Mathematical idea: preserve the proven Fourier deltas but replace the single
  32-operand output Einsum with an explicit contraction tree. Contract case/side/
  offset axes first, never allowing an intermediate larger than `3x5x5x30`;
  output folding is retained only for the final spatial/channel expansion.
- ONNX/dtype: small fp32 `Einsum`/`MatMul` stages, `Cast(fp16)` only after a proven
  sign margin, and a final fp32/fp16 Einsum directly to free output.
- Expected saving: budget is up to **7,975 bytes** of new tensors against the actual
  artifact while retaining +0.1. A tree with two <=3,600-byte intermediates qualifies.
- Risks: ORT may still choose a pathological internal path; fp16 mixed with the
  free fp32 input is rejected by ORT 1.26; static shape inference must remain exact.
- Minimum first experiment: contract a single pivot and one axis, profile the
  largest ORT tensor and wall time, then scale to three pivots only if the first
  inference is under one second and no tensor exceeds 3,600 bytes.

## 6. 현재 작업 상태

- Rebuild entrypoint: `candidates/task285/build_discovery_candidate.py`.
- Full candidate: `candidates/task285/discovery_bounded_affine.onnx`, static
  memory **3,600**, params **5,321**, cost **8,921**.
- Prior exact candidates/builders retained:
  `build_affine_reflection.py`, `affine_reflection.onnx`,
  `build_bounded_pivot.py`, `bounded_pivot_reflection.onnx`, and
  `bounded_pivot_only.onnx`.
- Tests: `scratch/tests/test_affine_reflection.py` -> **3 passed**; the hash pivot
  component at `scratch/tests/hash_pivot_probe.py/.onnx` -> stored certificate
  **265/265**.
- Isolated full inference: no output / process termination in the output
  contraction. Gate: **REJECT**, `fail=None`, `cost=None`, `error=no output`.
- Complete control candidate: `hashscatter.onnx`, gate **265/0**, memory 12,190,
  params 8,687, cost 20,877, rejected only for cost.
- Failure examples: none have been observed for the affine mathematics because the
  graph does not reach its first output. The failure pattern is resource/scheduling
  termination. The historical incumbent has separate hidden-risk examples around
  row/column zero sentinels; do not use those disagreements to reject a candidate
  without checking the generator oracle.
- Reusable sources: current exact builder `src/custom/task285.py`; rule/failure
  history `state/tasks/task285.md`; stored corpus `data/task285.json`; component
  tests under `candidates/task285/scratch/tests/`.

### 2026-07-15 implementation result — runnable manifest recovery

- Root cause is now reproducible with
  `scratch/tests/reproduce_bounded_timeout.py`: on the same stored input the
  pivot-only graph returns a `[1,30,30]` plane in **8.64s** (provisional under
  concurrent repository work), while the full graph times out after **12s** with
  no output. The serialized 32-operand output equation places four dense input
  operands before their anchor factors; its prefix live-index set peaks after
  operand 4 at **59,049,000,000,000,000 fp32 elements = 236.196 PB**. The failure
  boundary is therefore the monolithic output contraction schedule, not the
  bounded pivot certificate or affine identity.
- A safe fallback was produced by SHA-pinned deployed-seed graph surgery at
  `candidates/task285/build_runtime_candidate.py` -> `runtime_candidate.onnx`
  (SHA-256 `33f21e87ce2a622b56995d27d50915ea4f6e7e12888d0a1877a211f1cb6c7b87`).
  The only transformable seed is the current deployed artifact SHA-256
  `bc3544388fab7276343dc6b4778fe8dabc7acc4cdf23ebe7b768d84027afb4bc`;
  unrecognized deployment bytes are rejected loudly. It reconstructs the lost
  `conv_bias_sentinel` mechanism from the ledger, fuses
  `Add(Add(tA,tB),ga)` to `Sum(tA,tB,ga)`, and deletes constants whose consumers
  disappeared with the old full-canvas sentinel tail. Post-adopt, when the deployed
  file already has the final SHA, the builder validates and returns an unchanged
  clone, so the documented rebuild command is idempotent rather than applying the
  surgery twice.
- Static result: memory **17,359**, params **416**, cost **17,775**, score
  **15.2144517453**. This is **23** below the stale manifest endpoint 17,798 and
  **899** below the artifact currently on disk (18,674). The actual score delta
  against the artifact is **+0.0493392**; the incremental delta against the
  manifest is only **+0.0012931**, so this is a recovery/shave, not the original
  +0.1 bounded-affine prize.
- Isolated gate: **PASS, 265/265, fail=0**. Candidate and deployed artifact are
  bit-identical on all **265/265** stored outputs. Provisional 265-example wall
  times were 0.173s candidate versus 0.092s artifact while unrelated full-board
  jobs were active; rerun runtime after quiescence.
- Proof suite:
  `uv run pytest -q candidates/task285/scratch/tests/test_runtime_candidate.py
  candidates/task285/scratch/tests/test_affine_reflection.py` -> **11 passed**.
  It checks cost, legal fp16/fp32 TopK feeds, exact replacement of the two-Add
  coordinate chain, exhaustive signed-sentinel clamp algebra, and absence of
  consumers for deleted constants. It also proves the canonical artifact is
  byte-identical to a fresh pinned-seed rebuild, post-adopt final-SHA input is
  byte-idempotent, unknown source SHA is rejected, and scratch cost profiling never
  overwrites the canonical ONNX.
- Fresh limitation must not be hidden: the historical arc-gen source referenced in
  the task ledger is absent. The deliberately broad reconstructed generator at
  `scratch/tests/fresh_oracle_check.py` exposes an unknown/OOD pattern: at seed285,
  example80, shape29x21, TopK padding can select an edge background and the inherited
  neighbour `Gather` receives index905 (>899). Do not tune that reconstructed
  generator merely to pass. Bundled gate is authoritative for 8000-overfit adoption;
  a future generalization session must recover the exact generator before claiming
  fresh safety.

## 7. 단독 세션 실행 순서

1. Reconcile the artifact/manifest mismatch without adopting anything:
   `shasum -a 256 submission/overfit_nets/task285.onnx`,
   `uv run ng score 285`, and inspect manifest entry 285.
2. First execute the already-passing recovery candidate:
   `uv run python candidates/task285/build_runtime_candidate.py`, then
   `uv run pytest -q candidates/task285/scratch/tests/test_runtime_candidate.py
   candidates/task285/scratch/tests/test_affine_reflection.py`, then
   `uv run ng gate candidates/task285/runtime_candidate.onnx --task 285`.
   The build command accepts only the pinned pre-transform seed or the exact final
   candidate; after adoption it safely reproduces the final bytes unchanged.
   The root session subsequently completed `ng adopt`; pack/submit remained deferred by
   the unrelated full-board runtime/integrity blockers.
3. For the larger prize, rebuild and re-run the bounded proofs:
   `uv run python candidates/task285/build_discovery_candidate.py` and
   `uv run python candidates/task285/scratch/tests/reproduce_bounded_timeout.py --timeout 12`.
4. Next bounded implementation: splice `hash_pivot_probe` into a copy of the incumbent,
   stop after fp16 `TopK(k=3)`, and compare pivot slots on all 265 stored cases.
5. If the hash/LUT byte price cannot reach **<=16,104**, second implementation:
   use the bounded quadratic pivot plane, immediately sparsify to three slots, and
   rebuild the renderer as `3x5x5` Gather/Einsum/Scatter stages.
6. Pivot if either first inference exceeds one second, any inferred/trace tensor is
   >3,600 bytes unexpectedly, or the projected full cost is >16,104. In that case,
   manually split one axis of the Fourier output contraction and re-profile before
   writing the second axis.
7. Success for the larger rewrite means: source-built full model, fresh-process bundled **265/265**,
   `uv run ng gate ... --task 285` PASS with cost **<=16,104** (the stricter known
   baseline), no signed/unsigned integer TopK, and a fresh generator differential
   against the numpy affine oracle before any adoption is considered.

## 2026-07-15 exact-source synchronization

- `src/custom/task285.py` now owns the adopted cost-17,775 graph without loading
  `submission/`, `candidates/`, `networks/`, or `.npy` artifacts.
- A fresh source build, `build_runtime_candidate.py`, and deployed
  `submission/overfit_nets/task285.onnx` are byte-identical at SHA-256
  `33f21e87ce2a622b56995d27d50915ea4f6e7e12888d0a1877a211f1cb6c7b87`.
- The exact-source generator now preserves populated `producer_name` metadata; this was
  required for byte identity with the adopted runtime-sentinel model.
- Regression: `candidates/source_sync_285_295/test_task285_295_source_sync.py` verifies
  byte identity, full ONNX checking, bundled **265/265**, cost **17,775**, and external-
  artifact independence.

## 2026-07-15 direct 5x5 sparse renderer — ADOPTED

- Root cause/mechanism: the cost-17,775 graph already exposed the visible pivot root `a`
  and both quadrant signs `(e,f)`, but discarded one sign into `var` and therefore gathered a
  conservative `3x5x9` local strip.  Both signs uniquely determine the visible quadrant, so
  `build_sparse_renderer_candidate.py` gathers only the root-corner `3x5x5` patch.  Two bounded
  5x5 MaxPool connectivity passes are retained.  A three-operand Einsum over only `3x9` live
  creature slots emits the three affine destinations; its output is 162B and no new full-grid
  intermediate is introduced.
- TDD/result sequence: the first structural RED required a source builder and explicit 5x5
  `mem2d`; its first GREEN cost was **16,230** and already passed bundled **265/265**.  A second
  RED fixed the +0.1 budget at **16,083**.  Casting only `orient[3,2]` to int32 before the local
  offset MatMul, plus sparse int8 pivot-direction arithmetic in place of `MatMul([3,8],[8,4])`,
  reached **memory 15,793 + params 245 = cost 16,038**.
- Separate pivot certificate: `sparse_pivot_slots.onnx` returns
  `(row,col,quadrant,c00,c01,c10,c11,valid)`.  In a fresh process it loaded in **0.013105s** and
  produced its first output in **0.000862s**.  All **265/265** stored examples matched an
  independent four-distinct-window plus 8-neighbour connected-component oracle, mismatch **0**.
  The first diagnostic oracle incorrectly assumed 4-neighbour connectivity and failed on stored
  train[0]; switching the oracle to the generator/MaxPool's 8-neighbour semantics fixed the
  diagnostic without changing the ONNX candidate.
- Full candidate verification before adoption: `onnx.checker.check_model(full_check=True)` and
  strict shape inference passed; fresh-process load/first inference were **0.007673s/0.000424s**;
  isolated bundled evaluation was **265/265**, fail=0.  The complete counted-tensor dump summed
  exactly to memory **15,793** and params **245**.  Every TopK feed remains FLOAT16.
- Mandatory gate/adopt: `ng gate` PASS at score **15.3172838146**, then `ng adopt` changed
  **17,775 -> 16,038**, score **15.2144517453 -> 15.3172838146**, gain **+0.1028320693**.
  Adopted/candidate/canonical-source SHA-256 are all
  `b201dd621c751a7a9a9352cf7fe6fbc6ac9d35c5c701857ee06953c037b148d4`.
  Post-adopt regression is **20/20** and fresh `ng score 285` remains **265/265**, fail=0,
  cost **16,038**.
- Fresh limitation is unchanged: reconstructed seed285/example80 can still drive the inherited
  K=31 neighbour Gather out of bounds.  Bundled is authoritative and this adoption makes no
  fresh-safety claim.  No generator weakening was used.
- Next pivot: another +0.1 from the new endpoint requires cost **<=14,511**.  The analytical
  bounded-affine cost-8,921 route remains live only with a new <1s pivot/contraction lowering;
  do not retry the 32-operand terminal Einsum or unchanged bounded plane.  A future attack should
  remove the K=31 enumeration or fold the two remaining 900B output carriers while keeping every
  intermediate <=3,600B.

## 2026-07-15 bounded FREE-output re-attack and exact connectivity fold — ADOPTED

- The sparse-pivot analytical re-attack was implemented rather than left as a paper design.
  `build_bounded_free_output_renderer.py` keeps the exact prefix through `(a, acol, e, f)`,
  emits one counted **300B** source-shape contraction, and writes the FREE fp32 output with a
  bounded terminal contraction.  After adding the INT64 casts required by ORT 1.26 OneHot, its
  static result was **memory 12,121 + params 2,294 = cost 14,415**, under the next +0.1 target.
  Checker, strict inference, FLOAT16-only TopK feeds, maximum counted tensor 3,600B, and NumPy
  greedy largest-intermediate estimate **9,000 elements** all passed.
- The executable candidate was rejected at the required runtime boundary.  In a fresh spawned
  process with ORT optimizations disabled it loaded in **0.004216s** but first inference took
  **2.503682s**, exceeding the one-second pivot limit.  It also produced **1,078** positive logits
  on stored train[0], whose exact one-hot target has 900 positives.  Root cause: the terminal
  identity branch uses sign logits `(+1,-1)`, while the proposed numeric colour delta
  `h(2z-h)` is valid only against the different baseline `1-z^2`; it therefore makes multiple
  wrong colour channels positive.  The same graph was not retried or operand-rotated.
- The planned terminal OneHot fallback was independently closed by a pinned-runtime micrograph:
  ONNX permits INT8/INT32 indices, but ORT 1.26 CPU returns `NOT_IMPLEMENTED` for both and runs
  only INT64 indices.  Casting the 900-cell INT8 output label plane to INT64 would add 7,200B,
  so OneHot cannot remove the two 900B output carriers at a lower cost.
- A separate exact strict-cheaper lowering remained.  The first connectivity pass computed
  `m0 = mem2d * singleton(0,0)` (**75B**) and then a 5x5/pad2 MaxPool `p1` (**75B**).  Its output
  is exactly the dynamic root scalar broadcast over the top-left 3x3 block.  The adopted builder
  replaces both tensors with `Slice(root)` (**3B**) -> `Expand(3x3)` (**27B**) -> `Pad(5x5)`
  (**75B**).  Default Transpose replaces the two `[1,3] -> [3,1]` Reshapes, and default Flatten
  replaces `[3,1,5,5] -> [3,25]`, deleting their shape initializers.
- Static result: **16,038 -> 15,982**, memory **15,793 -> 15,748**, params **245 -> 234**,
  score **15.3172838146 -> 15.3207816321**, gain **+0.0034978174**.  The focused suite passed
  **32/32** before gate, including isolated bundled **265/265**, fail=0.  Mandatory `ng gate`
  passed 265/265 at cost15,982, and `ng adopt` recorded the same result.
- Candidate, deployed graph, and self-contained `src/custom/task285.py` rebuild are byte-identical
  at SHA-256 `dc509d96955eea40d6984648934e9b345293057aa61b6f2bf0d575106fa09120`;
  source synchronization is **4/4**.  The post-adopt 400-graph singleton-seed scan found no
  remaining deployed match; its historical seed self-check found task285 and estimated the
  connectivity portion's 52-cost saving correctly.
- The inherited reconstructed-fresh seed285/example80 neighbour-Gather OOB remains unchanged.
  No fresh-safety claim is made.  From cost15,982, the next +0.1 threshold is **14,461**, requiring
  another **1,521** saving.  Reopen the analytical suffix only with a different margin-correct
  colour basis and a runtime plan that is priced below this threshold before execution.

## 2026-07-15 residual sparse exact folds — ADOPTED

- The target-sentinel chain was redundant on all **265** stored examples: its two negative target
  lookups already had invalid `-1` updates.  Removing four 81B tensors plus scalar `i89` changed
  **15,982 -> 15,657**, score **15.3207816321 -> 15.3413266197**.  Focused tests passed **3/3**;
  mandatory gate/adopt both passed **265/265**, fail=0.
- Replacing `si2[3,1]` with direct `Gather(t,si)` / `Gather(c,si)` followed by compact rank
  restoration changed **15,657 -> 15,648**, score **15.3413266197 -> 15.3419016077**.  Focused
  tests passed **2/2**; mandatory gate/adopt again passed **265/265**, fail=0.  Final
  memory/params are **15,415/233**.
- Candidate, deployed graph, and self-contained source rebuild are byte-identical at SHA-256
  `6ce989475ccd96a943a8c8676551a98448b74f0771c20f5e67607d6b4af66213`.  The post-adopt
  structural regression passed and the 400-graph target-sentinel scan loaded every artifact with
  errors0 and found no remaining hit; its historical control found task285 as expected.
- The next +0.1 threshold is **14,158**.  The old cost14,415 FREE-output control is now **257**
  above that threshold before its known sign failure (1,078 rather than 900 positives) and
  2.503682-second first inference.  It was not rebuilt or rotated.  Reopen only with a different
  margin-correct colour basis and bounded contraction satisfying cost<=14,158 and first
  inference<1s.
- The inherited reconstructed-fresh seed285/example80 neighbour-Gather OOB remains unchanged.
  No fresh-safety claim is made, and no generator weakening was used.
