# NeuroGolf rebuild playbook (S8 consolidated — read THIS first, follow file pointers only as needed)

Score = 25 − ln(mem+params). This file is the single onboarding doc for per-task rebuild agents.

## Grader counting model (proven on LB, data/neurogolf_utils.py)
- Only NODE OUTPUTS are counted (inferred static shape bytes, maxed with ORT profiler trace).
  Graph `input`/`output` are FREE; a FREE `>0` threshold is applied to `output` by run_network.
- Everything INSIDE one op is uncounted → a multi-operand Einsum does unlimited internal work.
- params = initializer ELEMENT count (dtype-independent) → fp32 tables/projections as
  initializers are ~5× cheaper than same-size counted planes.
- Banned ops: Loop/Scan/NonZero/Unique/Script/Function/Compress + Sequence + subgraphs.
- 🚨 NEVER feed TopK an unsigned dtype (uint8 TopK passes ALL local gates but ERRORS the whole
  Kaggle submission). float/fp16/int64 feeds only.

## Proven mechanism toolbox (each: one-line + reference implementation)
1. **Walk einsum** (iterative flood → ONE einsum; #walks>0 = reachability): `src/custom/task187.py`,
   recipe+traps `reports/tasklog/task187.md`. 8-conn step = S[r,r']·S[c,c']·mask (2 letters, ≤23
   steps/einsum); 4-conn alternating axis = 1 letter/slot. Nonneg operands ONLY; dyadic or 1.0
   scales; ORT contracts pairwise LEFT-TO-RIGHT → order operands along the walk chain.
2. **Free-input traversability** (no counted mask plane): repeat graph `input` as operand with
   (10,) colour selectors; all traversability slots share ONE colour letter pinned by P0[q].
   `src/custom/task243.py`. Chain einsums for distance (slot cost 2D−1 across a chain boundary).
3. **S=1.0 exactness trick**: reached ⇒ W≥1 exactly → single `Greater(t, W)` epilogue node.
   Ring seeding as 4 nonneg (G,H) rank-1 pairs. `src/custom/task002.py`.
4. **Checkpoint constraints**: triangular/band operands bind interior walk positions to output
   indices (row(p_k)≤r etc.) → bbox/interval logic inside ONE einsum. `src/custom/task077.py`.
5. **Gates-as-einsum-operands**: data-dependent selection = tiny einsums → [K] vectors → one-hot
   gate g = valid·[U·valid==0] multiplies INSIDE the big einsum; heterogeneous fallback = stacked
   operand plane. `src/custom/task110.py`.
6. **Exact-count multiplicity-free walks** (run-length/area/size): 3-phase monotone machine
   {Stay, Move…} — each cell reached by exactly ONE walk ⇒ einsum value = the count. fp16 exact
   ≤2048. Self-loop walks CANNOT count (C(N,d) multiplicity). `src/custom/task145.py`.
7. **Counting-model rebuild kit** (non-iterative nets): free-input einsum bounds/profiles
   ('bchw,c->h' etc.), spread-based uniformity tests, separable two-stage axis Gathers,
   single-tap valid-Conv label read (crops in-op for free), QLinearConv i32-bias folds,
   Pad-with-negative-crop. `src/custom/task209.py`.
   S11 output-routing crossover: final one-hot emission = Equal-then-Pad (10·h·w bytes,
   bool Pad legal) vs Pad-then-Equal (fixed 900B index carrier) → pick Equal-then-Pad iff
   content area < 90 cells (task259 vs task041). S11 recast trap: fp16/u8 recast is INVALID
   for a float plane whose producer consumes the fp32 FREE input directly (ORT binds output
   dtype to input dtype; a Cast after it ADDS a plane) — PRODUCER_BOUND class, needs
   producer-replacement surgery instead.
8. **Moment-statistics detection**: per-colour n, Σr, Σc, Σr², Σc² from five [10]-output
   free-input einsums → O(1) counted bytes for "find the confined/small component colour".
   `src/custom/task158.py`.
9. **Batched-K block + placement einsum**: N copy-pasted per-band/per-case blocks → one block
   with leading dim K; placement/accumulation = 'kjr,ks,jsc,k->rc'. `src/custom/task219.py`.
10. **Parallel-plane conv-channel union**: K parallel dilation/detector planes → ONE conv with K
    output channels + count union (Min/(>0)). `src/custom/task349.py`.
11. **Sparse-edit ScatterND chain**: write line/ring cells as 255 with reduction='max' →
    duplicate writes = idempotent unions; final Equal vs 255-adjusted table. `src/custom/task054.py`.
12. **Checkerboard/parity paint**: rank-1 stacked-parity einsums 'nqrc,sr,sc,q->nq'. `src/custom/task286.py`.
13a. **EPILOGUE FOLD (S8 breakthrough)**: heterogeneous output (copy input colours + fixed
    recolours) folds into ONE einsum by riding a stacking index s through the walk chain with
    a signed mixer T[s,v,w] (T[0]=identity-copy, T[s]=δ(v,k) terms; every term must contain an
    input factor at (r,c) to silence off-canvas; S entries 1.0 for exact 1−W signs; ellipsis
    batch dim frees a letter). Reference: src/custom/task187.py (7200+2502 vs 15300+1176).
    GRADER STATUS: PROVEN on LB (54267065 COMPLETE 7197.62) — safe to mass-propagate.
    APPLICABILITY: only wins when it ELIMINATES separate pre-output planes (label/flood);
    measured NEUTRAL when the output pad merely swaps 1:1 with a small-grid one-hot (task044).
13. **Fresh-fail budget spending**: if the incumbent fails X% fresh, a CHEAPER rule that fails
    ≤X% (subset ideally) is landable — gate is candidate ≤ incumbent, not 0. `src/custom/task023.py`.
14. **Separable-remap einsum (S9, from kojimar teacher)**: any FIXED spatial remap/upscale
    out = P·in·Pᵀ per channel, with P factored as U@S (U = out-row→latent block table,
    S = latent→source-col table), is ONE 5-operand einsum 'ra,ai,zcij,bj,sb->zcrs' —
    mem=0 (single node writes the free output), params = |U|+|S|. Killed task108 864B→0
    (+1.175). Candidate lever for any fixed block-upscale/tile/sublattice-read task.
    `src/custom/task108.py`.

15. **Signed-channel priority overlay (S11)**: grader decodes `(out > 0.0)` per channel
    (src/harness.py:218) ⇒ overlap/paint-order priority is LINEAR — no [30,30] label/priority
    carrier. Each fill class q contributes fill_q(r,c)·W[q,v] with SIGNED channel vectors
    (loser classes get suppressed negative at overlaps: horizontal q → e_q−e_0, winner
    vertical q → (M+1)e_q−M·𝟙, M ≥ max fill multiplicity; bg = extra slot e_0). Separable
    axis-aligned fills ride ONE free final einsum `sr,sc,sv->vrc`. Killed task092's 3-plane
    scatter epilogue (6872→5399, +0.241; fresh 7000/7000 bit-identical). Reference:
    `src/custom/task092.py`. FLOOR it hits: data-dependent 2-D interval fill still needs
    [10,30] band profiles + compare transients ≈3000B (einsum rejects uint8 → fp16 operands).
    APPLICABILITY: output = union of separable rect/segment fills with a FIXED per-class
    overlap ordering; wins where the incumbent pays for scalar-label/priority/canvas planes.
    S11 cohort sweep verdict: 233/285/370/133/054/366 ALL KILL (costs sit in detection
    reads / assignment / sprite-stamp machinery, which signed-W cannot touch; sparse_scatter
    class label ≠ separable fills). Hits so far: 092/234/335. Screen candidates by RENDERING
    the output and checking separability + constant colour roster, not by class labels.
    S11 COMPOSITION CONSTRAINT (task084 pricing): only ONE op writes the free output —
    free-einsum + residual-scatter hybrids cannot compose (counted [1,10,30,30] bridge
    = 18-36KB); the fold is ALL-OR-NOTHING. If any output component is non-separable and
    data-dependent (e.g. A-dependent anti-diagonal), the whole fold pays a counted
    [K,30,30] fp32 operand → ScatterElements-into-FREE-input beats it. Also:
    ScatterElements updates are dtype-bound to data (fp32 input ⇒ fp32 updates, no recast).

## Reject-checks (priced floors — don't re-attempt without a new idea)
- S11 Conv→int8 QLinearConv on PRODUCER_BOUND colour-index Convs: DRY WELL (0/32 net wins,
  3 measured refutation builds at reports/candidates/task{074,080,383}_qconv.py). QLinearConv
  needs a QUANTIZED input → free 10-ch fp32 input forces a counted 9000B uint8 copy >> any
  single-channel output saving (≤2700B). The S10 wins (264/184/365/191) fed it ≤2-channel
  integer planes — that's the only working shape. The per-cell colour read off the free input
  stays floored at G²×4 fp32 (8th independent confirmation of the detection floor).
  Full data: reports/int8_ranking_scan.{md,json}, reports/dtype_overpay_scan.{md,json}.
- Parallel fan-out MaxPools/convs (per-size anchors) ≠ collapsible chain (task204).
- Sub-400B uint8 conv banks are einsum-proof: einsum entry ticket = fp32 spatial output
  (1600–3600B) + step params (task204/023 — but 023 shows ROUND GOLF can still win there).
- Radius-gated growth (window↔dilation coupling) needs phase-gated shift tensors ≥7–9k params (task349).
- COPY-class label epilogue: SOLVED by toolbox 13a (epilogue fold) — the old "~8–12KB uint8
  label epilogue floor" no longer applies where the pre-output computation can join one einsum.
- Single-tap valid-Conv crop (S9): wins ONLY when the 30×30 plane is a COUNTED entry read
  (task396 +0.147, task193 +0.136, task222 +0.086). It BACKFIRES on free-input walk-einsum
  nets: cropping makes formerly-free in-einsum input reads counted, forces a 900B output
  re-Pad, and splits shared square T-tables into rectangular pairs (task077: cand 9167 vs
  6331 floor). Dense kernel params (k²×10) also eat the win — check the trade curve first.
  EXCEPTION (task187 +0.153): when the walk einsum IS the final op, re-embed the cropped
  index to the 30-dim output axis INSIDE the einsum via P[r',R]·P[c',C] identity embeds
  (free input supplies the 30-dim axis) — costs 1 walk step + ~750 params, avoids both
  the counted fp32 pre-Pad plane and the task077 blocker. Verified generator bounds
  table: reports/grid_crop_bounds.md.

- LSTM/GRU/RNN (S9 scout, priced DEAD for current pool): grader-accepted; omitting the Y
  output makes ALL per-step states uncounted (Yh = 120B, 30 free nonlinear steps/node,
  RNN-ReLU sweep bit-exact 500/500) — a real loophole, BUT entry ticket = counted 3-D
  X-prep plane (≥4G²) + consumed Y for painters ⇒ ≥2× walk-einsum cost on everything
  expressible (task002: 6100+589 einsum vs ~14-19K RNN 2-D flood). Keep ONLY for a future
  sequential phase/reset machine no multilinear walk can express; canary-submit first.
  Evidence: scratchpad lstm_bet exp1/exp2 (S9).

## ORT/grader gotchas (all hit in S8)
- No bool `Where` kernel on ORT CPU → mux via u8. u8 Min/MaxPool (incl. global) OK. fp16
  Einsum/Equal OK. fp16 einsum with masking can NaN via 0·inf — prove ranges or use fp32.
- `np.ascontiguousarray` promotes 0-d → shape (1,) (breaks strict shape inference); use np.asarray.
- Rank-0 initializers OK but every stale value_info must be rewritten to [].
- sanitize_model requires globally-unique node output names.
- Latency: repeated-operand SIZE dominates multi-operand einsum cost (9000-elem input repeated
  46× = 330ms; one 625-elem mask plane = 31ms). Keep <100ms/run. Report per-run latency.
- Bundled (original-ARC) examples may VIOLATE generator guarantees (task077/023/054) — validate
  on BOTH fresh AND bundled; some incumbents memorize originals via trigger probes (task133).
- TopK/ArgMax tie-break = ascending index; scan-order tricks rely on it (task233).

## Procedure & gates (run from repo root; .venv/bin/python ALWAYS)
1. Read generator (`reports/arc_mapping.json` → `arc-gen/tasks/task_<arcid>.py`) + incumbent
   `src/custom/taskNNN.py` + `reports/tasklog/taskNNN.md`. Byte-rank the counted planes first.
2. numpy-validate the rule/replacement on ≥20000 fresh. Import generator with
   `sys.path.append('arc-gen')` (NOT insert-0). Measure the incumbent's own fresh-fail rate.
3. Build candidate in the session scratchpad as `cand.py` with `build(task)` (fresh_verify execs
   it with `__package__='src.custom'`; relative imports `._exact`/`..harness` work).
   NEVER overwrite `src/custom/taskNNN.py` or `networks/taskNNN.onnx` (fresh_verify self-trap).
4. Stored gate: `src.harness.evaluate` → fail=0 AND memory+params strictly below incumbent.
5. Fresh gate — FAST ITERATION vs FINAL:
   - iterate with the cache: `.venv/bin/python reports/scripts/fresh_verify.py N N N --cache`
     (pre-generate once: `PYTHONPATH=. .venv/bin/python reports/scripts/fresh_cache.py TASK 2500`).
   - FINAL gate must include one UNCACHED run (≥500) so the candidate can't overfit the cache:
     candidate fail ≤ incumbent fail. (VALIDATED: a cache-greedy-fitted variant showed 1 fail
     cached vs 23 uncached on task017 — cached fail understates fitted-parameter fail ~3×.)
6. Report raw data: candidate path, mechanism, stored eval dict, fresh_verify verbatim,
   measured distances/margins, per-run latency, traps hit. Negative results (priced floor
   decompositions) are valuable — report them fully.
