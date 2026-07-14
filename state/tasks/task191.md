---
deployed_cost: 11044
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task191 — 7df24a62

**Rule:** A blue square (channel-1 frame) encloses a small yellow pattern (tall×wide, tall∈{1,2,3},
wide∈{2,3}, touches all 4 bbox edges, exactly max(tall,wide) yellow cells) on a 23×23 grid littered
with scattered yellow noise dots. For every grid position and every dihedral orientation (4 rot × 2
xpose) where the yellow noise EXACTLY equals the oriented pattern (all pattern-yellows present AND no
extra yellow inside the oriented bbox), draw a blue box = oriented-bbox dilated by 1. Overlay the
yellow dots on top. (The reference sprite reproduces itself.) Generator only emits non-illegal
instances, so boxes never collide with the sprite frame / off-grid.

**Current (deployed):** 14.25 pts (ext:kojimar7113 crowd net). Prior custom 13.77 (mem 74258) was
WORSE than the crowd net. → **14.62 pts new custom**, mem 31276, params 844 (beats 14.25 by +0.37).
**Target tier:** detection (8-orientation template match) — NOT a multi-object-correspondence BAIL:
the match is a pure binary correlation expressible as a single stacked Conv.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | fp32, two Convs (corr+tot), D5 ConvTranspose | det | 199974 | 1113 | 12.79 | 200/200 | ok |
| 2 | + fp16 working planes | det | 129296 | 1114 | 13.22 | 200/200 | ok |
| 3 | + crop conv canvas to 23×23 grid | det | 94864 | 1128 | 13.53 | 200/200 | ok |
| 4 | + stamp mask3 (3×3) then single 3×3 dilate | det | 93394 | 1120 | 13.54 | 200/200 | ok |
| 5 | + COMBINED match kernel (fold tot into corr) | det | 74258 | 1122 | 13.77 | 500/500 | superseded |
| 6 | ConvTranspose+ReduceMax(8ch) -> forward grouped-SUM Conv (1ch) + PAD 2->1 | det | 53206 | 2034 | 14.08 | 200/200 | ok |
| 7 | bbox via blue profile-Convs; Y fp16 path; in-grid constant | det | 47248 | 2627 | 14.18 | 200/200 | ok |
| 8 | match via biased-Conv + Relu (drop Equal-bool & fp16 Cast) | det | 39704 | 2628 | 14.35 | 200/200 | ok |
| 9 | output via ONE uint8 colour-index + Equal (drop 7 bool Concat) | det | 37904 | 1742 | 14.41 | 200/200 | ok |
| 10 | whole pipeline at 23x23; uint8 Pad-99 to 30x30; clamp K3 gather idx | det | 31276 | 844 | **14.62** | 500/500 | **adopt-candidate** |

## Best achieved
**14.62 @ mem 31276 params 844 — beats deployed kojimar7113 (14.25) by +0.37 (≥+0.3 ✓).
fresh 500/500 + 267/267 stored.**

## Irreducible-floor analysis (new 31276 build)
The match pair `corrm`=Conv [1,8,23,23] + `M`=Relu = 16928B (54% of total) is the HARD floor: 8
dihedral orientations MUST be matched (measured: xpose-only matches contribute in 70% of fresh
instances, so cannot drop to 4 channels), each a 23x23 fp16 plane, and the stamp Conv needs a float
copy of the indicator (corr fp16 + relu fp16). The match scores are integer & fp16-exact (corr<=npat,
range [-200,npat]). All else is tiny: Y 23x23 fp32 slice (2116, forced — fp32 input), three 1058
fp16 planes (Yg/placed1/boxsum), one 900 uint8 colidx, and a handful of 288B scalar planes.
NOTE: ORT sometimes fuses corrm->M (Relu in-place) so the trace counts the pair ONCE (8464) — that
flips the score to ~14.9; it's graph-order-dependent and not relied upon here (we count both = 14.62).

## OPEN ANGLES (further compaction, not needed for the win)
- Fold the per-orientation ReduceMax(placed) earlier or stamp into ONE channel via summed
  ConvTranspose to drop the 8-ch `placed` plane (~11KB → ~1.5KB).
- fp16 the output-assembly fp32 planes (B/Y/ingrid 30×30) — compute ingrid on the 23×23 crop.
- Drop PAD top/left if a re-derivation shows edge anchors never go negative (would shrink the
  conv canvas 27→25).

## 2026-06-28 re-attack notes
- Tried replacing `Relu(corrm)->M fp16` plus fp16 stamp Conv with
  `Greater(corrm)>0 -> Cast uint8 -> QLinearConv`. Direct uint8 MaxPool is invalid in ORT
  (`MaxPool` rejects tensor(uint8)).
- Tried `QLinearConv -> Cast fp16 -> MaxPool`; stored/fresh passed (120/120), but score fell
  from 14.622859 to 14.608577 (`memory=31733`, `params=846`). Do not adopt.
- Tried changing active grid `G` to 20/21/22/24/25; all fail stored. `G=23` remains fixed by
  generator/output alignment.

## 2026-06-29 orientation-group probe

Hypothesis: reduce the 8 orientation match channels before stamping, because
the drawn blue box might depend only on oriented bbox shape.

Result: rejected as a general mechanism for this task.

- On square `3x3` patterns, all 8 stamp masks are identical.
- On non-square or sparse patterns, the dynamic `Mconv` stamp can split into
  4 or even 8 distinct masks depending on the oriented footprint.
- Therefore a fixed 2-group or 4-group `ReduceMax(corrm)` before `Relu/stamp`
  is not semantics-preserving.

The main 8-channel `corrm` and `Relu(M)` floor remains real unless a different
operator can threshold positive matches without materializing the 8-channel
post-activation plane.  Prior `Greater -> QLinearConv -> Cast fp16 -> MaxPool`
passed but was slightly worse.

## 2026-06-29 no-Relu stamp probe

Hypothesis: remove the `Relu(corrm)->M` 8-channel fp16 plane and feed `corrm`
directly into the stamp Conv. This would drop memory from `31276` to roughly
`22812` if valid.

Result: rejected immediately.

- Temp graph redirected `placed1` Conv input from `M` to `corrm` and removed the
  Relu node/value_info.
- Stored eval: `pass=0`, `fail=267`, `memory=22812`, `params=841`.

Conclusion: negative non-match correlation scores are not harmless; they
poison the positive stamp accumulation. A threshold/nonnegative activation
before stamping is semantically required. Future attempts must replace Relu
with a cheaper thresholding/stamping primitive, not simply delete it.

## INSIGHT (transferable)
⭐ **8-orientation dihedral template matching is NOT a shape-correspondence BAIL** — it is a stacked
Conv: extract the small pattern as a 3×3, build the 8 oriented kernels as FIXED gather-permutations
of the 9 flattened elements (rot90/T = constant index maps PERMS), and run all 8 as the
output-channels of ONE Conv weight [8,1,3,3].
⭐ **Fold a two-predicate window match into ONE correlation kernel via signed weights:** to test
"all K pattern cells present AND no extra inside the bbox" in a single Conv, use
`combk = Ko*(1+B) - B*mask3` (pattern → 1+B, extra-in-bbox → −B); `Conv==npat` is exact (fp16-safe
for B=100, sums < 2048). Removes the separate "total-in-bbox" Conv plane.
⭐ **A data-dependent small-window readout (Gather a 3×3 at a runtime bbox corner) can over-read** —
the 3×3 frame exceeds a smaller tall×wide sprite and silently captures adjacent noise; mask
rows≥tall / cols≥wide (derived as scalars from the bbox extent) before using it.
⭐ ConvTranspose(M, stamp, group=C) is the clean "scatter a fixed stamp at every firing anchor"
primitive; reduce over channels then a single MaxPool dilates the union.

## Safe-golf pass (S4, 2026-06-30)
Bit-identical dtype narrowing: Gather index intermediates `ridx` [3] and `cidx` [3]
came from `Cast(to=int64)`, each feeding ONLY a `Gather` index input (bbox-corner coords,
fit int32). Narrowed both `Cast`→int32 (`to=6`) + matching value_info.
- **mem 31276 → 31252** (−24B), params 841 (unchanged), **pts 14.6229 → 14.6236 (+0.0007)**.
- Gate: bundled fail=0; equivalence vs incumbent = **0 divergences / 1602** random
  in-domain recolorings. Grader-safe (int32 Gather index, not *ND, not TopK).
  (Task remains floor-bound — dominated by a ~31KB fp32 detection plane; this is the
  only landable lever.)

## S8 (2026-07-02) — dihedral-match-in-einsum (+0.420) ADOPTED, div 0
Detect/threshold/stamp block (~23KB: corrm Conv [1,8,23,23] + Relu + glue) → ONE 49-operand
44-letter einsum producing the box plane [23,23] directly. NEW TRICK: exact 8-orientation
pattern match as PRODUCT-OF-SUMS inside the einsum — per window cell a 2-branch factor
(α_c[o]·CONST + β_c[o]·Y[shifted]) with α=1−K, β=2K−M; the CONST branch reads input at fixed
on-grid (0,0) with all-ones channel weights (≡1), giving exact pad-black semantics for
don't-care cells sticking off-grid (naive [1−Y,Y] basis silently kills those matches).
Stamp folded via Q[g,i,y]=[−2≤y−i−g≤0] reused for rows+cols. 14070+6576 vs 22971+8464 →
14.644→15.065. Fresh 2500 (uncached) + 2500 cached + 500 uncached + 600 vs live onnx: div 0.
Latency 27.9ms. Old "8-orientation grouping rejected" verdict applied to pre-stamp REDUCTION,
not to keeping o as an einsum axis. Product-of-sums = general template for k-orientation/
k-template matching without materializing per-orientation planes.

## S9 (2026-07-03) — kojimar teacher REJECTED (fresh 23/2500 = 0.92%)
Teacher = QLinearConv int8 template match with RUNTIME-built dihedral kernels
(Slice/Transpose/flip/Concat of gathered 5×5 motif) + int32 bias threshold → params 171
vs our 6576. But int8 quantization not bit-exact: 23/2500 fresh fails vs our 0.
Stored +0.44 is illusory. KEEP exact fp32 product-of-sums einsum.
⭐ QLinearConv runtime-kernel matching = real param lever ONLY for tasks already
spending a fresh-fail budget (toolbox #13) — cannot displace an exact 0-fail einsum.


## S10 (2026-07-03) — bobmyers7186 teacher ADOPTED (+0.456, policy-gated)

**Gate-policy note:** the fresh gate was relaxed this session — bundled fail=0 stays
mandatory (public LB grades bundled), but the fresh gate drops from "cand ≤ inc" to
"~98%+ fresh pass → adopt and verify by real LB submission" (fresh-gate = private-LB
insurance only; the bobmyers/kojimar packs already survived the public LB at 7185+). This
net was **rejected in S8/S9 under the old strict rule** and is adopted now with its
fresh-fail rate recorded for private-LB risk tracking. A verification LB submission is planned this session.

**Mechanism diff (STRUCTURAL SWAP, retired vs new):** the incumbent was our S8 exact fp32
**product-of-sums dihedral einsum** (one big `Einsum` fed by orientation tables
`Q[3,23,23]` + `RS0/RS1/RS2[2,23,30]` ≈ 6576 params). The teacher replaces it with an int8
**`QLinearConv`×2 template match using RUNTIME-built dihedral kernels** (Slice/Transpose/flip/
Concat of the gathered motif) + `ArgMax`×4 + a small `Einsum`×2 assembly; orientation tables
collapse to runtime construction. params 6576→171 drives almost the entire −7556 cost drop.
This is exactly the S9-rejected kojimar-style int8 approach (bobmyers variant); int8 quant is
NOT bit-exact → a small fresh-fail budget. Directly reverses the S8/S9 "KEEP exact fp32
product-of-sums einsum" decision, now justified by the relaxed gate + the large cost win.

**Cost:** mem 14070→12919, params 6576→171, pts 15.0647→15.5204 (**+0.456**, cost 20646→13090 −7556).

**Gate evidence:** bundled 267/267 fail=0 (both nets). Fresh 2000: candidate **19 fails
(0.95%)** vs incumbent **0 fails**. TopK audit: no TopK in either net (match via QLinearConv+ArgMax).
Chosen over the kojimar variant (0.70% fresh but only −911 cost); bobmyers is far cheaper.

**Backup + provenance:** incumbent → `reports/retired_networks/task191_pre_s10.onnx`;
candidate source `public_candidates/bobmyers7186/task191.onnx` → `networks/task191.onnx`;
source regenerated via live_to_exact_source --write-src, src↔live reconciled fail=0.

Adopted under S10 relaxed gate (bundled=LB gate; fresh ≥98% → submit-verify); private-LB
risk = 0.95% fresh fail rate.

⭐ TRANSFERABLE: int8 `QLinearConv` with **runtime-built dihedral/oriented kernels** is a
massive PARAM lever (6576→171) for k-orientation template-match nets — but it is NOT
bit-exact (per toolbox #13 / the S9 note), so it is only viable now under the relaxed gate.
Selection: any k-orientation / k-template match net still carrying large fixed `[k,H,W]`
orientation-kernel initializers feeding a match einsum/conv, where a ~1% fresh-fail budget is
acceptable. Do NOT use it to displace an already-exact 0-fail einsum unless the relaxed gate
and a real cost win both hold.


## S15 (2026-07-06) — ADOPTED from urad public bundle 7225.82 (submission 54367833): 11882 -> 11320 (+0.048)
Mechanism: value_info Slice crop + QLinearConv.
Gate (fresh_verify, inc/cand fail on 1500-2000): 17/17 -> adopted under safe rule (cand fail <= inc fail AND cheaper).
Source-owned via live_to_exact_source --write-src; re-measured grader-side fail=0. Backup in scratchpad/backup_networks.
See memory [[neurogolf-urad-7225-bundle-vein]]. 
## S16 (2026-07-06) — fp16-recast reduce plane: 11320 -> 11282 (+0.0042)
Mechanism: motif_i32_sum was [1,1,5,5] int32 (100B) only to feed ReduceSum. motif in {0,1}, sum<=25 -> fp16 exact. Cast to=6 -> to=10 (fp16, 50B) + scalar Cast(fp16->int32) after ReduceSum. mem 11178->11130. Source-owned (src/custom/task191.py), rebuilt via rebuild_networks_from_source. Fresh-gate 2000/2000 cand!=inc=0 (bit-identical), fail=0. fp16 of the input-derived f32 planes (yellow_f32/scores) = FLOOR (needs 18000B input cast, measured mem 27880 LOSS); valid_u8_q 6ch = orientation floor.


## S16 adoption (2026-07-06) — yuu111111111 public-bundle net (+0.016)
- Source: yuu111111111/neurogolf-6-failure-modes notebook (total 7235.05, embedded 400-net archive; MINED per-task despite lower total).
- New grader cost = 11092 (mem 11009 + params 83), fail=0 bundled.
- Fresh-gate 1500: incumbent fail = 13 | candidate fail = 13 | candidate != incumbent = 0  -> cand_fail <= incumbent_fail (safe rule PASS).
- Mechanism: structural golf: fewer counted node-output intermediates (graph rewrite, functionally equal on fresh).

## S? re-confirm 6ch bank floor (2026-07-09, opus agent) — NO BUILD
- Ran: per-channel necessity table on valid_u8_q (drop ch -> bundled fails): ch0=100, ch1=41, ch2=40, ch3=59, ch4=1, ch5=1. ALL 6 load-bearing (even ch4/ch5 each carry a whole painted shape: 12-cell in arc-gen #223, 22-cell in #27).
- Verdict: channel-ablation DEAD (no dead/redundant channel). task367-style single-fail repair does NOT apply: the miss is a full shape not an edge cell, and validity/exclusion kernels are computed dynamically per-example (Concat of motif transposes) so there is no static weight to tweak — re-adding detection = restoring the channel. Two convs un-fusable (uint8 clamp between them is essential nonlinearity; uint8 already min dtype). yellow_f32 (2116B) = minimal irreducible fp32 carrier crop. No reduction >=800B survives.
- Tool+date: opus agent (candidates/task191/ablate.py necessity harness), onnx 1.21.0 / ort 1.26.0, 2026-07-09.
- Reopen triggers: (1) a primitive that detects a motif in multiple orientations within a single conv output channel without summation cross-talk (collapses the 6ch bank); (2) a way to get the uint8 yellow mask directly from fp32 input without any >=529-elem fp32 intermediate.
- Falsification history: prior S16/motif fp16 recast wins are exhausted; this entry adds the full per-channel necessity proof that the 6ch bank itself is floored.

## 2026-07-11 — fresh-tail diagnosis → (a) FIXABLE RULE-GAP (int8 under-detection, self-inflicted by S10)
ran: verified deployed submission/overfit_nets/task191.onnx == src.custom.task191 (0 divergences/400 fresh,
  isolated ORT DISABLE_ALL). It is the S10 bobmyers QLinearConv int8 template-match net (cost 11044,
  15.690 pts). Reproduced the tail on 6000 fresh generate() draws → **50/6000 = 0.833% fresh-fail**
  (matches the reported 0.8%). Failure-mode decomposition over all 50 fails (blue ch1 / yellow ch4 diff
  vs ground-truth, connected-component analysis): **50/50 (100%) = pure blue-box UNDER-detection** —
  exactly ONE whole oriented-match box missed per fail (1 connected component each, 12–22 cells = a full
  dilated bbox; missed outer dims 4×4/5×4/4×5/5×5/3×5/5×3 ⇒ oriented (t,w) dilated+1, spanning all valid
  pattern sizes/orientations). ZERO spurious boxes, ZERO yellow-overlay errors, ZERO wrong-dilation. The
  net simply drops one exact match whose int8 QLinearConv correlation score fell just under threshold.
  The generator output is a DETERMINISTIC function of the input (draw() computes `matches` purely by
  exact-equality scan of the grid — no random tie-break in the output), so there is NO generator
  ambiguity: every missed box is a genuine exact match a correct net must fire. Empirically confirmed a
  fixable class: the on-disk exact fp32 MatMul/Equal net (dumps/evgendvorkin_eda/task191.onnx, no int8)
  **fixes 50/50 of the deployed net's misses and scores 0/1500 fresh-fail** (bundled 267/267).
tool+date: direct-ONNX repro harness (scratchpad diag191.py/analyze191.py/detmap191.py/testexact191.py)
  vs arc-gen generator, 6000+ fresh draws, onnx 1.21.0 / ort 1.26.0 DISABLE_ALL, 2026-07-11 (fork).
verdict: (a) FIXABLE RULE-GAP — the 0.83% tail is **self-inflicted** by the S10 relaxed-gate swap that
  replaced our S8 exact fp32 product-of-sums einsum (0 fresh-fail over 5600) with the cheaper int8
  QLinearConv (−7556 cost, +0.456 pts) carrying a documented ~0.95% int8 non-bit-exact fail budget. This
  re-measures that budget at 0.833% and proves it is 100% under-detection, not ambiguity. FIX DESIGN:
  revert to an EXACT (non-int8) matcher. Cheapest known exact form = REBUILD the S8 product-of-sums
  einsum (mem 14070 + params 6576 = cost 20646, 15.065 pts) — NOTE it is NOT retained on disk (reports/
  tree retired), so this is a rebuild not a swap. PRICE: cost 11044→~20646, pts 15.690→~15.065 =
  **−0.625 CERTAIN pts** to remove the 0.833% hard-zero tail. (The on-disk exact MatMul/Equal net is a
  ready alternative but far pricier: cost 44794, 14.290 pts = −1.40 certain — do NOT use it; rebuild the
  einsum.) ECONOMICS: task scores 0 if ANY hidden draw fails (all-pass). Keep-int8 EV = 15.690·(1−p)^k,
  fix EV = 15.065, p=0.00833. Break-even **k ≈ 4.9 hidden draws/task** (fix net-POSITIVE iff hidden set
  draws ≥5 instances/task; at k=1 it is EV-negative by ~0.5 pt but removes a hard-zero private-LB tail).
  Same open question as task205/002: the fix is worth building only if hidden-draw-count/task ≥5 is
  established, or as pure private-LB variance insurance.
reopen: build the exact-einsum revert (tail→0) if hidden-set draws/task ≥5 is ever established, or a
  public dump ships an exact task191 net cheaper than ~20646; re-measure if a lower-cost 0-fresh-fail
  matcher (e.g. int16/fp16-exact correlation avoiding int8 quant) is found that keeps cost near 11044.
falsification history: first fresh-tail diagnosis of task191. CONFIRMS and quantifies the S9/S10 ledger
  note that int8 QLinearConv is not bit-exact (~0.95% → measured 0.833%); ADDS the proof that 100% of
  the tail is single-box under-detection on a deterministic (non-ambiguous) generator, so — unlike
  task157 (heuristic plateau) and task209 (0.83% genuine ambiguity floor) — this tail is fully
  removable by an exact feedforward net at a known certain-pt price.

## 2026-07-12 — EXACT HEDGE BUILT: 8-orientation QLinearConv (cost 12348, tail→0) — ROOT-CAUSE REVISED
ran: (1) verified the deployed 0.83% tail is NOT int8 inexactness — dumped valid_u8_q (the QLinearConv
  match plane) vs an integer numpy correlation using the SAME runtime kernel/bias on EVERY deployed fresh
  failure: planeDiff=0 (bit-exact) every time. (2) Decomposed 6 deployed under-detections: each missed box
  sits at an anchor where NO valid channel fired ⇒ MATCH-DETECTION gap, not stamp/arith. (3) Structural
  proof: bobmyers builds valid_u8_q as [1,**6**,23,23] — its runtime dihedral kernels apply only
  {I,rot90,rot180,rot270,flipV,T} and OMIT {flipH, antiT}; on asymmetric motifs those 2 are distinct
  templates, so a generator match in an omitted orientation is silently dropped (0 spurious, 100%
  under-detect — matches the observed tail). (4) BUILT the fix: extend both runtime kernels 6→8 channels
  (validity_flipH=flipH(valid_small), validity_antiT=antiT(valid_small); paired excl_flipH=flipV(excl_small),
  excl_antiT=T(excl_small) via the reverse-engineered rule excl_τ=rot180∘τ(excl_small)); bias Expand 6→8.
  Kept QLinearConv (proven bit-exact @ scale1/zero0 AND dynamic-weight-safe). Candidate =
  candidates/task191/exact_8orient.onnx (builder candidates/task191/build_exact.py).
  ALSO ruled out fp Conv for the correlation: an fp16-Conv exact variant passed ISOLATED single-shot but
  failed 40/267 bundled in the sequential same-session grader — ORT PRE-PACKS the fp Conv weight on run-0
  and reuses it, but the kernel here is a per-example runtime tensor ⇒ deterministic wrong output. This is
  why every prior exact design used MatMul/Einsum/QLinearConv (dynamic-weight-safe), never a data-dependent
  fp Conv. (Same class as the knife-edge conv-flip / isolated-eval rail.)
tool+date: direct-ONNX repro + integer-corr oracle + arc-gen generator (8000 fresh draws, isolated
  ORT DISABLE_ALL CPU), onnx 1.21.0 / ort 1.26.0, 2026-07-12 (BUILDER fork).
gate: `ng gate candidates/task191/exact_8orient.onnx --task 191` → ok=true, pass=267, fail=0,
  memory=12252, params=96, cost=12348, points=15.5788. REJECT reason = "not strictly cheaper
  (cand=12348, deployed=11044)" ONLY (acceptable for HEDGE; correctness gate PASSES fail=0).
fresh A/B 8000 (isolated): deployed_fail=71 (0.887%) | candidate_fail=0 (0.000%) | fixed 71/71 deployed
  under-detections | 0 regressions. Candidate fixes every miss and breaks nothing.
cost/points: 11044→12348 (+1304) ; 15.690→15.5788 = **−0.111 pts** to remove the 0.887% hard-zero tail.
  (Far cheaper than the earlier estimated einsum-revert price of −0.625; the omitted-orientation fix costs
  almost nothing because it only adds 2 match channels, not a whole materialized detection plane.)
break-even: keep-int8 EV=15.690·(1−p)^k, fix EV=15.5788, p=0.00887 ⇒ break-even **k≈0.80 hidden
  draws/task**. Since every task draws ≥1 hidden instance, the fix is EV-POSITIVE at k=1 already
  (k=1 keep EV=15.551 < 15.579; k=2:15.413; k=3:15.276). Unlike the −0.625 einsum this needs NO
  ≥5-draws assumption — it is EV-favourable for the exact net at any realistic hidden-draw count, and a
  strict private-LB zero-tail insurance regardless.
verdict: EXACT HEDGE candidate READY (not adopted, per instruction). REVISES the 2026-07-11 diagnosis:
  the tail is a 6-of-8-orientation STRUCTURAL gap in bobmyers, NOT int8 non-bit-exactness (QLinearConv is
  bit-exact here). The exact revert does NOT require the S8 einsum rebuild (~20646/−0.625) — extending the
  existing QLinearConv to 8 orientations gets exactness at cost 12348/−0.111.
reopen: if adopted to HEDGE (or MAIN — EV-favourable), source-own via live_to_exact_source + grader-side
  re-measure; to push cost below deployed 11044 (MAIN by raw points) the only lever left is folding the
  [1,8,23,23] valid plane into an einsum (S8-style) — the 12252 mem is dominated by valid_u8_q (4232) +
  yellow_f32 (2116) + colour-assembly tail.
NOTE ON FORK STATE: this fork did NOT produce any `exact_v4_fused.onnx` / `build_exact_v4.py` / a
  cost-7716 net; those referenced artifacts are not from this fork and are not on disk here. This fork's
  sole deliverable is exact_8orient.onnx (cost 12348) + build_exact.py above.

## ADOPTED 20260711T152043Z (SAFETY PRICE-EXCEPTION, 005/188A-pattern)
- cost: 11044 -> 12348 (points 15.6904 -> 15.5788, -0.1116)
- source: candidates/task191/exact_8orient.onnx
- note: 8-orientation dihedral completion (flipH+antiT added to bobmyers 6-channel
  QLinearConv bank; excl_tau = rot180.tau rule); root cause was orientation OMISSION,
  not int8 rounding. Fresh 8000: 0.887% -> 0.000%, 71/71 fixed, 0 regressions.
  Break-even k~0.8 => EV-positive at k=1; orientation-coverage tail is not a
  curation-filtered class. Bundled fail=0 gate UNCHANGED (267/267 isolated).
  DURABLE GOTCHA (also in ledger): runtime-weight fp Conv is BROKEN in sequential
  grading (ORT pre-packs run-0 weight) — exact designs must use MatMul/Einsum/QLinearConv.

## PORTFOLIO REVERT 20260711T152157Z
- The price-exception fix adopted above is MOVED to the HEDGE slot only; MAIN restored
  to the pure-max net. Reason (portfolio math): every task >= 14.6pt > HEDGE-v3 public
  handicap ~6.8pt, so ANY single silent-zero on MAIN already makes HEDGE the better
  selected slot — MAIN-side insurance changes the best-of-two in NO world and costs
  its price in the lenient world. DOCTRINE: insurance belongs exclusively on HEDGE;
  MAIN carries only strict wins. (005-scale ~0.001pt repairs remain fine on MAIN.)

## ADOPTED 20260713T144124Z
- cost: 11044 -> 10811 (points 15.7117)
- source: candidates/public_dumps/20260713_highroi/king77578_neurogolf-udit22-single-zips-public/task008/task191.onnx
- note: Udit22 public-LB min-merge; bundled fail=0

## ADOPTED 20260713T150906Z
- cost: 11044 -> 10811 (points 15.7117)
- source: candidates/public_dumps/20260713_highroi/king77578_neurogolf-udit22-single-zips-public/task008/task191.onnx
- note: isolated residual-public LB probe; bundled fail=0
