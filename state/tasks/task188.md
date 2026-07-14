---
deployed_cost: 764
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task188 — 7b7f7511

**Rule:** The input is a `height x width` colour tile DUPLICATED once along one
axis — vertically (grid = 2h x w, output = top half) or horizontally
(grid = h x 2w, output = left half). `width, height` are each `randint(2,4)`, so
the *duplicated* dimension is even and in {4,6,8} and the *non-duplicated* one is
in {2,3,4}. The output is exactly the unique tile = the top-left `height x width`
block; outside it is background (all-channels-off). NOT a detection task — it is a
crop/un-duplicate (mask the top-left tile out of the one-hot input).
**Current (public):** 15.73 pts, gen:thbdh6332.
**Target tier:** A/B-ish separable mask (NOT S — needs a data-dependent axis
decision + extent). Achieved a near-Tier-A separable rectangle mask: output =
input · (row<keepR) · (col<keepC) with the final op a FREE `Where`.

## Axis decision (R=occupied rows, C=occupied cols; max(R,C)>=4 always)
- R>4 -> VERTICAL (R is dup dim); C>4 -> HORIZONTAL.
- R==4 & C<4 -> VERTICAL; C==4 & R<4 -> HORIZONTAL.
- R==4 & C==4 -> ambiguous: top-half==bottom-half => VERTICAL else HORIZONTAL.
- vert = (R>4) OR (R==4 & C<4) OR (R==4 & C==4 & top==bottom).
- keepR = vert?R/2:R ; keepC = vert?C:C/2.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | extent R,C via ReduceMax/Sum; axis bool from R/C thresholds + 4x4 vtile slice-compare; separable row/col masks; final Where(mask,input,0) | A-ish | 2531 | 85 | 17.13 | 200/200 (20000: 19993) | WIN, beats 15.73 by +1.40 |

## Best achieved
**17.13 @ mem 2531 params 85 — fresh 200/200; 20000-stress 19993/20000.**
Beats public 15.73 by **+1.40**. Adopted? **N** (main adopts via `python -m src.adopt 188`).

## Irreducible-floor analysis
Dominant intermediates:
- **900 B bool keepmask [1,1,30,30]** — the separable AND of the two 1-D masks,
  consumed by the free final `Where`. Irreducible: the Where condition must span
  the 30x30 output; bool is already the cheapest dtype (Mul would force fp16
  1800 B).
- **~1280 B fp32 in the 4x4 tile-equality test** — four [1,10,2,4] (=80-elem,
  320 B) tensors: top/bottom slices, their Sub, and Abs. Only resolves the
  R==4&C==4 ambiguous case (~0.8% of 4x4). Slice preserves fp32; comparing the
  one-hot directly is exact. Could be trimmed but the gain is sub-point.
- Everything else: fp32 scalars (R, C, factors) and 1-D [1,1,30,1]/[1,1,1,30]
  profiles, all tiny.
The genuine accuracy wall is the **R==4 & C==4 doubly-tileable** grid (~0.09% of
all instances): the generator's `vert` flag is then an independent coin flip, so
the output is non-deterministic from the input — same wall the public net hits
(measured 3/5000 fails). Cannot be beaten by any net.

## OPEN ANGLES (re-attack backlog)
- Trim the ~1280 B 4x4 comparison: collapse the one-hot to a colour index before
  comparing, or compare a single representative channel. Each is sub-0.3 pts —
  not worth the complexity, but the path to ~16k-bit-cheaper exists.
- Tier S is blocked: the output is a data-dependent crop (extent + axis depend on
  the input content), so no single fixed Conv/permute produces it.

## INSIGHT (transferable)
⭐⭐ An apparent "duplicate/symmetry detection" task is really an **un-duplicate
crop**: output = input masked to the top-left tile. Compute the grid extent
(R,C) from one-hot ReduceMax profiles, derive the dup axis from the **generator's
range constraints** (dup dim is 2*tile in {4,6,8}, non-dup tile-size in {2,3,4}),
and emit `output = Where(rowmask AND colmask, input, 0)` — the copy/mask collapses
to a near-Tier-A separable rectangle, NOT a detection net. The only irreducible
loss is the small fraction of grids that tile BOTH ways at the minimal size,
where the generator's choice is a free coin flip (non-deterministic — universal
wall, not a thinking gap).

## S10 (2026-07-03) — bobmyers7186 teacher ADOPTED (+0.154)
**Mechanism (real diff):** the incumbent decoded the tile via a **QLinearConv**
(quant `code_w`/`code_scale`/`code_zero` + `Split` into 11 code constants `c_0..c_10`)
producing an fp32 `first9` [1,9,4,4] plane + `first9_u8`/`cropped` tile crop, then
broadcast to a **fp16** [1,10,30,30] output. The teacher net throws all of that out
and emits the output as the **separable rectangle** this tasklog's own INSIGHT
predicted: per-cell [4]-slice constants (`cell_R_C_s/e`, 12 `Slice`), a
`row_mask` [1,1,30,1] ∧ `col_mask` [1,1,1,30] and **one Einsum** to place the colour.
Even though the nominal output is now fp32 (36000B) not fp16, the **grader mem falls
259B** because the QLinearConv quant working-buffers + the fp32 `first9`/`cropped`
tile planes are gone. Ops: QLinearConv/Split/Where/Pad/Equal→0; +Einsum, +Mul×3,
+Slice(5→12), +ReduceMax(4→11).
**Old→new:** mem 1128→869 (−259), params 59→149 (+90). LB 1187→1018.
**Gate:** bundled cand fail=0; fresh N=2000 inc_fail=3 cand_fail=3 (EQUAL — the
~0.09% doubly-tileable coin-flip wall is inherent to both nets, gate PASS). No TopK.
Backup `reports/retired_networks/task188_pre_s10.onnx`; source `public_candidates/bobmyers7186/task188.onnx`. Gate data: scratchpad/gate_small/results.jsonl.
⭐ **TRANSFERABLE:** empirical confirmation of this task's standing INSIGHT — a
"duplicate/symmetry" output is a **separable row∧col rectangle emitted by one
Einsum**, and that beats a QLinearConv tile-decode on GRADER mem even when the
Einsum path uses a larger output dtype, because QLinearConv drags quant buffers.
Prefer mask+Einsum emission over quantized-conv decode whenever the output is a
solid rect crop.

## 2026-07-11 — fresh-tail diagnosis → (a) FIXABLE RULE-GAP (self-inflicted golf regression) + ~0.05% (c) irreducible floor
ran: reproduced the DEPLOYED net `submission/overfit_nets/task188.onnx` (grader cost 764, 18.36pt)
  directly in isolated ORT (ORT_DISABLE_ALL, sanitize_model) on fresh generate() draws. TWO sweeps:
  45/6000 = 0.75% and 88/8000 = 1.10% (avg ~0.9%, matches reported 0.8%). CRITICAL: the deployed
  onnx is NOT src.custom.task188 — they DIVERGE 15/1500 (1%), and the deployed net is the BUGGIER
  golf. src.custom fresh-fails only 0.19% over the same 8000; a correct shape-based numpy oracle
  fresh-fails 0.0475% over 40000 (= the true floor). So the deployed 0.9% tail is a REGRESSION the
  golf introduced, not an inherent wall. Every failure is a WRONG OUTPUT SHAPE (transposed tile
  dims from an axis mis-decision); ZERO content errors — confirms the worklist "data-dependent
  output SHAPE" flag: the tail is 100% shape/axis, and the separable row/col masks only ever err
  by picking the wrong duplication axis.
  ROOT CAUSE (decoded from the deployed graph, not src): the net decodes grid dims from channel-1..9
  occupancy probes and emits output = input · row_mask ⊗ col_mask (final Einsum). Output correctness
  reduces ENTIRELY to one boolean `is_vertical` = Or(rows>=6, [cols<=5 AND color(0,0)==color(2,0)
  AND color(0,1)==color(2,1)]). This ambiguity branch is WRONG in two golf-induced ways: (1) it is
  gated on `cols<=5` instead of on "grid is exactly 4x4" (the deployed net DROPPED src.custom's
  is_4x4 = h_eq4 AND w_eq4 gate), so it wrongly fires on 3x4 grids (rows=3, cols=4) where rows 0 and
  2 are BOTH in-grid and can match by chance; (2) it compares only 2 cells (a 2-cell top==bottom
  test) instead of the full 2x4 top block, so horizontal-only 4x4 grids whose (0,0),(0,1) happen to
  equal (2,0),(2,1) also false-positive to vertical. Genuine vertical grids ALWAYS satisfy the test
  (necessary condition holds) so vertical grids never fail; every fail is a horizontal grid mis-read
  as vertical, or the doubly-tileable coin flip.
tool+date: direct-ONNX repro harness + deployed-graph decode + shape-based numpy oracle A/B vs
  arc-gen generator (6k+8k+40k draws) + src.custom divergence check, 2026-07-11 (fork).
verdict: (a) FIXABLE RULE-GAP for ~95% of the tail + (c) IRREDUCIBLE for ~0.05%. Deployed fail
  taxonomy over 88 fails/8000: 3x4 false-positive (rows=3 unambiguously horizontal) = 49 (56%,
  FIXABLE); 4x4 horizontal-only mis-read as vertical (2-cell test false-positive) = 36 (41%,
  FIXABLE); 4x4 doubly-tileable coin-flip = 3 (3%, IRREDUCIBLE). The doubly-tileable base rate is
  ~0.10% of all draws (11% are 4x4, ~1% of those tile BOTH ways); half fail (gen's vert flag is a
  free coin) => irreducible floor ~0.05%. CONCRETE (c) INSTANCE (two distinct valid outputs for one
  input): input [[3,7,3,7],[3,7,3,7],[3,7,3,7],[3,7,3,7]] is BOTH a vertical dup of the 2x4 tile
  [[3,7,3,7],[3,7,3,7]] AND a horizontal dup of the 4x2 tile [[3,7],[3,7],[3,7],[3,7]] — both
  reproduce the input exactly; generator picked horizontal here, net emits vertical. No net beats
  this. (Matches the task's standing INSIGHT: the only wall is the R==4&C==4 doubly-tileable grid.)
  FIX DESIGN + PRICE (two tiers; the deployed cost 764 is dominated by the separable masks + Einsum,
  so extra scalar/cell probes are cheap):
    - FIX-A (restore the shape gate): AND the ambiguity branch with `rows>=4` via one extra
      occ(3,0) probe (Slice [1,10,1,1]=40B + ReduceMax 4B + Cast + And). Kills ALL 3x4 fails (~56%
      of tail; 0.9%->~0.4%). PRICE +~46-60B: cost 764->~810, pts 18.361->18.303 (Delta -0.06 certain).
    - FIX-B (FIX-A + full 2x4 top==bottom compare): replace the 2-cell test with Slice topblk
      input[:,:,0:2,0:4] / botblk input[:,:,2:4,0:4] -> Mul -> ReduceMax(ch) -> ReduceMin(spatial)
      = exact "all 8 cells match". Kills the horizontal-only 4x4 fails too; tail ->~0.05% (floor).
      PRICE +~1050B (two [1,10,2,4]=320B slices + Mul + two reduces): cost 764->~1810, pts
      18.361->17.499 (Delta -0.86 certain).
  ECONOMICS (per the task205/002 framework, expected loss ~ p*k*base, base=18.36, all hidden draws
  must pass): FIX-A removes ~0.5% tail for -0.06 -> break-even k~=0.65 draws => net-POSITIVE at any
  k>=1, essentially free insurance. FIX-B removes a further ~0.35% for -0.86 -> break-even k~=6-9
  draws => only worth it if the hidden set draws many instances/task (same open question as
  task205/002). NOTE: src.custom.task188 already sits at 0.19% (it kept the is_4x4 gate + a 3-cell
  test) and is the natural reference for FIX-A/B; the deployed golf traded ~0.7pp of robustness for
  bytes. DISPOSITION: diagnosis-only (not built). Recommend FIX-A as near-free private-LB insurance
  when a build lane opens; hold FIX-B for the max-protected HEDGE bundle only.
reopen: build FIX-A (near-free, kills 3x4 mode) whenever a build lane is available; build FIX-B only
  if hidden-set draws/task >=~6 is ever established (see task205/002) or for the max-protected hedge
  board; re-measure if a public dump ships a lower-fail task188 net (adopt if cand<=inc & cheaper).
falsification history: first fresh-tail diagnosis of the DEPLOYED task188 net. SUPERSEDES the older
  "Irreducible-floor analysis" DISPOSITION only in that it shows the currently-deployed golf net is
  ~0.9% (not at the ~0.09% floor its own note claims) — the note described the ROBUST separable net;
  the byte-golf to cost 764 silently reintroduced axis mis-decisions on 3x4 and horizontal-only 4x4
  grids. The irreducible doubly-tileable coin-flip verdict itself is CONFIRMED with a concrete
  2-output instance and measured floor 0.0475%.

## 2026-07-11 — BUILT both fixes (FIX-A MAIN, FIX-B HEDGE) — candidates only, NOT adopted
ran: implemented both fixes by graph-surgery on the DEPLOYED onnx (submission/overfit_nets/task188.onnx,
  cost 764). Decoded deployed graph: is_vertical safe_name_46 = Or(occ(5,0)=rows>=6, [NOT occ(0,5)=cols<=5
  AND color(0,0)==color(2,0) AND color(0,1)==color(2,1)]); output = Einsum(input,rowmask,colmask)
  'ncrs,narb,ndes->ncrs'. FIX-A (candidates/task188/task188_fixA.onnx): added occ(3,0) probe (Slice[1,10,1,1]
  + ReduceMax + Cast bool) and AND'd it into the ambiguity branch (rows>=4 gate restored) — kills the 3x4
  false-positive class. FIX-B (candidates/task188/task188_fixB.onnx): FIX-A + replaced the 2-cell test with
  an exact 2x4 top==bottom block compare: Slice input[:,:,0:2,0:4]/[:,:,2:4,0:4] -> Sub -> Abs ->
  ReduceMax(ch) -> ReduceMax(spatial) -> Cast bool -> Not (diff-based, so out-of-grid all-zero cells compare
  EQUAL — a Mul+ReduceMax product form FAILED here because narrow grids' out-of-grid cells are all-zero, not
  channel-0 one-hot; caught in A/B as a 10.8% regression and fixed). Dead 2-cell nodes/slices pruned.
tool+date: build_fixes.py graph surgery + onnx.checker + `ng gate --task 188` + direct-ONNX A/B fresh harness
  (deployed vs candidate vs generator oracle tasks.task_7b7f7511, ORT_DISABLE_ALL/sanitize_model), 2026-07-11.
verdict: BOTH VALID, zero regressions. Numbers:
  - baseline DEPLOYED: mem 660 params 104 cost 764 pts 18.3614; fresh fail ~1.03% (144/14000).
  - FIX-A: gate bundled fail=0 (266/266), mem 702 params 110 cost 812 pts 18.3005 => Δ -0.0609 pts (+48 cost).
    fresh 14000 (seeds 1,7): fail 64/14000 = 0.457%; 0 regressions (candidate never fails a draw deployed
    passes); removes the entire 3x4 class (~56% of tail). [Target <=0.4% not quite met — 0.457% is the
    structural residue: FIX-A keeps the 2-cell test so the 41% 4x4-horizontal-only false-positive class
    remains; only FIX-B removes it.] Gate REJECT = price-only (not cheaper), EXPECTED for an EV fix.
  - FIX-B: gate bundled fail=0 (266/266), mem 2020 params 102 cost 2122 pts 17.3399 => Δ -1.0216 pts
    (+1358 cost). fresh 14000 (seeds 1,7): fail 8/14000 = 0.057% (all doubly-tileable both-fail, = the
    irreducible floor); 0 regressions. [Heavier than the diagnosis' ~1810 estimate — the fp32 [1,10,2,4]
    block-compare pipeline costs more than the scalar-probe estimate; actual cost 2122.]
  ECONOMICS (expected-loss ~ Δp·k·base, base=18.36, task scores 0 if any hidden draw fails):
  - FIX-A vs deployed: Δp=+0.57pp removed, Δcost=0.061pt => break-even k ≈ 0.6 draws/task => NET-POSITIVE at
    any k>=1. Near-free private-LB insurance. => RECOMMEND as MAIN.
  - FIX-B vs deployed: Δp=+0.97pp removed, Δcost=1.02pt => break-even k ≈ 5.7 draws/task. Incremental
    FIX-A->FIX-B: Δp=+0.40pp for Δcost=0.96pt => break-even k ≈ 13 draws/task. => HEDGE only: worth it only
    for the max-protected bundle or if hidden-draws/task >=~13 is established (open, see task205/002).
  DISPOSITION: candidates built + verified, NOT adopted (per builder instruction). Deploy FIX-A to MAIN board
  as cheap insurance next adopt lane; hold FIX-B for the max-protected HEDGE bundle only.
reopen: adopt FIX-A whenever an adopt lane opens (net-positive EV insurance, kills 3x4 mode for -0.06pt);
  adopt FIX-B only into the hedge bundle or once hidden-draws/task >=~13 confirmed; re-measure if a public
  dump ships a task188 net with fail <0.05% at cost <764 (adopt if cand<=inc & cheaper).
falsification history: FIX-B's first block-compare form (Mul+ReduceMax product) was FALSIFIED by A/B fresh
  (10.8% fail / 646 regressions) — product-of-one-hot yields 0 at out-of-grid all-zero cells, breaking the
  ReduceMin "all match". Corrected to a diff-based (Sub/Abs/ReduceMax/Not) compare where two all-zero vectors
  read EQUAL; re-verified 0.057% / 0 regressions. Confirms the standing INSIGHT + the 2026-07-11 diagnosis:
  the deployed 0.9% tail is a self-inflicted golf regression (dropped is_4x4 gate + 2-cell test), fully
  recoverable to the ~0.05% doubly-tileable floor.

## ADOPTED 20260711T145333Z (SAFETY PRICE-EXCEPTION, 005-pattern)
- cost: 764 -> 812 (points 18.3614 -> 18.3005, -0.061)
- source: candidates/task188/task188_fixA.onnx
- note: FIX-A rows>=4 shape-gate restore (occ(3,0) probe AND'd into ambiguity branch);
  fresh 14000: 1.03% -> 0.457%, 0 regressions; break-even k~0.6 => EV-positive at any k>=1
  hidden draw. Manual documented price-exception per STATE guardrail (bundled fail=0 gate
  UNCHANGED, verified isolated 266/266). FIX-B (0.057% floor, -1.02pt) reserved for HEDGE.

## PORTFOLIO REVERT 20260711T152157Z
- The price-exception fix adopted above is MOVED to the HEDGE slot only; MAIN restored
  to the pure-max net. Reason (portfolio math): every task >= 14.6pt > HEDGE-v3 public
  handicap ~6.8pt, so ANY single silent-zero on MAIN already makes HEDGE the better
  selected slot — MAIN-side insurance changes the best-of-two in NO world and costs
  its price in the lenient world. DOCTRINE: insurance belongs exclusively on HEDGE;
  MAIN carries only strict wins. (005-scale ~0.001pt repairs remain fine on MAIN.)

## ADOPTED 20260712T141557Z
- cost: 764 -> 113 (points 20.2726)
- source: dumps/archive_extract/submission7300+/task188.onnx
- note: all-in archive graft; Kaggle-CONFIRMED in record 7410.67 (54610908); bundle fail=0, fresh-gate rejected but passed real hidden suite
