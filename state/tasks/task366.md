---
deployed_cost: 31559
logged_costs_match: match
migrated: 2026-07-09
---

# task366 — e6721834

## Current live

`memory=35927`, `params=490`, `points=14.497209022434086`.
The deployed graph is a high-quality heuristic; prior fresh sanity was `39/40`, so any rewrite needs
fresh validation.

## Semantic rule

Input contains two same-sized panels, stacked or side-by-side.
One panel is the template: a background plus 2–3 solid rectangles in a single `forecolor`, each with
1/2/3 same-coloured dots punched into it.
The other panel contains only the dot stencils at new positions.
Output is the non-template panel background with the missing full rectangles reconstructed at those
dot positions.

## Bottlenecks

- `label30` one-hot-to-colour Conv: ~3600B.
- template colour-present machinery, including int32 template label lookup: ~1KB plus several
  255/510B tensors.
- repeated `k0/k1/k2` stencil matching blocks: roughly 7KB.
- repeated placement/stamping blocks: roughly 5.5KB.
- final label path: roughly 5.7KB.

## Re-attack angle

Generator fact: punched dot colours exclude both backgrounds and `forecolor`.

Therefore a cheaper template-dot mask may be:

`T_dot = T_non_background AND T != forecolor`

instead of the current “template cell colour appears in placement dots” machinery. If `forecolor`
can be derived cheaply from the rectangle mask, this may remove 2–4KB. If deriving `forecolor`
requires full per-colour counting, it probably gives the savings back.

Larger idea: use the generator guarantee that rectangle `idx` has `idx+1` dots to map 1/2/3-dot
templates to placement clusters and delete much of the `k0/k1/k2` matching. This is much riskier
because count/color collisions exist; treat as research, not an immediate adoption path.

## 2026-06-29 forecolor/dot-mask probe

Hypothesis: replace the current placement-colour membership path

`pos_color -> T_color_present -> GatherElements(T_idx_for_present) -> T_present`

with `T_dot = T_non_background AND T != forecolor`, using the generator fact
that dot colours exclude both backgrounds and `forecolor`.

Generator probes:

- First non-background template cell is a dot colour in about `30/300` samples,
  so it is unsafe as a cheap `forecolor` proxy.
- Component first-cell/majority proxies are also unsafe (`~8-40%` failures in
  1000-sample probes depending on proxy).
- Template non-background mode is safe in tested samples (`0/300` dot-colour
  collisions), because rectangles dominate the dot pixels.

Conclusion:

The semantic fact is valid, but the safe `forecolor=mode(non-bg)` route likely
needs per-colour counting over the template panel.  That may cost as much as or
more than the current `T_present` path.  Do not patch until a cheap mode extractor
is designed.

## 2026-06-29 fresh rare-failure capture

- Public candidates for task366 (`boristown`, `lucifer`, `biohack_mix`, `urad`) are identical to live/source on stored eval: `pts=14.497209`, `mem=35927`, `params=490`, `pass=255/255`.
- Fresh sanity in this session: `40/40`, then `199/200`.
- Captured a failing eligible fresh case (`input_shape=(26,11)`, `output_shape=(13,11)`, `diff_count=20`) where the target should reconstruct a 5x4 rectangle from a single placement dot color `3`, but prediction left the placement panel unchanged.
- The visible template panel contained the same dot color `3` in another box/dot-count context. This confirms the earlier risk: dot colors are not unique per rectangle index, so color-present routing alone can bind the placement dot to the wrong template evidence.

Conclusion: a score-improving semantic rewrite must route by geometric dot stencil/count and placement cluster, not only by dot color membership. The cheap `T_dot = T_non_background AND T != forecolor` idea remains useful only if paired with a reliable stencil/count association.

## 2026-06-29 — uint8 TopK mask enumeration trim adopted

- Previous source/live: `pts=14.497209022434086`, `mem=35927`, `params=490`.
- Rewrite: five bool-mask `TopK` inputs (`A_nb8_flat_f`, `P_nb_flat_f`,
  `k0/k1/k2_anchor_flat_f`) were changed from bool→fp16 full-vector casts to
  bool→uint8 full-vector casts. The small `TopK` value outputs are then cast
  back to fp16 for existing downstream `Greater`/`ReduceSum` consumers.
- Stored eval: `255/255`, `mem=34678`, `params=490`, `pts=14.532108142796652`.
- Fresh truth nuance: both incumbent and candidate have rare failures against
  the current local generator, but the candidate matched the incumbent exactly
  over `20000/20000` eligible fresh examples.

Insight: bool-mask `TopK` cannot run directly in ORT, but bool→uint8→TopK plus
small value-output recast can remove large fp16 mask casts while preserving the
same selected indices and downstream semantics.

## 2026-06-29 dot-count semantic reference probe

Read the generator directly.  Strong constraint confirmed: template index `idx`
has exactly `idx+1` punched dots, and placement boxes reuse the same relative dot
stencil.  This suggests a colour-collision-safe rewrite should route by
1/2/3-dot geometry rather than dot colour.

A quick Python semantic reference using `forecolor=mode(non-bg)`, forecolor
components as template rectangles, and per-colour relative-stencil placement
matching was **not** exact: `1137/2000` fresh.  The simplification fails because
punched dot holes can split a forecolor rectangle into multiple 4-connected
components, and same-colour/multiple-placement cases need a stronger rectangle
reconstruction step before matching.  Do not lower this to ONNX yet.

Current conclusion: dot-count/stencil routing remains the right high-ceiling
direction for task366, but the semantic compiler needs a robust template
rectangle extractor (probably bbox/gap based, not component-only) before it can
replace the live 640-node heuristic.

## 2026-06-29 exact-cover semantic compiler probe

Upgraded the Python reference from component extraction to a more faithful
mathematical compiler:

1. infer candidate template panel/background/forecolor,
2. enumerate all 2..7 by 2..7 rectangles whose cells are non-background and are
   exactly `forecolor` plus `k in {1,2,3}` same-colour punched dots,
3. choose one rectangle for each dot count `k` by exact cover of all template
   non-background cells,
4. enumerate placement anchors whose relative dot stencil exactly matches and
   whose rectangle interior contains no extra placement dots,
5. exact-cover all placement dots and reconstruct the output rectangles.

This is a much better semantic model than the component-only probe: **987/1000**
fresh on the first run.  Adding a simple prior that the template panel has more
non-background cells than the placement panel gave **2950/3000**, so the remaining
errors are not solved by a crude panel-size prior.

Interpretation: the high-level mechanism is real, but rare ambiguity remains in
template/placement selection or in exact-cover tie-breaking.  The next useful
step is to capture failing cases and inspect them in the trace/transform viewer,
not to lower this compiler to ONNX yet.  A successful version would be a genuine
large rewrite candidate because it replaces much of the current colour-membership
and repeated k0/k1/k2 heuristic routing with bounded rectangle/stencil exact
cover.

## 2026-06-29 exact-cover bg-candidate fix

The `987/1000` failure mode was not primarily dot-count ambiguity.  The wrong
assumption was `template_background = mode(template_panel)`: large foreground
rectangles can cover more cells than the template background.

Probe update:

- keep placement background as panel mode, because the placement panel contains
  only sparse dots on its background;
- enumerate template background candidates among colours with count `> 3`
  (dot colours occur at most three times);
- choose foreground and 1/2/3-dot rectangles by bounded exact cover;
- use an uncovered-cell exact-cover recursion so false background candidates do
  not explode combinatorially.

Fresh reference results:

- `1000/1000` with seed 0;
- `5000/5000` with seed 1.

Interpretation: task366 now has a clean source-level mathematical specification:
split panels, infer the placement side by sparse dot panel, enumerate candidate
template backgrounds rather than trusting the mode, extract one rectangle per
dot count by exact cover, then stamp matching dot stencils into the placement
panel.  This is not adopted yet because lowering this exact-cover compiler to a
smaller ONNX graph is still the hard part.  The key transferable lesson is that
majority-colour/background heuristics are unsafe when generated rectangles can
dominate the panel area.

## S8 (2026-07-02) — output-preserving surgery wave (+0.143) ADOPTED
NO iterative flood here (~60×255B bool planes + 8.5KB scalar soup) — literal walk-einsum N/A.
Landed: has8-count as ONE 4-op Einsum vs free input (deletes 6 planes + fp16/TopK-8 chain);
bgT-mux recompute of T_nb0/P_nb; k-anchor TopK feeds shared as one fp16 presence Gather + Where
(deletes 3×(mask+flat+cast); TopK feeds stay fp16 = grader-safe); c-block first-cell via 4D
ReduceMax/ArgMax; mask re-association (255B→15B); Not(Less)→GreaterOrEqual fusion ×26;
row/col-any planes → Einsums vs free input. 30983+576 vs 35927+490 → 14.497→14.640.
Divergence 0 on ~9.5k fresh + stored + 500 random (arithmetic identities). TRAPS (transferable):
ORT CPU has NO bool Where kernel; np.ascontiguousarray promotes 0-d→(1,) (use np.asarray);
rank-0 initializers need value_infos rewritten to [] for strict inference.
Floors: label30 3600B, T_idx 1020B (Gather idx must be int), out_label30 900B Pad.
Adopted via ONNX materialization + live_to_exact_source.

## S9 (2026-07-03) — fold 2nd pass: FLOOR re-confirmed (no change)
13a N/A (no walk/flood einsum; output = index-shift Gather stamping). Batched-K on the
12-fold rectangle banks = byte-identical (outputs counted, nodes free; params only 576).
Byte-rank floors: label30 3600 fp32 Conv read, label30_u8/out_label30 900 each,
T_idx_for_present 1020 int32 (GatherElements needs int), TopK fp16 feeds 5×510,
B_rows_gather 450 (row-first already minimal). Remaining ~21.5KB = flat tail of needed
255B masks/int32 scalars across replicated rect machinery. Not→GEq fusion exhausted;
free-input einsum inapplicable (masks also Gathered; derive from T_nb0 not input).
Only ceiling-lifter = exact-cover semantic compiler (un-lowered, research). DO NOT re-probe.

## S11 (2026-07-03) — signed-priority overlay (playbook 15) scout: KILL — stamped template rects with punched dots are 2-D non-separable AND all colours are instance-dependent (no constant signed W exists); ~21.5KB = replicated rectangle-extraction/stencil-match blocks (assignment machinery). Priority already via Where chain, no [30,30] carrier.

## 2026-07-06 golf re-attack — NEGATIVE (batching is byte-neutral)

Confirmed incumbent = cheapest known: measured ALL public dumps (kojimar/urad7225/bobmyers/
lucifer, incl. 7220-7225 tier) — every one is 33.8K-35.9K mem. Our `networks/task366.onnx`
(30983) is the global minimum. No borrow available.

**Key correction (was the whole premise, and it was WRONG):** grader memory =
`sum over EVERY intermediate tensor of (num_elements * dtype_itemsize)` (harness
`calculate_memory`). Collapsing the 3x unrolled `c0/c1/c2`, `k0/k1/k2`, `r0/r1/r2` blocks
into one batched block over a size-3 axis turns three `[1,255]` planes into one `[3,255]`
plane = **identical bytes**. Batching cuts tensor COUNT, not total element-bytes. So
"vectorize the unrolled blocks" gives ZERO memory reduction. Do not re-attempt.

**What actually costs (measured):** 3600B `label30` detection Conv (FLOOR, static 30x30
colour read) + ~10.5KB of ~41 full-panel `(1,1,15,17)`=255B uint8/bool mask planes + a
~16KB tail of ~200 small int32/bool coordinate tensors. Memory is a function of the NUMBER
of distinct plane/coord tensors the algorithm materializes, i.e. algorithm structure.

**dtype golf ≈ 0:** the only >255B downcastable planes (`k*_anchor_flat_f`, `presF_flat`,
510B f16 each) all FEED TopK — uint8 TopK crashes the grader. `label30_u8` already minimal.

**Leaner rewrite: high-risk, not smaller.** From-scratch correlation solver prototype
(match template dot-pattern vs stencil, stamp rect) scored 220/2855 = 7.7% error, 38x the
0.2% bar. Failures = dual-background + large-forecolor cases that defeat most-common-colour
bg detection — exactly what the incumbent's corner-voting `A_tl_eq_tr`/`A_bg` machinery
(~40 nodes) exists to handle. A *correct* correlation solver still needs several full-size
planes per box (color mask + correlation response + rect mask) => memory floor near the
incumbent.

**16.0 (mem+params <= 8103) is INFEASIBLE.** Hard floor of the current structure alone =
3600 (Conv) + 15503 (1-byte masks) = ~19.1K => ceiling ~15.15 even with perfect dtype golf.
The whole public field at 30K+ corroborates. VERDICT: keep incumbent, de-prioritize task366.

## 2026-07-07 — local-only REJECTED signed INT8 TopK feed probe (+0.0083)

Candidate: `reports/candidates/task366/task366_int8_topk_greedy.onnx`.
Recast `P_nb_flat_f` TopK feed/value output to signed INT8.  This is not the
exact-cover breakthrough, but it recovers dtype memory while avoiding unsigned
TopK submission errors.

Bundled gate: fail=0.  Unsigned TopK scan: clean after adoption.  Cost: 31559
-> 31299 (memory 30983 -> 30722, params 576 -> 577).

## ADOPTED 20260709T041329Z
- cost: 31559 -> 30159 (points 14.6858)
- source: candidates/public_dumps/20260709/neurogolf-7266-72-w-visualizations/nets/task366.onnx
- note: min-merge from nets

## 2026-07-09 — 0.X re-attack (fresh-eyes, post-22219): NO-WIN, colf 3600 = structural DETECTION floor

Byte breakdown of deployed cost=22219 (mem 21990 + params 229, pts 14.9913):
- `colf` Conv [1,1,30,30] fp32 = **3600** (single dominant plane, 16% of total)
- `padded` u8 [30,30] = 900; `gU` u8 [1,1,30,30] = 900 (label plane)
- 3x fp16 TopK feeds (cfH/fdfH/ndfH [1,255]) = 510 each = 1530
- `Brow` u8 [1,1,15,30] = 450
- ~35 tensors at 255B (bool/u8 [15,17] or [1,255] mask planes) + ~180 small int32/bool coord
  tensors across the 3 replicated stamp blocks = ~15.5K tail
- static-VI dtype totals: u8 6333, bool 6461, fp16 3612, fp32 4316, int32 1092, int64 176

CARRIER-vs-DETECTION classification: `colf` is a pure **fp32 DETECTION** plane. Its ONLY consumer
is `Cast->gU` (uint8 label plane); gU's ONLY consumers are the panel-A slice (`A4`) and the
data-dependent panel-B gather (`Brow`). Per the fold heuristic, a fp32 detection read that must be
materialized to be gathered CANNOT be folded into a free op -> NO-WIN by construction. It is not an
avoidable carrier of output content.

Why colf 3600 is a floor (all routes measured/derived WORSE):
- Conv/Einsum reading the free fp32 input has output type == input type == fp32 (ONNX type
  constraint T); a full 30x30 label read is 3600 min.
- uint8 label via QLinearConv/ConvInteger needs uint8 input => Cast(input)[1,10,30,30] = **9000B**. Worse.
- ArgMax(input,axis=1) label = int64 [1,30,30] = **7200B** peak. Worse.
- Per-panel decode (Slice input->[1,10,15,17] fp32 2550 + Conv 1020 + Cast 255, x2 panels) = **7650**. Worse.
  (The incumbent's decode-ONCE-to-u8-then-gather-cheap is provably optimal; gather source must be u8.)
- Dual dilated dead-tap crop ([1,1,15,30]=1800 for horiz + [1,1,30,17]=2040 for vertical) = 3840,
  AND the two shapes can't merge for the shared downstream slice/gather. Worse + infeasible.
- fp16 label: Conv output = fp32 (input dtype); casting input->fp16 = **18000B**. Worse.
- Transpose-normalize (make occupied region always 15xN): transpose output materializes a full
  [1,10,30,30] fp32 = 3600. Worse.
- GENERATOR FACT forcing this: panels tile the grid with NO gap; height in [10,15], width in [8,17];
  horiz uses <=15 rows x <=30 cols, vertical uses <=30 rows x <=17 cols. Neither axis is statically
  <=15 in both cases, and panel B start (Hh or Wh, in [10,15]) is data-dependent => a full uint8
  label plane is required for the panel-B gather => a full fp32 label read is required.

Tail (~15.5K of 255B masks + small coords) is the 3x replicated stamp-block machinery; batching is
byte-neutral (2026-07-06 log, re-confirmed). dtype golf already exhausted last session (all >255B
downcastable planes feed TopK; uint8-TopK is a grader-killer). No params lever (kernels already
1x1/2x2; params=229).

VERDICT: incumbent 22219 is at the structural floor of the current decode-first algorithm. NO fold /
downcast / crop win exists via free-output-einsum, stamp-conv, or bitpack. Consistent with S9/S11/
2026-07-06 floor verdicts; last session's 30159->22219 rewrite already captured the available insight gains.

Reopen trigger (ledger):
1. what was run: node-level output-byte dump of deployed net + CARRIER/DETECTION classification of
   colf; derivation/measurement of 6 alternative label-plane constructions.
2. tool+date: scratchpad breakdown.py (ORT 1.26 trace-mem, mirrors scoring.calculate_memory) +
   scoring.calculate_params, 2026-07-09.
3. reopen-trigger: (a) exact-cover semantic compiler successfully LOWERED to ONNX with fewer full-size
   planes than incumbent (prior Python probe 5000/5000 fresh but never lowered; a correct correlation
   solver still needs several 255B planes/box per 2026-07-06 -> floor near incumbent); (b) a new public
   dump for task366 measured < 21990 mem; (c) a general mechanism that produces a full uint8 label
   plane at peak < 3600B from an fp32 one-hot input (would refute the colf floor globally).
4. falsification history: "AT byte floor / no endogenous lever" verdicts were reversed elsewhere
   (task011 +1.52); treat this colf-floor as tool-vs-net-at-T, not durable truth. Durable physics only:
   Conv/Einsum output dtype == fp32 free-input dtype; uint8 decode requires a 9000B input cast.

## ADOPTED 20260709T051157Z
- cost: 30159 -> 22219 (points 14.9913)
- source: candidates/task366/cand.onnx
- note: public-insight generalize: probe_then_build (Scatter histogram -> free-input Einsum rank-1 indicators) + QLinearConv corner stencil (8-plane chain -> 510B) + merged single-pass stamp + fp16/u8 dtype golf on gated mux; equivalence-golf, 0 divergence over 3255 examples (incumbent's own 13/1600 fresh-fail rate matched exactly)

## 2026-07-10 representation-inversion re-audit (task233 lens, opus agent) — NO BUILD
- Ran: full graph dump + per-node byte map (onnx shape-inference / scorer trace), arg-select op trace, generator read, bundled-usage probes.
- Verdict: **INJECTIVE-BUT-UNFAVORABLE-BYTES**. arg-select structure EXISTS and relation IS injective (box containment, dot counts 1/2/3 distinct) — but match matrix already collapsed to assign [6,3] bool = 18B; ScatterElements inverse table (~1KB) is byte-negative. 3x 510B fp16 TopK feeds are sparse-coordinate extraction, not match matrices. b2 lane exercised by 66.5% of examples (3000-sample probe) — not overrideable.
- Tool+date: opus triage agent, onnx 1.21.0 / ort 1.26.0, 2026-07-10.
- Reopen triggers: a future net re-introducing a fat (N>=50) Equal([K,1],[1,N])->TopK plane; int8-TopK acceptance (3x510B->255B); mixed-dtype Conv (colf 3600B->1800B); new public net < 21990 mem.
- Falsification history: this is the systematic 233-lens sweep prescribed by STATE.md Active Vein 1 after the 2026-07-10 task233 win falsified its own 07-09 CLOSED verdict; lens applied and did not fire here.

## 2026-07-11 — fresh-tail diagnosis (~0.7% tail) → (b) HEURISTIC PLATEAU (rule recoverable, exact fix unaffordable); NOT irreducible, NOT cheaply fixable
ran: ran the DEPLOYED net (submission/overfit_nets/task366.onnx, cost 22219, sha matches manifest —
  NOT src.custom) directly in isolated ORT_DISABLE_ALL on fresh generate() draws. Reproduced tail:
  34/4734 = 0.72% (also 16/2855 = 0.56% earlier) — consistent with the logged ~1.06%/13-per-1600.
  Captured 34 failures + taxonomy: 15 gross (ndiff≥25, whole-rectangle/panel errors) + 19 small
  (ndiff 5-17, mis-sized/unstamped rectangle); orientation 23 stacked / 11 side-by-side (~proportional,
  not orientation-specific); only 4/34 leak a foreign color. Dominant mode = UNDER-RECONSTRUCTION: the
  net leaves the placement-panel dots un-stamped (no forecolor rectangle) or stamps a wrong-size/
  wrong-anchor rectangle; a few gross cases leak the TEMPLATE panel's background color into the output
  (panel/forecolor-vs-background confusion). This is exactly the documented dot-color-collision +
  template/placement + mode-background failure family (color-membership routing binds a placement dot
  to the wrong template evidence, or the mode-bg heuristic mis-fires when a forecolor rectangle
  dominates the panel).
tool+date: isolated ORT (deployed onnx) A/B vs generator ground-truth, 4734 draws + failure taxonomy
  (scratchpad diag.py/an366.py), onnx 1.21.0 / ort 1.26.0, 2026-07-11 (fork).
verdict: (b) HEURISTIC PLATEAU. The rule is NOT irreducible — the task's own 2026-06-29 exact-cover
  semantic compiler (candidate template-bg enumeration + bounded rectangle exact-cover + dot-stencil/
  count geometric routing) scored 1000/1000 seed0 and 5000/5000 seed1 fresh, i.e. the output IS fully
  determined by the input and the collisions are resolvable by GEOMETRY (stencil/count), not color. So
  this tail is a heuristic approximation gap, NOT generator ambiguity (rules out (c)). But it is also
  NOT a cheap (a) fix: every partial patch tried is incomplete (mode-bg prior 2950/3000; panel-size
  prior fails; forecolor=mode(non-bg) needs full per-colour counting), and the only 100%-correct route
  is the full bounded exact-cover solver, which prior floor analysis (S9/S11/2026-07-06/2026-07-09)
  proves cannot be lowered to ONNX at or below 22219B (a correct correlation solver needs several
  full-size 255B planes PER box → memory floor near/above the incumbent; batching the 3× stamp blocks
  is byte-neutral). So no net-positive fix exists under the op/byte budget — the deployed 22219 net is
  the best affordable heuristic (plateau), and this is NOT a self-inflicted regression (unlike
  task205/task233).
reopen: (a) exact-cover semantic compiler successfully LOWERED to ONNX with fewer full-size planes
  than incumbent (never achieved — the hard part); (b) a public dump for task366 measured < 21990 mem;
  (c) int8-TopK acceptance or mixed-dtype Conv shrinking the detection tail enough to fund a robust
  2-D stencil-match plane; (d) a cheap dot-count/geometric router that beats color-membership on the
  collision cases without full exact-cover. Any of these would move the tail below ~0.7% — none known.
falsification history: first fresh-tail diagnosis of task366. Consistent with the 2026-06-29 dot-color
  -collision captures and the 2026-07-09/07-10 floor verdicts; confirms the deployed 22219 net inherits
  the incumbent heuristic's ~0.5-1% tail (as noted at adoption). Durable: rule is recoverable
  (exact-cover 100%), so the tail is a build-affordability plateau, not an information floor.

## 2026-07-11 — offense-lens note (366 = board's 3rd-most-expensive net, dual-purpose scan)
- Cost structure (deployed 22219): dominant single plane = `colf` 3600B fp32 one-hot→label DETECTION
  (16%); ~15.5KB tail = 3× replicated stamp-block 255B mask/coord planes (structure-driven, batching
  byte-neutral); 3× fp16 TopK feeds 510B (uint8-TopK = grader-killer, blocked).
- OFFENSE READ: NO new dual-purpose lens found. The `colf` 3600B fp32-detection floor is SHARED across
  the decode-first family (task205 cgrid 3600, task187 label 3600) — a sub-3600B uint8-label mechanism
  would be a BOARD-WIDE offense lever, but it is proven infeasible under onnx 1.21 type constraints
  (Conv/Einsum output dtype == fp32 free-input dtype; uint8 label via QLinearConv needs a 9000B uint8
  input cast; ArgMax label = 7200B int64). The ~15.5KB bulk is NOT an avoidable carrier (structural
  replication, byte-neutral to batch), so it yields no offense multiplier.
- Reusable primitives ALREADY harvested here (in insights, transferable to other nets): QLinearConv
  corner-stencil compressing an 8-plane bool chain → 510B (stencil/pattern-match compression lens); and
  probe_then_build (Scatter histogram → free-input Einsum rank-1 indicators) count-to-indicator fold.
  Verdict: 366's expense is irreducible-detection + structural-replication, so it does not itself open a
  new offense lens beyond these already-known primitives.
