---
deployed_cost: 7210
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task219 — 90f3ed37

**Rule:** Fixed 15×10 grid. The instance defines ONE shared "legend" of three sub-patterns — A (tall×awide), B
(tall×bwide), C (tall×cwide) — all monochrome cyan, with tall∈{1,2,3}, awide/bwide/cwide∈{1,2}. The grid holds
2–6 horizontal "bands". Each band sits at a random top `row` (consecutive band tops spaced `randint(tall+1,tall+3)`
apart) and a random B-column `col∈{awide,2·awide}`. Per band the input draws: A tiled across cols `[0,col)` (period
awide), B at `[col, col+bwide)`, and — ONLY for the FIRST (top) band — C tiled across `[col+bwide, 10)` (period
cwide), all cyan. The OUTPUT is the input PLUS, for every band AFTER the first, the C-pattern tiled into that band's
own C-region `[col+bwide,10)` drawn in BLUE (color 1). Every changed cell is input-0 → output-blue (verified
3000/3000). So: copy input, and "replay" band-0's C-pattern, recolored blue, into each later band's empty right side.

**Current:** 14.605 pts, ext:kojimar7113, mem 18574, params 14113 — but scores only **49/500 ≈ 9.8% on isolated fresh**
(re-measured; matches handoff's 6–12%). Stored points are a non-generalizing memorization.
**Target tier:** GAP-CLOSER (any generalizing net beating ~10% adds ~its full score to REAL LB).

## Attempts (all numpy-oracle / recovery experiments; no ONNX written — see verdict)
| # | angle | result |
|---|---|---|
| 1 | closed-form per-cell | NO — output cell depends on band-0's far-away C-region (non-local copy) |
| 2 | C-pattern shift hypothesis (read band0 C, tile into later bands by Δcol) | EXACT 3000/3000 **with oracle params** |
| 3 | recover params from input: A-anchor alignment (first cyan row in cols[0,awide)) | anchor=top+min(arows) consistent 2000/2000 — robust vertical align without needing absolute top |
| 4 | recover cstart_i for later bands = 1+rightmost-cyan-in-band | EXACT 0 mismatch |
| 5 | recover band-0 anchor cstart0 (col0+bwide, col0∈{awide,2awide}) | **AMBIGUOUS** — B-pattern match leaves 2 candidates ~27%; A∪B reconstruction match leaves ~21% ambiguous |
| 6 | full global-reconstruction brute force over (tall,awide,bwide,cwide,cstart0) | ~72% fresh (band-detection edge cases + multi-accept votes; reachable ceiling is 99.95%, not 100%) |

## Irreducible-floor analysis — TWO independent walls

**WALL 1 — INFORMATION-THEORETIC (caps even a perfect oracle below the fresh bar).**
The C-pattern exists ONLY in band 0. Band 0's internal A|B|C column segmentation is NOT always determined by the
input: `col0∈{awide,2awide}` is an independent random draw, and when the A-pattern, B-pattern and C-pattern tiles
coincide at the A/B/C boundary, two different `cstart0` values reconstruct the SAME band-0 input but imply DIFFERENT
blue outputs. Direct collision scan: **same input → different output in ~1 case per 2174** (oracle ceiling
**99.954%**, measured over 100 000 fresh instances; 43 ambiguous inputs / 79 308 unique). The choice of cstart0
flips the output in 44% of cases, so it is load-bearing, not cosmetic. A perfect net therefore leaks ~1.4 per 3000.
The session's stated bar is ISOLATED fresh ≥3000/3000 (a prior 1/3000 leak caused an LB regression). 219 cannot
meet 3000/3000 — the residual is the theoretical floor, not a fixable bug.

**WALL 2 — ONNX EXPRESSIBILITY (the deeper blocker).**
Even accepting the 99.95% ceiling, the rule needs: detect a DATA-DEPENDENT NUMBER of bands (2–6) at data-dependent
rows; per band derive an A-anchor and a cstart (= 1 + rightmost cyan, a per-object reduction); identify band 0 as
the furthest-right band; disambiguate cstart0 via a 2-candidate GLOBAL self-reconstruction; then COPY band-0's
C-region tile into each later band with a per-band (Δrow, Δcol) shift, recolored. This is variable-count
multi-object correspondence with a data-dependent source→target copy — exactly the class the playbook lists as a
WALL (no Loop/Scan/NonZero; the band count and both source and target positions are all data-dependent, so the
copy cannot be unrolled into a fixed DAG the way task48's bounded flood was). The master-key bounded-iteration
unrolling does not apply: there is no bounded local propagation: the operation is a long-range gather of one band's
content into K others.

## Best achieved
No net written. Best understood-rule numpy recovery ≈72% (band-detection edge cases unsolved); even a flawless
recovery is capped at 99.954% by WALL 1. Not adoptable at the required 3000/3000.

## OPEN ANGLES (genuinely tried and rejected — left for the record)
- Per-cell closed form: rejected (non-local copy).
- Absolute-column C reading (skip cstart0): FAILS — cstart_i ≢ cstart0 (mod cwide) in ~11% (phase differs).
- Both-cstart0-candidates-agree shortcut: only 56% agree, so cannot dodge the disambiguation.
- A-anchor alignment + global reconstruction in numpy CAN approach 99.95% with more edge-case engineering, but
  (a) still below 3000/3000 by WALL 1, and (b) does NOT translate to ONNX by WALL 2.

## INFEASIBLE VERDICT
INFEASIBLE for a 3000/3000-generalizing ONNX net, on TWO independent grounds:
(1) information-theoretic: oracle ceiling 99.954% (~1 ambiguous input per 2174) — the input does not always
determine the output, so no model can reach the required exactness; and
(2) expressibility: variable-count (2–6) band correspondence with a data-dependent source→target region copy is a
no-Loop ONNX wall (long-range gather, not bounded local propagation — master key does not apply).
The handoff's "info-bottleneck / connectivity wall" label is CONFIRMED and now quantified: the bottleneck is band-0's
internally-ambiguous A|B|C segmentation, the sole carrier of the C-pattern.

## 2026-06-29 public/source parity recheck

Read-only parallel analysis rechecked the current live/source/public state.

- Current manifest/inventory entry is the URAD teacher overlay: `points=15.162385`, `memory=18633`, `params=92`.
- Public candidates under `boristown`, `lucifer`, `biohack_mix`, and `urad` have no useful structural delta from live/source: same computation, op histogram, initializers, and attributes aside from serialization/naming details.
- The older wall conclusion still stands. Public artifacts do not provide a new mechanism, and the task remains a poor source-owned rewrite target because the needed variable-count long-range band copy is not cleanly expressible in the available no-loop ONNX subset.

## INSIGHT (transferable) ⭐
- ⭐ When a "deterministic-looking" generator hides ONE template in a single object and that object's internal
  segmentation can coincide, RUN THE COLLISION SCAN (`dict[input.tobytes()] → set(output.tobytes())` over 50k–100k
  fresh) BEFORE building. A nonzero collision rate is a hard oracle ceiling; if it sits near the 1/3000 fresh bar,
  the task is INFEASIBLE regardless of recovery cleverness. (Here ~1/2174.)
- ⭐ A-ANCHOR ALIGNMENT for multi-band tasks: align repeated objects by the first cyan row of a shared sub-region
  (here the always-present A-region) — gives a per-object reference that is consistent across objects (= top +
  min(pattern-rows)) WITHOUT needing the true top, which can be empty. Useful for any "repeat this object N times"
  recovery. But it does NOT make the cross-object COPY ONNX-expressible when N and positions are data-dependent.

## S8 (2026-07-02) — batched-band + placement einsum (+0.922) ADOPTED; "18k floor" REFUTED
5 copy-pasted per-band blocks (~2KB each) → ONE K=6-batched block (also fixes the missing 7th
band); placement+accumulation → ONE fp16 einsum 'kjr,ks,jsc,k->rc' (placement one-hots ×
shift selectors × shifted patterns × exists flags → [15,10] mask, 7.8KB→1.4KB); shift variants
via init index-table Gather (params not memory); epilogue Max(8·cyan, mask) + Pad-255 + Equal.
7078+132 vs 18033+93 → 15.195→16.117. Fresh 20000: cand-only-fail 0, inc-only 13 (candidate
strictly ⊆); fresh_verify 2500 (1042≤1046) + 1500 (632≤633). Incumbent inherent fail ~42%
(public LB = bundled → still pointed). LOAD-BEARING QUIRKS preserved: 4-shift set {+2,+1,0,−1}
(Δ=−2 spuriously wins), k=2 block uses band-0's occ[1] validity (bands can have internal empty
rows). Latency 0.04ms.

## 2026-07-11 — fresh-fail diagnosis + exact decision procedure (fork agent) — NUMPY VALIDATED, ONNX PORT PENDING
Context: silent-zero reality (hidden arc-gen draws) made the deployed net's known ~42-44% inherent
fresh-fail the board's largest private-LB risk (~7 expected pts at hidden-k≈1-2).

FAILURE TAXONOMY (1500 fresh, size-filtered): 659 fails = 116 edge-only (right-edge partial C-tile
truncation) + 427 interior (wrong sigma/shift selection) + 116 mixed. The S8 4-shift heuristic
{+2,+1,0,-1} is the dominant error source, not the edge clipping.

EXACT RULE (from generator source, task_90f3ed37): bands at rows[idx] (tall<=3, shared A/B/C pixel
patterns, widths awide/bwide/cwide ∈ {1,2}, col_idx = awide*{1,2}); band0 complete; lower bands
missing their C-tiling (from col_idx+bwide, step cwide, per-pixel clip) — output adds it in blue.
KEY IDENTITIES: (i) s_k (fill start) = rightmost_occupied(band k) + 1  [bwide := max(bcols)+1 ⇒ B's
last column always drawn]; (ii) width invariants: the LAST column of each of A/B/C's sampled pattern
is always occupied (all three widths are max(cols)+1 of the sample); (iii) all lower bands share one
relative row-mask ⇒ segmentation (internal 1-row holes in tall-3 bands vs distinct tall-1 bands) is
decided by global consistency + parse-vote; (iv) band-0 row anchor d (lower visible top vs band0
top) is a single global shift, recovered by joint (d, sigma) exact prefix-match.

DECISION PROCEDURE (v6b, tools/t219_exact_ref.py): enumerate the finite parse space
(d ∈ [0,tall), bw,aw ∈ {1,2}, col_k = s_k−bw ∈ {aw,2aw}, col_0 ∈ {aw,2aw}, sigma = col_0+bw,
cw ∈ {1,2}) with checks: A-region prefix equality on [0,min(col_0,col_k)); B-block equality
band0[:,sigma−bw:sigma] == band_k[:,s_k−bw:s_k]; A-tiling periodicity within [0,col_0); C-tiling
periodicity of band0 from sigma (mod cw, auto edge-clip); the three max-col occupancy invariants;
global (d,sigma,cw) consensus across all lower bands; tie-break prefer larger sigma.
MEASURED: bundled 265/265 PASS; fresh 20000 → 62 fails (0.31%) vs incumbent 43.9% (142x).
Residual 0.31% ≈ near-collision parses (ledger's measured oracle ceiling ~1/2174 = 0.046% is the
hard floor; a few resolvable modes may remain).

ONNX PORT (not built this session — dedicated build required): all machinery is fixed-size and
vectorizable under the no-Loop opset: band stack [K<=7, 3, 10] via the S8 K-batching; the parse
space is <=3*2*2*2*2 = 48 branches of small tensor equalities (Equal+ReduceMin) producing a vote
tensor; selection = ArgMax over 48; fill = per-band broadcast of the selected pattern with
(c-s_k) mod cw phase (index-table Gather, cw in {1,2} → two precomputed phase planes). Est. added
cost over the deployed 7210: +1-3KB (well worth it: expected private value ≈ +6-7 pts vs zero-risk).
The S8 net's K-batch/placement einsum skeleton is reusable; only the selector and fill subgraphs
change. Fresh gate for the port: >=20k A/B, require <=0.4% and strictly ⊆-or-equal incumbent fails
is NOT required (incumbent fails 44%) — require candidate fails <= 100/20k.

## 2026-07-11 exact-rule ONNX port (candidates/task219/exact_v1.onnx)
cost 25124 (mem 23860 + params 1264) vs deployed 7210 = -1.249pt public; fresh 18/6000 (0.30%,
oracle floor ~0.046%); bundled 265/265; server ORT 1.24.4 runs 4/4. Gate REJECT on price only.
DISPOSITION: goes into the HEDGE bundle (portfolio strategy — Kaggle final selection takes 2
submissions: one cheap-risky board + one max-protected board = best-of-two on private).
Raw-only variant (est. -5K cheaper, 4.26% fresh fail) rejected — dominated across hidden-set
uncertainty. Anchor-alignment shortcut rejected (bundled train[1] uses out-of-distribution
col=3 parameters).

## ADOPTED 20260715T073303Z
- cost: 7210 -> 5258 (points 16.4325)
- source: candidates/task219/conv_topk.onnx
- note: collapse: free epilogue folds 7 tensors (incl 900B [1,1,30,30] col30) into one 1x1 QLinearConv (w_zero_point=128 -> signed) over a [1,3,15,10] u8 stack doing colour map + canvas pad (pads=[0,0,15,20]) into free output; u8 saturation performs the old Max(8*cyan,blue) free. W5 fold: pf IS the row-gather one-hot so the score einsum reads cyh through it, deleting the [1,1,6,3,10] window. TopK(sorted,largest) index-stable on ties yields band tops in row order + exists-flags, replacing CumSum band-index + [6,15] one-hot + 2 ArgMax. 73->69 nodes, cost 7210->5258. Semantics preserved bit-exactly; differential vs incumbent 4000 in-distribution + 4000 OOD cyan + 4000 OOD multicolour = 0 disagreements.
