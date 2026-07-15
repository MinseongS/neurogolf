---
deployed_cost: 7775
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task209 — 8a004b2b

**Rule:** Input has (a) a YELLOW box marked only at its 4 corners at (brow,bcol), size wide×tall;
inside the box, a sprite is magnified mag× and placed at offset (irow,icol) but ONLY a random subset
`shows` of its cells is drawn; (b) the FULL sprite (every cell, native 1× resolution, arbitrary colors
from {1,2,3,8}) drawn at the bottom of the grid (rows ≥ height−3, strictly BELOW the box). OUTPUT =
the box region (wide×tall) with yellow corners + the COMPLETE mag× magnified sprite at (irow,icol).
So the task = "recover the full sprite from the bottom, recover (mag,irow,icol) from the partially-
shown magnified blocks, then re-stamp the complete magnified sprite into the box."

**Current:** 13.357 pts, `gen:wguesdon6315` (imported overfit), mem 113691, params 144.
Base net FAILS isolated fresh (~38/40) → scores ~0 on real Kaggle LB. Gap-closer candidate.
**Target tier:** detection / multi-object shape-correspondence — non-separable.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | Python reference solver (full brute search over mag∈{2,3,4}×irow×icol + complete-block validation + arbitrary-color sprite re-stamp) | n/a (analysis) | — | — | — | 266/266 STORED ; ~495-499/500 FRESH | irreducible generator ambiguity caps fresh <100% |

## Best achieved
No ONNX net built. The IDEAL Python solver passes 266/266 STORED examples (incl. the official mag=4
test) but only ~98.5-99.6% of FRESH instances. ONNX implementation judged INFEASIBLE as a clean
generalizing win (see below). Not adopted.

## Irreducible-floor analysis (two independent walls)

**WALL 1 — generator is genuinely ambiguous (input does NOT determine output).**
Only a random subset `shows` (≥2) of sprite cells is drawn magnified in the box. When the shown blocks
lie in a SINGLE sprite row (or single column) — common; the official TEST example is exactly this case:
shown blocks occupy sprite-row-0 only (`8 . 3`), mag=4 — the translational offset (icol, sometimes irow)
of the magnified sprite is under-determined. Measured over 8000 fresh instances: only **98.45% of inputs
uniquely determine the output**; the other 1.55% admit ≥2 distinct consistent outputs (truth always among
them, but unrecoverable). Confirmed by constructing two parameter sets that differ only in icol yet are
both valid. Best deterministic tie-break (max-mag, then min-irow,min-icol) reaches ~99.3-99.6% fresh —
so even a PERFECT solver fails ~1-5 per 500 and would clear genverify's 40/40 gate only ~85% of the time.
NO net (not even #1) can exceed this; it is a property of the generator, not of the encoding.

**WALL 2 — no separable / single-op tensor form (ONNX construction is a detection-floor blowup).**
The reconstruction is intrinsically a CORRESPONDENCE + SEARCH problem with NO separable structure:
(1) mag is not directly readable (adjacent same-color blocks merge; run-gcd recovery only ~98% and gcd
isn't an ORT op); (2) (irow,icol) require matching shown blocks to the bottom sprite's cells, which has
no row⊗col factorization; (3) the re-stamp is an arbitrary-COLOR data-dependent Kronecker magnify by a
RUNTIME factor into a data-dependent offset in a data-dependent-size output. A faithful ONNX build must
enumerate ~3 mags × ~15 irow × ~15 icol candidate full reconstructions, validate complete-block coverage
per candidate, and select — materializing many full ≤20×20 multi-color planes (≫100KB intermediates),
landing at the ~13-14 detection floor at best, while STILL capped by Wall 1 below 100% fresh. Banned ops
(Loop/Scan/NonZero/Unique) make the search/argmax-of-candidates expression especially costly.

## OPEN ANGLES (exhausted for a CLEAN generalizing win)
- Direct scalar recovery of (mag,irow,icol) without search: BLOCKED — single-shown-row/col cases are
  genuinely ambiguous (Wall 1), so no scalar formula can be exact.
- Assume blocks span ≥2 rows AND ≥2 cols (then mag,offset pin uniquely): FAILS the official stored test
  (shown blocks are single-row) → evaluate() = 0 points. Non-starter.
- Heavy brute-search ONNX at the detection floor: even if buildable (~13-14 pts, below current 13.357),
  Wall 1 still drops fresh below the genverify gate ~15% of runs — not a reliable +13.

## INSIGHT (transferable)
⭐ Some arc-gen generators are INTRINSICALLY AMBIGUOUS: a partially-shown magnified sprite (random
`shows` subset) leaves the translational offset under-determined whenever the shown cells are collinear
(single sprite row/col). This is detectable cheaply — construct two parameter sets identical except for
one offset and check the inputs collide. When the input→output map is not a function (1.55% here), NO
encoding can pass strict fresh 500/500; the ceiling is the unique-determination rate (~98.5%), or the
best-tie-break rate (~99.6%). 209 is a genuine wall on BOTH the determinism axis (Wall 1) AND the
no-separable-form axis (Wall 2) — the earlier "suspected near-wall" verdict is confirmed: it is a wall.
The base net's 13.357 stored is unbeatable AND non-generalizing; there is no clean generalizing
replacement. (Lesson mirrors task255/198 connectivity walls but here the wall is generator non-determinism.)

## S8 (2026-07-02) — counting-model rebuild (+0.349) ADOPTED, div 0
Not iterative — the win is the COUNTING-MODEL rebuild: free-input Einsum contractions for all
row/col bounds ([30] f32 profiles 120B replace ~55 400B bool planes); spread-based S-detection
(MaxPool(lab)+MaxPool(9−lab)−9, global-max==9 test); separable two-stage axis Gathers (kills
[20,20] i32 flat-index plane); 11×11 valid-Conv label read at 20×20 (1600+400 vs 4900, +1200p);
QLinearConv int32 bias folds +1; Pad-with-negative-crop replaces Slice+Pad.
21298+1418 vs 32027+185 → 14.620→14.969. Fresh 2500/1500/800: div 0; inherent fail ~9.8%
(generator-ambiguity wall) unchanged. ORT-OK under strict inference: u8 MaxPool (incl 20×20
global), u8 Where/Add/Equal, QLinearConv+i32 bias, repeated-free-input einsums.

## S9 (2026-07-03) — fold 2nd pass: FLOOR re-confirmed (no change)
13a N/A: output = runtime-factor/offset Kronecker gather (not fixed-mixer einsum).
All candidate plane-merges measured byte-NEUTRAL with only 2 magnification sizes
(S-detect keep-mask, vote, stamp, box). W11 1210p = optimal trade vs 3600B 1×1-conv
plane. Existence-based mag detector: 1/20000 fresh errors + saves only ~800-1200B
(+0.04-0.05) on a net at 10.1% ambiguity wall — rejected under cand≤inc gate.
Clean-corner+universality detector numpy 0/20000. DO NOT re-probe.


## S15 (2026-07-06) — ADOPTED from urad public bundle 7225.82 (submission 54367833): 22716 -> 7927 (+1.053)
Mechanism: Terminal GridSample (fp16 [1,30,30,2] grid vs fp32 input) = gather+mask+zero-pad in one free node.
Gate (fresh_verify, inc/cand fail on 1500-2000): 216/55 -> adopted under safe rule (cand fail <= inc fail AND cheaper).
Source-owned via live_to_exact_source --write-src; re-measured grader-side fail=0. Backup in scratchpad/backup_networks.
See memory [[neurogolf-urad-7225-bundle-vein]]. our incumbent fresh-failed 216/2000 (already ~0 on private LB); urad both cheaper AND more robust — strict win.

## S15b (2026-07-06) — ADOPTED from prvsiyan 7235.05 min-merge: 7927 -> 7862 (+0.008); gate inc/cand=34/34 (safe). See [[neurogolf-urad-7225-bundle-vein]].

## 2026-07-11 — fresh-tail diagnosis → MIXED (c-irreducible floor + expensive-(a), NO-FIX)
ran: reproduced deployed net (submission/overfit_nets/task209.onnx, the S15 urad GridSample solver,
  164 nodes — NOT the old overfit) on fresh generate() draws. 2000-draw + 6000-draw sweeps. For every
  net-fail I enumerated ALL params (mag∈{2,3,4}×irow×icol) that reproduce the exact input via a ≥2
  full color-matching-block cover (sprite shape/colors read unambiguously from the bottom native copy,
  box geom from the 4 yellow corners), then rendered each candidate's output. Analyzer validated:
  ground-truth output reproducible from a consistent triple 47/47. Bundled stored train+test 4/4 pass.
tool+date: direct-ONNX repro harness + full-enumeration ambiguity oracle vs arc-gen generator
  (6000 fresh draws) + reference/arc-code-golf-solutions/task209.py cross-check, 2026-07-11 (fork).
verdict: net fresh-fail = **2.30%** (138/6000; matches the 2.2% sweep). SPLIT:
  • **0.83% (50/138) = class (c) IRREDUCIBLE generator ambiguity.** Same input admits ≥2 distinct
    valid outputs. CONCRETE INSTANCE: box(brow,bcol,wide,tall)=(2,4,14,11), sprite pix
    [(0,0,8),(0,1,2),(0,2,2),(1,0,3),(1,1,3)]; consistent triples (mag=3,irow=1,icol=1) AND
    (mag=3,irow=1,icol=4) — differ only in icol, both reproduce the input exactly. 9/12 of the
    ambiguous fails have COLLINEAR shown blocks (single sprite row/col) → offset under-determined
    (confirms the old Wall-1). NO net can beat this floor; net already tie-breaks 33/83 ambiguous
    draws correctly. Base ambiguity rate = 1.38% (matches ledger's ~1.55%).
  • **1.47% (88/138) = uniquely-determined but net-wrong.** Full enumeration finds exactly ONE
    consistent triple, so a smarter feedforward solver COULD get these (NOT Loop/Scan/NonZero-blocked
    — it is a bounded ~3×15×15 candidate grid + per-candidate complete-block validation + argmax
    Gather). Failure profile: mag∈{3,4} (30/35 sampled) with exactly 2 collinear shown blocks
    (27/35) — the cheap GridSample heuristic mis-recovers scale/offset when the magnified footprint
    is large and evidence is minimal. Reference oracle (arc-code-golf task209.py) ALSO fails 45/47 of
    these → the deployed net already tracks the best-known feedforward heuristic. This is technically
    class-(a) FIXABLE but the ONLY exact fix is the Wall-2 full-enumeration solver (materializes ~700
    candidate planes → detection floor ~13–14 pt) which is **−2 to −3 pt below the current ~16.0 pt
    net** and STILL capped at the 0.83% irreducible floor. NET-NEGATIVE at any k≥1 hidden draw
    (16.0×0.977=15.6 > 13.5×0.992=13.4). NO-FIX shipped.
reopen: a public dump net with measured fresh-fail <2.3% on THIS generator (adopt if cand≤inc &
  cheaper, per S15 routine); OR evidence the hidden/private set is curated/filtered (not raw
  generate()) — as found for task002 — which would make the 0.83% floor an overstatement; OR a
  cheap targeted scale/offset-recovery fix for the mag∈{3,4}+2-collinear-block mode that regresses
  neither the 265 bundled nor mem (none found — the binding step is a runtime-factor Kronecker
  re-stamp, already GridSample-encoded).
falsification history: SUPERSEDES the pre-redesign S8/S9 "13.357 stored unbeatable + fresh ~9.8%
  ambiguity wall" verdict — those were measured against the OLD overfit base; the current deployed
  net is the S15 urad GridSample SOLVER at 2.30% fresh-fail (far better), and the true irreducible
  floor is 0.83% (not the earlier ~1.55–9.8% figures, which conflated ambiguity with heuristic
  plateau). Wall-1 (collinear-shown offset under-determination) CONFIRMED with a concrete 2-output
  instance; Wall-2 (no cheap exact form) CONFIRMED (best-known oracle also plateaus, exact needs
  detection-floor enumeration).
## ADOPTED 20260712T141558Z
- cost: 7775 -> 7609 (points 16.0629)
- source: dumps/archive_extract/submission7300+/task209.onnx
- note: all-in archive graft; Kaggle-CONFIRMED in record 7410.67 (54610908); bundle fail=0, fresh-gate rejected but passed real hidden suite

## ADOPTED 20260713T112224Z
- cost: 7609 -> 7579 (points 16.0669)
- source: candidates/task209/kcollapse.onnx
- note: kernel-collapse: sparse Conv single tap -> 1x1+pad; bit-identical params -30

## 2026-07-13 — byte-golf axis floor CONFIRMED (QLinearConv color-decode recast = measured net-NEGATIVE)
ran: per-tensor cost audit of the DEPLOYED archive net (7579) — never plane-audited before (S8/S9
  audits were vs our OLD net). Top counted intermediates: color_f 1600B f32 (Conv color-decode ->Cast u8),
  Oidx30 900B u8, BmCell 576B bool, color_u8 400B u8; everything else <=400B, ALL already min-dtype.
  Only overpay candidate = color_f fp32 1600B. Built QLinearConv recast (Cast input->u8 + QLinearConv,
  bit-identical 4/4 stored) to kill the fp32 plane.
tool+date: tools/per_tensor_cost.py-style static audit + ng gate isolated measure, 2026-07-13.
verdict: REJECT — measured cost 7579 -> 14981 (memory 6895 -> 14295). The integer path must materialize a
  COUNTED u8 copy of the full [1,10,30,30] input = 9000B; that dwarfs the 1600B fp32 Conv output it removes.
  fp32 Conv is OPTIMAL for color-decode because it reads the FREE fp32 input directly. Confirms our own
  cost-rule ([[neurogolf-detection-floor-costmodel-proof]] Cin_read=10 >= 3*Cout=3 => net-negative) by
  direct measurement. All other planes already at minimal dtype/size. NO endogenous byte-golf win remains;
  kernel-collapse (params -30, same day) was the last bit-identical lever.
reopen: external (public/archive) net with measured fresh-fail <2.3% & cheaper (S15 routine) ONLY.
  Do NOT re-attempt QLinearConv/integer color-decode recast on 209 (or any 1x1 color-decode reading >=3
  input channels) — the free->counted input flip makes it strictly worse.

## ADOPTED 20260715T021526Z
- cost: 7579 -> 7495 (points 16.0780)
- source: candidates/task209/factor_qc_rank8.onnx
- note: exact Qc axis-2 rank8 factorization: shared one-hot selector [12,8] + prototype [3,3,8,5] in both row/col Einsums; no added nodes

## ADOPTED 20260715T031437Z
- cost: 7495 -> 7330 (points 16.1003)
- source: candidates/task209/factor_qcb_second_stage.onnx
- note: direct exact second-stage QcB rank-5 factorization across (s,r)|(k,p)

## ADOPTED 20260715T084402Z
- cost: 7330 -> 6821 (points 16.1722)
- source: candidates/task209/u8idx_qsplit.onnx
- note: collapse: (1) uint8 recast of every geometry/index chain (-473B) — all placement arithmetic is ring-ops mod 256 so intermediate sign is irrelevant; the only non-ring steps are Div/Clip, where a negative wraps to >=239 and Clip sends it to the HIGH clamp instead of the low one, but Lcpad row 0 == row 4 and col 0 == col 6 (both the zero pad-ring from Pad(Lc2,[1,1,1,1],0)) so both clamps gather IDENTICAL bytes. (2) Exact Qc refactorization (-36 params): Qc[s,k,m,p]=delta(p==k+m//(s+2)) was QcA[12,8]*QcC[3,8,5]*QcD[5,3,5]=291; the (s,m)|(k,p) rank-5 split gives QA[s,m,j]=delta(m//(s+2)==j) [3,12,5] + QS[k,j,p]=delta(p==k+j) [3,5,5] = 255, dropping an operand. cost 7330->6821. Differential: 13330 inputs, 0 disagreements (12000 fresh over 3 seeds + 1064 colour permutations + 266 stored); risky wrap path verified live (rr0<0 in 19.0%, cc0d<0 in 22.4%, ~100% hit the low clamp); suite is harsher than real arc-gen (incumbent fresh-fail 10.25% vs ledger-measured 2.30%). u8 lever EXHAUSTIVELY proven: over a in [-120,120] x sval{2,3,4} x i, 5400/11568 row and 6630/14460 col combos differ between int32 and u8 paths and 0 land off the zero-pad set {0,4}/{0,6}; recast exact for a in [-242,240]/[-238,232] and the graph STRUCTURALLY bounds a_r2 in [-12,19] (>20x margin, not empirical).
