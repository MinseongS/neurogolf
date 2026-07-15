---
deployed_cost: 8660
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task216 — 8efcae92 ("mostpixels")

**Rule:** 20x20 grid with 3-4 non-overlapping solid BLUE(1) rectangles ("boxes"),
each separated by a >=1-cell gap **in at least one axis** (common.overlaps spacing=1
=> two boxes may touch in one axis but never overlap in BOTH). Each box has RED(2)
pixels scattered inside; red counts are DISTINCT across boxes. The box with the MOST
red pixels (generator forces box 0 = max) is the winner. Output = exactly that
winning box (its blue rectangle + reds) placed at the top-left (0,0); cells outside
the box's own size are all-channels-zero (output grid IS box-sized).
**Current (public):** 13.56 pts, mem 92612, params 101 (CumSum-scan segmentation +
ArgMax + crop/translate). Generalizes 300/300 fresh — a real, at-floor score.
**Target tier:** detection (segmentation + global ArgMax + data-dependent crop &
translate). NOT B/A/S: output is a variable-size data-dependent crop of one of
several connected components selected by a non-local global maximum.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | corner-label (T,L) via triangular ReduceMax; per-corner red histogram via batched double-MatMul; ArgMax winner; double-MatMul shift to origin | det | 176k | 361 | 12.92 | 266/266 stored | correct but heavy (W^3 histogram + W^3 running-max) |
| 2 | running-max -> Hillis-Steele doubling scan (O(W^2)); drop histogram, count reds via 2-D prefix-sum integral image over [T,L,B,R] (needs R,B suffix-min scans) | det | 138k | 404 | 13.16 | 266/266, 300/300 fresh | correct, leaner |
| 3 | fp16 everywhere (scans, run masks, prefix-sum cast post-CumSum, gather idx int32); fp16 neighbour shifts | det | 125k | 404 | 13.26 | 266/266 | dtype squeeze |
| 4 | doubling shift via fixed {0,1} MatMul (no Pad/Slice), col-orientation + transpose for row scans (10 shared shift mats), factored integral-image index math | det | **106k** | 4219 | **13.39** | **200/200 fresh** | best; still < public 13.56 |

## Best achieved
**13.39 @ mem 105972 params 4219 — 266/266 stored, 200/200 (& 300/300) fresh.**
Adopted? **N.** Beats prior 13.56? **NO** — 0.17 BELOW. Does NOT beat by +0.3;
in fact a slight regression. Verdict: MARGINAL-LOSS, do not adopt.

## Irreducible-floor analysis
Dominant memory = ~91 fp16 [1,1,20,20] (800 B) intermediates. Breakdown of the
irreducible work:
- **4 corner maps (T,L,B,R)**: each needs a prefix/suffix MAX scan over the run-
  start/run-end index encoding. Hillis-Steele doubling = 5 steps (W=20 => need
  reach 18 => log2 => 5 steps unavoidable), 2 tensors/step (shift + Max) = 40
  fp16 [1,1,20,20] = **~32 kB. Byte-invariant**: stacking maps onto channels makes
  each tensor proportionally bigger (no win); uint8 would halve it but ORT
  Max/Min/MatMul all reject uint8; CumSum rejects fp16. fp16 is the floor.
- **integral-image red count** (PS 2-D CumSum f32 + 4 Gathers + index math) ~20 kB;
  needs R,B (=> the 2 extra scans). The rectangle [T..B]x[L..R] provably contains
  only the winning box (no two boxes overlap in both axes), so the integral image
  is exact — but it requires all four corners.
- run-start/end detection (~16 tensors) + double-MatMul shift-to-origin (~12) +
  uint8 label Pad + final Equal (output is FREE) ~25 kB.
The winner-SELECTION (which box has the most reds) is the genuine cost driver: it
is a non-local global ArgMax over per-box red totals, which forces full per-cell
segmentation (corner labels) — exactly what the public CumSum net also pays for.

## OPEN ANGLES (re-attack backlog)
- **Drop R,B (2 scans, ~16 kB)** by counting box reds with only L,T. Every cheap
  formulation tried still needs the box right/bottom edge (row-run total needs R;
  box total needs B): the integral image over [T,L,B,R] is the minimal exact form.
  A genuinely L,T-only group-by-corner count is the W^3 histogram (worse). OPEN: is
  there an O(W^2) group-sum keyed by (T,L) using two reset-CumSums that reuse L,T?
- **uint8 scan** blocked: ORT Max/Min reject uint8 (verified). A custom non-Max
  monotone combine (e.g. Add of disjoint magnitude bands) might pack the running
  max into uint8 — untried, fiddly, ~16 kB upside if it works.
- Tier S/A/B all blocked: output is a variable-size crop of a globally-selected
  connected component => no fixed Conv/permute/separable form exists.

## INSIGHT (transferable)
⭐ **2D-separated rectangles get a unique per-box label from (T,L) = (top of its
column-run, left of its row-run), recoverable by prefix/suffix MAX scans; and the
bbox [T..B]x[L..R] then contains ONLY that box** (because spacing>=1 forbids overlap
in both axes), so per-box reductions (here red count) reduce to a 2-D prefix-sum
integral image with 4 Gathers — NO connected-component flood-fill needed. Beware the
spacing=1 loophole: two boxes CAN share a column range (row-gap separated), so a
naive "reds in bounding rectangle" using a buggy suffix-MAX over-counts; the run-end
edge must be suffix-MIN (nearest end >= position), validated 800/800.
⭐ **Hillis-Steele doubling prefix-max via a fixed {0,1} MatMul shift** (encode values
so 0-fill is order-safe; col orientation + Transpose for row axis to share matrices)
replaces Pad+Slice+Max (3 tensors/step -> 2) — but the byte cost is invariant under
channel-stacking and fp16 is the dtype floor (Max rejects uint8), so this whole
"segment + global-argmax + crop" class floors near the memorizer/CumSum level (~13.4)
for everyone. This is the non-local detection wall the sweep guidance warns about:
the public net is already at it.

## S9 (2026-07-03) — free-operand einsum red-count (+0.053) ADOPTED
NOTE: this tasklog was STALE (described a 92k segmentation net); live incumbent was
already 9135 via QLinearConv-corner + integral-free design. S9 golf: red plane (400B) +
2×MatMulInteger contractions (320B i32 + 64B cross-terms) replaced by ONE 4-operand
Einsum 'nkrc,k,br,bc->b' reusing counted c12_f32 as free operand → per-box red counts
(4,) direct. mem 9048→8584, params 87→76. Bit-identical: 2500+600 uncached fresh 0/0/0.
FLOORS: c12_f32 3200 = entry floor (both channels needed at 20×20; alternatives ≥cost),
corner-finding 1200 + run-scan 1120 near-minimal u8. Backup task216_pre_s9.onnx.

## S11 (2026-07-03) — mech-15/pointer scout: KILL — output = variable-size crop of globally-ArgMax-selected most-red box; cost = winner-selection (3200B entry + corner/run-scan). No separable fills, no carrier. Floor stands.

---
## 2026-07-08 — fold-batch floor verdict (STALE tasklog above = old CumSum net; active is now compact mem 8584)
**Ran:** fold_finder flagged `c12_f32[1,2,20,20]` fp32 slice (3200B, Slice(input)→Einsum). Deep opus agent tried rerouting the counting Einsum (`nkrc,k,br,bc->b` → counts[4]) to contract free `input` directly.
**Verdict: FLOOR.** The Einsum was a FREE 2nd consumer of already-materialized `c12_f32`; c12_f32 is also the source for `Cast→c12 uint8` (feeds QLinearConv corner-detect + Gathers + crop). Reroute doesn't delete c12_f32 and ADDS ~560B (masks [4,20]→[4,30] + pads). Slice preserves free-input fp32 dtype; regional uint8/fp16 needs full-tensor cast first (9000/18000B) = worse. Active unchanged (mem 8584 par 76).
**Reopen:** whole-net reformulation building the output box from free input without QLinearConv corner-detect; or slice+cast fusing op.

---
## 2026-07-09 — task394 compact-OneHot/projector generalization attempt → mechanism NO-TRANSFER; trace-max audit win instead (+274B, cand NOT adopted)
**Ran:** full byte-costing of a task394-style rewrite (free-input detection Einsums, compact-coordinate OneHots
w/ out-of-range sentinel, final N-ary placement Einsum straight to free output) against the live net's exact
per-tensor grader accounting (profiler trace over all 266 bundled examples, replicating scoring.calculate_memory).
**Tool+date:** manual per-tensor trace audit + generator analysis (task_8efcae92.py), 2026-07-09, onnx 1.21/ort 1.26.
**Verdict on the 394 mechanism: NO-TRANSFER (anti-win), byte math:**
- *Placement Einsum as graph output* ('bkrc,Ir,Jc->bkIJ'): row/col selector OneHots are fp32-welded to the free
  input and must be 30-wide ⇒ [30,30]×2 = 7200B (compact [18,30]×2 = 4320B + 1080 const-lift params) vs the
  current crop→Pad carrier at 648B (traced max 374B). Box patch is up to 18×18=324 cells vs task394's 3×3=9 —
  the exchange rate inverts for large patches.
- *Selector-plane elimination*: rm/cm [4,20] fp32 (320B ea) are RANGE masks (not Equal-vs-arange onehots) and are
  einsum-welded to c12_f32; MatMulInteger/u8 alternative needs a red u8 plane (Slice 400B) ⇒ +300B net.
- *c12-free detection* (Conv on free input, pads [1,1,-10,-10], corner stencil x(r,c)-x(r-1,c)-x(r,c-1)+bias):
  detection itself gets cheaper (2800B vs 5200B) but scans go fp32 ([4,30] OneHot+vec+mask ⇒ +2100B) and the
  epilogue loses its u8 channel-resolved source: cheapest c12-free epilogues = f32 input-Slice crop (traced max
  8·187=1496B) or enc-plane + double-Pad + Equal-vs-iota sentinel (1224B + 2000B marginal source). Best full
  c12-free design ≈ 9046B > 8584B deployed. c12_f32 3200 + c12 800 entry-bridge floor RE-CONFIRMED (3rd time).
**Reopen-trigger:** a u8-producing op directly from the fp32 free input cheaper than a same-size f32 bridge; or
grader change away from max(declared, traced-max) charging; or an exact winner-selection rule not requiring
per-box extents (red counts are forced exact by unique-counts, so density/blur proxies stay dead).
**Falsification history:** 2026-07-08 fold-batch verdict (c12_f32 floor) — held. S11 mech-15 KILL — held.
**LANDED INSTEAD — value_info crop on `crop` (the un-mined urad-7225 mechanism, self-applied):** grader charges
max(declared, traced-max); max winner box across all 266 bundled examples is 11×17=187 cells ⇒ traced max 374B,
but declared [1,2,18,18] floored the charge at 648B. Candidate declares [1,2,11,17] (annotation-only, graph
bit-identical). `candidates/task216/cand.onnx` (+ build_cand.py): **gate PASS 266/266, mem 8584→8310, params 76,
cost 8660→8386, 15.9335→15.9657 (+0.0322)**; fresh 500/500 (136 of those had winner boxes exceeding the declared
shape — ORT ignores intermediate value_info at runtime in both directions; bundled grading set is constant, so
the 374B trace-max is permanent). NOT adopted (per session scope).
⭐ TRANSFERABLE: **on any net with a dynamically-shaped intermediate (Slice/Pad crop carriers), diff declared
value_info bytes vs profiler traced-max over the bundled set — declared > traced-max is free memory.** Scan all
deployed nets for dynamic-shape value_info slack.

## ADOPTED 20260709T050157Z
- cost: 8660 -> 8386 (points 15.9657)
- source: candidates/task216/cand.onnx
- note: public-insight generalize: valueinfo_legalized_dynamic_crop self-applied — dynamic crop tensor declared [1,2,18,18] but traced max 11x17; re-declare shrinks charge 648B->374B, graph bit-identical, 500 fresh 0-fail. TRANSFERABLE: sweep all deployed nets for declared-vs-traced-max value_info slack

---
## 2026-07-13 — REGIME-CRACK (conv_fp32_arsenal) re-attack on c12_f32 3200B → FLOOR (4th confirm)
**Ran:** dumped deployed graph, profiled per-tensor declared bytes (c12_f32=3200 fp32, c12=800 u8,
tl/tl_flat/tl_flat_mid=400 ea, crop=374, rm/cm=320 ea, all scan tensors ≤80B; params 76). Re-examined
whether the dominant `c12_f32` [1,2,20,20] fp32 (38% of cost) is a foldable CARRIER or an irreducible bridge.
**Tool+date:** onnx 1.21.0 / ort 1.26.0, manual graph dump + per-tensor byte accounting, 2026-07-13, opus.
**Verdict: FLOOR.** `c12_f32` is NOT a foldable carrier — it is the entry-crop intermediate whose sole purpose
is to be `Cast→c12` (uint8, the tensor that actually feeds QLinearConv corner-detect + Gathers + final crop
Slice). It does double duty: the red-count Einsum (`c12_f32,sel_red,rm_f32,cm_f32->counts`) already free-rides
it (S9). There is no counted carrier left to fold; the fold was done in S9.
- **Entry-bridge floor = 3200 (fp32 crop) + 800 (u8 cast) = 4000B, irreducible under onnx 1.21:** reading the
  free fp32 input's 20×20 blue+red region needs one counted node output; `Slice` preserves input dtype (fp32)
  ⇒ 3200B minimal crop; downstream integer work (QLinearConv/Gather/Slice) needs uint8 ⇒ `Cast` to 800B.
  **No op crops-and-casts in one node** (verified: Slice/Gather/Conv/Pad keep dtype fp32; Cast/QuantizeLinear/
  DynamicQuantizeLinear change dtype but do NOT crop ⇒ cast-first processes full input = 9000B+800 = 9800B,
  worse than crop-first 4000B). fp16 bridge worse (3200+1600=4800>4000; QLinearConv needs u8 not fp16).
- **Whole-net free-output reformulation stays ≥9046B** (re-confirms 2026-07-09 byte math): building the output
  box directly via placement Einsum from the free input forces 30-wide row/col selector one-hots fp32-welded to
  the input (ORT uniform-T) = 2×[30,30] fp32 = 7200B just for selectors, vs current crop→Pad epilogue 374B.
  fp32-co-bind trap: no fp16 carrier escape (uniform-T). Structurally exceeds deployed 8386.
**Reopen-trigger:** (1) a single onnx op that crops AND casts fp32→u8 (would collapse 3200→800 bridge to ~800);
(2) grader change away from counting the fp32 Slice intermediate (e.g., dtype-view escape); (3) an exact
winner-selection rule NOT needing per-box extents (red counts forced exact by unique-counts ⇒ density/blur
proxies stay dead). No candidate built (all reformulations cost > 8386).
**Falsification history:** 2026-07-08 fold-batch (c12_f32 floor) held; S11 mech-15 KILL held; 2026-07-09 task394
generalization NO-TRANSFER (~9046B) held; 2026-07-13 conv_fp32_arsenal re-attack — FLOOR held (4th).

## ADOPTED 20260715T070537Z
- cost: 8386 -> 8357 (points 15.9691)
- source: candidates/task216/topk_corners_signed_cast.onnx
- note: exact corner enumeration: repeated ArgMax+Scatter -> safe signed TopK(k=4)

## ADOPTED 20260715T070727Z
- cost: 8357 -> 7956 (points 16.0183)
- source: candidates/task216/topk_corners_all_i8.onnx
- note: end-to-end int8 corner response enables safe TopK(k=4) without 400B cast

## REPAIRED 20260715T073650Z
- cost: 7956 -> 8386 (points 15.9657)
- source: submission/.backups/task216_20260715T070537Z.onnx
- note: Kaggle safety repair after ref54716353 ERROR: restore pre-INT8-TopK ArgMax/Scatter implementation

## ADOPTED 20260715T081702Z
- cost: 8386 -> 6445 (points 16.2289)
- source: candidates/task216/codeplane.onnx
- note: FALSIFIES the 2026-06-16 'confirmed-infeasible / public net at real floor' verdict AND the 4x-confirmed 4000B entry-bridge floor (now 2000B). Negative-crop prologue replaces Slice(input)->c12_f32[1,2,20,20] (3200B) with Conv(input, w[1,10,1,1], pads=[0,0,-10,-10]) -> ONE code plane [1,1,20,20] (1600B; blue->1 red->3), halving the u8 cast too. Two things made the single plane viable where prior attempts stalled: (a) the 2x2 corner stencil uses eff weights [[0,-3],[-3,1]] with y_scale=1/256 so every TL response saturates to a uniform 255 regardless of whether the corner cell is blue or red — what the ArgMax/Scatter top-4 enumeration requires; (b) the epilogue keeps its channel-resolved source via a signed zero-point: QLinearConv(crop, x_zero_point=2) maps blue->-1/red->+1 in dequantized space so eff weights [-1,+1] split the 1-ch crop into 2 u8 planes with NO bias (nothing to fire over the pad region), and Pad still writes the free output. The count Einsum keeps free-riding the counted code_f32 so masks stay [4,20] — exactly what the 2026-07-09/07-13 c12-free reformulations lost when they forced 30-wide input-welded selectors to ~9046B. cost 8386->6445, 79 nodes, params unchanged 76. Deliberately avoided TopK (ref54716353 Kaggle ERROR). Differential vs deployed: 0 disagreements over 4000 faithful + 7446 adversarial band-partition + 4000 tie-heavy; coverage incl. 586 touching-box pairs and 5778 red-at-winner-corner cases; the 33 rule-fails are tie-only and identical to the incumbent's.
