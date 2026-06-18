# task096 — 4290ef0e

**Rule:** Input H×W grid (13..19), bg = most-frequent colour. K (=4..6) non-bg colours each
own a UNIQUE ring index idx∈0..K-1 (a permutation). Ring idx draws its colour at 4 corners
(±idx,±idx) about a per-shape RANDOM centre, each corner with two inward arms of length L_idx;
shapes are scattered and CLIPPED at edges (gen guarantees ≥2 quadrants drawn). Output is a
(2K-1)² concentric reassembly: offset (a,b) from centre (K-1,K-1), m=max(|a|,|b|), n=min(|a|,|b|)
→ colours[m] iff (m<K and n ≥ m−L_m+1) else bg.
**Prior bar:** 12.97 (weak gen-import, 248 nodes). +0.3 bar = 13.27.

## Result of THIS session
**pts 13.196, mem 132013, params 1825, fresh 200/200, stored 4/4 → BEATS 12.97 by +0.226 (MARGINAL, <+0.3).**
The prior agent's "INFEASIBLE — exact net floors at 12.54" verdict was based on a STALE fp32-conv
claim. Re-tested: **fp16 Conv keeps fp16 output under ORT_DISABLE_ALL on the current ORT** (the
"ORT upcasts fp16 conv → 0 pass" claim is false now). That + crop-to-WORK19 + sig-as-1×1-conv +
type-0 drop + uint8 output dropped the exact net from 12.54 → 13.196.

## The net (exact, 0-err over 2100+ fresh + 4 stored)
1. Crop input to [0:19,0:19], cast fp16. `sig_k = ingrid − 2·mask_k` for all k in ONE 1×1 conv
   (W = ones(10,10) − 2·I). Reshape channels→batch [10,1,19,19].
2. Matched filter: `cdo = Conv(sig, K_t)` = og − 2cm, 11 type kernels (idx≥1), fp16, pads
   [5,6,6,5] → [10,11,20,20] so centres reach 1 cell off the left/bottom edge (hand-authored ARC
   examples place a centre at col −1 / row 19). `mind = ReduceMin(cdo)`; `dist = tot + mind`;
   `match = (dist==0)`. **min-idx tiebreak**: ArgMax over the idx-ordered type axis returns the
   first True (a clipped large stamp also fits smaller types; the true type is the MIN-idx match).
3. idx0 = single-pixel colour (tot==1; the ≥2-quadrant rule gives idx≥1 ⇒ tot≥2), handled by a
   Where override (its kernel dropped from the conv to save one type).
4. Scatter (k, L_k) by recovered idx into length-6 ring vectors; K = max(visible idx)+1 (robust to
   bg-coloured invisible inner rings in the hand-authored ARC examples — bg excluded by gen for
   fresh). Invisible (bg-coloured) rings default to bg colour & L=6 so their cells render bg.
5. Closed-form synthesis on an 11×11 centred canvas (gather ring colour/L by m=max(|a|,|b|), gate
   by n>m−L), Where bg fill, crop/shift to (2K-1)² at top-left (Gather by srci = i+cen+1−K, clamp),
   valid-mask off-grid → sentinel 99, uint8 Pad → Equal → BOOL one-hot.

## Memory floor analysis (irreducible for the EXACT net)
| plane | bytes | why irreducible |
|---|---|---|
| matched-filter conv [10,11,20,20] fp16 | **88000** | 10 ch (per-colour masks needed, bg/absent can't be dropped statically), 11 types (every (idx,L) needs its own kernel — corner-only=75% idx err, all 11 occur in fresh), 20×20 = centre range rows[0..19]×cols[−1..18] forced by the off-grid hand-authored centres; uint8/int8 Conv unsupported by ORT ⇒ fp16 is the dtype floor |
| fp32 input Slice [1,10,19,19] | 14440 | the one fp32 entry plane (Slice preserves fp32; must feed the fp16 cast) |
| sig chain (inwf cast + sig 1×1 conv + batch reshape), each [.,.,19,19] fp16 | 3×7220 | front-end; the reshape is layout-only but counts; a grouped conv would drop it but costs +11.9k params (net worse) |
Total ≈ 132k. **+0.3 needs mem+params ≤ 124244** — the gap (~10.8k) is below the conv plane; no
dtype/crop/type-drop trick closes it without breaking exactness.

## OPEN ANGLES (could break the floor, all currently blocked)
- Cheap exact idx recovery (to cut the 11-type conv): bbox/2 = 10% err (clip underestimate); rank
  by extent = 18% instance err; corner-only matched filter = 75% err. The full per-(idx,L) 2-D
  match is the only verified-exact recovery → conv channels×types is load-bearing.
- Off-grid stored centres force conv 20×20 (vs 19×19 fresh-only). Special-casing the 3 boundary
  centres could shrink to 19×19 (~−8.8k) but is fragile / not generalizing-clean.
- Reducing 10 channels (only ≤7 present) needs a data-dependent gather-compact → symbolic-dim trap.

## INSIGHT (transferable)
⭐ **fp16 Conv now keeps fp16 output under ORT_DISABLE_ALL** — re-test before trusting any
"fp32-conv-at-floor" verdict; halved the matched-filter plane (173k→88k). ⭐ **sig = ingrid−2·mask
for all channels = ONE 1×1 conv** (W = ones − 2·I) — folds the cross-channel ingrid sum + the −2·mask
into a single op, no per-channel Mul/Sub. ⭐ **min-idx tiebreak via ArgMax over an idx-ordered type
axis** cleanly resolves the nested-stamp ambiguity (a clipped large stamp matches smaller types; the
true one is the smallest-idx match). ⭐ For a permutation-indexed concentric figure, **K = max(visible
idx)+1**, robust to an invisible (bg-coloured) inner ring — count-of-present is WRONG when bg ∈ colours
(the hand-authored ARC examples) even though the random generator excludes bg.
