# task002 (ARC 00d62c1b — fill enclosed regions / "honey pots")

## Verdict
INFEASIBLE for the EXACT (3000/3000) bar required to beat the deployed flood net via src.adopt.
Best deterministic input-pure model reaches **97.7%** (2924/3000), which still fails the all-fresh
adopt gate (~70/3000 fail) ⇒ zero real LB gain, same as the current 94% net.

## Exact generator rule (verified)
bg=black(0), green(3)=static noise (5% density) + box outlines, yellow(4)=fill.
Boxes are **cornerless** hollow rectangles: top/bottom rows green at cols 1..w-2, left/right cols
green at rows 1..t-2 — the 4 CORNERS are NOT drawn (gaps are diagonal, so 4-conn enclosure holds).
Output = two steps, in order:
1. **box-fill**: every box's interior (rows 1..t-2 × cols 1..w-2) → yellow, UNCONDITIONALLY (box list).
2. **single row-major surround pass** (in place): for each black cell in raster order, if all 4
   ortho neighbors are >0 in the PARTIALLY-UPDATED grid (off-grid = -1) → yellow. ONE pass only.
   (up/left read updated values, down/right read pre-pass values.) Verified: my `surround_once`
   replication == generator's `is_surrounded` pass 0/3000 mismatch when fed the true post-box-fill grid.

## Why it's a near-wall
- `surround_once` is NOT flood-fill. A single pass cannot fill a thin (1×L, L≥2) black pocket
  (every cell keeps a black neighbor along the strip). So thin enclosed regions only become yellow
  if they are a REAL box (box-filled in step 1). Flood-from-edge (the deployed net) OVER-fills these
  thin pockets → 161/3000 mismatches (~94.6%).
- Output IS a pure deterministic function of the input (0 contradictions / 30000 distinct inputs).
- BUT real vs false thin boxes are LOCALLY IDENTICAL. Noise green pixels frequently complete a
  perfect cornerless box outline around an enclosed 1×L strip; that strip is pixel-for-pixel
  indistinguishable from a real box (confirmed by side-by-side patches). The generator leaves the
  noise-completed one BLACK (not in its box list) and fills the real one.
- Discriminator is GLOBAL, not fixed-radius: 1-ring signature leaves 1 ambiguous pair / 15000;
  only the FULL grid disambiguates. No local conv/window rule resolves it.

## Angles tried (all measured on isolated fresh, generator loaded by file path)
1. Pure flood-from-edge: 94.6% (= deployed net behavior; overfills thin pockets).
2. Solid-rect enclosed-component (scipy.label) box detect + surround_once: 97.2% (83/3000 fail).
3. Cornerless-outline box detect + surround_once: **97.7% (2924/3000)** — best; FN=0, all errors are
   FP from noise-completed thin false boxes (interiors only ever (1,2),(2,1),(1,3),(3,1); never both-dim≥2).
4. + require interior flood-enclosed: WORSE (97.2%) — false boxes are enclosed too.
5. + corners-black / edge-not-extended pruning: huge FN (real thin boxes routinely have green
   corners/extended edges from noise/adjacency).
6. Outer-ring green-count: FP boxes always have ring-green ≥4 (vs TP mean ~3) but TP overlaps up to 14
   → probabilistic only, not separable.
7. Outline-removal-leak (per-box: blank this box's outline, re-flood, real iff interior now leaks to
   border): FP 54 / FN 2 — the most principled GLOBAL discriminator, still NOT exact, AND requires one
   full flood PER candidate box (hundreds of unrolled floods → mem explosion, infeasible to build).
8. Skip thin boxes entirely (thick-only fill + surround): 21.9% (real thin boxes are common).

## Why no buildable exact net
The only exact discriminator found (outline-removal-leak) is (a) not exact (~98.2%) and (b) needs a
separate ~size-cap unrolled flood for every candidate rectangle position — combinatorially many full
floods, blowing the mem budget far past any positive-score tier while STILL not reaching 3000/3000.
The residual ambiguity is the generator's private box list, which is not recoverable from the rendered
image by any feasible local-or-bounded-global computation.

## Lesson (transferable)
"Fill enclosed regions" with a single-pass `is_surrounded` (not flood-to-fixpoint) + an UNCONDITIONAL
box-prefill from a hidden object list is a near-wall: the single pass diverges from flood on thin
pockets, and noise that completes a box outline is pixel-identical to a real box. Output is
input-deterministic but the disambiguator is global and not boundedly computable. A flood net caps
~94.6%; cornerless-detect+single-surround caps ~97.7%; neither clears an all-fresh adopt gate.
