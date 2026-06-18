# task096 — 4290ef0e

**Rule:** Input H×W grid (13..19), bg = most-frequent colour. K (=4..6) non-bg colours each
own a UNIQUE ring index idx∈0..K-1 (a permutation). Ring idx draws its colour at 4 corners
(±idx,±idx) about a per-shape RANDOM centre, each corner with two inward arms of length L_idx;
shapes are scattered and CLIPPED at edges (generator guarantees ≥2 quadrants drawn). Output is a
(2K-1)² concentric reassembly: offset (a,b), m=max(|a|,|b|), n=min(|a|,|b|) → colours[m] iff
n ≥ m−L_m+1 else bg. Diagonal out[K-1-i][K-1-i]=colours[i].
**Current:** 12.97 pts, weak gen-import (248 nodes), the bar to beat is 13.27 (+0.3).
**Target tier:** detection/B — synthesis is closed-form; the WALL is recovering (idx,L) per colour.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | matched-filter conv NT=12, 2 convs + bool chain | — | 633k | 3409 | 11.64 | 200/200 | exact but heavy |
| 2 | + fp16 ReduceMax of matchpos | — | 547k | 3410 | 11.78 | — | minor |
| 3 | + drop cm>0 (present-gate tot>0) | — | 460k | 3412 | 11.95 | — | minor |
| 4 | single conv cdo=Conv(ongrid−2·oh,K), dist=tot+min(cdo) | — | 256k | 1957 | 12.54 | 200/200 | best exact; still < 12.97 |

## Best achieved
12.54 @ mem 255690 params 1957 — adopted? **N** (does NOT beat the existing 12.97; it is *below* it).
The recovery is EXACT (fresh 200/200) — the blocker is purely memory.

## Irreducible-floor analysis
The ONLY recovery verified exact (0/400 op-level, fresh 200/200) is a **generative matched filter**:
for each colour-channel k and each of the **12 distinct (idx,L) shape types**, test whether some
translation makes the clipped stamp cover colour-k exactly — `cm==og AND cm==tot` (cm=Conv(mask,K_t),
og=Conv(ingrid,K_t), tot=Σmask). Folded to ONE conv via `cdo=Conv(ongrid−2·oh,K)=og−2cm` and
`min_space(dist)=tot+ReduceMin(cdo)`. The dominant intermediate is that single conv plane
**`cdo` [10,12,19,19] fp32 = 173 280 B**, which is IRREDUCIBLE:
- **channels 10**: per-colour masks are required; bg/absent colours can't be dropped statically.
- **types 12**: every (idx,L) needs its own kernel — corner-only / min-L / max-L kernels are all
  too permissive or too strict (tested 35–73 % idx error); only the full per-(idx,L) shape match is exact.
- **W=19**: grid side is randint(13,19); the match centre can sit anywhere in-grid → SAME-pad 19×19.
- **fp32**: ORT upcasts an fp16 Conv output back to fp32 in the trace (PrecisionFreeCast), so fp16
  neither shrinks the plane nor stays correct (verified: fp16 conv → 0 pass, mem unchanged).
173 KB alone ⇒ score ≤ ~12.8; with unavoidable overhead the exact net lands at 12.54.
**The bar 13.27 needs mem+params ≤ ~124 KB, below the single conv plane.** ⇒ INFEASIBLE to beat +0.3.

## OPEN ANGLES (re-attack backlog)
- Profile-symmetry centre (palindromic 1-D row/col profile, integer centre, min-idx) recovers idx
  on TINY [10,19] tensors and is exact ONLY when each candidate centre is VERIFIED by a full
  shape-match; without verification it is 16–26 % wrong (spurious palindromic centres win min-idx).
  If the per-candidate verification could be done on a [10,19,19] (≈14 KB) tensor instead of the
  [10,12,19,19] conv, the net would drop to score ~15. The verification currently needs the 2-D stamp
  ⇒ unresolved. This is the one angle that could break the floor.
- Reduce types: if idx were recoverable from a cheap signal and only L needed the conv (or vice
  versa), NT could drop. No cheap exact idx signal found.

## INSIGHT (transferable)
⭐ "Reassemble scattered CLIPPED symmetric sprites into a canonical figure" is a recovery WALL when
the per-sprite (size,shape) parameter is only pinned by a full 2-D generative shape-match: the
matched-filter conv plane `[colours × shape-types × W × W]` fp32 is irreducible (ORT upcasts fp16
conv; channels/types/W all load-bearing) and floors the score *below* a generic gen-import. Cheap
1-D symmetry/profile recoveries are NON-exact (clipping breaks symmetry; spurious palindromic
centres), so they fail fresh-200 even though they pass ~99 %. ⭐ Useful op trick discovered:
`og − 2·cm = Conv(ingrid − 2·oh, K)` collapses the two matched-filter convs + the (cm==og ∧ cm==tot)
bool chain into ONE conv plane plus `tot + ReduceMin` (dist≥0, ==0 ⇔ exact placement) — halves mem
for any "exact-stamp-placement" detector.
