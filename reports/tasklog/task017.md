# task017 — 0dfd9992

**Rule:** A 21×21 grid is filled with a doubly-periodic pattern of period `length`
(4..9, same offset/length on both axes): `v(r,c)=((rr²+cc²)%mod)+1`,
`rr=(offset+r)%length-length//2`, `cc` likewise. mod∈4..9, length∈4..mod,
offset∈1..length. The input has 5 black (colour 0) rectangle cutouts stamped over
the pattern; the output is the SAME pattern with the cutouts removed. The pattern
is fully determined by the 3 scalars (mod,length,offset) — only 106 valid tuples.
**Current:** 15.30 pts, ext:kojimar7113 (crowd net), mem 13500, params 2827
**Target tier:** B (closed-form formula rebuild after scalar-parameter recovery)

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | period-detect + max-fold + class-gather (prior custom file) | B | 63256 | 469 | 13.94 | 200/200 | worse than P; detection planes blow up; also leaks when a small class is fully cut |
| 2 | template-match params (kojimar idiom) + free-output routing | B | 10890 | 2826 | 15.47 | 199/200 | +0.17, below thresh |
| 3 | + fold +1 into channel_values (drop labels21 plane), uint8 pad sentinel-200, channel0=255 wrap | B | 9549 | 2825 | 15.58 | 200/200 | +0.28 |
| 4 | + drop 1 weak sample (NS=16→15, greedy) | B | 9182 | 2679 | 15.62 | 500/500 | **+0.32 ADOPT-CANDIDATE** |

## Best achieved
15.62 @ mem 9182 params 2679 — beats prior 15.30 by +0.32 (✓ ≥+0.3). Fresh
500/500; 3996/4000 (99.90%) at scale = identical to the kojimar baseline's
99.90% (shared inherent leak).

## Irreducible-floor analysis
Dominant intermediates: `matches_h` fp16 [1,106,15]=3180B and `matches_b` bool
[1,106,15]=1590B (the per-sample×per-candidate comparison + its fp16 cast for the
score ReduceSum). This [106,NS] plane pair (4770B, ~52% of mem) is the floor of
the template-match: ReduceSum rejects bool/uint8 (only int32/fp), so the bool
Equal MUST be cast to fp16 to count agreements; a MatMul-onehot reformulation
moves the cost into a 16960-element param table → strictly worse in log-score.
The formula planes `rrcc`+`pat0` (2×882 fp16) are the minimal 21×21 closed-form
rebuild (Add then Mod, both must be fp16: ORT Mul/Mod reject uint8). GatherND
`sample_planes` fp32 [10,15]=600B inherits the fp32 input dtype (unshrinkable).

## OPEN ANGLES (re-attack backlog)
- Two-stage parameter recovery (recover `length` then `(mod,offset)`) to shrink
  the 106-candidate axis of the match plane — blocked by the parameters being
  coupled in the sample colours; no clean separable signal found.
- Reduce GatherND index param (sample_nd_idx 10×NS×4) via batch_dims to drop the
  10× channel replication — minor (~param only).
- NS=14 greedy = 99.91% fresh (~1.3/200 expected fail) — too risky for strict
  200/200; NS=15 is the safe floor matching baseline robustness.

## INSIGHT (transferable)
⭐ For "fill cutouts in a parametric pattern with only K valid parameter tuples":
TEMPLATE-MATCH the global scalars (precompute each tuple's colour at ~16 fixed
sample cells; read those cells via GatherND+ArgMax; majority-vote via
ReduceSum(Equal)→ArgMax — cutout-robust because cut cells read colour 0 which no
candidate sample contains, so they vote for nobody) THEN rebuild by CLOSED-FORM
formula. This beats per-cell period-detection+max-fold (huge detection planes AND
leaks when a small class is fully cut). Routing the 10-ch one-hot into the FREE
output (pad the 1-ch colour-index plane to 30×30, Equal→output) saves the crowd
net's onehot_raw [1,10,21,21]=4410B. ⭐ uint8 pad-back with channel0 compare-value
= (−1 mod 256)=255 lets the +1 colour offset fold into channel_values for free
(off-grid sentinel 200 matches no channel → background-free off-grid; channel0's
255 never appears in-grid). ⭐ ReduceSum accepts int32 but NOT bool/uint8 (re-
confirmed) → a bool match plane is pinned to an fp16 cast before counting.
