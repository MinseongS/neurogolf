# task270 — ae3edfdc

**Rule:** Fixed 15x15 grid, background 0, two "flowers". Flower 0 = centre colour 2 with
petals colour 3; flower 1 = centre colour 1 with petals colour 7. Each flower's centre is a
single pixel; in each of the 4 orthogonal directions a petal MAY exist, placed somewhere along
that ray at distance >= 2 from the centre (it "flew off"), at most one petal per ray. The OUTPUT
keeps both centres in place and moves every existing petal to the cell immediately ADJACENT to
its centre in that direction. Because flower 0 is the only source of colour-3 pixels and flower 1
the only source of colour-7, "does a petal exist in direction d from this centre" reduces to "is
there a petal-colour pixel anywhere along that ray" — closed-form, no flood-fill, no shape
correspondence.

**Current (prior adopted):** 14.85 pts.
**Target tier:** A — separable directional reconstruction; output colours are a FIXED known set
(1,2,3,7) so slice+place, not a Conv colour-index plane.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | CumSum prefix-OR presence + MatMul shifts, fp32 | A | 52200 | 961 | 14.12 | — | below P |
| 2 | strict-triangular MatMul presence (skip Greater/Cast: <=1 petal/ray so gated exclusive count is {0,1}) + MatMul shifts, fp16, Sum-fold | A | 21600 | 1858 | 14.94 | — | MARGINAL |
| 3 | stacked 2-flower channel batch + grouped Conv shift | A | 32850 | 1109 | 14.57 | — | worse (multi-ch planes bloat mem) |
| 4 | #2 + matrix dedup (all 8 matrices are Aup/Sup or their transpose) | A | 20700 | 958 | 15.02 | — | MARGINAL |
| 5 | #4 + uint8 label entry plane (Cast L->uint8, Pad sentinel 99, uint8 Equal) | A | 20025 | 958 | **15.05** | 500/500 | MARGINAL (+0.20) |

## Best achieved
15.049 @ mem 20025 params 958 — adopted? **N** (below the +0.3 bar). Beats prior 14.85? YES (+0.199),
and generalizes perfectly (fresh 500/500). MARGINAL by the adopt rule.

## Irreducible-floor analysis
~34 fp16 [1,1,15,15] working planes (450B each) dominate. The per-flower pipeline needs 13 planes:
4 directional-presence MatMuls (Aup@P above, AupT@P below, P@AupT left, P@Aup right) — these
contract DIFFERENT axes/sides so they cannot be fused into one op; 4 centre-gated planes; 4
one-step shift MatMuls to the neighbour; 1 Sum. Removing the gate-then-shift pair by gating with a
shifted centre instead (petUp = shiftUp(C) (.) (Aup@P)) is plane-count-neutral (still 4 presence +
4 shifted-centres + 4 products). Stacking the two flowers on the channel axis is mem-neutral
(planes double in size, halve in count) and adds fp32 Concat planes, so it is strictly worse here.
Input prep is 4 fp32 channel Slices (3600B, Slice preserves fp32) + 4 fp16 casts (1800B); a 1x1
colour-index Conv would instead pay a 30x30 fp32 plane (3600B) so it doesn't help. The uint8 label
entry plane (900B) is already minimal. Net: mem floor ~20KB -> ~15.05, ~0.1 short of the +0.3 bar.

## OPEN ANGLES (re-attack backlog)
- A single fused op that yields all four directional presences at once (they need 4 distinct
  triangular contractions) would cut ~4 planes (~1.8KB) and likely clear +0.3 — none found in opset 11/13.
- Encode the 4 direction flags additively at the centre into ONE plane and expand with a per-bit
  Conv: blocked because one scalar value can't be split back into 4 directional taps by a linear conv.
- A bounded directional Conv instead of the full-length triangular MatMul (petal distance is bounded
  by grid size 15, no tighter generator bound) would not shrink params/planes meaningfully.

## INSIGHT (transferable)
- ⭐ When a directional prefix/suffix indicator is read ONLY at a single gated pixel and at most one
  hit exists along the ray, the strictly-triangular MatMul output IS already {0,1} at that pixel —
  drop the Greater+Cast (saves 2 planes/direction).
- ⭐ For axis-aligned row/col shifts, ALL four one-step shift matrices and ALL four strict-triangular
  presence matrices collapse to ONE base matrix + its transpose (Sdn=Sup.T, Sleft=Sup.T, Sright=Sup;
  Adn=Bl=Aup.T, Br=Aup). Store base+transpose as two inits (params, no runtime Transpose planes) —
  halves the matrix-param budget at zero mem cost.
- ⭐ uint8 label entry plane: Cast the fp16 colour-index plane to uint8 (900B vs 1800B fp16 at 30x30),
  Pad with an out-of-range sentinel (e.g. 99), and Equal(uint8, uint8 arange) — ORT supports uint8
  Pad and uint8 Equal under ORT_DISABLE_ALL; off-grid sentinel keeps the harness's all-zero off-grid
  target satisfied.
