# task202 — 855e0971

**Rule:** The grid is fully painted with stacked horizontal colored "strata"
bands (each band = `height` rows of one DISTINCT color, colors from
`random_colors` so never repeat). Sparse black(0) pixels sit inside the bands.
For every black pixel at column c within a band, the output paints the ENTIRE
vertical extent of that band at column c black. When `xpose=1` the whole grid
(input+output) is transposed, so bands run vertically and the fill is
horizontal. Because band colors are distinct, two rows are in the same band iff
they share a non-black color — no contiguity/flood-fill needed.

**Current:** 14.79 pts, custom:task202 (prior adopted), mem 27063, params 40
**Target tier:** A (separable per-band row/col contraction routed into a bool
mask + a single Where into the free output; no per-cell colour-index plane).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | Slice-ch0 black + [10,30] colblk/rowblk contraction, all fp16 | A | 22863 | 33 | 14.96 | — | marginal (+0.17) |
| 2 | full-4D batched matmuls, drop `black` reshape + obb reshape | A | 20163 | 29 | 15.087 | — | +0.295 (just short) |
| 3 | rank-3 occupancy [1,10,30] + native-4D black, MatMul rank-broadcast (no reshapes) | A | **18963** | **25** | **15.148** | 500/500 | **beats +0.3** |

## Best achieved
15.148 @ mem 18963 params 25 — adopt recommended Y. Beats prior 14.79 by **+0.356** (≥+0.3). Fresh 500/500.

## Irreducible-floor analysis
Memory now dominated by:
- 3600B fp32 `blkslice` — the channel-0 (black) read. This is the unavoidable
  single fp32 "entry plane" for reading a colour spatially (the 3600B rule).
- 4×1800B fp16 full planes: `black` (the fp16 black map), `obR`, `obC`
  (the two orientation candidates), `ob` (the orientation Where-select).
- 2×1200B fp32 `mrf`/`mcf` — the per-colour row/col occupancy entry reductions
  (ReduceMax on the free input outputs fp32; [1,10,30] not [1,10,30,30] so only
  1200B each).
The remaining cost is the TWO orientation candidates obR/obC (3600B together):
the non-xpose op multiplies black on the LEFT (`Rrow^T@Rrow@black`) and the
xpose op on the RIGHT (`black@Rcol^T@Rcol`), so they cannot be unified by one
matmul, and selecting operands earlier would require a full `black^T` plane
(+1800B for the transpose, +1800B for the Where) — strictly worse than just
computing both [30,30] candidates. So we are at the practical floor for this
two-orientation closed form.

## OPEN ANGLES (re-attack backlog)
- Single-orientation: if a future op let you select the orientation BEFORE the
  black plane is read (e.g. an orientation-conditioned Slice axis), you'd drop
  obC + RcolT + rowblk (~3000B). No opset-11 op does a data-dependent axis pick
  without a Where on full planes, so currently blocked.
- The `ob` Where (1800B) + `mask` Greater (900B) could in principle fold into
  per-branch bool masks (maskR/maskC + Where), but that nets to the same 2700B.

## INSIGHT (transferable)
⭐ ONNX MatMul **rank-broadcasting** lets you keep occupancy vectors at their
native ReduceMax shape `[1,10,30]` (rank-3) and the colour slice at its native
`[1,1,30,30]` (rank-4) and contract them directly — the lower-rank operand is
left-padded with 1s, batch dims broadcast, and the matrix dims contract. This
removed FOUR explicit Reshape intermediates (each a full extra plane) vs forcing
both operands to a common rank. General lever: when chaining per-channel matvecs
through the free fp32 input, do NOT reshape operands to a uniform rank — let
MatMul broadcast, and only reshape the final bool mask into [1,1,30,30] for the
output Where (and even that is free if the last matmul already lands 4D).
⭐ "Distinct per-band colour" (from `random_colors`) collapses same-band testing
to same-colour testing → the band-similarity routes through a tiny [10,30]
band×col count, never a [30,30] similarity matrix or any flood-fill.
