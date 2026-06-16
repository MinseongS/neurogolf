# task134 — 5ad4f10b

**Rule:** Input is a 20–30 grid with two non-bg colours: scattered single-pixel NOISE
of `color` (density 0.05) and a 3×3 conway sprite (every row/col has ≥1 on-cell;
4–9 on-cells) drawn in `megacolor`, magnified by `magnifier`∈{2..6} (each on sprite-cell
= a solid mag×mag block), top-left at (rowoffset,coloffset). Output is the 3×3 sprite
**downscaled**, recoloured to the NOISE colour: out[R][C] = `color` iff sprite cell (R,C)
is on, else 0.

**Current:** 15.576 pts, method `gen:thbdh6332`, mem 12300, params 80
**Target tier:** B-ish (label/MatMul-pool with one full-grid mask) — the sprite is a
GENERAL 3×3 pattern (not row⊗col separable), so it needs true 2-D pooling of one
full-grid megacolor mask; that [1,1,30,30] plane is the floor.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | colf-Conv + 2×2-solid-block detect (dilated) + bbox-pool MatMul | B | 45715 | 218 | 14.27 | (stored ok) | too many full planes; fp16 doesn't help (ORT upcasts trace to fp32) |
| 2 | density-by-bbox-area (per-channel ramp planes) + Equal mask + label-Pad-Equal | B | 24505 | 180 | 14.89 | — | per-channel ramp planes (8×1200) + fp32 colf + L all expensive |
| 3 | **presence-density (cnt/(nrows·ncols)) + Gather mm + Floor-Equal selectors + 3×3 one-hot Pad** | **B** | **9459** | **110** | **15.83** | **400/400** | exact, generalizes; final |

## Best achieved
15.834 @ mem 9459 params 110 — adopted? N (orchestrator gates). Beats prior 15.576 by
**+0.26** → MARGINAL (below the +0.3 adopt bar) but a clean generalizing custom net
replacing a non-custom `gen:` method. 266/266 stored, 400/400 fresh isolated (2 trials).

Key exact pieces (all verified independently to 40k fresh instances):
- **megacolor = argmax_k presence-density** `cnt[k] / (n_present_rows[k]·n_present_cols[k])`,
  channel 0 zeroed. Megacolor fills its bbox densely (density ≥0.13 above noise's ~0.05,
  min gap 0.137 over 40000); noise is sparse+spread. Uses only `cnt`[1,10,1,1] +
  `rowpres`/`colpres` reductions — NO ramp-weighted full-channel planes.
- **mm = Gather(input, megacolor, axis=1)** — the megacolor mask is literally one input
  channel; no `colf` Conv / Equal plane needed to build it.
- **mm bbox → magnifier = bbox_height // 3**; selectors `Srow[R,r]=Equal(floor((r−rmin)/mag),R)`,
  `Scol` likewise. Out-of-bbox rows get a wrong sprite-index but mm=0 there, so the
  MatMul ignores them → a single Floor+Equal builds each selector (no lo/hi pair).
- `P = Srow @ mm @ Scol` → [1,1,3,3]; `noisecolor = chi+clo−megacolor` (the two present
  non-bg channels); one-hot the 3×3 label then Pad to 30×30 with 0.

## Irreducible-floor analysis
Memory ≈ mm(3600, the single [1,1,30,30] megacolor mask) + rowpres/colpres(2×1200, needed
for the exact 2-D presence-density discriminator) + ~3.4k selector/bbox/one-hot overhead.
The sprite is NOT row⊗col separable (general 3×3 conway pattern), so the pooling genuinely
needs a full-grid 2-D mask → the [1,1,30,30] mm plane is unavoidable. Even with zero other
overhead, mm alone caps this architecture near pts≈16.8 (25−ln(3600+params)). The presence
vectors (2400) are the cost of the only exact 2-color discriminator found; all 1-D variants
(cnt/nrows, cnt/nrows², bbox-span, bbox-area, 2×2-block-count by value) fail on rare
clustered-noise or noise>mega instances.

## OPEN ANGLES (re-attack backlog)
- **Crop mm to the ≤18×18 active bbox** to break the 3600 floor → ~16.8+. Blocked by the
  data-dependent-Slice symbolic-dim trap; a Gather-based fixed-18×18 window needs the
  channel-gathered [1,1,30,30] to exist FIRST (cropping 10-channel input first is worse),
  so no win was found. A clever index-Gather that fuses channel-select + crop could help.
- Eliminate one presence vector: an exact megacolor discriminator using only `rowpres`
  (or a single combined plane) would save 1200. None found that survives 30k+ fresh
  (cnt/nrows² fails 2/30000).
- Shave the ~720B one-hot tail / selector bool+float pairs (ORT upcasts bool→fp32 in the
  trace, so these don't shrink with dtype).

## INSIGHT (transferable)
- ⭐ **fp16/bool/uint8 do NOT shrink full-grid working planes in the ORT profiling trace** —
  ORT inserts PrecisionFreeCast and records fp32 shapes, so a [1,1,30,30] of ANY dtype
  costs ~3600 in the score. The real lever is FEWER full-grid intermediates, not narrower
  ones. (This reverses the usual "fp16 the small planes" advice for the *measured* memory.)
- ⭐ **"which scattered colour is the sprite" = PRESENCE-DENSITY argmax** `cnt/(nrows·ncols)`:
  a magnified/solid shape fills its presence-bbox densely while noise is spread thin. Exact
  and cheap (two ReduceMax presence vectors + reductions), beats bbox-area/span and 2×2-block
  heuristics that fail on rare clustered noise.
- **mask = Gather(input, color_scalar, axis=1)** gives a single-colour full-grid mask for
  free (it IS an input channel) — no colf-Conv + Equal needed when you already know the
  colour index.
- **Selector matrices for a regular block tiling can be a single Floor+Equal** when the
  underlying mask is zero outside the bbox: spurious out-of-range selector entries are
  annihilated by the zero mask in the MatMul, so no in-bbox gating is required.
