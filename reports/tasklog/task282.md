# task282 — b60334d2

**Rule:** Fixed 9x9 grid, background 0. Input places 3-4 gray(5) pixels at (r,c), r,c in [1,7],
pairwise non-overlapping (their 3x3 stamps never collide). For EACH gray input pixel the OUTPUT paints
blue(1) at the 4 orthogonal neighbours, gray(5) at the 4 diagonal neighbours, and the centre becomes
background (the input gray is NOT copied). Pure local stamp; no detection, no argmax. Stamps never
overlap so every output cell receives at most one contribution.
**Current:** ~18.2 stored but FRESH-RATE 0.00 (non-generalizing base net — scores ~0 on real LB).
**Target tier:** A — local stamp = ONE 3x3 Conv on a single cropped colour channel routed into the
free BOOL output. (Not S only because output colours [1,5] are constants painted by a fixed kernel, but
the 10-ch expansion never materializes — it lands in the FREE Equal output.)

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | slice ch5 (9x9) + ONE banded 3x3 Conv (ortho=1,diag=2) + 2 Greater + 2 Where + Pad + Equal | A | 1872 | 42 | 17.44 | 200/200 | ADOPTED candidate |

## Best achieved
17.44 @ mem 1872 params 42 — adopted? candidate only (leave to src.adopt). Beats real-LB 0.00 by ~+17.4. YES. fresh 200/200.

## Irreducible-floor analysis
Dominant intermediate is the single fp32 9x9 channel slice / Conv response (~324B each) plus the small
uint8 label planes and the bool output (free). Already tiny. A single banded Conv (blue=weight1,
gray=weight2) collapses what would otherwise be two convs into one plane — stamps never overlap so the
response is exactly 0/1/2 per cell, making the magnitude-band readout exact. Hard to push much lower
without removing the Conv entirely; the 9x9 active canvas keeps every plane sub-1KB.

## OPEN ANGLES (re-attack backlog)
- Could fold the two Greater+Where into a single arithmetic label `L = is_any + 4*is_gray` (blue=1,
  gray=5) via one Greater pair + Add, shaving a Where plane — marginal (~tens of bytes), not worth risk.
- Tier-S spatial-copy is impossible: output paints NEW colours at NEW positions (the input pixel itself
  is erased), so there is no identity copy of input cells.

## INSIGHT (transferable)
⭐ "Stamp a fixed local pattern (ring/cross/X) around each marked pixel" with NON-overlapping stamps is a
pure Tier-A single-Conv transform: pack DISTINCT output colours into ONE banded kernel (orthogonal=1,
diagonal=2) so the conv response is a disjoint magnitude code (0/1/2), recovered by stacked Greater+Where
into the free Equal output. Non-overlap (generator's overlaps-guard) is what makes the band code exact —
no double-counting. Crop to the generator's fixed small canvas (9x9) first to keep every plane sub-1KB.
