# task034 — 1f0c79e5

**Rule:** A 2×2 colored seed square at top-left corner (R,C); its 4 corners map 1:1 to the 4 diagonal
directions. A corner painted RED(2) in the input sprouts an outward width-3 diagonal staircase to the
edge in the output. Closed form rel. to chosen corner: painted iff a>=0 & b>=0 & |a-b|<=1, where
a=(r-r0)·dr, b=(c-c0)·dc. size=9 grid.
**Current:** 15.16 pts (custom:task034, adopted from gen:thbdh6332 13.99), mem 18666, params 70
**Target tier:** B (label-map) achievable; rule is per-cell closed-form so B floor ~3600 is the real
target, NOT 18666. Possibly A (separable per-direction) with more work.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | closed-form predicate per 4 dirs on 9×9 fp32 planes → uint8 L9 → Pad → Equal | B-ish | 18666 | 70 | 15.16 | 200/200 | ADOPTED (+1.17) |

## Best achieved
15.16 @ mem 18666 params 70 — adopted? **Y**. Beats prior 13.99? Y (+1.17).

## Irreducible-floor analysis
NOT at floor. mem_profile: 3240B one fp32 [1,10,9,9] Pad + **~50 × 324B fp32 [1,1,9,9] planes**
(the per-direction staircase predicate arithmetic: Sub/Mul/Abs/Cast/Add chains, ×4 directions).
The 324B planes are fp32 but hold booleans/small ints → 4× waste. The label map L itself is only
81→900 uint8 (fine). The bloat is entirely the un-downcast predicate chain.

## OPEN ANGLES (re-attack — high yield, well-specified)
- **Downcast every predicate plane fp32→uint8/bool** (324→81B). ~50 planes ⇒ ~12k saved ⇒ mem ~5–6k ⇒ ~16.8 pts. (floor-break guide §uint8.)
- **Share the 4-direction arithmetic:** the predicate is the same function of (a,b); compute one set of
  |a-b|<=1 / a>=0 / b>=0 planes on signed coords, fold the 4 corners via a single transform instead of
  4 separate chains ⇒ ~halve the plane count.
- **Build L directly on a small canvas** (already 9×9) but in uint8 throughout; avoid the fp32 [1,10,9,9]
  Pad (3240) — Pad the uint8 L only.
- Stretch (Tier A): per-direction staircase = separable in rotated coords? a>=0&b>=0&|a-b|<=1 is a
  diagonal band — may express as row/col-shifted outer structure. Lower priority.

## INSIGHT (transferable)
⭐ A "working adopt" at 15 with high memory is a HALF-win: agents tend to leave the predicate scaffold in
fp32. ALWAYS mem_profile after adopt; if a long tail of equal-size fp32 planes appears, they are
boolean/small-int and downcasting to uint8 is a free 4× cut. Bake "downcast all mask/predicate planes to
uint8, fp32 only where Conv/MatMul truly needs it" into every build prompt.
