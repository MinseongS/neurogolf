# task366 — e6721834

## Current live

`memory=35927`, `params=490`, `points=14.497209022434086`.
The deployed graph is a high-quality heuristic; prior fresh sanity was `39/40`, so any rewrite needs
fresh validation.

## Semantic rule

Input contains two same-sized panels, stacked or side-by-side.
One panel is the template: a background plus 2–3 solid rectangles in a single `forecolor`, each with
1/2/3 same-coloured dots punched into it.
The other panel contains only the dot stencils at new positions.
Output is the non-template panel background with the missing full rectangles reconstructed at those
dot positions.

## Bottlenecks

- `label30` one-hot-to-colour Conv: ~3600B.
- template colour-present machinery, including int32 template label lookup: ~1KB plus several
  255/510B tensors.
- repeated `k0/k1/k2` stencil matching blocks: roughly 7KB.
- repeated placement/stamping blocks: roughly 5.5KB.
- final label path: roughly 5.7KB.

## Re-attack angle

Generator fact: punched dot colours exclude both backgrounds and `forecolor`.

Therefore a cheaper template-dot mask may be:

`T_dot = T_non_background AND T != forecolor`

instead of the current “template cell colour appears in placement dots” machinery. If `forecolor`
can be derived cheaply from the rectangle mask, this may remove 2–4KB. If deriving `forecolor`
requires full per-colour counting, it probably gives the savings back.

Larger idea: use the generator guarantee that rectangle `idx` has `idx+1` dots to map 1/2/3-dot
templates to placement clusters and delete much of the `k0/k1/k2` matching. This is much riskier
because count/color collisions exist; treat as research, not an immediate adoption path.

## 2026-06-29 forecolor/dot-mask probe

Hypothesis: replace the current placement-colour membership path

`pos_color -> T_color_present -> GatherElements(T_idx_for_present) -> T_present`

with `T_dot = T_non_background AND T != forecolor`, using the generator fact
that dot colours exclude both backgrounds and `forecolor`.

Generator probes:

- First non-background template cell is a dot colour in about `30/300` samples,
  so it is unsafe as a cheap `forecolor` proxy.
- Component first-cell/majority proxies are also unsafe (`~8-40%` failures in
  1000-sample probes depending on proxy).
- Template non-background mode is safe in tested samples (`0/300` dot-colour
  collisions), because rectangles dominate the dot pixels.

Conclusion:

The semantic fact is valid, but the safe `forecolor=mode(non-bg)` route likely
needs per-colour counting over the template panel.  That may cost as much as or
more than the current `T_present` path.  Do not patch until a cheap mode extractor
is designed.
