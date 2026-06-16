# task222 — 91714a58

**Rule:** A 16×16 grid holds ONE solid axis-aligned rectangle of a single `color`
(width,height ∈ [2,8], 9 ≤ area ≤ 16, placed strictly interior) on top of a field of
random single-pixel noise of arbitrary colours. The generator guarantees the box
`color` has NO same-colour 4-neighbour anywhere outside the box (noise of the box
colour is always isolated). OUTPUT = INPUT with everything but the box zeroed (keep box
cells at their colour, blank the rest to background).
Exact local rule (brute-verified 0 fails / 50000 fresh): keep(r,c) iff cell (r,c)
belongs to a fully-filled single-non-zero-colour block of shape **3×3 OR 2×4 OR 4×2**.
Every valid box contains such a sub-block (a 2-wide box must be ≥2×5 ⇒ has a 2×4; else
both dims ≥3 ⇒ has a 3×3) and these windows tile a solid rect fully; noise needs ≥8
same-colour pixels in that exact shape (~0 probability) to false-fire.

**Current (prior):** 14.68 pts, separable-bbox via "box colour = unique colour forming
≥2 solid 2×2 blocks" (fragile colour-ID), mem ~? params ~?
**Target tier:** B+ (single-cell-local but non-separable: 2×2/3×3/2×4/4×2 membership
couples a cell with its orthogonal+diagonal neighbours → not row⊗col separable A, not a
pure per-cell-of-input S). Achievable above the B floor via cheap MaxPool detectors.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | brute: part-of-filled-2×2 | — | — | — | — | 51/3000 fail | spurious noise 2×2 (always exactly 4 cells) |
| 2 | brute: filled 2×3 OR 3×2 | — | — | — | — | 1/20000 fail | rare spurious noise 2×3 |
| 3 | brute: filled 3×3 OR 2×4 OR 4×2 | — | — | — | — | 0/50000 | exact + robust rule found |
| 4 | MaxPool min/max uniform-window detector, 3 shapes, f16 16×16 canvas, single Where free output | B+ | 18502 | 65 | **15.17** | **500/500** | FINAL (off-grid kept via pad-True so input's all-zero off-grid wins, not bg one-hot) |

## Best achieved
15.17 @ mem 18502 params 65 — adopted? N (orchestrator gates). Beats prior 14.68 by
**+0.49**. Generalizes: stored 266/266, isolated fresh 500/500.

## Irreducible-floor analysis
Dominant: `colf30` [1,1,30,30] f32 = 3600 B — the 10→1 colour-index Conv entry plane
(the per-cell value reduction must be f32 per FLOOR_RESEARCH; cropping to 16×16 happens
*after* because ORT Conv inherits the input's 30×30 extent). After that the active
canvas is sliced to 16×16 and everything is f16 (≤512 B planes). The remaining ~14.9 KB
is the SHEER COUNT of small f16 detection planes — 3 block shapes × {mx, nmn(=-min),
rng, seed, seedpad, dil} (~6 planes × ~400-512 B each) plus the two 30×30 final-select
planes (keep30_u8 900 + keep_b 900 = 1800, forced because Pad rejects bool but Where
needs bool). Three shapes are irreducible for coverage: a 3×3 box (area 9) contains
neither a 2×4 nor a 4×2; a 2×5 box contains only a 2×4; a 5×2 box only a 4×2. No two
shapes cover all boxes.

## OPEN ANGLES (re-attack backlog)
- Replace the 3 separate MaxPool min/max detectors with ONE pass: a 3×3 ones-Conv on v
  and on v² then Cauchy-Schwarz `9·S2==S1² ∧ S1>0` finds 3×3 uniform windows (task165
  idiom); still needs the 2×4/4×2 cases, so ~same op count — unlikely to beat 18.5 KB.
- Fuse the three dilation MaxPools' footprints: a single max over the OR of the three
  TL-seed maps then one 4×4 dilation MaxPool (super-set footprint) — would over-dilate
  by 1 cell at edges, breaking exactness near a box edge; untried, risky.
- Drop colf16 f32 (1024 B) by Conv→f32→Cast f16 over 30×30 then Slice: measured WORSE
  (1800 cast plane > 1024 f32 slice). Entry chain is already minimal.

## INSIGHT (transferable)
⭐ "Keep the one big SOLID rectangle, drop random multi-colour noise" generalises the
task193 "part-of-a-filled-2×2" lever to the **multi-colour, area-bounded** case: detect
"fully-filled uniform-COLOUR h×w window" with `MaxPool(v)==−MaxPool(−v) ∧ min>0` (the
range==0 test handles arbitrary colours, vs task193's occupancy-count which only works
single-colour), then dilate the TL-anchored seed back over its footprint with an
opposite-anchored MaxPool. Pick the block-shape SET from the generator's area/dimension
bounds so noise can never reach the same fill size: area≥9 with both dims≥2 ⇒ the union
{3×3, 2×4, 4×2} covers every box yet demands ≥8 same-colour pixels (noise FP ~0), where
a plain 2×2 (FP ~1/60) or 2×3 (FP ~1/20000) leaks. The off-grid handling is the task193
`selcond = keep OR off-grid` idiom: pad the keep mask with **True** outside the active
grid so the final Where selects the input's all-zero off-grid cells (NOT the bg one-hot,
which would wrongly set channel-0 high → silent fail on every fresh instance).
