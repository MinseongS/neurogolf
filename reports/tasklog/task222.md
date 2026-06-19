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

## RE-VISIT 2026-06-19 (baseline now ext:kojimar7113 = 15.616, mem 11451 par 444)
The prior 15.17 custom (re-measured 15.50/13326) is BELOW the adopted kojimar crowd net.
Target to beat = 15.616 + 0.3 = 15.92  =>  mem+par <= exp(25-15.92) ~= 8806.

kojimar net = colour-channel route: Slice(input, ch1..9, r/c 1..14) -> **f32 [1,9,14,14]
= 7056B** (its dominant plane) + uint8 cast 1764 + a 6x7 patch QLinearConv (colour argmax)
+ a 3x3 cross degree QLinearConv (deg>4 == cell with >=2 same-colour neighbours) + Pad.
The 7056 f32 Slice + its 1764 uint8 cast (8820B) ARE the floor of the one-hot route (Slice
inherits f32; casting the input first is 9000B, worse).

OUR colour-INDEX route avoids that 8820: colf30 1x1-Conv = **3600B f32** entry instead.
Cuts applied this pass (sum-conv 3-shape detector): dropped the per-shape non-bg gate
(mx>0) entirely -- an all-bg uniform window dilates only over bg cells and a "kept" bg
cell routes Where->input==bg == the false branch, so it is HARMLESS (re-verified 0 fails/
12000). That removed 6 bool planes. Result: **15.563 @ mem 12466 par 72, 266/266 stored,
200/200 isolated fresh.** Still 0.05 BELOW kojimar and 0.36 below the +0.3 bar.

## Verdict: MARGINAL / effectively at floor for this rule
Irreducible budget: colf30 (3600 f32, the 10->1 colour-index reduction MUST be f32 -- fp16
needs an 18000B input cast) + colf slice (784) + the two final 30x30 carriers (keep30_u8
900 + keep_b 900: Where needs bool, Pad rejects bool, off-grid must be kept-True to pick the
all-zero input while in-grid border must be kept-False -> two pad values -> two planes) =
**6184B fixed**, leaving only ~2550B for ALL detection to reach 8806. The 3-shape solid-block
detector needs ~6 small fp16/bool 14x14 planes (~6000B); the alternative degree+colour-argmax
route (Rule C, proven 0/8000: keep = (colf==bc) AND deg_bc>=2, bc = argmax_k count of colour-k
cells with >=2 same-colour nbrs) needs a per-colour 9-channel reduction for the argmax that is
>=3528B fp16 (ReduceSum/Max reject bool+uint8, so the histogram cannot stay narrow) -- both
blow the ~2550B budget. Separable-bbox output does NOT help: the Where false-branch must be
"ch0 in-grid / 0 off-grid", which is not a constant, so the input-based Where + full keep mask
is already optimal. ⇒ no path to mem+par <= 8806; kojimar's 11451 is essentially the structural
floor and our 12466 ties it. BEATS-BY-0.3 = INFEASIBLE.

## NEW transferable insights
- ⭐ HARMLESS-FALSE-POSITIVE GATE-DROP: when the final op is `Where(keep, input, bg_onehot)`,
  any keep=True on a BACKGROUND cell is free (Where picks input == bg == the false branch).
  So a detector's "non-background" gate can be DELETED whenever its only spurious hits are
  bg cells -- removed 6 planes here (15.50->15.56). Generalises to any keep/input/bg Where.
- (colf==boxcolour) AND degree>=2 == the box EXACTLY (deg>=1 fails ~50%: a single box-colour
  noise pixel may abut the box edge -> deg 1; every true box cell incl corners has deg>=2).
  boxcolour = argmax_k count(colour-k cells with deg>=2) is robust 0/8000 (argmax-by-max-degree
  is NOT: noise can hit deg 4, 73/10000 fail).
