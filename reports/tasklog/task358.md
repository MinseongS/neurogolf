# task358 — e21d9049

**Rule:** Grid is H×W (W∈10..20, H=W or W+1). A "cross" is drawn: every cell on
row `row` OR column `col` gets colour `colors[(r+c) % n]`, n=len(colors)∈{3,4}
(diagonal-stripe colouring restricted to the cross; each arm cycles with period
n). INPUT shows only a contiguous n×n window of the cross around the
intersection; OUTPUT redraws the FULL cross. Optional horizontal flip mirrors
both.
**Current (stored):** ~14.22 pts (public net).
**Target tier:** A (separable arms; per-arm colour is a periodic 1-D profile;
off-grid mask is a separable rectangle). Not S (output colour is not a
per-cell local function — needs arm detection + periodic extension).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | label-map, Mul-product extension, ReduceMax in-grid | A/B | 31514 | 91 | 14.64 | 265/265 stored | ok, heavy |
| 2 | MatMul arm-extract + MatMul Teq extension | A | 22506 | 92 | 14.97 | — | leaner |
| 3 | separable in-grid (rowany⊗colany) + uint8 label chain | A | 15666 | 92 | 15.34 | — | big cut |
| 4 | Clip indicator (drop nzB bool) | A | 14766 | 93 | 15.39 | — | |
| 5 | Gather periodic extension (drop Teq 30×30 entirely) | A | 13098 | 65 | 15.51 | 200/200 | |
| 6 | indicator-Conv [0,1..1] for counts + reduced-one-hot arm colours (drop colour-index plane) | A | 12504 | 74 | **15.56** | **500/500** | ADOPT-CANDIDATE |

## Best achieved
15.56 @ mem 12504 params 74 — fresh 500/500. Beats prior ~14.22 by **+1.34** (Y).

## Recovery (flip-agnostic, verified 0/5000 numpy + 200/200 ONNX fresh)
Colour on each arm is PERIODIC with period n, so flip/offset/colour-list are
never recovered. From the input only:
- in-grid mask = any-channel-hot (off-grid cells are all-channels-off → grid
  extent visible). Computed SEPARABLY: rowany⊗colany 1-D occupancy profiles.
- colour-index plane G = Σ k·input_k (1×1 Conv → fp32, cast fp16).
- per-row/col coloured counts (ReduceSum of Clip(G,0,1)); the arms are the row &
  column whose count == n = max(rowcount,colcount).
- arm colours via MatMul (contract the arm axis of fp16 G → 1-D, no product
  plane). Periodic extension by **Gather**: the n coloured cells are n
  CONSECUTIVE positions, so out colour at i = arm[first + ((i−first) mod n)]
  (Mod fmod=0, all 1-D int32 indices — no 30×30 equality matrix).
- assemble uint8 label L = where(rowarm,rowprof, where(colarm,colprof,0)),
  off-grid→sentinel 99; final Equal(L,0..9) → BOOL output (free).

## Irreducible-floor analysis
Dominant intermediates (final): coloured-indicator Conv nz (3600 fp32) + two
reduced-one-hot arm planes (1200 each) + ingrid And (900 bool) + three uint8
Where label planes (900 each = 2700). The Conv is fp32 because input is fp32
(any linear combo of the free input is fp32; casting input→fp16 first would
materialise an 18000B one-hot). ReduceSum rejects uint8/bool so the per-cell
coloured-count indicator must be ≥fp32-from-conv (3600 is the entry floor). The
arm-OH MatMuls are 1200 each (fp32, forced by the fp32 input). The three label
Wheres each broadcast a separable mask into a full [1,1,30,30] plane. Floor ≈
3600 + 2·1200 + 900 + 2700 + smalls ≈ 12500 ⇒ ~15.56.

## OPEN ANGLES (re-attack backlog)
- Fuse the 3-Where label assembly to 2 (fold sentinel base into an arm default)
  — every attempt still needs 3 full planes; no clean win found.
- The two arm-OH MatMuls (1200 each) are fp32 because input is fp32; no obvious
  way to fp16 them without a huge input cast.
- The indicator-Conv 3600 fp32 entry looks irreducible without a free-input fp16
  path. If a future task shares this "find the periodic arm" shape, the count
  indicator is the binding constraint.

## INSIGHT (transferable)
PERIODIC arm/profile reconstruction is flip/offset/colour-list-AGNOSTIC: recover
one period from the data and extend by `Gather(arm, first + (i−first) mod n)` —
the n shown cells are consecutive, so a 1-D modular Gather replaces a 30×30
equality matrix (saved ~2700B vs the MatMul-Teq idiom). Pairs with the now-proven
"grid extent is visible because off-grid is all-channels-off" lever → separable
rowany⊗colany in-grid mask (120B vs 3600B ReduceMax plane).
