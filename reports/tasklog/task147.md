# task147 — 67385a82

**Rule:** Input = green (colour 3) pixels scattered on a small (W,H∈3..6) black grid,
top-left anchored on the 30×30 canvas. Per cell: a green pixel with ≥1 orthogonal
green neighbour → cyan(8); an isolated/edgefree green pixel (0 orthogonal green
neighbours) → green(3); in-grid background stays background(0); off-grid stays all-zero.
A GENUINE orthogonal-4-neighbourhood per-pixel recolor (the output colour of a green
cell depends on its neighbours), NOT a 1×1 recolor.

**Current:** 18.187 pts, generic conv3×3+b, mem 0, params 910
**Target tier:** detection/at-floor — minimal encoding is one dense 3×3 Conv→output.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | multi-plane: slice ch3 + plus-Conv + cond + Equal/sentinel route | B | 31500 | 33 | 14.64 | (passed train) | worse — too many full planes |
| 2 | ANALYTIC single dense 3×3 Conv→output (exact weights) | — | 0 | 910 | 18.187 | 200/200 | matches floor, provably general |

## Best achieved
18.187 @ mem 0 params 910 — adopted? N (== current). Beats prior 18.187? N (tie).

## Irreducible-floor analysis
Single dense Conv [10,10,3,3]+bias = 910 params, mem 0 (output IS the conv output).
910 params is the floor because:
- Conv input channels MUST equal 10 (input is the full 10-ch one-hot). Slicing to the
  2 active channels {0,3} would cut the weight to [10,2,3,3]=190 but the sliced
  [1,2,30,30] f32 intermediate costs 7200B → 25−ln(7390)=16.1, far worse.
- Output must be 10 channels with the colour bits at FIXED positions {0(bg),3(green),
  8(cyan)}; can't shrink M or relocate via Pad (Pad only zero-pads edges).
- Grouped Conv (the only single-op <910-param route, e.g. group=2 → [10,5,3,3]=460 →
  18.87) FAILS: green is input channel 3 but cyan is output channel 8, and no
  contiguous group of ≤5 channels contains both 0 and 8 (span 0..8 = 9 > 5), so the
  group barrier blocks routing green→cyan.
- Any decomposition (plus-Conv feature + 1×1 recolor) needs ≥1 full plane (≥1800B f16
  / 3600B f32); the isolated-green branch even needs a BAND (feat==exact), not a
  halfplane, so it can't fold into a 1×1 Conv at all → ≥2 feature planes. All
  decompositions land ~14–16, below the 18.187 single-conv floor.

## OPEN ANGLES (re-attack backlog)
- None viable. The {0,3,8} output-channel spread vs single green input channel is the
  hard structural blocker for both grouped-conv shrink and channel-slice shrink. If a
  future scoring change ever made Pad-with-interleave or sparse-initializer params
  free, the analytic conv could drop to ~190 params (16 nonzero weights).

## INSIGHT (transferable)
⭐ A genuine orthogonal-4-neighbourhood recolor whose CURRENT net is one generic dense
3×3 Conv→output (mem 0, params 910) is AT FLOOR — do not chase it. The decisive test
before attempting: can the needed output-colour channels + the carrying input channel
all fit in ONE contiguous group of size ≤ M/group? Here {bg=0, green=3, cyan=8} span 9
channels, so no grouped-conv shrink is possible, and channel-slicing to fewer input
channels always costs a ≥7200B f32 plane that dwarfs the param saving. The analytic
exact-weight version (10·gc+nbr−10.5 for cyan, 10·gc−nbr−9.5 for isolated-green via the
harness `out>0`) is worth keeping as documentation of the exact rule, but it only TIES
the trained generic conv.
