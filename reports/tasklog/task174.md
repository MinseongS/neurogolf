# task174 — 72ca375d

**Rule:** The 10x10 grid holds exactly 3 monochrome "boxes" (creatures) in 3 distinct colours.
Box-0 (`colors[0]`, the `if not idx` box that is copied to the output) is constructed to be BOTH
horizontal(column)-mirror symmetric AND 180-rotationally symmetric; boxes 1 and 2 are constructed to
be NEITHER. The output is box-0 cropped tight to its bounding box, placed at the top-left corner of a
fresh grid (channel-0 fills the holes inside the HxW bbox; every cell outside the bbox is
all-channels-off). **Key invariant (verified 0/8000 + fresh 500/500):** box-0 is the UNIQUE present
colour (c!=0) whose bbox-cropped shape equals its own HORIZONTAL mirror — `hflip`-symmetry alone
identifies it (rot-symmetry alone also works; vflip alone does NOT).

**Current (prior):** 14.60 pts, mem 32787, params 81 (108-node CumSum/ArgMax-style net).
**Target tier:** B (variable crop + data-dependent translate-to-origin). Identification collapses to a
closed-form per-channel reflection MatMul (NOT a symmetry-search/flood); the crop+shift is the
task036 idiom, landing well below the detection floor.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | reflect-MatMul hsym + task036 crop/shift, all fp32 | B | 26566 | 104 | 14.81 | 266/266 | works, <+0.3 |
| 2 | + cast active region to fp16, whole symmetry pipeline fp16 | B | 18480 | 104 | 15.17 | — | win |
| 3 | + overlap(A,Mf)==count(A) (1 plane) instead of diff+diff^2 (2 planes) | B | 16500 | 103 | **15.28** | 500/500 | win |

## Best achieved
15.28 @ mem 16500 params 103 — adopted? N (write-only). Beats prior 14.60 by **+0.68** (≥0.3 ✓).
GENERALIZES: isolated fresh 500/500 against freshly-generated instances; the hflip-symmetry identifier
is 0/8000 exact.

## Irreducible-floor analysis
Dominant intermediates (all over the 10x10 active region, [1,10,10,10]):
- `A32` 4000 B fp32 — the `Slice(input)` of the active region; Slice inherits the fp32 input dtype, so
  this single fp32 entry plane is the documented 3600B-style floor (cannot slice straight to fp16).
- four 2000 B fp16 planes: `A` (fp16 cast of A32), `Cmat` (per-channel reflection matrix), `Mf` (=A@Cmat),
  `AMf` (=A*Mf overlap). Each is a genuine full-region working plane; the reflection axis a=c0+c1 is
  per-channel so Cmat and the MatMul cannot be shared/shrunk, and Mf is the MatMul output. The
  overlap-count symmetry test already fuses what would otherwise be two planes (diff + diff^2) into one.
Everything else is tiny (1-D [1,10,1,1] scalars, the WORK=5 crop window, the 900B uint8 Pad output).

## OPEN ANGLES (re-attack backlog)
- Kill the 4000B fp32 `A32`: would need a Slice that emits fp16 directly (ORT has none) or a way to feed
  the MatMul without a named fp32 region — untried, likely structural.
- Collapse `Cmat`+`Mf` 4000B: a single fused per-channel column-reflection op (data-dependent reverse)
  would remove the explicit reflection matrix, but per-channel axes block a single Gather; no opset-11
  op does a per-batch data-dependent reversal cheaply.
- The reflection MatMul could in principle run on a per-channel-cropped 5x5 window if the box column
  bbox were gathered first, shrinking every plane ~4x — but the window position is itself per-channel
  data-dependent (circular: need the box to crop the box), so it cannot precede identification.

## INSIGHT (transferable)
⭐ "Odd-one-out by SYMMETRY among monochrome shapes" is NOT a symmetry-search wall: per-channel
horizontal-mirror symmetry is a closed-form **reflection MatMul** — reflect each channel's columns about
its own axis `a = c0 + c1` via `Cmat[k,c',c]=Equal(c'+c, a_k)` (task112/250 reflection-matrix idiom,
batched over the channel axis), then test equality of binary masks cheaply by `overlap(A,Mf)==count(A)`
(reflection preserves pixel count, so equal-count + full-overlap ⟺ identical mask — uses ONE extra full
plane `A*Mf` instead of two `diff`+`diff^2`). Once the box colour is a scalar, the variable crop +
translate-to-origin is the task036 Gather-shift + uint8 sentinel-Pad + final Equal finish.
Also: the canvas was a fixed 10x10 → slice to the active region and run every working plane at fp16
(whole-geometry fp16 works under ORT_DISABLE_ALL); the lone irreducible fp32 cost is the entry Slice.
