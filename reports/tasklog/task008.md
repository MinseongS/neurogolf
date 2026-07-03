# task008 — 05f2a901

## 2026-06-29 mechanism screen

Rule: move red pixels relative to cyan marker structure under flip/transpose
variants.  The current graph uses fixed channel slices and compact geometric
logic rather than a full colour-index plane.

Current source score: 16.365306 @ mem 5561 params 65.  Dominant tensors are the
two fixed-colour slices (`c2f`, `c8f`) and a 30x30 label/mask carrier; the palette
is fixed enough that direct channel slices are cheaper than a generic 1x1
colour-index Conv.

No rewrite adopted.  This is not a good transfer target for colour-LUT mechanisms:
the two relevant colours are fixed, and the remaining cost is geometric routing
plus full-canvas output conditioning.

## 2026-06-30 S1 — FLOOR confirmed (deep structural re-route attempt failed)
Hypothesis: colours fixed (red/cyan), cyan unchanged, red rigidly translates → route cyan
from FREE input + native-cropped red plane, drop the 30×30 carrier (target ~+1pt). Built a
correct from-scratch geometric closed form: oracle 266/266, bundled fail=0, fresh 2500/2500
== incumbent (fully generalizing). But it measured 5938B > incumbent 5561B.
**Floor proof (per-tensor, irreducible):** 2×16×16 fp32 slices = 2048B (red nibble holes need
the exact mask read; no slice-with-cast — uint8/bool mask requires materialising the fp32
slice first; cyan locator likewise); ONE 30×30 uint8 carrier = 900B (cheapest one-hot
expansion is Equal(carrier[1,1,30,30],chans[1,10,1,1]); Concat/Scatter alternatives are
WORSE — a [1,10,30,30] base = 9000B); 7×256B working planes = 1792B INCLUDING the **channel-0
background ingrid-rectangle plane** (the "route everything via free input" idea overlooks that
output ch0 must be 1 across the whole in-grid rect → forces the carrier + ingrid plane);
+~820B strips/scalars = ~5561 = incumbent. VERDICT: incumbent (ext:biohack) already optimal,
no per-task lever. No change.

## 2026-07-01 sequential deep pass

Fresh recheck: **1000/1000 pass**.

Memory profile still matches the prior floor analysis:

- `c2f`, `c8f`: two `[1,1,16,16]` fp32 channel/spatial slices = **2048B**.
- `lab30`: final scalar label carrier = **900B**.
- 16x16 mask/route planes (`m2`, `m2b`, `m8b`, `base`, `lab2`, `lab`,
  `mvr`) = **7 x 256B**.

Rechecked possible substitutions:

- Fixed colours mean generic colour-index or task001-style colour factorization
  is not useful; direct channel slices are cheaper.
- `Gather` channel-first would materialize a 30x30 fp32 channel plane before
  cropping, worse than the current precise 16x16 `Slice`.
- Replacing the scalar label carrier with 10-channel construction is much larger.

Conclusion unchanged: no adoptable improvement found.

## 2026-07-03 cyan profile rewrite — ADOPTED

Human review questioned whether the cyan 2x2 marker needs a full `[1,1,16,16]`
channel slice.  It does not: the marker is always a solid 2x2 block, so row and
column occupancy profiles are enough to recover both the locator and the cyan
overlay mask.

Adopted source-owned rewrite:

- remove `c8f [1,1,16,16]` fp32 slice (**-1024B**) and its direct bool cast;
- compute cyan row/column profiles from the free full input with two `Einsum`
  contractions against `ch8f [10]`, `ones30 [30]`, and a singleton;
- crop those profiles back to 16 and rebuild the cyan 2x2 mask as
  `Cast(rp8) AND Cast(cp8)`;
- keep the red 16x16 slice and label-carrier output path unchanged.

Stored verification: **266/266 pass**,
`memory=4809`, `params=104`, `points=16.50035996783135`.
Fresh generator verification: **3000/3000 pass**.

Net improvement over the previous deployed source:
`memory 5561 -> 4809`, `params 65 -> 104`, total counted cost
`5626 -> 4913`, score `16.364846 -> 16.500360` (**+0.1355**).

Border-crop note: dropping a fixed two-cell border is not safe.  Over 20k fresh
samples, cyan and red input/output bboxes each reached all four borders after
the generator's flip/xpose transforms.  The persistent 16x16 crop remains needed.
