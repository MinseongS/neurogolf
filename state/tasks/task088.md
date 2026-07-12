---
deployed_cost: 1872
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task088 — 3de23699 (crop box interior, recolour sprite → corner colour)

**Rule:** A rectangular "box" is marked ONLY by 4 corner pixels in colour `colors[1]`
at the four cells just OUTSIDE the interior: (brow-1,bcol-1), (brow-1,bcol+wide),
(brow+tall,bcol-1), (brow+tall,bcol+wide). Inside the interior
[brow..brow+tall-1] x [bcol..bcol+wide-1] are `wide+tall+randint(-1,1)` sprite pixels
in colour `colors[0]`. Output = the tall×wide interior cropped to the top-left of a
fresh grid, with every sprite pixel painted the CORNER colour `colors[1]` and every
other interior cell background (0); outside tall×wide everything is off.
Colour ID (exact, verified 0/5000): corner colour = the non-bg channel with pixel
count == 4; sprite colour = the non-bg channel with MAX count (sprite ≥ 5 since
wide+tall ≥ 6). Box geometry from the corner colour's bbox: brow=rmin+1, bcol=cmin+1,
tall=rspan-1, wide=cspan-1.
**Current:** 13.846 pts, gen:biohack_new, mem 69698, params 120
**Target tier:** B (data-dependent crop + translate-to-origin). Not Tier A: output is a
data-dependent TRANSLATE of a recovered window to the origin (needs a Gather-shift, not
separable). Both the crop window AND the fill colour are data-dependent, but every
parameter collapses to a closed-form scalar (count==4 / argmax / 1-D bbox), so it lands
well below the detection floor. Same shape as task036 (crop+shift).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | counts→cornercol(==4)/spritecol(argmax); corner bbox from 1-D occupancy; Gather(input,spritecol,ch) full plane → 10×10 window shift; label map L→Equal | B | 12791 | 134 | **15.533** | 200/200 (+500/500) | WIN |
| 2 | regime crack (2026-07-09): kill 900B class_pad — generator proves tall,wide∈[3,10] ⇒ fixed [1,1,10,11] tri-level u8 code plane (sprite=0/off=1=x_zp/bg=2, Pad value 1); terminal ConvInteger(1×1, runtime u8 w=1+δ(e,0)−m̂, w_zp=1, x_zp=1, static pads [0,0,20,19]) does one-hot expansion + 10→30 placement inside FREE int32 output; moment-bbox front-end unchanged | B | 1031 | 93 | **17.975** | 500/500 | WIN (gate PASS, not adopted — orchestrator gates) |

## Best achieved
**17.975 @ mem 1031 params 93 (cost 1124) — candidates/task088/cand.onnx, adopted? N (orchestrator gates).**
Beats deployed 17.465 (cost 1872) by **+0.510**. Fresh 500/500; bundled 267/267 (gate PASS).
Note: deployed net had meanwhile evolved to the moment-bbox + Slice-crop + Pad(900B)+Equal
design (cost 1872); attempt 2 keeps its front-end and cracks the 900B tail.

### Attempt-2 mechanism (regime crack, free-output ConvInteger)
- Generator (`arc-gen/tasks/task_3de23699.py`): `wide, tall = randint(3,10)` ⇒ interior ≤10×10
  ALWAYS, grid ≤24×24 (never >30). Licenses a FIXED compact canvas — no 30-space label plane.
- Tri-level u8 code plane: crop bg-channel interior [1,1,t,w] fp32 (read floor 400B) → Cast u8
  → `label2 = bg+bg` ∈ {0=sprite, 2=bg} → `xfix = Pad(label2 → [1,1,10,11], value=1)` (110B).
  Code 1 = off-window = x_zero_point.
- Terminal `ConvInteger(xfix, w_u8[10,1,1,1], x_zp=1, w_zp=1, kernel 1×1, static pads
  [0,0,20,19])` → int32 [1,10,30,30] = FREE graph output. Runtime weight `w−w_zp = δ(e,0)−m̂`:
  bg→ch0=+1, sprite→ch_m=+1 (others ≤0), off-window/pad→0. Decode (out>0) exact; ORT 1.26
  ConvInteger pads with x_zero_point (verified numerically).
- ⚠️ ORT buffer-reuse gotcha: shape-changing Pad output declared same size as its input's
  declared value_info ([1,1,10,10]) gets planned into the input's (smaller actual) buffer →
  runtime "Shape mismatch attempting to re-use buffer". Fix = give the Pad output a UNIQUE
  declared size ([1,1,10,11]) and absorb the extra column in the terminal op's static pads.

## Irreducible-floor analysis
Dominant intermediates (same floors as task036, ~15.6):
- **3600 B fp32 `splane` = Gather(input, spritecol, axis=1)** — the full 30×30 sprite
  mask plane. Irreducible: the crop window position (brow,bcol) is data-dependent, so the
  full plane must exist before the windowed Gather (circular: window pos needs the corner
  bbox, plane needs spritecol). fp16 does NOT shrink it (ORT upcasts full planes to fp32).
- **two 1200 B fp32 occupancy profiles** `rowocc`[1,10,30,1] + `colocc`[1,10,1,30] from
  ReduceMax(input) — needed to recover the CORNER colour's bbox cheaply (avoids a SECOND
  full 3600 B corner plane). They stay fp32 (inherit input dtype).
- **900 B uint8 padded label map** L[1,1,30,30] — Pad rejects bool so the 30×30 label map
  must be uint8 before the final Equal→BOOL output.
Everything else (counts 40 B, scalars, fp16 ramps) is small.

## OPEN ANGLES (re-attack backlog)
- Avoid materialising splane: gather the 10-col / 10-row window straight from the 10-ch
  input then contract — tried mentally, [1,10,10,30]=12 KB is WORSE; collapse-to-1-channel
  first wins. No cheaper path while window pos is data-dependent (~3600 B is the floor).
- Fuse the two Gather(axis2)+Gather(axis3) window steps via one GatherND (≈ saves the
  300-elem `Vr` intermediate, ~+0.05, marginal).
- The two occupancy profiles could in principle be one ReduceMax if corner bbox came from a
  single combined scan, but row & col mins/maxes need separate axes — no clean fusion.

## INSIGHT (transferable)
⭐ **(attempt 2) Generator-proven fixed compact canvas + terminal ConvInteger placement:**
when arc-gen source bounds the dynamic window (here ≤10×10), pad the compact code plane to
that FIXED size with a dynamic Pad, then let ONE ConvInteger with STATIC pads do both the
one-hot channel expansion (runtime u8 1×1 weight, w_zp=1 signed routing) and the
compact→30×30 placement inside the free output. Tri-level code {lo, x_zp, hi} makes both
active channels linear-separable with NO bias: extremes map to ±1 through (x−x_zp)(w−w_zp),
the middle/pad value to 0. Kills Pad(900B)→Equal tails on every crop-to-origin task whose
window is generator-bounded (task036 family). Sibling of canvas_crop_shrink +
bitpack_code_plane_arithmetic_decode(4) + ConvInteger-as-free-output (task392).

⭐ "4 corner markers define a variable crop box" is the task036 crop+shift idiom with a
cleaner colour-ID: count-based discrimination (corner = count==4, sprite = argmax-count)
beats geometric span analysis when the marker colour has a FIXED small pixel count. Recover
the crop window from the marker colour's 1-D occupancy bbox (no second full plane), then
Gather-shift the OTHER colour's plane to origin and recolour via a label map. Two distinct
data-dependent scalars (window colour vs fill colour) cost only 2 cheap ArgMax/Equal over
the 40 B counts vector — no extra planes.

## ADOPTED 20260709T055657Z
- cost: 1872 -> 1124 (points 17.9754)
- source: candidates/task088/cand.onnx
- note: regime vein batch7: generator-proven 10x10 canvas cap + tri-level u8 code + terminal ConvInteger (runtime u8 weight, x_zp=1) doing one-hot expansion AND compact->30x30 placement in the free output; 900B class_pad + 600B crop chain deleted. 500 fresh 0-fail. TRANSFERABLE: task036 crop+shift family + any generator-bounded lost-Pad:900B window; ORT buffer-reuse gotcha (unique declared size for shape-changing Pad)

## ADOPTED 20260712T030136Z
- cost: 1124 -> 1122 (points 17.9771)
- source: /Users/minseong/.codex/worktrees/56ef/neurogolf/submission/overfit_nets/task088.onnx
- note: min-merge from overfit_nets
