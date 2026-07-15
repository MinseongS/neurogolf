---
deployed_cost: 1610
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task036 — 1f85a75f

**Rule:** The grid holds ONE small connected blob ("celestial object") drawn entirely in the
special colour `colors[0]` — a 3..5 × 3..5 region at 0.75 density, guaranteed connected. All
other pixels are scattered single-pixel NOISE in colours `colors[1..]`, sprinkled over the whole
grid but kept OUT of the blob's bbox+1 border (so noise never touches the blob). The output is the
bounding box of the blob, cropped to the top-left corner of a fresh grid (channel-0 fills the holes
inside the bbox; everything outside the HxW box is all-channels-off). **Key invariant (verified
0/20000):** the blob colour is the colour whose pixels are spatially CLUSTERED, i.e. the colour with
the minimum bbox span `max(rowspan,colspan)` — noise colours span the whole grid (span up to 29),
the blob colour spans ≤4. Picking the min-span colour identifies the blob EXACTLY; no
connectivity/flood-fill is needed.

**Current (prior):** 13.71 pts, gen:vyank6322, mem 79832, params 111
**Target tier:** detection→B (variable-crop + translate). The output is a data-dependent TRANSLATE
of a recovered bbox to the origin → needs a Gather-shift; not separable/single-conv. But the colour
identification collapses to a closed-form 1-D per-channel argmin (NOT connectivity), so it lands well
below the detection floor.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | min-span colour → full-30×30 shift+label | B | 24723 | 133 | 14.88 | 200/200 | win |
| 2 | + 5×5 working-window shift + Pad | B | 14273 | 126 | 15.43 | — | win |
| 3 | + fp16 per-channel occupancy/span work | B | 11641 | 126 | 15.63 | — | win |
| 4 | + Gather(input, blobcol, axis=1) instead of colour-Conv+uint8-cast | B | 11274 | 118 | **15.66** | 200/200 (+1000/1000) | win |

## Best achieved
15.66 @ mem 11274 params 118 — adopted? N (orchestrator gates). Beats prior 13.71 by **+1.95** (≥0.3 ✓).
GENERALIZES: fresh 200/200 AND a separate 1000/1000 in-memory run; the core "min-span colour = blob"
idea is 0/20000 exact.

## 2026-06-29 live-frontier refresh

Current live/source is far beyond the older semantic build: **17.314071 pts @
mem 2140 params 37**.  It avoids the full 30x30 blob-plane Gather entirely:
`MaxPool(input, kernel=[30,1])` gives per-channel column occupancy (1200B), the
minimum-width colour identifies the blob, a scalar `target_idx` drives a tiny
5x5 `Slice`, and `Equal` builds only a 10x5x5 bool crop before final bool `Pad`
to `output`.

Rechecked for a +0.3 lever.  The dominant 1200B occupancy profile is the actual
colour-selection statistic; it replaces the old 3600B full selected blob plane.
The next largest tensor is only the 250B crop one-hot.  A label-plane Pad would
reintroduce a 900B full-canvas label and is worse.  This is no longer a high-ROI
candidate unless a way appears to compute per-channel min/max column spans
without materialising the 10x30 fp32 profile.

## Irreducible-floor analysis
Dominant intermediates: (a) the 3600 B fp32 `bplane` = `Gather(input, blobcolor, axis=1)` — the full
30×30 blob mask plane, irreducible because the 5×5 crop WINDOW position (min_row,min_col) is
data-dependent, so we must materialise the full plane before the window Gather (circular: window pos
needs blobcolor, blobcolor needs the per-channel spans); (b) two 1200 B fp32 `ReduceMax(input)`
occupancy profiles [1,10,30,1]+[1,10,1,30] — per-channel occupancy is required to find which channel
is clustered, so 10×30 elements are irreducible; they stay fp32 because ReduceMax inherits the fp32
input dtype (casting input to fp16 would cost an 18000 B [1,10,30,30] plane). Everything else is fp16
1-D aggregates or 5×5 / sentinel uint8 planes.

## OPEN ANGLES (re-attack backlog)
- Shrink the two 1200 B occupancy ReduceMax to fp16 (→600 each, ~1200 B / ~0.1 pt) — would need a
  cheap fp16 cast of a *reduced* tensor without paying for a full fp16 input plane; not obviously possible.
- Collapse `bplane` 3600 B: if a single op could both select the blob channel AND window-gather in one
  shot (e.g. a fused GatherND with computed indices) the full plane might be skippable — untried.
- Drop the sentinel-10 Pad path by emitting the 5×5 one-hot then Pad the BOOL output (blocked: Pad
  rejects bool; current uint8-label + final Equal is the cheaper route).

## INSIGHT (transferable)
⭐ A "find the connected object among noise" task is NOT necessarily a connectivity/flood-fill BAIL:
when the object is a SINGLE clustered colour and noise uses OTHER colours, the object is identified in
CLOSED FORM as the **minimum-bbox-span colour** (per-channel 1-D occupancy → argmin of
max(rowspan,colspan)). This replaces flood-fill entirely and is exact. General lever for variable
crop-and-translate-to-origin: recover (min_row,min_col,H,W) as scalars, then `Gather(axis=2,
arange(W)+min_row)` + `Gather(axis=3, …+min_col)` shifts a small WORK×WORK window to the top-left;
build a uint8 label map on that window, `Pad` to 30×30 with sentinel 10, finish with
`Equal(L, arange[0..9])` into the free BOOL output. Watch the scorer trap: a Reshape-to-scalar target
must be a length-1 (`[1]`) initializer, NOT an empty `[]` array — an init with a 0 dim makes
`calculate_params` return None ("performance could not be measured").

## S8 (2026-07-02) — matrix-sweep verdict: priced FLOOR (block-1/2 opus agents; occupancy/max-semiring reductions or sub-400B u8 banks). Do not re-attempt without a new mechanism.


## S16 adoption (2026-07-06) — yuu111111111 public-bundle net (+0.036)
- Source: yuu111111111/neurogolf-6-failure-modes notebook (total 7235.05, embedded 400-net archive; MINED per-task despite lower total).
- New grader cost = 2100 (mem 2061 + params 39), fail=0 bundled.
- Fresh-gate 1500: incumbent fail = 0 | candidate fail = 0 | candidate != incumbent = 0  -> cand_fail <= incumbent_fail (safe rule PASS).
- Mechanism: structural golf: fewer counted node-output intermediates (graph rewrite, functionally equal on fresh).


---

## 2026-07-08 — 8000-mode adoption (public re-mine, LB-confirmed 7264.26)

**+0.2606 LB** (mem 1502, par 114; from franksunp-7249.50). Mechanism: 36-node MaxPool+GatherElements plane pipeline -> 59-node Einsum-heavy form (7 Einsum); wide pooled/gathered planes contracted into free-input Einsums.

Gate: isolated `evaluate` bundled fail=0 + strictly cheaper than our incumbent + uint8-TopK scan clean.
Artifact: `submission/overfit_nets/task036.onnx` (backup in `.minmerge_backup/`). Source dumps + notebook code under `reports/candidates/public_mine_20260708/`.
This is a direct 8000-mode ONNX overlay (constant dataset => permanent). Source `src/custom/task036.py` NOT regenerated: the public net is optimizer-emitted (base64 in-notebook), no per-task Python builder to lift; the mechanism above IS the transferable insight.

## ADOPTED 20260709T041335Z
- cost: 1610 -> 1589 (points 17.6291)
- source: candidates/public_dumps/20260709/7261-53-lb-compact-onnx-artifact-starter/nets/task036.onnx
- note: min-merge from nets

## ADOPTED 20260712T140205Z
- cost: 1589 -> 1051 (points 18.0425)
- source: /Users/minseong/project/neurogolf/dumps/archive_extract/submission7300+/task036.onnx
- note: archive.zip submission7300+ net; fresh 2000/0 fail; mechanism-graft

## ADOPTED 20260715T110100Z
- cost: 1051 -> 940 (points 18.1541)
- source: candidates/task036/poly_convinteger.onnx
- note: output_folding: replace counted 5x5 onehot+Pad epilogue with three u8 features and padded ConvInteger writing FREE output

## ADOPTED 20260715T113857Z
- cost: 940 -> 718 (points 18.4235)
- source: candidates/task036/scalar_power_base8.onnx
- note: mathematical_rewrite: replace counted row/column score planes and ArgMax tails with FREE-input base-8 power-sum scalars plus Log coordinate decode; bundled exact and fresh 10k A/B identical

## ADOPTED 20260715T121548Z
- cost: 718 -> 670 (points 18.4927)
- source: candidates/task036/scalar_qlinear_tail.onnx
- note: output_folding: preserve scalar power coordinate decode and replace three-feature polynomial ConvInteger with shifted two-feature padding-safe QLinearConv bias; fresh1600 A/B identical
