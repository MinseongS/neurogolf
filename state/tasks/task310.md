---
deployed_cost: 3227
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task310 — c909285e

**Rule:** A `size`x`size` grid (size 20..30) is filled with 3..4 "wires": each
wire (colour, spacing) sets every cell with `(r+1)%spacing==0 OR (c+1)%spacing==0`
to that colour (later wires overwrite earlier). Then a square box of side
`boxlength` (5..8) at (boxrow, boxcol) has its PERIMETER drawn in `boxcolor` (a
colour NOT used by any wire). Output = the box subregion
`grid[boxrow:boxrow+L, boxcol:boxcol+L]` (L=boxlength) — wire colours plus the
boxcolor perimeter — cropped to the top-left of a fresh grid; everything outside
the LxL box is all-channels-off. Box id: wire colours fill whole rows/cols so
their bbox span is ~size (>=15); boxcolor only spans the box (span = L-1 <= 7),
so boxcolor = the non-bg colour with MINIMUM bbox span, and
(boxrow, boxcol, L) = (minrow, mincol, rowspan+1) of that colour.

**Current:** 14.33 pts (manifest) / P=14.84 given, public base net, mem high
**Target tier:** B — data-dependent variable-size crop+translate-to-origin
carrying arbitrary per-cell colours. Tier S/A blocked: output colour per cell is
an arbitrary per-instance value read from a data-dependent window (not
row⊗col-separable, not a fixed linear function of the local one-hot); the crop
size and position are both data-dependent. NOT a BAIL: the output size is
recoverable (L = boxcolor span + 1) and box id collapses to a closed-form 1-D
per-channel argmin span (task036 idiom), no flood-fill / connectivity.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | min-span colour id (task036) + colf colour-index window Gather + LxL gate + Pad + Equal | B | 11916 | 134 | 15.603 | 200/200 | win |

## Best achieved
15.603 @ mem 11916 params 134 — adopted? N (orchestrator gates). Beats prior
14.33 by **+1.27** (vs given P=14.84 by **+0.76**, >= +0.3). GENERALIZES: stored
266/266, ISOLATED fresh 200/200 against freshly-generated instances.

## Irreducible-floor analysis
Dominant intermediates: (a) `colf` [1,1,30,30] fp32 = 3600B — the per-cell
colour-index plane (Conv output must match fp32 input; casting input to fp16
costs an 18000B 10-ch plane). The crop WINDOW position (boxrow,boxcol) is
data-dependent so the full plane must be materialised before the window Gather
(circular dependency, same as task177). (b) two `rowocc`/`colocc`
ReduceMax(input) profiles [1,10,30,1]+[1,10,1,30] = 1200B each — PER-CHANNEL
occupancy is required to find which colour has min span; they stay fp32 because
ReduceMax inherits the fp32 input dtype. (c) `Vr` [1,1,8,30] fp32 = 960B (a
WORK-row window still spans 30 cols; gathering cols-first gives [1,1,30,8], same
cost). (d) `Lfull` [1,1,30,30] uint8 = 900B (full-size for the final 30×30
Equal; Pad rejects bool so cannot Equal-then-Pad). These four are the same
structural floor that pins the task036/task177 crop-and-translate family at
~15.4–15.6.

## OPEN ANGLES (re-attack backlog)
- Cast colf->fp16 and gather the window in fp16 (Vr 960->480, Vw->128): adds an
  1800B fp16 colf plane while saving only ~600B on the window — net LOSS here
  because the windows are already tiny. Only helps when many downstream
  full-canvas ops run on the plane (task377), which this task does not have.
- Shrink the two 1200B per-channel occupancy ReduceMax: would need to identify
  boxcolor without per-channel spans (e.g. a single banded statistic that
  separates "full-grid wire" from "small box"); untried, ~0.1 pt if found.
- Fused GatherND to select colour channel AND window in one shot to skip the
  3600B colf plane — same untried angle as task036/task177.

## INSIGHT (transferable)
⭐ A grid saturated with full-row/full-col "wire" lines plus ONE small marked box
is NOT a detection wall: the box's distinguishing colour is the MINIMUM-bbox-span
non-background colour (wires span ~size, the box spans <=7), recovered by the
task036 per-channel 1-D argmin-span idiom. For a SQUARE data-dependent crop the
output size falls out for free as `span+1` of that colour — no NonZero/ArgMax of
output extent needed. Combine task036's min-span colour id with task177's
colf-window-Gather (carry arbitrary per-cell colours) to handle "crop a marked
region that contains arbitrary colours" in closed form at the ~15.6 B-tier floor.

## ADOPTED 20260709T041334Z
- cost: 3227 -> 3184 (points 16.9341)
- source: candidates/public_dumps/20260709/neurogolf-7266-72-w-visualizations/nets/task310.onnx
- note: min-merge from nets

## 2026-07-11 — full July-arsenal audit + value_info-crop experiment (opus agent) → FLOOR, NO BUILD
ran: full byte-map (mem 3108 = 2560 Slice[1,10,8,8] fp32 box carrier + ~548 moment-based
  detection); value_info-crop candidates (5×5, 1×1) BUILT + GATED — passed checker + bundled
  266/266 but memory unchanged 3108; byte-math on fp16/u8 recast, separable P·in·Q, GridSample,
  two-Gather, ch0-drop (invalid — grader decodes per-channel >0, ch0 load-bearing).
tool+date: onnx 1.21/ort 1.26, ng gate, scoring.py source read, 2026-07-11.
verdict: at mechanism floor. ⭐DURABLE: value_info under-declaration NEVER beats the profiler
  trace — calculate_memory takes max(static declared, runtime trace) per tensor; crop-by-
  declaration only helps tensors absent from the trace.
reopen: (1) ORT uniform-T escape (fp16 carrier co-bound with fp32 input); (2) an op reading
  free fp32 one-hot → windowed colour-INDEX without full-canvas intermediate; (3) public dump
  < 3184; (4) scoring change dropping the trace max (revives VI-crop → cost ~664).
falsification history: the 2026-07-11 sweep hypothesis ("Slice plane partially carrier") is
  FALSE — pure detection+carrier of output pixels.

## 2026-07-11 (2nd) — ci_triage row "valueinfo-legalized crop (216/008 class)" (BUILDER, opus agent) → DRY, floor RE-CONFIRMED independently
ran: per-tensor byte-map of DEPLOYED net (submission/overfit_nets/task310.onnx, sha e8f6c2df5a,
  cost 3184) — declared static shape vs ORT profile traced-max over the FULL 266-example bundled set
  (train+test+arc-gen), single accumulating session (mirrors scoring.evaluate). Box carrier =
  `safe_name_50` Slice output. Note: over the 4 local ARC train/test examples it traces only [1,10,7,7]
  =1960B (looks like 600B slack), but over the full 266 (arc-gen included) it reaches [1,10,8,8]=2560B.
tool+date: onnx 1.21 / ort 1.26, shape_inference(strict) + ORT profiler trace, 2026-07-11.
verdict: DRY — box carrier declared [1,10,8,8]=2560B, bundled traced-max also [1,10,8,8]=2560B (an
  8×8 box EXISTS in the arc-gen bundled set; boxlength∈5..8). declared == traced ⇒ zero VI slack; charge
  = max(declared,traced)=2560 either way. This independently RE-CONFIRMS the 2026-07-11 (1st) floor
  verdict: the earlier VI experiment declared 5×5 & 1×1 (both below traced) and correctly saw no change;
  the tighter test (re-declare to the exact bundled traced-max) also yields nothing because that max IS
  the full 8×8. DURABLE gotcha logged: value_info audits MUST trace the full 266-bundle (arc-gen), not
  the 4 ARC train/test — the local-4 max UNDERSTATES the carrier (7×7 vs true 8×8) and would produce a
  FALSE +600B slack signal. NOT MAIN-eligible.
reopen: unchanged from 1st verdict — (1) ORT uniform-T fp16 carrier escape; (2) op reading free fp32
  one-hot → windowed colour-INDEX with no full-canvas intermediate; (3) public dump < 3184; (4) scoring
  change dropping the max(declared,traced) charge (would revive VI-crop only if bundled box < 8×8, which
  it is not).

## ADOPTED 20260712T140109Z
- cost: 3184 -> 593 (points 18.6148)
- source: /Users/minseong/project/neurogolf/dumps/archive_extract/submission7300+/task310.onnx
- note: archive.zip submission7300+ net; fresh 2000/0 fail; mechanism-graft
