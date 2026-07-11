---
deployed_cost: 3183
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task201 — 846bdb03

## 2026-06-30 — Conv dual-axis crop (ADOPTED, +0.32)

Live was already a better `live_exact` reconstruction (**16.563 @ mem 4476**,
266/266) than the 15.46 attempt below — that table is stale.

Found and adopted a structural win: the colour-index entry Conv had
`kernel [2,1] dil [17,1]` → output `labels_f [1,1,13,30]` (1560 B fp32), then a
`Slice` cropped width 30→13. The +17 vertical tap only ever reads the empty
region (all 266 inputs are fixed **13×13**, content ≤ row/col 12), so it
contributes nothing. Extending to `kernel [2,2] dil [17,17]` crops BOTH spatial
dims to 13×13 inside the Conv, so `labels_f` → 676 B and the width `Slice`
(plus `slice_starts`/`slice_ends`) disappears.

**Result: 16.881 pts @ mem 3202, params 154 — 266/266 (exact by construction).**
Same domain assumption as the original (already sliced to [0:13,0:13]); the
extra taps read provably-empty cells, so this is mathematically identical, not a
heuristic. Source-owned in `src/custom/task201.py`; `networks/task201.onnx`
rebuilt; source/live parity confirmed. Fresh-gen N/A (generator absent locally,
like task001) — but exactness is structural, not sample-gated.

**Remaining floor (3202):** `labels_f` 676 (fp32 Conv entry, irreducible),
`output_small` 560 (bool one-hot at the 7×8 max output size, bound by the scored
set), `crop_labels` 240, and six 169 B (13×13) detection planes. Next gains are
<0.1 each (fuse the transient `yellow_mask`→`yellow_u8` Cast, etc.) — low ROI.

⭐ **TRANSFERABLE:** a colour-index entry Conv should crop EVERY bounded spatial
dim in one op via `kernel[2,2]/dilation[gridsize]` — never emit a half-cropped
`[H,30]` plane and then `Slice` the width. Scan all tasks whose entry is
`Conv → Cast → Slice`: each wasted `[H,30]`/`[30,W]` fp32 entry plane is ~900 B
of free memory.

## (stale) prior analysis below

**Rule:** INPUT has two disjoint objects on black: a hollow h×w BOX (h=max(rows)+3, w=2*(max(cols)+2)) with YELLOW(4) corners, left col=colors[0], right col=colors[1]; and a SPRITE CLUSTER = two conway sprites side-by-side (idx0 in colors[0], idx1 in colors[1]), horizontally FLIPPED inside its own (h-2)×(w-2) bbox iff flip==1. OUTPUT (exactly h×w) = the box border + its interior filled by the cluster DE-FLIPPED (colors[0] sprite left, colors[1] right). Verified 500/500 numpy: yellow bbox gives (r0,c0,h,w) exactly; colors=colf[r0+1,c0]/[r0+1,c0+w-1]; cluster = nonzero outside the box bbox, its bbox is exactly (h-2)×(w-2); flip iff mean-col(colors[0] px in cluster) > mean-col(colors[1] px); interior[i-1,j-1]=colf[sr0+i-1, sc0+(flip? w-2-j : j-1)].
**Current (prior public net):** 13.72 pts, 137-node ArgMax/Gather/Where/Pad chain, mem 79002, params 308.
**Target tier:** B (closed-form scalar recovery + spatial copy/mirror into a fixed small index plane). Not Tier-S because the colour-index entry plane is an irreducible 30×30 f32 Conv.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | colf-index + 1-D bbox scalars + fixed 6×8 gather + Pad/Equal | B | 16516 | 140 | 15.28 | 200/200 | win (after OH 6→8: train can hit 7×8) |
| 2 | + fp16 flip-centroid chain | B | 14818 | 153 | 15.39 | — | win |
| 3 | + fp16 interior gather & value chain | B | 14184 | 153 | 15.43 | — | win |
| 4 | + reuse single colf16, fp16 occupancy/colours, drop dead occ | B | 13808 | 153 | 15.46 | 200/200 | BEST |

## Best achieved
15.46 @ mem 13808 params 153 — adopted? N (write-only). Beats prior 13.72 by **+1.74** (>> +0.3). 266/266 stored, fresh 200/200.

## Irreducible-floor analysis
Dominant intermediate = `colf30` [1,1,30,30] f32 = 3600B: the entry colour-index plane (Σ k·input_k via one Conv[1,10,1,1]). Per FLOOR_RESEARCH this 10→1 reduction must be f32 and full-canvas (3600B floor). Next: `OUT30` 900B uint8 (the 30×30 Pad sentinel plane feeding the FREE Equal — minimal at uint8). Then four 676B 13×13 f32 planes (`colf`, `Y`, `sp`, `spv`-path) forming the f32 detection core: `colf` must stay f32 as the Cast source; `Y` & `sp` feed the f32 `bbox` helper (ramp-dot ReduceMin/Max). Everything downstream (centroid, interior gather, value chain) already fp16.

## OPEN ANGLES (re-attack backlog)
- Make the `bbox` helper fp16 (fp16 ramp/penalty consts) so `Y` and `sp` go fp16 → save ~676B (≈+0.05). Low payoff, moderate refactor risk (shared f32/f16 paths).
- Avoid the 676B `colf` slice by slicing the input to 13×13 channel-subset first — but a 10-ch slice is 6760B, strictly worse; no win.
- colf30 3600B is the genuine floor; further gains are <0.1 and not worth the dtype-fragmentation risk.

## INSIGHT (transferable)
⭐ A "reassemble two detected objects into a variable-size output" task that LOOKS like shape-correspondence is closed-form when the geometry is a pure **spatial copy + optional 1-D mirror**: recover all geometry as 1-D-profile scalars (yellow-corner bbox → output size; nonzero-outside-box → sprite bbox), build a FIXED-MAX-SIZE (OH×OW) colour-index plane by Gather-ing source coords (with per-axis flip select = `Where(flip, w-2-j, j-1)`), overlay the border + an off-output **sentinel(99)** via a tiny Where chain, then Pad+Equal into the FREE output — no per-object plane army, no ArgMax/Gather/Pad scaffold. 79k→13.8k (+1.74).
⚠️ **OH/OW bound lesson:** the random generator distribution (heights 4-6) UNDER-represents the hand-crafted train examples (train[1] is 7×8). Always set the fixed-output max from the actual scored set (train+test+arc-gen), not from sampling the generator — sampling 5000 instances never produced the train[1] size.

## ADOPTED 20260709T041329Z
- cost: 3183 -> 3046 (points 16.9784)
- source: candidates/public_dumps/20260709/7261-53-lb-compact-onnx-artifact-starter/nets/task201.onnx
- note: min-merge from nets

## ADOPTED 20260709T061947Z
- cost: 3046 -> 3016 (points 16.9883)
- source: candidates/task201/kcollapse.onnx
- note: kernel-collapse: single-position Conv kernel collapse after public/regime overlays

## NEGATIVE 2026-07-11 — full July-arsenal audit, 0 wins (fp32-input co-bind floor)
- **what was run:** grader-exact byte map (scoring.py model: mem = Σ node-output num_el×dtype_itemsize
  except free input/output; params = Σ init element-count, dtype-free). Counted floor = mem 2876 + params 140.
  Dominant counted planes: `labels_dil` fp32 [1,1,13,13]=676 (1×1 colour-index Conv, neg-pad crop 30→13),
  `output_small` bool [1,10,7,8]=560 (decode Equal→Pad), three 13×13 detection planes 169 ea
  (`labels` u8 / `rect_mask` bool / `pat` u8), `yrow`+`ycol` fp32 [1,30]=120 ea (yellow-corner profiles).
  Every July mechanism checked with byte math (below).
- **tool+date:** uv run ng gate + onnx shape-infer byte map + scoring.py read, opus per-task agent, 2026-07-11.
- **verdicts (why each fails):**
  - fp16-recast / QLinearConv / s8port tail-fold: BLOCKED by fp32-input co-bind. Harness feeds
    `np.float32` (scoring.py:53) → graph input MUST be fp32. Every entry reduction (`labels_dil` Conv,
    `yrow`/`ycol` Einsum) inherits fp32 → 676/120/120 are dtype-locked. QLinearConv/ConvInteger need
    uint8 input; a Cast(input→u8) = [1,10,30,30]=9000B counted → strictly worse. No fp32 dying plane
    clears the ≥480 fold gate.
  - free-output N-ary Einsum: FLOOR. Output interior = data-dependent SPRITE CONTENT stamped at a
    data-dependent position (optionally flipped) = POSITIONED-CONTENT per vein taxonomy, not global-state
    routing. `output_small` is a DECODE discriminator, not a Where ROUTING mask. All-or-nothing composition
    → the un-foldable sprite floors the whole tail.
  - signed-einsum separable routing: FLOOR (task233-class). Cost is in detection/sprite-stamp, not a
    separable-fill priority/label carrier; box border is separable but the sprite interior is not.
  - width-crop `yrow`/`ycol` [1,30]→[1,13]: IMPOSSIBLE. Einsum keeps the non-reduced spatial full-30;
    can't slice inside einsum, and Slice(fp32 input→13×13)=6760B. Deriving extents from `labels` needs
    ReduceMax-on-bool (unsupported) or new ≥169B planes → net ≥ break-even loss.
  - dtype/index golf: ArgMax outputs int64-locked (8B); Gather indices int32-minimal (no int16 in ONNX);
    all u8/bool planes already 1B. No redundant/dead tensor (0 dead outputs). kernel-collapse already applied.
- **reopen-trigger:** (1) new public task201 dump beating cost 3016 (min-merge); (2) a grader change letting
  input be declared uint8 (would unlock QLinearConv u8 colour-index → kills the 676B fp32 `labels_dil`,
  ~+0.23); (3) mixed-dtype Einsum accepted by ORT (fp16 carrier under fp32 input co-bind) — the vein-wide
  top residual lever; (4) a positioned-content stamp primitive (dynamic_kernel_stamp on fp32 input) proven
  cheaper than the 507B 13×13 detection core.
- **falsification history:** June "labels_f 676 irreducible" claim RE-CONFIRMED here under July arsenal
  (input-dtype root cause identified, not just asserted). Prior redesign falsified generic "floor" claims
  board-wide, so this stays a tool-conditional verdict, not durable truth.
