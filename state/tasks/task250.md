---
deployed_cost: 2516
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task250 — a48eeaf7 ("pull each gray pixel onto the ring around the red box")

**Rule:** A 2x2 red(2) box sits at (boxrow,boxcol)..(boxrow+1,boxcol+1) on a 0
background; gray(5) pixels are scattered.  Output = red box copied unchanged, plus
every input gray at (r,c) snapped toward the box: R=clamp(r,boxrow-1,boxrow+2),
C=clamp(c,boxcol-1,boxcol+2) — i.e. each gray lands on the 4x4 ring
[boxrow-1..boxrow+2]x[boxcol-1..boxcol+2].  Multiple grays may collide; original
gray locations are dropped.  br=boxrow=min red row, bc=boxcol=min red col.  Grid is
always exactly 10x10; colours {0,2,5}.
**Current:** was 15.61 pts (public ext:biohack_new). Now 16.92 pts, mem 3164, params 61.
**Target tier:** B (data-dependent clamp = scatter-collapse). Box position is a
global aggregate (min red row/col) and each gray's output cell is an input-derived
clamp → not S (no fixed conv/permute window). NOT row x col separable: outgray[R,C]
requires the SAME pixel to map both coords, so a rowcond ⊗ colcond would create
cross-pixel false positives → Tier A out. B is the highest admissible tier.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | bg/red/gray slices + clamp-matrix double-MatMul + in-grid Or chain + Pad + Equal | B | 5564 | 93 | 16.36 | 200/200 | works |
| 2 | grid is always 10x10 → drop bg slice + entire in-grid mask; build CmatT pre-transposed (drop Transpose) | B | 4264 | 84 | 16.62 | 200/200 | trim |
| 3 | clamp matrices + gray + MatMuls cast to fp16 (values {0,1}, exact) | B | **3664** | **85** | **16.77** | **200/200** | FINAL |
| 4 | clamp matrices + gray + MatMuls as uint8 QLinearMatMul; onnxsim | B | **3164** | **61** | **16.92** | **500/500** | ADOPTED |

## Best achieved
**16.92 pts @ mem 3164, params 61 — stored exact, fresh 500/500.** Adopted as
`custom:task250+qlinear+onnxsim`. Beats prior live 16.77 by +0.144.

## Irreducible-floor analysis (after attempt 2)
The data-dependent clamp is realized as a boolean-semiring double MatMul:
`outgray = Rmat @ gray @ CmatT` where Rmat[R,r]=[clamp(r,br-1,br+2)==R] and
CmatT[c,C]=[clamp(c,bc-1,bc+2)==C], both built from the scalar br/bc via a clamped
arange + Equal, then summed (collisions harmless, thresholded `>0` at the end).
After the fp16 MatMul trim the dominant intermediate is the 900 B uint8 Pad (the
30x30 label feeding the FREE final Equal) — output spans 30x30, only the Pad makes
off-canvas cells all-channel-0; irreducible. Remaining cost:
- red + gray channel slices, [1,1,10,10] fp32 = 400 B each. red is load-bearing
  twice (ReduceMax min-index for br/bc; box mask) and must be fp32 (ReduceMax
  rejects uint8; Slice preserves fp32). gray is cast to fp16 then fed to MatMul.
- Rmat + CmatT clamp matrices + gray8 + rowmapped + colmapped, now uint8 [1,1,10,10] =
  100 B each. QLinearMatMul is exact here because values are {0,1}, sums are small, and
  the only downstream use is `Greater(count, 0)`.
- the 200 B bool clamp masks + scalar reductions + 2-level Where are ≤200 B.

## OPEN ANGLES (re-attack backlog)
- **Shrink the MatMul canvas.** Output gray + box occupy only the 4x4 ring + 2x2
  box (≤4x4 = 16 cells), but the box position varies (br,bc∈2..6) and grays read
  the full 10x10, so the input slice can't shrink. A data-dependent crop to the
  ring would be a Gather (its own ≥100 B), net neutral. Untried in detail.
- **Cast clamp matrices to fp16 for the MatMul.** DONE (attempt 3), then superseded by
  uint8 QLinearMatMul in attempt 4.
- **Avoid the red fp32 slice.** br/bc need ReduceMax (fp32); the box mask reuses the
  same slice. No cheaper min-index over uint8 (ReduceMax/Min reject uint8). Blocked.

## INSIGHT (transferable)
⭐ **A per-pixel data-dependent COORDINATE REMAP that is independent in row and col
(here a clamp toward a box) is NOT Tier-A separable — but it IS a boolean-semiring
double MatMul:** outgray = Rmat @ src @ CmatT, where each remap matrix Mat[out,in]
= Equal(remap_vector[in], out_arange) is built from the scalar parameters via a
clamped/shifted arange. Sum semantics are fine because collisions only over-count
and the final `>0` threshold flattens them. This generalizes any
"scatter each marked pixel to f_row(r), f_col(c)" rule (clamp, shift, fold,
modulo) without a Gather/Loop — two 400 B fp32 MatMuls land it solidly below the
Tier-B 16.8 ceiling.
⭐ 2026-06-28 dtype update: if the double MatMul only implements boolean-semiring OR
via `sum > 0`, use uint8 QLinearMatMul instead of fp16 MatMul. This halves every remap
matrix/intermediate and is exact with scale=1/zero-point=0.
⭐ **When the generator's grid is a FIXED full size (here always 10x10), the entire
canvas is in-grid → delete the bg slice and the whole in-grid Or chain; the 30x30
Pad sentinel alone produces the off-canvas all-zero cells.** (−1300 B, +0.26 pt
here.) Always check `input sizes` over 500 fresh instances before paying for an
in-grid mask.

## ADOPTED 20260709T041317Z
- cost: 2516 -> 2123 (points 17.3394)
- source: candidates/public_dumps/20260709/7261-53-lb-compact-onnx-artifact-starter/nets/task250.onnx
- note: min-merge from nets

## 2026-07-10 s8port fold probe (runtime-spend axis, opus) — NO BUILD
- Ran: graph dump + per-tensor mem breakdown (2052 exact); tail map (F4->IM->Equal(colors6)->onehot6
  bool 600B->Pad(contiguous)); ORT 1.26 Einsum dtype probe at opset 13 (fp16/int32 OK, uint8/bool
  NOT_IMPLEMENTED); gate arithmetic 600 − 0 fp32 − 330 params = 270 < 480; best-case realized cost
  2253-2453 > 2123.
- Verdict: BLOCKED on bytes — onehot6 is BOOL (no fp32 to reclaim); non-contiguous channel placement
  needs a [10,30] embed (+330 params); einsum forces an fp16 basis that re-spends the dying bytes.
- Tool+date: opus agent, onnx 1.21.0 / ort 1.26.0, 2026-07-10.
- Reopen: uint8/bool Einsum kernel; public task250 < 1920 tail.

## 2026-07-13 REGIME-CRACK re-probe (onehot6 indicator_fold, opus) — NO BUILD, FLOOR CONFIRMED
- Target: deepfold `onehot6` [1,6,10,10] bool = **600B** (top tensor, dpts 0.3321),
  produced by Concat (node40), consumed by output Pad (node41). Incumbent grader cost 2123.
- **Byte-math floor (two hard constraints):**
  1. **bg channel 0 is REQUIRED.** scoring.convert_to_numpy encodes color 0 as channel-0=1;
     run_network = `(output>0)` compared with `np.array_equal` on the FULL [1,10,30,30] one-hot.
     In-grid background cells (color 0) MUST have channel 0 set → bg cannot be dropped/argmax-defaulted.
  2. **Pad is edge-only → interior channel spacers 1,3,4 unavoidable.** Active output channels are
     {0=bg, 2=red, 5=gray}; span 0..5 = 6 channels. Channels 1,3,4 (never-present colors) sit INTERIOR
     to the span and cannot be created by an edge Pad → must be materialized as false in the compact concat.
  - ⇒ compact one-hot span = 6 channels × 10×10 = **600B is the mechanism floor.** Spatial 10×10 fixed
     (bg spans full grid, grays read full grid — no crop). bool already minimal dtype (no fp32 to reclaim).
- **Alt mechanisms ruled out by byte math:** label-pad ordering = padded [1,1,30,30]=900B > 600 (onehot-pad
  already the cheaper side of the task308/382 break-even: 6×100 < 900). free_final_onehot_equal/Where-over-input
  fails — grays RELOCATE (removed at source, added at ring) so input isn't a valid base (differs in ch0 AND ch5,
  not a pure overlay). direct-Gather (task343) needs axis-aligned column select — N/A for scatter. Einsum-fold
  hits fp32-co-bind wall: per-cell placement routing matrix co-bound to free fp32 input ≥4B/cell > 600B bool.
- Tool+date: opus agent, onnx 1.21.0 / ort 1.26.0, 2026-07-13. Independently re-derives the 2026-07-10 s8port
  BLOCKED verdict via the grader-decode proof (bg-required) rather than tail-map inspection.
- Reopen-trigger: uint8/bool Einsum CPU kernel lands in ORT (would allow a bool-basis free-output fold); OR a
  public task250 dump measures < 2050 (foreign mechanism we haven't found).

## ADOPTED 20260713T143837Z
- cost: 2123 -> 2098 (points 17.3513)
- source: candidates/public_dumps/20260713_7281/extracted/task250.onnx
- note: Ryosuke 7281.18 public-LB confirmed per-task min-merge; bundled fail=0

## ADOPTED 20260713T151005Z
- cost: 2123 -> 2098 (points 17.3513)
- source: candidates/public_dumps/20260713_7281/extracted/task250.onnx
- note: Ryosuke-7281 isolation B; task047 explicitly excluded; bundled fail=0

## ADOPTED 20260713T151947Z
- cost: 2123 -> 2098 (points 17.3513)
- source: candidates/public_dumps/20260713_7281/extracted/task250.onnx
- note: Kaggle-isolated safe: group delta +2.05 exactly (sub 54651291 minus 54651270); task047 excluded
