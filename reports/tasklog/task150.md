# task150 — 67a3c6ac

**Rule:** A square grid of side `size` (3..9), every cell coloured from {6,2,1,7}
(never colour 0), sits at the top-left corner of the 30×30 canvas; off-grid cells
are ALL-ZERO. The output is each row reversed (horizontal mirror):
`out[:,:,:,c] = in[:,:,:,size-1-c]` for c<size, all-zero for c>=size. Pure spatial
permutation along the column axis.
**Current:** 18.44 pts, public CumSum/Where index + Gather (mem 668, params 36)
**Target tier:** S — output is a pure column permutation of the input → ONE Gather
whose output IS the free graph output; only the int32 index vector materialises.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | ReduceSum col-profile → size → Where(in-grid, rev, 29) idx fp32 → Gather | S | 788 | 37 | 18.28 | 200/200 | below target |
| 2 | same, fp16 working vectors, reuse occ as in-grid mask | S | 454 | 36 | 18.81 | 200/200 | beats +0.37 |
| 3 | drop Where: negative idx wraps to off-grid zero cols (no clamp) | S | 394 | 35 | 18.94 | 200/200 | +0.50 |
| 4 | size = Sqrt(ReduceSum(input)) [=size²] → scalar, kills [30] col-profile | S | 192 | 31 | 19.59 | 200/200 | +1.15 |
| 5 | integer index arithmetic (int32 arange, no fp16 rev) | S | 136 | 31 | 19.88 | 500/500 | ADOPTED |

## Best achieved
19.88 @ mem 136 params 31 — adopted? Y. Beats prior 18.44? Y (+1.44).

## Irreducible-floor analysis
Dominant intermediate is the int32 column-index vector `idx` ([30] = 120B). Gather
indices reject uint8 (ORT) and int64 is wider, so int32×30 = 120B is the floor for
ANY Gather-based column permutation. The remaining ~16B are three scalars
(total/size/size-1). No full-grid (30×30) plane ever materialises — the result plane
is the FREE output, and `size` is recovered from a single scalar reduction.

## OPEN ANGLES (re-attack backlog)
- A negative-step Slice (steps=[-1], axis=3) reverses the WHOLE 30-col axis with 0
  params/0 mem, but lands the grid at the right edge (shifted by 30-size); undoing
  that shift is itself data-dependent and needs a Gather/Slice, so no net win.
- Strict floor for permutation-by-Gather is the 120B int32 index; only a 0-index
  closed-form reversal (none exists for a left-anchored variable-size block) beats it.

## INSIGHT (transferable)
⭐ Two reusable levers landed here:
1. SIZE FROM TOTAL-COUNT, NOT A PROFILE: for a fully-filled k×k grid of nonzero
   colours, `size = Sqrt(ReduceSum(input))` is a single 4B scalar (fp32 sqrt of a
   perfect square 9..81 is exact, truncating Cast→int32 is safe) — avoids the [30]
   column-occupancy profile entirely. Use whenever a scalar dimension equals √(pixel
   count) or (pixel count)/known-width.
2. NEGATIVE-INDEX WRAP AS A FREE OFF-GRID CLAMP: a reversed index `size-1-c` goes
   negative for c≥size; ONNX Gather wraps `idx+dim`, which for this left-anchored
   block lands every out-of-grid column on columns [size..29] — all off-grid/zero —
   so the output zero-fills with NO Where/clamp/fallback constant. Replaces a
   Where(mask, rev, fallback) (drops a bool mask + a const + an op).
