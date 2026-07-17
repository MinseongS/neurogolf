---
deployed_cost: 1065
logged_costs_match: match
migrated: 2026-07-09
---

# task301 — beb8660c

**Rule:** Input grid is W×H with W = num_colors (3..9), H = W + gap (gap 0..3). Each color k
appears as ONE horizontal bar of a DISTINCT length L (lengths 1..W, one per color; cyan/8 is
always the longest = W) at a random row/col. So pixel-count(color k) == its bar length, distinct
per color. Output is a right-justified staircase triangle: the color whose bar length is L fills
output row (L-1+gap), right-aligned over cols [W-L..W-1]. Rows 0..gap-1 empty. Equivalently a grid
cell (r,c) is occupied iff r+c >= H-1, with color = the color whose count == (r-gap+1).
**Current:** 15.84 pts (prior custom net, 30×30-plane formulation, mem 9385, params 89)
**Target tier:** B — per-cell deterministic color-index plane, but every parameter (H, W, gap,
per-row color) reduces to scalars/tiny vectors and the active canvas is bounded 12×9.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | prior: 30×30 inrect/inbar/geSplit/Lrect/L planes + rocc[1,10,30,1] | B | 9385 | 89 | 15.84 | 200/200 | baseline |
| 1 | small 12×9 canvas; H/W via 1-D occupancy; rowcolor via cnt==lenOfRow; occupied=r+c≥H-1; uint8 L→Pad99→Equal | B | 3905 | 58 | 16.715 | 200/200 | pass |
| 2 | cast all work planes (rcsum/occval/Lf/rowcolor) to fp16 | B | 3211 | 70 | 16.904 | 500/500 | best |

## Best achieved
16.904 @ mem 3211 params 70 — adopted? N (main adopts). Beats prior 15.84? Y (+1.06).

## Irreducible-floor analysis
Dominant intermediate is `L30` = the padded 30×30 uint8 label map (900B). It is irreducible:
the final 10-channel expansion must be `Equal(L30, arange[1,10,1,1])` routed into the FREE output,
which requires a 30×30 carrier broadcastable against the [1,10,30,30] input. uint8 (900B) is the
minimum carrier — fp16 would be 1800B and bool cannot be Equal'd against arange to yield 10 channels.
This is the standard tier-B label-map floor (~16.8); the small-canvas work planes push slightly above
it. Off-grid cells carry sentinel 99 (≠ any color 0..9) so Equal yields all-zero there (off-grid is
all-zero); in-grid empty cells = 0 → channel-0 background set correctly.

## OPEN ANGLES (re-attack backlog)
- Slice cnt to channels 1..9 (drop bg ch0) → match/matchf/kmatch become [1,9,RH,1] (216B vs 240B
  each); ~marginal, ~+0.02.
- Tier A is blocked: the occupied region is the anti-diagonal triangle r+c≥H-1 (couples r and c, not
  a row⊗col separable rectangle) AND the per-cell color is a per-row lookup keyed on data-dependent
  counts — both forbid a pure rowcond⊗colcond routing. The label-map is the right tier.

## INSIGHT (transferable)
⭐ "Length-coded bars sorted into a triangle" collapses to scalars: a color's pixel-COUNT uniquely
identifies its rank (counts are distinct), so per-row color = Σ_k k·(cnt[k]==lenOfRow[r]) with NO
positional matching of input bars to output rows. The triangle footprint is a single anti-diagonal
comparison r+c≥H-1 (NOT a separable rectangle, but still one Less op). Combined with the
guide's H/W-from-1-D-occupancy lever and a bounded 12×9 active canvas, the whole task is a tiny
label-map + one Pad-to-30×30 uint8 carrier → 16.9, ~+1pt above the naive 30×30-plane formulation
purely by shrinking every full plane to the generator's bounded grid size.

## 2026-07-12 — rank-machine/free-output-einsum regime enumeration → NO-BUILD (kill bar +0.1)
ran: full final-op regime enumeration (Greater/And/Or/Xor/Where/Mul/BitwiseAnd-u8/sign-decode-einsum/
ConvInteger-placement/ScatterElements/MatMulInteger) + brute-force (K,S) u8-wrap constant search +
deployed byte walk reconciled to 1077+64=1141 exactly (zero fat in its regime).
verdict: deployed outer-threshold design Greater(a[d,r],b[d,c]) is regime-optimal. Structural fact:
bg channel = 9-level staircase [c<=H-2-r], boolean rank min(W,H-1)=9 — only threshold comparison
renders 9-level staircases at 1B/element; u8 bit-planes give 8<9 (fails 10.7% draws), einsum operands
fp32-co-bind >=4B/el => >=~1350. Best honest alternative = u8-wrap row-gating 1065 (+0.069, below bar;
constants verified K=10,S=246, b-side off-column patch +28B included — build fully specified if bar drops).
tool+date: fable agent regime enumeration + ng gate re-measure, onnx 1.21/ort 1.26, 2026-07-12.
reopen: (1) mixed-dtype/fp16 ORT Einsum support (sign-decode revives at 2B/el); (2) public task301 <1141;
(3) final op with >=9-level nesting <1B/element; (4) grader decode relaxes from strict >0; (5) kill bar
drops to +0.05 (1065 wrap variant ready).
falsification history: first regime-enumeration pass on this task; prior "900B label carrier floor"
claim was superseded (deployed is NOT label-carrier — it is already the cheaper outer-threshold form).

## ADOPTED 20260715T122904Z
- cost: 1141 -> 1065 (points 18.0293)
- source: candidates/task301/u8_wrap.onnx
- note: UINT8 modular row gating K=10/S=246; Max-patched shifted column threshold; removes Equal+Where row carrier
