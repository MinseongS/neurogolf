---
deployed_cost: 683
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task254 — a61f2674

**Rule:** 9x9 grid with `num` (4 or 5) vertical GRAY (5) bars; bar `idx` is a solid
bottom-anchored run of height `val` in column `2*idx+offset`. Heights are sampled
DISTINCT from 1..9 so min/max are unique. Output: shortest bar recoloured RED (2),
tallest bar recoloured BLUE (1), all other bars erased to background.
**Current:** 16.02 pts (prior public net)
**Target tier:** A (closed-form recolour of the gray plane; no detection).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | full-30x30 col-count recolour | A | 10298 | 79 | 15.75 | - | passes but < P |
| 2 | crop to 9x9 active grid, label+Pad | A | 1736 | 37 | 17.52 | 200/200 | ADOPT |

## Best achieved
17.52 @ mem 1736 params 37 — beats prior 16.02 by +1.50. Y.

## Irreducible-floor analysis
Dominant intermediate = the Pad output L [1,1,30,30] uint8 = 900B (the one-hot
carrier the final `Equal(L, chan[1,10,1,1])` needs at full canvas to emit the
all-zero off-grid region). uint8 is already the minimum dtype; the plane cannot be
narrower. Second = gray fp32 9x9 slice (324B, ReduceSum needs float). Casting gray
to fp16 saves ~162B → ~17.62 but adds ops for +0.10; skipped as not worth the risk.

## OPEN ANGLES (backlog)
- fp16 gray slice (162B vs 324B) → ~17.62. Marginal.
- Route blue/red into the FREE output WITHOUT the 900B carrier would need a
  separable row⊗col form, but blue=gray∧maxcol is 2-D arbitrary (the bar shape),
  so not separable — the uint8 carrier is the right floor.

## INSIGHT (transferable)
Bottom-anchored solid bars mean bar HEIGHT = per-column pixel COUNT and the bar
SHAPE is already exactly the input gray cells — so "recolour the shortest/tallest
bar" is a pure recolour of the gray plane gated by Equal(colcnt, max)/Equal(colcnt,
min-over-nonzero), never a position reconstruction. Cropping a 9x9-bounded task to
its active grid (escape 3) cut every working plane 11x and moved 10298→1736.

## ADOPTED 20260712T030152Z
- cost: 683 -> 351 (points 19.1392)
- source: candidates/task254/rankeinsum.onnx
- note: rank-machine transfer from task010 template: pair-slot summing (Sel5 [30,5] shared between free-input height einsum + free-output emission einsum), Where(nz,h,ReduceMax) constant-free min-over-nonzero, GE(mx,h5) all-ones guard row, 6-operand sign-decode einsum; 683->351 (+0.666); fresh 2500/2500 div 0 vs incumbent

## ADOPTED 20260712T140216Z
- cost: 351 -> 254 (points 19.4627)
- source: /Users/minseong/project/neurogolf/dumps/archive_extract/submission7300+/task254.onnx
- note: archive.zip submission7300+ net; fresh 2000/0 fail; mechanism-graft
