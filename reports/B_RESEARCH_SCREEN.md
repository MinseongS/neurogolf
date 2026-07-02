# B-research screen — can any sub-17pt task convert to 20+?

Run: 2026-06-30. Method: 6 parallel agents screened ALL 119 sub-17pt tasks
(every task with a tasklog), classifying whether the RULE can be reformulated to
delete every full [30,30] intermediate (the precondition for 20+ = mem+params ≤ ~148).

## Headline result

**TRUE_20+ candidates found: 0 / 119.**

| verdict | count | meaning |
|---|---:|---|
| TRUE_20+ | 0 | genuinely needs zero full planes |
| NEAR_18 | 33 | only +0.3..+1.8 incremental possible (most < +0.3) |
| FLOOR_BOUND | 70 | a full [30,30] label/entry plane is structurally required (~16pt cap) |
| HARD_WALL | 16 | information-loss / global discriminator / exact-infeasible |

## Root cause (independently cited by all 6 agents)

1. **Entry-plane floor (3600B).** Extracting a per-cell colour from the
   [1,10,30,30] one-hot input requires ReduceSum/ReduceMax/Conv, and ORT type
   rules force the output to fp32 → 3600B. Casting to uint8 still costs 900B.
   Seen in 004, 014, 029, 035, 091, 138, 162, 177, 222, 280, 310, 333, 387, 396, …
2. **Label/colour-index plane floor (900B).** Any data-dependent recolour or
   placement needs a full [30,30] index plane.

20+ needs mem+params ≤ 148; a single full plane is 900–3600B, so **any task that
must read per-cell colour is capped at ~17.9 pts.** The 48 tasks already at 20+
are precisely the ones whose rule never needed a per-cell plane (scalar
predicate / fixed LUT / input-channel-direct crop). No such sub-17 task remains.

## Implication for 7800

- Per-task mechanism transfer (the HIGH_SCORE_FRONTIER program) yields +0 in 20+
  conversions. This is an evidence-based dead end, not a "needs more effort" case.
- The only residual is incremental NEAR_18 golf; genuine ≥+0.6 beats are ~6 tasks
  (264 +1.84, 270 +1.11, 354 +0.95, 086 +0.78, 037 +0.72, 368 +0.68), all
  low/med confidence — matches the "per-task hand-opt vein EXHAUSTED" memory.
- The real lever is the **shared floor itself**: a representation that avoids the
  forced fp32 entry plane (the two standing bets — lossless bit-packing sweep,
  directional-cummax global propagation — attack the floor, not single tasks), or
  a stronger public base.

## Genuine incremental (NEAR_18) shortlist, if pursuing small wins anyway

| task | claimed beat | conf | lever |
|---|---|---|---|
| 264 | +1.84 | low | matched-filter glyph detect + template stamp |
| 270 | +1.11 | med | scalar pull-back (12-scalar rebuild, drop directional MatMuls) |
| 354 | +0.95 | med | CumSum reset on run boundaries (untried, "most promising lever") |
| 086 | +0.78 | low | L-parametric morphology + colour-index → FREE Equal |
| 037 | +0.72 | low | direction-separable diagonal 7×7 Conv reused by flip |
| 368 | +0.68 | low | solid-rectangle offset via run-length product-chain |

## HARD_WALL list (do not retry without a public teacher / new primitive)

002 018 044 077 090 096 118 157 209 219 255 319 361 366 (+ 008, 286 partial)

## Addendum 2026-06-30 — "attack the shared floor" (path 1) also DEAD

PoC `reports/scripts/poc_entry_floor.py` measured minimal graphs at 30×30:

| graph | mem | pts | note |
|---|---:|---:|---|
| identity (input>0→output) | 0 | 25.0 | pass-through is free; only *computing* costs |
| fp32 entry + label + Equal | 4500 | 16.6 | classic floor |
| packed extract (input→f64 Cast) | 72240 | 13.8 | **bit-packing dead: input cast = full 72KB plane** |
| packed + unpack | 94980 | 13.5 | worse |
| fp32 strips → Greater direct out | 2400 | 17.2 | best general per-cell mechanism |
| uint8 strips → Greater direct | 3000 | 17.0 | cast ADDS cost (fp32 reduce already counted) |
| uint8 1ch separable (single colour) | 1200 | 17.9 | channel-expand forces 900B |

**Verdict:** the two standing 7800 bets are refuted. bit-packing requires casting
the FREE input to a packable dtype, which materializes a full [1,10,30,30] plane
(≥72KB) — strictly worse; directional-cummax shares this (ops on the float input
are forced fp32). Reading any per-cell colour forces an fp32 ReduceMax/Einsum
output that is already counted; uint8 casting only adds cost. Min entry for a 10ch
per-cell-colour task ≈1200B (separable)..3600B (non-sep) → **hard ceiling ~18pt,
20+ mathematically impossible under this grader.** Both reformulation paths
(per-task transfer + shared-floor) are now closed. Only remaining 7800 lever: a
stronger public base.
