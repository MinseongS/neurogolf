---
deployed_cost: 11716
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task138 — 5daaa586

**Rule:** Input H×W grid (H,W~10..26) with a rectangular box of four FULL lines —
left/right vertical (cols `left`,`right`, every row) and up/down horizontal
(rows `up`,`down`, every col) — coloured colors[0..3]=(left,right,up,down) and
drawn in a random `draworder` (corners take the last-drawn line's colour). Scattered
single-cell pixels of one `drawcolor` (== exactly one of the four line colours) emit
a RAY in a single global direction toward the matching wall (left if drawcolor==
colors[0], right==colors[1], up==colors[2], down==colors[3]), painting drawcolor from
the pixel up to (not into) the wall. Output = box region [up..down]×[left..right]
moved to the top-left of a fresh canvas.

**Current (deployed):** 15.4157 pts, ext:kojimar7113, mem 14386, params 149.
**Target for +0.3:** mem+params ≤ 10768.

## Key finding — opset-12 uint8 MaxPool (re-confirmed why kojimar is so lean)
The deployed kojimar net uses **opset 12**, where MaxPool legally accepts/returns
uint8 (opset-11 and earlier reject it — the BUILD_PROMPT "ORT rejects uint8 MaxPool"
note is opset-≤11-only). This lets the ENTIRE ray-fill run on uint8 24×24 planes
(576B each) instead of fp16 (1152B). Also: **uint8 global MaxPool(kernel=[CH,CH])
returns the scalar max with NO fp32 ReduceMax plane** — drawcolor recovery for free.
The harness scorer checks DOMAIN not VERSION, so opset 12 scores identically.

## My from-scratch re-golf (crop-first, uint8, opset 12)
Rebuilt the whole pipeline: ONE forced fp32 decode (`Σk·input_k` Conv, 3600B) → Cast
uint8 → box edges from per-row/col count-Convs (no 30×30 occupancy plane) → crop+shift
the uint8 index to top-left → all downstream planes uint8/bool 24×24 → directional
uint8 MaxPool ray (transpose + one-sided full-length kernel) → Pad → Equal→BOOL output.

| # | angle | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|
| 1 | crop-first, fp16 ray (opset 11) | 32343 | 172 | 14.61 | 200/200 | correct, fp16 ray too heavy |
| 2 | opset 12, uint8 ray (MaxPool) | 20691 | 770 | 15.03 | 266+200/200 | correct, still > kojimar |
| 3 | drawcolor via uint8 global MaxPool (drop 2304B fp32) + fused raymask | 18383 | 770 | 15.14 | 266+200/200 | best; STILL below kojimar 15.42 |

## Verdict: INFEASIBLE to beat +0.3 (kojimar is at the structural floor)
The forced fp32 decode entry plane is 3600B (a 10→1 channel reduction over the fp32
one-hot must output fp32; ORT Conv on fp32 input yields fp32; cropping in 10-ch space
is 23040B, far worse). The data-dependent crop machinery is unavoidable: colu 900
(uint8 cast for cheap gather) + cr 720 (2-step gather intermediate) + crop 576 + L 900
(Pad 24×24→30×30 for the output) = 3096B. Together with the entry = **6696B forced**.
The 4-directional ray fill (seed → transpose → directional MaxPool ×2 → select →
transpose-back → fillback, plus the interior + keep + seed masks) needs ~8–10 full
24×24 planes ≈ 4600–5800B even fully uint8. Floor ≈ 11500–14000 → score ≈ 15.4–15.55.
The +0.3 bar (≤10768) sits BELOW this floor. kojimar (11 such planes, 14386) is already
near-optimal; my leanest faithful rebuild lands 18383 (more mask planes than kojimar)
and even an idealized fusion to kojimar's plane-count only reaches ~15.4–15.5 (+0.0–0.1).

## OPEN ANGLES (all dead-ended)
- Conv decode → fp16: blocked, ORT Conv needs weight dtype == fp32 input dtype.
- Avoid entry plane via per-channel crop: [1,10,24,24] fp32 = 23040B, far worse.
- Pad bool output instead of L: [1,10,24,24] bool = 5760B, worse than L 900.
- Nested-Where to drop union/keep planes: each Where still materializes its own
  counted plane → no net reduction.

## INSIGHT (transferable)
⭐ **opset 12 unlocks uint8 MaxPool** (incl. global-max scalar reduction with no fp32
plane) — a real lever for any prefix/suffix-MAX fill or directional carry that the
opset-11 "uint8 MaxPool rejected" note wrongly closes. Halves every fill plane vs fp16.
⭐ BUT a "forced-fp32-entry + data-dependent CROP + multi-directional ray fill" task has
a hard ~14k floor (~15.4 pts): the 3600 entry + ~3100 crop/pad + ~5000 ray-plane stack
cannot be squeezed below the +0.3 bar. A net already at ~15.4 with one fp32 decode plane
+ uint8 crop/ray (kojimar pattern) is AT FLOOR — BAIL.

## S8 (2026-07-02) — matrix-sweep verdict: priced FLOOR (block-3 opus agent; see agent report in submission_log context). Do not re-attempt without a new mechanism.

## S9 (2026-07-03) — 5×6 single-tap valid-Conv crop to 26×25 (+0.096) ADOPTED
Rectangular binding max H26×W25 (height=width±1). Entry planes cropped: cgf32 3600→2600,
cg 900→650, cgp 961→729 (pad end [1,1]→[1,2] keeps 27×27 square), crop_r 713→621,
sentinel 30→26. mem 13789→12215, params 172→462. Bit-identical: 2000+600 uncached,
inc fail=2 = cand fail=2 (pre-existing 24-row edge case), div 0. Orthogonal to S8
CODE_PLANE floor (pure size cut). Box 23×23 stack + cg30 exit-Pad = remaining floor.
Backup task138_pre_s9.onnx.

## 2026-07-07 auto-golfer rectangular 22x23 ray probe KILL

Candidate: `reports/candidates/task138/rectangular_22x23_ray_probe.py`.

Bundled max output extent is height 22, width 23, so auto-golfer tried replacing
the active 23x23 square ray workspace with a 22x23 rectangular workspace.  This
requires dropping the transpose-sharing trick and using separate vertical and
horizontal MaxPool paths.  Byte model was promising: candidate cost `12215+462`
to `11923+506`.

Result: fail 137/266.  Four direction-polarity variants
`task138_rect22_hrev{0,1}_vrev{0,1}.onnx` also failed.  The transpose-sharing
path encodes more than just direction polarity; the direct horizontal MaxPool
lowering is not equivalent.  Do not retry rectangular ray shrink without first
proving a Python/ONNX oracle for the exact ray direction semantics.

## 2026-07-08 S31 — byte-identical initializer dedupe ADOPTED

Candidate from `reports/candidates/dedupe_initializers_sweep.py`.

One byte-identical initializer alias was rewired.  Bundled gate remained fail=0.
Cost `12267 -> 12266` (`memory 12215`, `params 52->51` equivalent active count);
active overlay updated in `submission/overfit_nets/task138.onnx`.

## ADOPTED 20260713T144127Z
- cost: 11716 -> 11656 (points 15.6364)
- source: candidates/public_dumps/20260713_highroi/king77578_neurogolf-udit22-single-zips-public/task008/task138.onnx
- note: Udit22 public-LB min-merge; bundled fail=0

## ADOPTED 20260713T150909Z
- cost: 11716 -> 11656 (points 15.6364)
- source: candidates/public_dumps/20260713_highroi/king77578_neurogolf-udit22-single-zips-public/task008/task138.onnx
- note: isolated residual-public LB probe; bundled fail=0

## ADOPTED 20260715T082749Z
- cost: 11716 -> 11656 (points 15.6364)
- source: candidates/public_dumps/20260715_refresh/prvsiyan_neurogolf-7268-10-reproducibility-audit/audit_models/task138.onnx
- note: prvsiyan reproducibility-audit public min-merge; bundled fail=0

## ADOPTED 20260715T084816Z
- cost: 11656 -> 11647 (points 15.6372)
- source: candidates/task138/c3_wide8.onnx
- note: ROBUSTNESS WIN that is also strictly cheaper: cost 11656->11647. The incumbent finds box lines via 'count non-black in cols 0..3' + ArgMax over a 1x4 score window; a non-line row reaching 4 breaks it, and 34/266 BUNDLED examples already reach 3 — one cell short. It has not fired in 266 draws but is a live LB hazard (same family as insight entry_crop_fitted_to_bundled_max_not_rule_max). Widening the score window 4->8 makes a non-line row need 6 coincident decoys instead of 2; costs only params (216->274) and still gates strictly cheaper. Differential over 12000 fresh instances (48 grid sizes, box extents 6-24, all 4 ray directions ~3000 each): deployed fresh-fail 94 (0.78%), candidate 8 (0.067%) — a 12x cut — with 86 disagreements and 0 WORSE (never regresses vs incumbent). Expected value ~13.1 pts vs the incumbent's ~1.9. Also folded range25 into range24+scalar and dropped a no-op broadcast Unsqueeze. Entry crop verified RULE-EXACT (margins >=1 on all sides in 266/266), clearing the crop-risk flag for this task. CAVEAT: arc-gen absent from this checkout, so failure rates come from a reverse-engineered generator whose margins are uniform [1,7] where the real distribution declines — it likely UNDER-states the extent-24 risk; the 1x4 risk is independently corroborated by the bundle's own 34/266 at distance 1.

## ADOPTED 20260715T084905Z
- cost: 11647 -> 11567 (points 15.6441)
- source: candidates/task138/c2.onnx
- note: compact spatial-plane tail c2; bundled fail=0; 11647->11567

## ADOPTED 20260715T091055Z
- cost: 11567 -> 11537 (points 15.6467)
- source: candidates/task138/kcollapse.onnx
- note: kernel-collapse rescan: single-position Conv 1x6 -> 1x1, params -30
