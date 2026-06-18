# Submission log (autonomous sweep — submit every 5 adopted wins)

Baseline before sweep: **LB 6384.61** (prev best, 2026-06-15 16:07 submit, 333/303/228 wave).
Stored at session start: 6445.88 (≈61pt stored-vs-LB gap pre-existing from overcounted base nets).

| # | time(UTC) | stored | wins since last | LB (publicScore) | Δ LB | notes |
|---|---|---|---|---|---|---|
| baseline | 06-15 16:07 | ~6384.6 | — | 6384.61 | — | pre-sweep |
| 1 | 06-15 17:32 | 6454.46 | 020,034,020R,034R,091,224,370 | **6393.20** | **+8.59** | session stored Δ +8.58 → LB +8.59 = **1:1 translation confirmed** |

## ⭐ KEY RESULT (submission 1): floor-break sweep translates 1:1 to LB.
Session wins +8.58 stored → +8.59 LB (baseline 6384.61 → 6393.20). The large floor-break compactions
(020/034/091/224, each ~+2) are REAL LB gains, NOT local-only. (task370 +0.06 was marginal noise.)
Pre-existing ~61pt stored-vs-LB gap (6454.46 stored vs 6393.20 LB) is UNCHANGED — it lives in the
inherited public base nets (overcounted/non-generalizing), not touchable by our custom sweep. So: stored
delta from a generalizing floor-break win ≈ LB delta. Keep grinding; trust stored for generalizing customs.

| 2 | 06-15 17:58 | 6461.51 | 012,245,035,061,250 | **6400.24** | **+7.04** | proj was 6400.25 → **0.01 error, tracker exact**. stored Δ +7.05 → LB +7.04 = 1:1 again. gap 61.27 STABLE |

## ⭐ Submission 2 confirms the model: gap tracker projected 6400.25, actual 6400.24 (0.01 error).
Two submissions now: both +stored ≈ +LB exactly, gap pinned at ~61.2. The PROJECTED LB (stored − gap)
is trustworthy to ±0.1 — no need to submit to know where we stand; submit only to re-anchor/lock.

| 3 | 06-15 18:21 | 6467.99 | 290,195,188,341,375 | **6406.72** | **+6.48** | proj was 6406.72 → **0.00 error**. stored Δ +6.48 → LB +6.48 = 1:1. gap 61.27 PINNED. 3rd consecutive exact projection. |

## ⭐ 3 submissions, all exact (errors 0.01/0.01/0.00). Gap pinned at 61.27. Stored is a perfect LB proxy
(minus the constant 61.27 base-net gap). Submit only to lock/re-anchor; the projected LB is the truth.

| 4 | 06-15 ~18:4x | 6470.67 | 119,362,342,360,225 | **6409.40** | **+2.68** | proj 6409.40 → **0.00 error** (4th exact). gap 61.27 pinned. Smaller Δ = thinning headroom (low-pt wins). |

## ⭐ 4 submissions, errors 0.01/0.01/0.00/0.00. Gap immovable at 61.27. The stored→LB ratio is exactly 1:1
for generalizing customs. lb_status.py projected LB is ground truth. LB so far: 6384.61→6393.20→6400.24→6406.72→6409.40.

| 5 | 06-15 19:xx | 6480.57 | 244,278,275,306,264,57 (re-triage reservoir) | **6419.29** | **+9.89** | proj 6419.30 → **0.01** (5th exact). First batch from the 50 mislabeled-feasible. gap 61.28. |

## ⭐ Submission 5: re-triage reservoir translates 1:1 too (proj 6419.30 → 6419.29). LB now 6419.29.
Trajectory: 6384.61→6393.20→6400.24→6406.72→6409.40→6419.29. 6 mislabeled-feasible recoveries landed +9.89.

| 6 | 06-16 13:0x | 6489.62 | 036,206,112,033,177 | **6428.35** | **+9.06** | proj 6428.34 → **0.01** (6th exact). Re-triage reservoir wave 2. gap 61.27. |

## ⭐ Submission 6: 6th consecutive exact (proj 6428.34 → 6428.35). LB now 6428.35.
Trajectory: 6384.61→6393.20→6400.24→6406.72→6409.40→6419.29→6428.35. Reservoir wave 2 (+9.06).
Note: task025 came back MARGINAL (+0.06, transpose-equiv cap) — skip-marginal, not counted.

| 7 | 06-16 13:4x | 6498.01 | 358,193,359,368,161,132,390 (7 wins) | **6436.74** | **+8.39** | proj 6436.74 → **0.00** (7th exact). Reservoir wave 3. gap 61.27. |

## ⭐ Submission 7: 7th consecutive exact (proj 6436.74 → 6436.74, 0.00 error). LB now 6436.74.
Trajectory: …→6428.35→6436.74. Reservoir wave 3 (+8.39, 7 tier-A/B wins). task216 confirmed-infeasible
(non-local detection wall, public net at real floor). Gap pinned at 61.27 across all 7 submissions.

| 8 | 06-16 14:5x | 6506.30 | 232,271,354,94,180,346,94R,(+others) 8 wins | **6445.03** | **+8.29** | proj 6445.03 → **0.00** (8th exact). Reservoir wave 4 + task094 re-attack. gap 61.27. |

## ⭐ Submission 8: 8th consecutive exact (proj 6445.03 → 6445.03, 0.00 error). LB now 6445.03.
Trajectory: …→6436.74→6445.03. Reservoir wave 4 (+8.29). Incl task271 17.01 (tier A) & task180 17.74.
Marginal-skips this batch: 365 (+0.27), 117 (+0.035), 134 (+0.26) — all below +0.3 bar, not counted.
Reservoir now nearly exhausted (remaining retriage gains <0.7); approaching productive-work-done stop.

| 9 | 06-16 15:0x | 6514.67 | 321,121,27,21,88,70,238 (7 wins) | **6453.40** | **+8.37** | proj 6453.40 → **0.00** (9th exact). PENDING-POOL opened. gap 61.27. |

## ⭐⭐ Submission 9: 9th consecutive exact (proj 6453.40 → 6453.40). LB now 6453.40.
KEY DISCOVERY: the curated reservoir (retriage_build_queue + sweep_wave) was NOT the end — the UNTRIAGED
"pending" pool in sweep_ledger (272 low-score tasks) has REAL HEADROOM, not just detection walls.
Proven: task088 13.85→15.53 (+1.69), task070 13.90→16.25 (+2.35), task238 13.93→15.34 (+1.41) — all from
"pending"/untriaged tasks the re-triage never looked at. Probe with EARLY FEASIBILITY CHECK to bail fast on
genuine walls. Trajectory: …→6445.03→6453.40. Runway re-opened — keep mining lowest-points pending.

## Procedure (folded into loop)
1. trigger: every 5 adopted wins.
2. `python -c "from src.pipeline import pack; pack()"` (networks/ only; never --pack flag).
3. `/opt/homebrew/Caskroom/miniconda/base/bin/kaggle competitions submit -c neurogolf-2026 -f submission/submission.zip -m "<msg>"`.
4. poll: `kaggle competitions submissions -c neurogolf-2026` until status COMPLETE; record publicScore.
5. compute stored→LB ratio for the batch (calibrates whether wins translate). Kaggle keeps BEST submission,
   so a flat/down result never loses standing — but a flat result means the wins didn't translate (re-examine).

| 10 | 06-16 16:0x | 6525.70 | pending-pool wave: 204,184,351,213,400,231,328,037,094,346 (10 wins) | **6464.42** | **+11.02** | proj 6464.43 -> **0.01** (10th exact). PENDING POOL is the engine. gap 61.27. |

## ⭐ Submission 10: LB 6464.42 (proj 6464.43). Pending-pool wave +11.02. Trajectory: ...->6453.40->6464.42.
The untriaged pending pool is the productive reservoir (gap-closing concluded dead — 219/255/209 are walls).
Hit rate ~85% on lowest-points pending probes. Session total: 32 wins, confirmed LB +45.13 (6419.29->6464.42).

| 11 | 06-16 16:3x | 6529.36 | session-final: 092,324,378,165,377 (+204 wave) | **6468.09** | **+3.67** | proj 6468.08 -> **0.01** (11th exact). SESSION-FINAL LOCK. gap 61.27. |

## ⭐ Submission 11 (session-final): LB 6468.09 (proj 6468.08). SESSION TOTAL: 6419.29 -> 6468.09 (+48.80).
33 wins adopted across submits #6-#11, all projection-exact (gap pinned 61.27). Pending pool is the engine;
gap-closing concluded structurally dead. Handed off via RESUME.md + project memory. Next: keep mining the
untriaged pending pool (lowest-points first, EARLY FEASIBILITY CHECK) until it mines out.

| 12 | 06-16 08:18 | 6539.71 | RESUME-session wave: 154,107,4,392,383,55,343,202,340,310,222 (11 wins) | **6478.44** | **+10.35** | proj 6478.44 -> **0.00 error** (12th exact). gap 61.27 PINNED. New session from RESUME handoff; pending-pool engine. |

## ⭐ Submission 12: LB 6478.44 (proj 6478.44, 0.00 error). New session resumed from RESUME.md.
11-win pending-pool wave +10.35, all generalizing 1:1. Trajectory: ...->6468.09->6478.44. gap immovable 61.27.

| 13 | 06-16 16:14 | 6548.00 | wave: 289,86,62,284,308,153,59,397 (8 wins) | **6486.73** | **+8.29** | proj 6486.73 -> **0.00 error** (13th exact). gap 61.27 PINNED. salvaged-while-idle batch translated 1:1. |

## ⭐ Submission 13: LB 6486.73 (proj 6486.73, 0.00 error). Trajectory: ...->6478.44->6486.73.
13 exact projections running. Pending pool still productive but points creeping to ~15.1 (easy sub-15 consumed).

| 14 | 06-17 14:37 | 6553.85 | session-final: 31,75,22,268,234,218,131,298,93,13 (+ earlier wave) | **PENDING** | — | proj 6492.58. SESSION-FINAL LOCK. Verify score next session (Step 0). gap 61.27. |

## ⭐ Submission 14 (session-final, RESUME-session): LB 6492.58 CONFIRMED (proj 6492.58, exact).
Session: 6486.73->6492.58 proj. Pending pool now at ~15.4+ pts (easy sub-15 mined out); hit rate
dropping (~50% wins vs 85% early), more skip-marginal/at-floor. gap-closer hunt was a FALSE ALARM
(single-process generator pollution; see reports/gap_closers.md) - gap is structural, no big closers.

## ⭐ Submission 15 (RESUME-session cont.): LB 6501.03 CONFIRMED (proj 6501.04, 0.01 error; 15/15 near-exact).
11 wins: 371(+1.18),159(+0.54),125(+0.367),374(+0.76),68(+0.73),281(+0.47),260(+0.89),345(+0.32),
97(+0.87),124(+0.33),329(+2.04). Stored 6553.85->6562.31 (+8.46). Hit rate this wave ~92% (11 wins +
1 marginal/355). Floor ~15.6+; wins still landing well (329 was a +2.04 outlier). gap pinned 61.27.

## ⭐⭐⭐ Submission 16: LB 6524.54 CONFIRMED — GAP-CLOSER HYPOTHESIS PROVEN (+16.01 over proj 6508.53!).
GAP DROPPED 61.28 -> 45.27. task274 base net was genuinely fresh-0/real-0; the custom closed ~16 of gap.
The 15-submission "gap is fully structural" conclusion is OVERTURNED: there are recoverable non-generalizing
base nets BEYOND the 219/255/209 walls. Each fresh-0 recovery is worth its ~full stored as real LB (~16 here)
vs ~+0.5 for a normal pending-pool win. ACTION: isolated-process scan for other fresh-0 base nets = reopened
high-value reservoir. (DO NOT trust single-process scans - generator pollution false-alarms, see gap_closers.md.)
[original pending note:]
8 wins since #15 anchor: 391(stored+1.21 GAP-CLOSER),175(+0.94),301(+1.06),24(+0.82),303(+0.39),
49(+0.54),212(+0.35) [wait: 391 was prior submit]. THIS batch: 175,301,24,303,49,212 (1:1 generalizing)
+ task274 GAP-CLOSER. task274 base net had fresh-rate 0 / real 0.00 (inflated stored 16.00) -> custom
17.21 generalizing. lb_status anchor-arithmetic projects 6508.53 (treats 274 as +1.21 stored), but REAL
LB gain from 274 is ~+17.21 if the local fresh-check reflects the hidden test. If actual LB ~6524 (+16
over proj), the "gap is fully structural" conclusion is WRONG — there are recoverable non-generalizing
base nets beyond the curated 219/255/209 walls, a reopened reservoir worth hunting (isolated-process scan).

## ⭐⭐⭐ Submission 17: LB 6543.68 CONFIRMED (predicted 6543.7 EXACT) — GAP-CLOSER #2 validated.
GAP recalibrated 45.27 -> 28.96. CLEAN RESULT: remaining gap 28.96 == 219(15.00)+255(13.95) EXACTLY —
the two genuine info-bottleneck/connectivity WALLS. The two gap-closers 274(+16.01)+332(+16.31) closed
the ENTIRE recoverable portion (~32.3 pts). Gap is now TRULY structural (only the 2 walls left). NOTE:
23/2/209 base nets mostly PASS Kaggle (rare-failures, genverify binary over-flagged them) — NOT real gap.
Session: 6492.58 -> 6543.68 (+51.10). Gap-closer reservoir exhausted; back to pending-pool grind (~+0.4/win).
Since #16: 254(+1.5),40(+0.31),228(+0.33) [1:1] + task332 GAP-CLOSER (conv1x59+b base fresh-0/real-0 ->
custom 17.00, real +17). lb_status proj understates by ~16.32 (332 gap closure). Expect actual ~6543.7.
Gap-closer scan results: 332 WIN; 23/2/209 confirmed WALLS (gen-imports). Discriminator: non-gen
fresh-fail base nets are solvable gap-closers; gen-imports are mostly walls. See gap_closers memory.

## ⭐ Submission 18: LB 6551.52 CONFIRMED (proj 6551.52 EXACT, 16th). task105 NOT a gap-closer (base passes Kaggle). 13 pending-pool wins + task105 (possible gap-closer).
Wins since #17: 39,141,137,240,288,293,327,45,348,63,7,263,189 (+~7.84 stored, all 1:1) + task105
(adopt base real=0.00). WATCH: if actual LB ~6567 (proj+16), 105 was a HIDDEN gap-closer the genverify
n=40 missed -> more may exist, re-open hunt. If ~6551.5, 105 base is a rare-fail that passes Kaggle
(consistent with the post-#17 gap==219+255 result). Either way 105 custom adopted safely.

## ⭐ Submission 19: LB 6559.24 CONFIRMED (proj 6559.24 EXACT, 17th). 11-win pending-pool wave (all 1:1 generalizing).
Wins since #18: 43,259,65,398,100,357,166,388,248,190,335,246,199,252,273 (+7.68 stored). gap pinned
28.96 (==219+255 walls). Session total so far: 47 wins + 2 gap-closers (274,332). LB trajectory
6492.58 -> 6501 -> 6524 -> 6543 -> 6551 -> (proj 6559). Pending pool floor now ~16.3, wins ~+0.4 steady.

## ⭐ Submission 20: LB 6561.68 CONFIRMED (proj 6561.68 EXACT, 18th). 3 HAND-BUILT wins (subagent infra down).
Subagent streaming infra stalled 7 agents at 600s watchdog. Pivoted to building nets MYSELF in the
main loop (my own tools work fine): task60 (16.37->16.92 mirror-fill), task292 (16.32->17.18 recolor),
task78 (16.32->16.62 bar-stack) + task176 salvage. All fixed-size simple-recolor/fill (small active
region, no full-channel read = beatable by hand). Variable-size tasks (256/353/109) need a full in-grid
read -> too memory-heavy to beat the ~16.4 nets by hand. gap pinned 28.96.

## ⭐ Submission 21: LB 6569.81 CONFIRMED (proj 6569.81 EXACT, 19th). 16-win wave after infra recovery.
Subagent infra recovered (probes 109/10 cleared). Wave: 109,10,28,353,41,136,226,302,160,291,305,104,
323,130,123,316 (all generalizing, ~+0.4-0.9 each; 305/123/316 +0.66-0.91). Truly-untouched pending pool
(no prior custom) is the productive frontier now. gap pinned 28.96. Session: 6492.58 -> proj 6569.81 (+77).

## ⭐ Submission 22: LB 6578.01 CONFIRMED (proj 6578.01 EXACT, 20th). ~21-win wave (infra healthy, scaled to ~10 agents).
Incl task191 +2.24 (8-orientation dihedral template match, weak base 11.53). TARGETING LESSON: rank by
MANIFEST points not stale ledger. Lowest-manifest(11-13) gen-imports mostly WALLS (158 multi-object-scatter,
286 unbounded-flood, 133 multi-sprite-correspondence confirmed-infeasible) BUT 191 solvable +2.24. Sweet spot
= mid-manifest(14-16.5) solvable-bloated imports. Session: 6492.58 -> proj 6578.01 (+85.4). 85 wins adopted.

## ⭐ Submission 23: LB 6585.61 CONFIRMED (proj 6585.60, 0.01; 21st near-exact).
Mid-manifest wave: 253,192,198,138,396,325,338,89,9,182 (+191 in #22). Mid-manifest(14-16.5) bloated
imports are the productive frontier — big wins 191(+2.24),325(+1.55),253(+1.62),192(+1.18),338(+1.08).
SESSION TOTAL: confirmed 6492.58 -> proj 6585.60 (+93.0). ~97 wins adopted. gap pinned 28.96==219+255.
NEXT SESSION: poll to confirm #23, then continue mid-manifest sweep (rank by MANIFEST points, ~38 left).

- #24 (2026-06-18 15:40): **6586.75** (proj 6586.76, EXACT ±0.01). task017 adopt-gate said real=0.00 (FALSE-POSITIVE gap-closer) but Kaggle scored galaxy_v1 base ~full; gap UNCHANGED 28.96==219+255. LB +1.14 came from stragglers, not 017 (017 swap was -0.63 stored). LESSON: proj-exact after a "gap-closer" submit => it was NOT one; adopt real=0.00 disagrees with Kaggle held-out. Real gap-closers (274/332) made proj JUMP; 017 did not.

- #25 (2026-06-18 16:12): **6595.39** (proj 6595.39, EXACT). 14 golf wins on the 17.45-18.2 ext-import pool (389,207,299,229,152,142,52,235,211,72,3,267,214,399). Driver = uint8 whole-pipeline dtype lever (out>0 threshold makes output dtype irrelevant, ~halves planes, often zero-algo) + closed-form rewrites of import argmax/gather/template-match. task399 +2.12 (count->fixed-pattern, mem 102B). Gap unchanged 28.96==219+255 (structural).

- #26 (2026-06-18 18:40): **6594.85** (proj 6594.85, EXACT). GAP-CLOSER TEST of task151 (deployed conv3x3 gen:thbdh6332, adopt real=0.00 from {5,8} colour fresh-fail) -> FALSE POSITIVE: Kaggle scored the conv its full 18.19, LB==proj exactly, gap unchanged. REVERTED 151 to original conv. Net #26 was 6594.85 (-0.54 vs best 6595.39) because the 151 -2.77 swap outweighed the bundled golf wins; Kaggle keeps best so #25 standing preserved. LESSON: our-own-conv real=0.00 is NOT a reliable gap-closer signal (274/332 were real, 017+151 false); ONLY a post-submit LB JUMP above proj confirms a gap-closer.

- #27 (2026-06-18 18:49): **6602.46** (proj 6602.46, EXACT). NEW BEST (+7.07 vs #25 6595.39). task151 false-positive REVERTED + 7 golf wins (339/386/167/026/249/318/380, all 18.2-18.5 ext pool) translated 1:1. Driver levers: COUNT->FIXED-PATTERN (339 strip +1.04, 167 nc-select +0.70) + uint8 whole-pipeline + stacked-halves NOR. gap unchanged 28.96==219+255.

- #28 (2026-06-18 18:58): **6610.58** (proj 6610.58, EXACT). NEW BEST (+8.12 vs #27). 9 golf wins on 18.4-19.0 ext pool (155/236/334/347/150/129/395/6 + earlier), all 1:1. Levers: count->fixed-pattern, AND/NOR-of-stacked-halves, flip-via-Gather (side=sqrt pixelcount, neg-index-wrap clamp). gap 28.96==219+255.

- #29 (2026-06-18 19:22): **6618.58** (proj 6618.58, EXACT). NEW BEST (+8.00 vs #28). 11 golf wins on 18.7-19.8 tail (314 dilated-conv +1.61, 67 crop-scalar +1.50, 150 flip-Gather +1.44, 334/129/186/103 count->fixed-pattern, 56 classifier-fingerprint, 322 gravity-conv, 144/393 etc). gap 28.96==219+255.

- #30 (2026-06-18 19:46, session-final): **6620.24** (proj 6620.24, EXACT). +73 dwconv-height-trim, +149, +352 grouped-conv sub-floor escape. NEW BEST. Evening session total: 6586.75 -> 6620.24 = +33.49, ~49 golf wins (#24-#30, all proj-exact). gap 28.96==219+255 structural. Productive 14-19 ext-import golf pool now FULLY MINED to near-optimal (final 13/13 probe all at-floor).

## #31 — 2026-06-18 21:00 — CONFIRMED 6628.62 (proj 6628.61, off +0.01)
Re-probe wave: 14 false-positive blank-note wins bundled (42,270,143,350,74,387,369,51,247,148,102,356,50,237).
6620.24 -> 6628.62 = +8.38. KEY LESSON: "confirmed-infeasible"/"skip-marginal" ledger labels with a BLANK note
(no documented reason) are ~80% FALSE-POSITIVES — re-probe lowest-points / highest-bloat first. True walls
(279,277,361) and at-floor (64,48) all had DOCUMENTED or now-documented structural reasons.

## #32 — 2026-06-18 23:06 — CONFIRMED 6635.63 (proj 6636.66, -1.03)
Wave-2 re-probe (16 wins): 58,48,208,265,85,29,333,162,134,382,178,30,355,117,80,110. 6628.62 -> 6635.63 = +7.01.
NOTE: proj was +1.03 HIGH (first non-proj-exact this session) — one wave-2 net passes arc-gen-fresh 200/200 but
scores ~1pt less on Kaggle's exact held-out set (gap 28.95->29.98). Acceptable; Kaggle keeps best so no loss.

## #33 — 2026-06-19 — CONFIRMED 6644.18 (proj 6644.18, EXACT)
Re-probe wave (6 wins): 174 (+0.68 symmetry-MatMul), 185 (+1.23 line-lattice), 196 (+0.63 bounded-unroll-flood
8-conn 11-iter), 300 (+2.51 crop+translate task036), 201 (+1.74 spatial-copy+mirror), 363 (+1.75 template-as-Conv).
6635.63 -> 6644.18 = +8.55, PROJ-EXACT (clean 1:1, gap stable 29.98). Walls re-confirmed/documented this wave:
046 (per-segment data-dep roll), 319 (magnified-sprite correspondence), 118 (cross information-loss ~99.8%),
366 (template-matching ~92%), 187 (box-vs-line-pocket fill), 76 (rotated-sprite reveal, golfed exact solver).
KEY: blank-note "confirmed-infeasible" labels remain ~50% false-positive; bounded-iteration unrolling (HARD_WALLS
master key) cracked flood task 196. Session start LB 6635.63 -> 6644.18.

## #34 — 2026-06-19 — CONFIRMED 6658.65 (proj 6658.64, EXACT +0.01)
13 re-probe wins: 069(+1.02 4-conn plus-min label), 071(+0.80 mirror-complete), 379(+0.99 ray-stop-on-cyan),
280(+0.36 beam-as-rect), 251(+2.06 hole-fill bounded-unroll), 168(+1.24 diagonal-ray), 079(+0.36 most-copied-sprite),
170(+1.64 2-obj correspondence), 169(+1.38 component-count recolor), 364(+0.32 shape-classify flag-floods),
090(+1.62 maximal-empty-rect), 145(+0.31 guillotine-rect-area), 183(+2.38 GatherND tier-A). 6644.18 -> 6658.65 = +14.47.
⭐ KEY: gap-region "skip-list walls" 251 & 090 (blank-note) were FALSE-POSITIVES — cracked via HARD_WALLS §1
bounded-iteration unrolling (crop-to-gen-size-cap) and closed-form (suffix-min MaxPool). Rejected: 243 (self-reported
1000/1000 but failed isolated adopt-gate). Walls documented this session: 046/319/118/366/187/076. Session start LB 6635.63.
