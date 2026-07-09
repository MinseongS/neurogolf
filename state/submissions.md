# Submission log (autonomous sweep — submit every 5 adopted wins)

Baseline before sweep: **LB 6384.61** (prev best, 2026-06-15 16:07 submit, 333/303/228 wave).
Stored at session start: 6445.88 (≈61pt stored-vs-LB gap pre-existing from overcounted base nets).

| # | time(UTC) | stored | wins since last | LB (publicScore) | Δ LB | notes |
|---|---|---|---|---|---|---|
| baseline | 06-15 16:07 | ~6384.6 | — | 6384.61 | — | pre-sweep |
| urad7174-remaining-overlay-7178.35 | 06-28 09:26 | 7178.2553 static | remaining URAD positive overlay after top15: 359/251/364/286/363/019/004/076/066/255/165/196/396/201/005/366/138/074/218/010/044/213/188/014 | **7178.35** | **+1.37 vs 7176.98** | submission 54135767 COMPLETE; public-teacher positive-only overlay; most candidates fresh-safe in 80-sample check, weak fresh noted for 363/076/255/005; source drafts and insight notes generated |
| urad7174-top15-overlay-7176.98 | 06-28 09:20 | 7176.8794 static | URAD 7174.10 teacher positive-only overlay: 023/002/209/157/349/219/018/205/161/198/277/202/335/182/233 | **7176.98** | **+4.55 vs 7172.43** | submission 54135642 COMPLETE; public-teacher overlay, not blind full import; all 15 passed stored; 9/15 passed fresh 120/120; source drafts and insight notes generated under `reports/public_teacher_*` |
| task017-source-probe-7172.43 | 06-28 09:10 | 7172.3327 static | source-owned task017 final-Equal/13-sample candidate; risky fresh 2998/3000 but probe-approved | **7172.43** | **+0.33 vs 7172.10** | submission 54135366 COMPLETE; non-public original source-owned change; public-probe id `t017_source_ahead_15631`; adopted after LB validation |
| equiv-compress-7170.59 | 06-27 18:47 | 7170.4880 static | onnxsim equivalent compression sweep over 11 tasks | **7170.59** | **+0.10 vs 7170.49** | submission 54118286 COMPLETE; non-public original compression; all adopted candidates matched current outputs on stored + >=503 fresh samples; biggest gains task379 +0.0403, task340 +0.0306, task080 +0.0124, task250 +0.0062 |
| task118-onnxsim-7170.49 | 06-27 16:16 | 7170.3931 static | task118 semantic-equivalent onnxsim param reduction | **7170.49** | **+0.04 vs 7170.45** | submission 54114451 COMPLETE; non-public original compression; base vs simplified task118 outputs matched 1000/1000 fresh; params 4788→3387, local task118 +0.0399 |
| task187-label-7170.45 | 06-27 08:38 | 7170.3532 static | task187 label-map flood-fill rewrite | **7170.45** | **+0.09 vs 7170.36** | submission 54102185 COMPLETE; non-public original change; replaced full 10-channel uint8 output construction with single-label flood map + final BOOL Equal; local task187 +0.0936 |
| task118-tail-7170.36 | 06-26 09:45 | 7170.2596 static | task118 tail output materialization optimization | **7170.36** | **+0.17 vs 7170.19** | submission 54074109 COMPLETE; zip SHA `69716db8...`; non-public original change; replaced task118 ScatterElements channel-update tail with direct `Where(mask, cyan_onehot, input) -> output`; previous and patched task118 outputs matched on official + 1000 fresh cases |
| custom-recover-7170.19 | 06-26 07:11 | 7170.09 static | recovered 15 fresh-gated repo custom nets over public base | **7170.19** | **+3.51 vs 7166.68** | submission 54070075 COMPLETE; zip SHA `02382f68...`; changed 009/055/080/128/174/191/202/204/222/250/338/340/379/383/398; all changed nets fresh-verified 300–500/500 |
| public-7166.68 | 06-26 02:51 | 7166.58 static | franksunp/7166.68 public rewire Mark I wholesale | **7166.68** | **+0.03 vs 7166.65** | submission 54064157 COMPLETE; zip SHA `71674769...`; changed 018/044/054/066/076/096/101/157/233/255/280/319 vs repo |
| public-7166 | 06-26 00:46 | 7166.56 static | franksunp/7166.65 public rewire Mark I wholesale | **7166.65** | **+7.18 vs 7159.47** | submission 54061465 COMPLETE; zip SHA matches public artifact `67c695f...`; local mismatch 5 tasks, keep public raw |
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

## #35 — 2026-06-19 — CONFIRMED 6661.38 (proj 6661.39, EXACT -0.01)
6 skip-marginal floor-breaks (new levers overturned "at-floor" verdicts): 365(+0.65 global-argmax->two-forward-
prefix-scans), 194(+0.47 GridSample->invert-to-source-index-Gather), 032(+0.63 colour-0==bg + crop-conv-on-free-
input), 330(+0.66 ScatterND-histogram per-component), 108(+0.12 ConvTranspose upscale), 115(+0.07 centroid-rank),
349(+0.08 fused variable-radius dilation), 381(+0.05 static-row-mask). 6658.65 -> 6661.38 = +2.73.
SESSION TOTAL 2026-06-19: 6635.63 -> 6661.38 = +25.75 confirmed, 27 wins / 8 walls / 1 reject, 3 submissions all
proj-exact. PRODUCTIVE RESERVOIR EXHAUSTED (blank-note + skip-list + skip-marginal + gap-closer all mined). Remaining
= documented tight floors (18.19 mem-0-conv cluster, fixed-crop fp32-slice floors, 19.09+ near-optimal, 21.6 do-not-resweep).
- 2026-06-19 #36: 6662.12 (+0.74 vs #35 6661.38) — 3 deep-custom wall wins 243/096/367; proj 6662.11 exact
- 2026-06-19 #37: 6667.42 (+5.30 vs #36 6662.12) — 11-win structural plane-free re-golf wave; 306 +1.25, 355 +1.26; proj-exact
- 2026-06-19 7k-HARVEST: 7107.01 (+439.59 vs #37 6667.42) — adopted 313 fresh-verified sajayr/neurogolf-7k public nets (keep-best + fresh-200 gate, 4 rejected non-general); ABOVE sajayr LB 7015 by ~92 (kept our better models on tasks we won). 400/400 solved.
- 2026-06-19 kojimar-7113.80: 7113.80 (+6.79 vs our 7107.01) — submitted public kojimar audited blend (neurogolf-7113-80-minimal-onnx-assets-v1) AS-IS. subB pure-sajayr=7092.28 (WORSE than our keep-best, confirming our fresh-gate+keep-best is sound). New best=7113.80. NEXT: overlay our fresh-verified wins (merge_E) to exceed.
- 2026-06-19 merge_E: 7121.23 (+7.43 vs pure-kojimar 7113.80, ABOVE public crowd) — kojimar audited base + 14 fresh-verified our-overlays + 5 fallbacks. New best=7121.23.
- 2026-06-19 merge_E v2: 7121.00 (−0.23 vs 7121.23) — kojimar 7113.80 base + 9 NEW marginal re-golf overlays (incl leaky 017) + 14 original. MEASURED LB-NEGATIVE.
- 2026-06-19 clean-8: 7121.00 (017 reverted, 8 clean B-type overlays) — IDENTICAL to v2 ⇒ regression was NOT the leak; the 8 clean overlays themselves cost −0.23 (arc-gen≠private). Overlay re-golf = DEAD lever.
- 2026-06-19 REBASE 7121.60 (+0.37, NEW BEST): new public kojimar 7114.66 blend base + the 14 PROVEN original overlays (dropped today's dead 9). Base improvement transfers to LB (crowd nets). New-blend lever ALIVE.

## #NEW-BEST 2026-06-21 — CONFIRMED 7127.10 (REBASE kokinnwakashuu-7125.30 + 6 EXACT overlays)
User flagged public notebook kaggle.com/code/kokinnwakashuu/7125-30-lb-neurogolf-audit-trail (LB 7125.30).
The .ipynb EMBEDS the full 400-onnx submission.zip as base64 (sha256 6c3c21..., 518487 bytes, ref 53912538).
Decoded + verified (sha match). Base = Ricardo [7120 LB] + 11 Frank-7116.79 overrides (364,338,366,255,191,
349,080,187,174,350,050). Scored it vs our manifest: 64 tasks where ours beats base on examples, but the
ext:kojimar7113 ones are the arc-gen≠private LB-DEAD trap (Ricardo 7120 base is higher-LB than kojimar 7113).
Overlaid ONLY our 6 proven-EXACT closed-form wins where ours strictly beats base AND smaller-mem:
396(14.72->15.29)/174(15.77->16.12)/340(15.69->16.00)/222(15.62->15.92)/377(15.69->15.93)/364(14.58->14.61).
local +1.81 -> LB +1.80 (7125.30->7127.10). THIRD ~1:1 confirmation: exact closed-form overlays are
base-independent & stack on ANY public base. +3.67 over prior 7123.43. Repeatable lever intact.

## #NEW-BEST 2026-06-21 — CONFIRMED 7133.77 (12 MORE EXACT-audited overlays via parallel code-audit)
Pursuing "mine MORE exact overlays" on the kokinnwakashuu 7125.30 base. Found 13 custom solvers beating the
base on examples (+7.03 potential). KEY METHOD UPGRADE: dispatched 13 parallel agents to CODE-AUDIT each for
PROVABLE exactness over the FULL generator input space (passing 3000-fresh is NOT proof of exactness — the old
dead overlays 191/009/251/278/383 passed fresh yet leaked private via edge-case bugs). Result: 12 EXACT (all
recommended overlay), 1 RISKY DROPPED (task205: ~0.0075% structural box-detection failure when noise forms
spurious boxcolor runs — exactly the leak class the audit targets). Overlaid 6 prior + 12 new = 18 EXACT nets.
NEW exact set: 250(+1.16)/055(+1.09)/080(+1.01)/355(+0.72)/352(+0.68)/202(+0.62)/128(+0.47)/398(+0.38)/
267(+0.24)/338(+0.17)/215(+0.11)/349(+0.02). local +8.48 over base -> LB +8.47 (7125.30->7133.77, vs prior
7127.10 = +6.67). FOURTH ~1:1 confirmation. Code-audit exactness gate is the durable discriminator; ~390 tasks
remain as a mining field. Silver ~7150 = +16.2 away.

## #NEW-BEST 2026-06-22 — CONFIRMED 7134.40 (+4 more EXACT overlays; build-from-scratch batch = walls)
Continued mining: 2 WIP solvers audited (204/379 both EXACT) + 6 build-from-scratch agents on bloated non-wall
targets (191/158/243/367/198/324). RESULTS: 4 EXACT wins overlaid = 191(+0.33)/379(+0.12)/204(+0.10)/324(+0.07);
4 NO_GO = 158 (12-template exact-cover, 4.6x base mem), 243 (flood-fill needs ~323 unrolled rounds), 367 (true
input->output collision, base only passes via non-exact approx), 198 (exact but ~13 full-canvas planes = mem-floored).
local +9.10 over base -> LB 7134.40 (+0.63 over 7133.77), FIFTH ~1:1 confirmation. ⭐ task191 FINDING: the
historically-'dead' 191 overlay was MISATTRIBUTED -- provably exact (94k differential cases, touches-all-4-edges
invariant kills the suspected leak); the old -0.23 was a bundled batch, never isolated to 191. ⛔ BUILD-FROM-SCRATCH
LESSON: bloated low-score base nets are bloated because the task is HARD (wall/memory-floored), not easy -- 0/6 new
builds. Cheap EXACT overlay vein now EXHAUSTED (22 nets deployed). Further gains need NEW algorithms or a newer
public base. Total deployed exact set (22): 396,174,340,222,377,364,250,055,080,355,352,202,128,398,267,338,215,349,191,379,204,324.

## 2026-06-27 negative probe — 220/230/294 local-exact replacements LOWER hidden/public
Tried aggressive local-exact replacements for three apparent 0-score slots from a misleading single-process
bulk eval: task220 fixed 3x3 halo Conv (mem 0, 18.1976, fresh 1000/1000), task230 2x2 gray-block satellite
stamper (v1 15.1857, v2 15.8629, v3 16.2756, fresh up to 5000/5000), task294 rectangle-interior recolor Conv
(mem 0, 18.1866, fresh 5000/5000). Submissions:
- `54108392` task220+230v1+294 → **7167.44**
- `54108431` task220+230v2+294 → **7168.12**
- `54108457` task220+230v3+294 → **7168.53**
All are BELOW best **7170.45**. Reverted `networks/task220.onnx`, `task230.onnx`, `task294.onnx` to HEAD.
Lesson: do not trust a one-process 400-task eval for mem-0/simple Conv models; verify suspicious slots with
`python -m src.harness networks/taskNNN.onnx NNN` in a fresh process. These three current public/base nets already
score strongly on hidden despite local exact replacements looking clean; do NOT reattempt as a gap-closer.

## #NEW-BEST 2026-06-28 — CONFIRMED 7171.00 (original QLinearMatMul dtype rewrite)
Built a non-public, structural improvement on live `custom:task055`. The old solver used fp16
one-hot `MatMul @ LUT @ MatMul` and carried three full 30x30 fp16 label planes. Rewrote the
same integer LUT selection with `QLinearMatMul` using uint8 one-hots/LUT and scale=1, zero-point=0.
This preserves exact labels and shrinks task055 from mem 8760 / params 46 / 15.9168 pts to
mem 5790 / params 48 / **16.3279 pts**. Fresh verification: **1000/1000**.

Submission `54127206` (`task055 qlinear uint8 matmul custom +0.411 local`) completed with
**publicScore 7171.00**, improving previous best **7170.59** by **+0.41**. This is a meaningful
original mechanism: for one-hot integer label LUTs feeding final `Equal(label, channel_ids)`,
try uint8 `QLinearMatMul` before assuming fp16 MatMul planes are the floor.

## #NEW-BEST 2026-06-28 — CONFIRMED 7171.25 (QLinearMatMul repeats on packed outer-product)
Applied the same uint8 quantized-matmul mechanism to live `custom:task340+onnxsim`. This task's
single packed outer-product MatMul assembles a small non-negative integer label plane, so fp16 was
not necessary. Rewrote A/B operands and `og` to uint8 `QLinearMatMul` (scale=1, zero-point=0),
then onnxsim reduced the final net to mem 5860 / params 279 / **16.2776 pts** versus previous
task340 16.0288. Verification: source 1000/1000 fresh, adopted sim 500/500 fresh.

Submission `54127287` (`task340 qlinear packed matmul custom +0.249 local`) completed with
**publicScore 7171.25**, improving previous best **7171.00** by **+0.25**. The repeated win upgrades
QLinearMatMul from one-off trick to a real search pattern for integer label assembly graphs.

## #NEW-BEST 2026-06-28 — CONFIRMED 7171.58 (QLinearMatMul band-count contraction)
Applied the same quantized-matmul mechanism to `task202`. The old graph used fp16 band contractions
to produce `obR/obC/ob`, but these values are small integer counts and only feed `Greater(ob, 0)`.
Changed `black`, `colblk`, `rowblk`, `obR`, `obC`, and `ob` to uint8 `QLinearMatMul` paths
(scale=1, zero-point=0), keeping the fp32 input slice/reductions only where unavoidable.
Result: task202 mem 18963 / params 24 / 15.1485 pts → mem 13563 / params 25 / **15.4831 pts**.
Fresh verification: **1000/1000**.

Submission `54127377` (`task202 qlinear uint8 band contraction +0.335 local`) completed with
**publicScore 7171.58**, improving previous best **7171.25** by **+0.33**. This is the third
consecutive successful QLinearMatMul floor break.

## #NEW-BEST 2026-06-28 — CONFIRMED 7171.91 (uint8 Gather + QLinear parity)
Continued the dtype floor-break scan:
- `task398`: value plane `V` from Gather was fp16 but only carries colour labels 0..9 or sentinel
  99 before final `Equal`. Changed `data6/vtab/V/ARANGE` to uint8; onnxsim result mem 1666 /
  params 1193 / **17.0418 pts** vs previous 16.7528. Fresh 1000/1000.
- `task338`: parity ray-cast count `Tl @ Hm` was fp16 and only feeds `Mod 2`. Changed the count
  MatMul to uint8 `QLinearMatMul`; onnxsim result **15.4142 pts** vs previous 15.3722. Fresh
  500/500.

Submission `54127652` completed with **publicScore 7171.91**, improving previous best **7171.58**
by **+0.33**.

## #NEW-BEST 2026-06-28 — CONFIRMED 7172.06 (QLinear clamp MatMuls)
Applied the same count-only QLinearMatMul conversion to `task250`: `Rmat @ gray @ CmatT` only
feeds `Greater(count, 0)`, so the fp16 clamp MatMuls were replaced with uint8 QLinearMatMul.
onnxsim result: mem 3164 / params 61 / **16.9213 pts** vs previous 16.7769. Fresh 500/500.

Submission `54127715` (`task250 qlinear clamp matmul +0.144 local`) completed with
**publicScore 7172.06**, improving previous best **7171.91** by **+0.15**. Total QLinear/uint8
series lift from 7170.59 is now **+1.47 public**.

## #NEW-BEST 2026-06-28 — CONFIRMED 7172.10 (QLinearConv bitmap count planes)
Applied the same dtype-floor logic to `task080`: three 10x10 bitmap count Conv planes
(`occ_cnt`, `edge_cnt`, `corner_cnt`) only feed equality/greater-than tests, so fp16 Conv
was replaced with uint8 `QLinearConv` using scale=1, zero-point=0. onnxsim result:
mem 10834 / params 454 / **15.6685 pts** vs previous 15.6252. Fresh verification: 300/300.

Submission `54127830` (`task080 qlinearconv bitmap counts +0.043 local`) completed with
**publicScore 7172.10**, improving previous best **7172.06** by **+0.04**. Total QLinear/uint8
series lift from 7170.59 is now **+1.51 public**.

## #NEW-BEST 2026-06-29 — CONFIRMED 7179.31 (task343 direct one-hot Gather output)

Applied the `direct_onehot_gather_output` semantic rewrite to `task343`.  The previous source
built a compact label grid and expanded it through `Equal(output_ids, channel_ids)` plus `Pad`.
The generator rule is a pure horizontal periodic remap, so the graph can instead concatenate
`chosen_cols = c mod period` with off-grid column `29` and make `Gather(input, final_cols,
axis=3)` the graph output.  This routes the original one-hot input directly into the free
output and deletes the counted one-hot rebuild path.

Local task343 result: `mem=1927, params=110, pts=17.380766` ->
`mem=1147, params=75, pts=17.891756`.  Fresh verification: **1000/1000** and **5000/5000**.
Stored total at submission: **7179.21**.

Submission `54172137` (`task343 direct onehot gather output local 7179.21`) completed with
**publicScore 7179.31**, improving previous best **7178.43** by **+0.88**.  This is a
source-owned mechanism, not a public ONNX import.

## #NEW-BEST 2026-06-30 — CONFIRMED 7182.09 (full-task improvement pass, after fixing uint8-TopK grader-killer)

Submitted the full source-owned improvement pass (33 modified tasks, local 7181.32). First
submission `54192655` and an identical resubmit `54192919` both returned **SubmissionStatus.ERROR**
(systematic, not transient — confirmed by submitting a known boristown ~7174 base which scored
cleanly at **7174.16**, proving the grader was healthy and the fault was in our zip).

Root cause isolated by bisection (replacing modified-task slots with boristown's grader-safe
networks, then binary-searching): **task208** errored alone; the trigger was the "safe-golf"
change that dropped the `Cast(uint8→fp16)` before `TopK`, feeding a **uint8 tensor directly into
TopK**. This passes local `onnxruntime==1.26` and even `onnx.checker.check_model(full_check=True)`
locally, but the **Kaggle grader raises on unsigned-int TopK input** → whole-submission ERROR
(uncaught, since `calculate_memory` calls `check_model` outside any try/except). A comprehensive
scan found 8 modified tasks with unsigned TopK input: **208, 285, 308, 316, 366, 368, 388, 397**.
Note `int64` TopK input is GRADER-SAFE (task366's HEAD `cnt_prio` INT64 TopK shipped in prior
working submissions); only **unsigned** dtypes are the killer.

Fix: reverted those 8 tasks' source to HEAD (float/fp16 TopK), rebuilt, reconciled. Cost was only
~0.24 local pts (the dropped-Cast golf gain on these 8 was negligible). Honest local total
**7181.08**.

Submission `54193588` (`FIXED: removed uint8-TopK grader-killer, local 7181.08`) completed with
**publicScore 7182.09**, improving previous best **7179.31** by **+2.78**.

Operational notes for next time:
- Kaggle CLI rejects the upload with `400 Bad Request` unless the file basename is exactly
  `submission.zip`.
- Before submitting a rebuild, scan all 400 nets for **unsigned-int TopK input** — it is invisible
  to local verification and errors the entire submission.

## #NEW-BEST 2026-07-01 — CONFIRMED 7182.46 (further improvement pass, after reverting a false-positive leak-fix)

The follow-up improvement pass (35 changed nets vs the 7182.09 set) as-is scored WORSE:
submission `54221614` (local bundled 7181.22) completed at **publicScore 7181.32** — DOWN 0.77
from best 7182.09, despite local going UP +0.14. Cause: local-measured memory != grader-measured
memory, and two "leak-audit fix" tasks dominated the delta.

Per-task grader-side diff (sanitize + ORT profiler trace + score_network, old-zip vs new) isolated
exactly two regressions; all other 33 changes were small positives (+0.001..+0.67):
- **task294: -1.137** — old net grader mem=0 (18.19pts). But 5000-fresh: **old=5000/5000 CORRECT**.
  The leak-audit "fix" was a FALSE POSITIVE; old was both higher-scoring AND fresh-safe. Its HEAD
  source `src/custom/task294.py` reproduces the mem=0 net source-owned. Reverted (git checkout HEAD).
- **task193: -2.826** — old grader mem=0 (18.19pts) but 5000-fresh: **old=4901/5000 (1.98% fail)** =
  REAL leak → old risks ~0 on private LB. New net = 1500/1500 correct, 15.36pts. User chose to KEEP
  the new (private-safe) version, accepting the public cost.

Final: reverted only task294 (+1.137, no private cost), kept task193 new. Local 7182.36.
Submission `54222245` completed with **publicScore 7182.46**, improving previous best **7182.09**
by **+0.37**, and this set is private-safer (task193 leak removed).

Lessons: (1) local mem reduction does NOT guarantee LB gain — verify grader-side memory (profiler
trace) on changed tasks before assuming a win; (2) leak-audit can false-positive — confirm the OLD
net actually fails fresh (5000+) BEFORE replacing it, since the replacement usually costs public LB.

## #NEW-BEST 2026-07-02 — 7186.61 (kojimar 7180.86 task-level positive overlay, fresh-gated)

Downloaded `kojimar/neurogolf-7180-86-minimal-onnx-blend-assets` (base_submission 400 + overrides 4),
ran `public_teacher_scan.py`: 26 base wins + 2 override wins vs our live nets (+7.5 gross).
Fresh-gated ALL candidates (1500 fresh; 5000 on known-leak families) with rule
"adopt only if candidate fresh-fail <= incumbent fresh-fail":

- ADOPTED 19 (+2.34 local): 152 35 336 277 397 92 165 237 218 260 31 (clean), 161 205 44 285 18
  (equal fail count, div=0 or equal rate), overrides 233(+0.363, 0 div!) 313 358.
- REJECTED on fresh: **193 (92/5000 fail — kojimar ships the leaky mem0 variant, as predicted)**,
  191 (12 vs 0), 17 (21 vs 2), 25 (6/5000 vs 0), 21 (5/432 vs 0), 2 (89 vs 80), 209 (174 vs 170),
  219 (kojimar worse than ours).
- task76 (+0.175) fresh-gate still running at session end (generator extremely slow) — pending.

Submission 54255339 (local 7186.74) **ERRORED**: full-400 pre-zip scan found uint8-TopK inputs in
task173(x3)/task208(x1)/task366(x5) — RE-INTRODUCED by earlier S6/S7 refit work that was never
submitted. My pre-scan only covered the 19 newly-adopted nets; the killer was in the OTHER 26
changed-since-last-submission nets. Reverted 173/208/366 to last-submitted nets via exact source
(cost -0.223; re-landable with float/int32 TopK feeds).

Submission 54255466 (local 7186.51) **COMPLETE publicScore 7186.61**, +4.15 vs previous best
7182.46. Delta = kojimar overlay +2.34 + previously-unsubmitted S6/S7 refits ~+1.8.

Lessons:
- The uint8-TopK scan must run on ALL 400 nets of the final zip (or at least all nets changed vs
  the last COMPLETE submission), never just the current session's additions.
- Grader-side per-task diff old-zip vs new (evaluate on both) predicted +4.36; actual +4.15
  (delta from the 3 reverts -0.22 happened after the diff). Prediction accuracy confirms
  evaluate() (ORT profiler) == grader memory model.
- opset 18/20/21 nets exist in prior COMPLETE submissions -> opset itself is NOT the grader-killer.

## #NEW-BEST 2026-07-02 — 7187.32 (walk-einsum mechanism PROVEN on LB)

Investigated (a) forum hints + (b) grader counting model per user direction.

(a) Forum (via Playwright agent): Fritz Cremer (#1) posted full per-bucket scores (~7580 at
6/15) → saved `reports/fritz_buckets.txt`; our gap (+394 vs 6/15-Fritz) is DIFFUSE: +4..+20
per 10-task bucket. Tony Li: 7600 via massive per-task LLM iteration, "not one simple idea".
Banned ops = Loop/Scan/NonZero/Unique/Script/Function/Compress (LSTM legal, unexplored).
Legit score-25 tasks exist (Deotte). 9th place: meta = "onnx tool profiling micro-interpretations".

(b) Read data/neurogolf_utils.py: only NODE OUTPUTS counted; op INTERNALS free; input/output
free; free >0 threshold. task313 (kojimar) = whole task in ONE 10-operand Einsum (input
repeated, tensor cores) = the precedent. Generalized to WALK-COUNTING: K flood steps = one
Einsum (see insight_registry `walk_einsum_iteration_collapse` + tasklog/task187.md).

task187 pilot: 14.580 (32850+665) → 15.290 (15300+1176), fresh 17≤19 incumbent, 20000-instance
numpy verification, 8-conn connectivity trap documented. Submission 54257728 (only task187
changed) **COMPLETE 7187.32 (+0.71 exactly as predicted)** → 74-operand Einsum grader-safe.

Next: fan out the template to 364/243/018/077/110/366/133/233/002/286 + directional-scan tasks.

## #NEW-BEST 2026-07-02 S8 — 7190.17 (walk-einsum fan-out wave 1)
- 54258484 COMPLETE **7187.53** (local 7187.43): task076 kojimar absorb (+0.175, fresh 24=24
  div0, exact-source) + task173 padded-coordinate TopK reorg (+0.034, bit-identical).
- 54258782 COMPLETE **7190.17** (local 7190.07, offset +0.1 exact): walk-einsum wave 1 —
  task110 +0.864 (period restoration → 3 einsums, gates-as-einsum-operands),
  task243 +0.918 (chained 46+47-slot 4-conn walk einsums, free-input traversability;
  FIXES incumbent's 0.04% deep-tail leak), task077 +0.854 (59-operand checkpoint-bbox einsum).
  All fresh-gated 2500+1500 fail 0 div 0. Grader accepts 59-operand einsums + einsum chains.
- Negative: task208 uint8-TopK re-landing = floor at safe dtypes (tasklog/task208.md S8);
  the old "+0.223 re-landable" estimate is retired.
- New pre-submit tool: reports/scripts/scan_unsigned_topk.py (all-400 unsigned TopK scan).

## #NEW-BEST 2026-07-02 S8 (cont) — 7191.92 → 7193.24
- 54259376 COMPLETE **7191.92**: task002 +1.296 (47-slot walk einsum, beats inc fresh 252≤256)
  + task018 +0.193 + task366 +0.143 (surgery; no-bool-Where/rank-0-init traps) + task133 +0.125.
- 54260115 COMPLETE **7193.24**: task286 +0.574 (4 chained walk einsums, inc fresh 20→0!)
  + task209 +0.349 (counting-model rebuild, div0) + task145 +0.230 (exact run-length einsum —
  multiplicity-free 3-phase walk, old FLOOR refuted) + task364 +0.159 (feature-seed reachability).
- task204 = genuine floor (parallel fan-out MaxPools ≠ chain; u8 conv banks einsum-proof).
- Day total so far: 7187.32 → 7193.24 (+5.92). Scanner reports/scripts/walk_einsum_scan.py
  drives the queue; blind-spot list (no repeat signature) pending different-lens audit.

## #NEW-BEST 2026-07-02 S8 (cont2) — 7196.37 (54261662 COMPLETE, local 7196.27)
Wave 4 (7 wins, +3.13): task219 +0.922 (batched-band placement einsum 'kjr,ks,jsc,k->rc';
"18k floor" REFUTED; cand-only-fail 0/20000) + task023 +0.618 (3-round unit-propagation golf +
residual majority rule, fresh fail 147→50) + task158 +0.524 (moment-statistics einsums: per-
colour n,Σr,Σc,Σr²,Σc² detection at O(1) bytes) + task066 +0.522 (free-input einsum plane
deletion, div0) + task054 +0.234 (sparse ScatterND chain, reduction=max idempotent unions) +
task233 +0.223 (single-Conv detector collapse, div0) + task349 +0.092 (conv-channel union).
DAY TOTAL: 7187.32 → 7196.37 (+9.05). Mechanism census: einsum-family ≈ +7.4, other counting-
model golf ≈ +1.6. Running: 018v2, 118, 191, 367 (FIXED_DELTA blind-spot wave).

## #NEW-BEST 2026-07-02 S8 (cont3) — 7197.62: EPILOGUE FOLD GRADER-PROVEN
- 54262620 COMPLETE **7197.09**: task191 +0.420 (dihedral product-of-sums einsum) + task118 +0.293.
- 54267065 COMPLETE **7197.62** (A/B B-leg): task187 epilogue fold +0.530 — whole net = Conv +
  ONE ellipsis-einsum (s-index rides the walk chain; signed mixer T[s,v,w]). **Ellipsis einsum
  + signed mixer CONFIRMED grader-safe** → mass-propagate to COPY class (285/025/044/017/074…)
  and every walk net still paying a label epilogue.
- Day: 7187.32 → 7197.62 (+10.30, 21 tasks rebuilt).

## #NEW-BEST 2026-07-02 S8 (cont4) — 7199.33 (54267540 COMPLETE, local 7199.23)
Wave 6: rect-recipe batch 4/4 (351 +0.487 free-input marker einsum, 280 +0.376 moment+Sqrt
closed form, 234 +0.307 profile bbox, 163 +0.266 scalar-einsum locate) + task101 +0.206
(+ fixes incumbent ORT-crash on 0.1% fresh — private-LB risk removed) + task367 +0.070.
Floors priced & logged: 025 (fold endpoint), 017 (NS=13 robust floor; cache-overfit warning
validated), 044 (fold measured neutral — 1:1 pad swap). Infra: fresh_cache RESHAPE bug fixed;
fresh_verify crash-tolerant. DAY: 7187.32 → 7199.33 (+12.01, 27 tasks rebuilt).

## #NEW-BEST 2026-07-02 S8 (cont5) — 7200.41: crossed 7200 (54267737 COMPLETE, local 7200.31)
Wave 7: task202 +0.784 (code_f/iszero/black planes DELETED — band colours via integer-Div
free-input einsums, orientation via Cauchy uniformity Σx²==(Σx)² einsum) + task064 +0.297
(per-row first/last via pow2-weight einsums + trunc(log2); u8 wraparound range fuse; NOTE:
count-profile ArgMax was WRONG for multi-dot rows — corrected mechanism).
DAY: 7187.32 → 7200.41 (+13.09, 29 tasks). Matrix sweep (recipe_matrix.json, est +8.9 over
25 tasks) running on 4 opus block agents.

## #NEW-BEST 2026-07-02 S8 (final) — 7201.18 (54268275 COMPLETE)
Wave 8 (matrix sweep): task251 +0.432 (walk einsum, 12×12 border flood) + task037 +0.280
(pow2-log extremes, ReduceSum==2 gate) + task089 +0.024 (chained-scatter fold) + task319
+0.022 / task205 +0.012 (reverse-ArgMax → select_last_index idiom). Wave 9 pending: task054
idiom +0.014 (local 7201.10).
MATRIX SWEEP VERDICT: 30 tasks examined → 7 wins (+1.22) / 23 priced floors (all logged in
tasklogs). Matrix est. +8.9 was ~7× optimistic: REPEAT_GROUP K-batching is BYTE-NEUTRAL
(grader sums elements — wins require ELIMINATING planes); REDUCE_ONLY converts only when
≤1 px per reduced line (pow2-exact) — occupancy/max-semiring reductions are floors.
DAY TOTAL: 7187.32 → 7201.18 (+13.86, 36 tasks rebuilt, 0 submission errors, 9 waves).
- 54268381 COMPLETE **7201.20** (wave 9 final, task054 idiom). S8 CLOSED: 7187.32 → 7201.20
  (+13.88, 36 tasks, 9/9 submissions clean).

## #NEW-BEST 2026-07-03 S9 wave 1 — **7206.71** (54270415 COMPLETE, local 7206.60)
Three engines this wave:
1. kojimar 7184.85 teacher sweep (+3.9 adopted / +1.1 rejected-by-fresh-gate): 108 +1.175
   (separable-remap einsum, mem=0 — playbook mech 14), 031 +0.761 (log-space bbox +
   ConvInteger), 029 +0.667 (fp16 GridSample crop + moment identity — old floor refuted),
   014 +0.626 (in-op Slice crop; fixes incumbent 0.1% bug), 021 +0.465 (REPAIRED height-cap
   bug), 303 +0.109 (fractional encoding), 155 +0.107 (Range swap; old ORT bug refuted).
   REJECTED by uncached fresh gate: 193 (2.36%), 017 (NS=9 1.27%), 191 (int8 0.92%),
   025 (K=4 undercount), 090 (15/2500). ⭐ overrides/ dir = the real teachers.
2. Native-crop sweep (verified generator bounds, reports/grid_crop_bounds.md): 243 +0.408
   (unified-passability letter-budget redesign), 187 +0.153 (in-einsum 25→30 index re-embed),
   192 +0.209 (never-materialize-30×30 QLinearConv), 193 +0.136, 138 +0.096, 222 +0.086,
   173 +0.082, 396 +0.147 (the S9 opener). REJECTED: 077 (crop backfires on free-input walk
   einsums — playbook reject-check added), 233/080/205 not croppable.
3. Micro: 018 fp16 recast +0.049, 216 einsum fold +0.053, 150 Range +0.135, 319/096/377.
Floors re-confirmed & logged: 158, 366, 133, 286, 338, 209, 233 (crop lens included).
DAY: 7201.20 → 7206.71 (+5.51, 21 tasks changed, submission clean).

## #NEW-BEST 2026-07-03 S9 wave 2 — **7208.43** (54270903 COMPLETE, local 7208.33)
Separable-remap einsum sweep (mechanism 14, from full-400 numpy scan
reports/separable_remap_scan.md): 152 +0.934, 142 +0.449, 211 +0.277, 083 +0.063 —
all mem=0 single 5-operand einsums. Scan projections were ~8× optimistic (output axis
must span full 30 → U tables [30,K]); 135/053/164/172/210/311 rejected (incumbent
Gather/Conv-pads already at/below the mech-14 floor). LSTM/GRU scout: DEAD, priced in
playbook. S9 CLOSED: 7201.20 → 7208.43 (+7.23, 25 tasks changed, 2/2 submissions clean).

## #NEW-BEST 2026-07-03 S10 — **7213.63** (54288054 COMPLETE, local 7213.52 isolated-corrected)
Daily public-teacher sweep: kojimar 7185.95 (overrides = real teachers) + NEW author
bobmyersthesecond 7186 (400-net dump; largely kojimar-lineage, several unique). 28 adopted:
1. Wave 1 (strict gate): 193 +2.556 (retrained single-Conv REVERSES S9 "inherent floor" —
   17.5k uncached fresh 0-fail), 021 +0.819 (rank-1 outer-product free output), 106 +0.328
   (Slice/Conv read → input-contracted einsum + [30,3] selector), 277 +0.047.
2. Wave 2 (⭐ RELAXED GATE, user policy: bundled=LB gate, fresh ≥98% → submit-verify):
   191 +0.456 (int8 QLinearConv dihedral template match, 0.95% fresh), 017 +0.294 (NS 13→9,
   1.10%), 025 +0.187 (slot_projector trim, 0.05%), 090 +0.006 (0.85%) — ALL S9/S8 rejects,
   all landed clean. Old strict fresh-gate was leaving ~1pt on the table.
3. Small batch: 188 +0.154 (separable-rect einsum emission), 173 +0.083 (0.4% fresh, relaxed),
   177 +0.054, 264 +0.113 / 184 +0.048 / 365 +0.024 (fp32 Conv→int8 QLinearConv where output
   feeds ranking only — ⭐ transferable), + 14 tiny trims ≈ +0.03. 157 skipped (UNGATEABLE).
Crop-bounds 400-scan: 163 flags, top-11 fan-out = 11/11 FLOOR (flagged planes are
free-output-axis welded; lever real only for counted entry reads — caveat in scan md).
🚨 Found: task220/230/294 single-Conv nets are 0.0-threshold knife-edge — batch local eval
under-counts ~54.6pt (ORT arena flip); grade in isolation. Hardening (epsilon bias) queued.
DAY: 7208.43 → 7213.63 (+5.20, 28 tasks changed, submission clean).

## #NEW-BEST 2026-07-03 S11 — **7214.42** (54295163 COMPLETE, local isolated 7214.32)
Family-first pivot session (user directive). New mechanism 15 (signed-channel priority
overlay: `(out>0.0)` per-channel grading ⇒ overlap priority is LINEAR via signed W —
no [30,30] label carrier) landed 092 +0.241; cohort sweep 6/6 KILL (233/285/370/133/054/366
— costs are detection/assignment/stamp, boundary recorded in playbook 15).
Global dtype value-range audit (dtype_overpay_scan, 326 nets, 0 err): recast seam mostly
harvested — landed 234 +0.172 (einsum-island fp16), 203 +0.061 (cnt16; u8 would overflow
on fresh). Carrier crossover sweep (Equal-then-Pad iff content<90): exactly 1 hit → 174
+0.189 (crossover −650B + fp16 subtrees, opset 13 bool Pad). User hand-landed 008 +0.136
(fresh 2000/2000 clean). int8 ranking-only QLinearConv on PRODUCER_BOUND: DRY WELL 0/32
(input quantization 9000B >> savings; 3 measured refutation builds). Floors confirmed by
build: 041 (crossover rule + PRODUCER_BOUND trap), 084 (single-free-output-writer
composition constraint), 162 (bundled train#2 violates generator guarantee — public-fatal).
294 knife-edge root cause = ORT 1.26 cross-session weight aliasing (task120 weights leak
into 294 session); initializer hardening impossible, PARKED. Teacher rescan: no new dumps.
DAY: 7213.63 → local 7214.32 (+0.80: 092/234/203/174 by loop, 008 by user).

## #NEW-BEST 2026-07-03 S12 — **7214.54** (54300052 COMPLETE, local 7214.44)
Single adoption: task370 runtime-parameterized stamp (NEW playbook mechanism 16,
S11 "does not exist" refuted): d pre-detection via 4 clamped GatherND probes →
parametric ScatterND kernel assembly (centered 31×31, direction baked into taps,
OOB clamped to trash cell) → ONE QLinearConv, replacing the 4-candidate dilated
bank + mux. 9669→8571 (+0.1205), bit-identical (fresh 5000 + orchestrator re-check
2000 vs networks incumbent), bundled 266/266.
Session S12 negative results (measured, logged): train-to-golf factory COMPREHENSIVELY
refuted (0/19 single-Conv + 0/2 hidden-C probes; k-locality ≠ linear separability;
LP infeasibility proofs 004/265/192 — reports/train_to_golf_report.md). Teacher
rescan kojimar 7186.82: max +0.006, skip. UNKNOWN-bucket dossiers ×12 (mostly real
floors; 3 fp16 lever suggestions all measured-refuted, Cast-boundary). fp16/dtype
S11 verdict re-confirmed.
DAY: 7214.42 → 7214.54 (+0.12, 1 task changed, submission clean).

## S14 (2026-07-05) — public-frontier merge: 7214.54 → 7232.24 (+17.70), NEW BEST
Submission 54360209, public 7232.24 (local isolated 7232.14, offset +0.10).
Source: lucifer19/tinyonnx-golf-forge notebook = base64 dump of a public 7221.43
submission (400 ONNX). Analysis: on OUR isolated grader their artifact = 7196.45
(BELOW our 7214.44); their public-LB edge was partly LB-leniency vs our strict
arc-gen gate (only task352 differs) + 89 genuinely-cheaper nets. Best-of-both:
keep our 311 better nets, adopt their 89 cheaper nets (ALL pass strict gate,
arc-gen 262 + train+test fail=0). Dominant technique = bilinear/shared-operand
Einsum vs FREE input: (1) reuse one low-rank factor N× in the equation (P×2,
R×2·C×2, m×4 — halves params vs distinct-but-equal matrices, exploits row=col /
encode=decode symmetry); (2) input appears TWICE (`input,input,...`) → pairwise/
auto-correlation features with ZERO counted intermediates; (3) collapse Conv/
QLinearConv/ConvTranspose/ScatterND/RoiAlign/GridSample → Einsum, killing big
activation planes (grader MEMORY term drops even when params rise, e.g. task325
mem 2010→518). Top gains: 292(+1.69),128(+0.79),142/152/83(+0.69),40(+0.68),
396(+0.67),146(+0.60),111(+0.54),254(+0.47),400/399(+0.42). Full list
reports/lucifer7221_adopt_targets.md. MEASUREMENT GOTCHA (logged): evaluating 400
nets in ONE process under-scores scattered nets (ORT profiler state leak) — batch
said 7141/7196; per-process isolated eval (matches Kaggle per-task scoring) = truth.
networks/ NOT yet formalized (89 source regens + tasklogs pending); live submission
built from submission/merged_nets/.

| 2026-07-09 04:14Z | public min-merge 20260709: 4 dumps (prvsiyan 7266.72 / ryosukeshiroshita 7266.48 / kutenk 7261.53 / jonathanncoletti merged91), margin-0, 78 adoptions +10.30 -> local 7289.59 | **LB 7289.71 확정 (sub 54481576, 신기록)** |
| 2026-07-09 05:26Z | public-insight generalize wave: 12 adoptions +3.14 -> local 7292.73 (294 rank1-qconv +0.35, 092 free-endpoint einsum +1.28, 064 axis-code +0.46, 366 equiv-golf +0.31, 009 maxpool-pack +0.21, 245 crop-decode +0.20, 086 perm-decode +0.14, 363/368 u8 +0.15, 216/008 value_info +0.05) |
| 2026-07-09 05:56Z | regime/public-insight follow-up: submitted current overlay set; post-status manifest local 7295.96 (task041/074/088/187/246/383 included; submit message said 7295.45 before status refresh) | **LB 7295.57 확정 (sub 54484027)** |

| 2026-07-09 06:03Z | regime vein batch7 (insight-arsenal fanout): 6/6 cracks +3.23 -> local 7295.96 (383 free-output 9-op einsum +0.81, 246 quadratic bands +0.61, 041 shared-triangular interval einsum mem0 +0.57, 088 canvas-cap ConvInteger +0.51, 187 epsilon-J walk +0.44, 074 ScatterND/GatherND shared-index +0.30). fail=0 topk clean |
| 2026-07-09 05:58Z | regime vein batch7 (insight-arsenal fanout): 6/6 cracks +3.23 (383/246/041/088/187/074) | **LB 7296.08 확정 (sub 54484248, 신기록)** |
