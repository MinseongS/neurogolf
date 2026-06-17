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

## ⭐ Submission 14 (session-final, RESUME-session): proj LB 6492.58, PENDING at handoff.
Session: 6486.73->6492.58 proj. Pending pool now at ~15.4+ pts (easy sub-15 mined out); hit rate
dropping (~50% wins vs 85% early), more skip-marginal/at-floor. gap-closer hunt was a FALSE ALARM
(single-process generator pollution; see reports/gap_closers.md) - gap is structural, no big closers.
