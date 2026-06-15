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

## Procedure (folded into loop)
1. trigger: every 5 adopted wins.
2. `python -c "from src.pipeline import pack; pack()"` (networks/ only; never --pack flag).
3. `/opt/homebrew/Caskroom/miniconda/base/bin/kaggle competitions submit -c neurogolf-2026 -f submission/submission.zip -m "<msg>"`.
4. poll: `kaggle competitions submissions -c neurogolf-2026` until status COMPLETE; record publicScore.
5. compute stored→LB ratio for the batch (calibrates whether wins translate). Kaggle keeps BEST submission,
   so a flat/down result never loses standing — but a flat result means the wins didn't translate (re-examine).
