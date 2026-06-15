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

## Procedure (folded into loop)
1. trigger: every 5 adopted wins.
2. `python -c "from src.pipeline import pack; pack()"` (networks/ only; never --pack flag).
3. `/opt/homebrew/Caskroom/miniconda/base/bin/kaggle competitions submit -c neurogolf-2026 -f submission/submission.zip -m "<msg>"`.
4. poll: `kaggle competitions submissions -c neurogolf-2026` until status COMPLETE; record publicScore.
5. compute stored→LB ratio for the batch (calibrates whether wins translate). Kaggle keeps BEST submission,
   so a flat/down result never loses standing — but a flat result means the wins didn't translate (re-examine).
