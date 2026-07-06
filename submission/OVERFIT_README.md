# Overfit submission (S17, 2026-07-06) — PUBLIC-LB ONLY, private-unsafe

`overfit_submission.zip` (400 nets in `submission/overfit_nets/`) = safe base + FULL-OVERFIT overlays.
Gate = **bundled fail=0 only** (fresh IGNORED, per user directive "리더보드만 넘기면 돼").

## Confirmed LB (2026-07-06)
- **Overfit v2 = 7248.76 public** (sub 54396589). Min-merge of {S16 overfit base, udit 7237.17, poby 7235.83}
  under bundled fail=0. Base overfit v1 = 7246.88 (sub 54396185); +1.886 from udit overlays.
- Safe hedge = **7245.33** (sub 54396297, fresh-gated, private-robust). Keep selectable at deadline.

## v2 overlays over the safe base (the paying ones)
- task206 3766->1795 (udit, +0.741) — ALSO adopted into SAFE tree (clean fresh-gate).
- task377 8209->5019 (udit, +0.492) — OVERFIT ONLY (0.1% fresh-fail vs clean incumbent, rejected for safe).
- task205 9982->6467 (udit, +0.434) — OVERFIT ONLY (17/1500 fresh vs inc 4/1500, rejected for safe).
- task076 15932->13235 (udit, +0.185) — OVERFIT ONLY (giant slow net, unverifiable fresh).
- + public leaky traps from S16 (319 poby 5852, 048, 188, 285, 101, 117, 018, 090) + own-net 286/233 cuts.
- + tiny udit/poby tail (215/019/044/... each ≤0.012).

## 🚨 SUBMISSION NAMING (critical, learned S17)
Kaggle REQUIRES the file be named exactly `submission.zip`. Any other name (e.g. `overfit_submission.zip`)
=> `400 Bad Request: "Submission files must be named 'submission.zip'"`. The old "daily limit is 5" memory
was a WRONG-CAUSE MYTH from this. **Real limit = 100/day** (Kaggle API `max_daily_submissions=100`).
To submit: copy the batch to a temp file named `submission.zip` first, then
`kaggle competitions submit -c neurogolf-2026 -f <tmp>/submission.zip -m "..."`.

## Rebuild
Min-merge script: `scratchpad/mine07/minmerge.py` (per-task cheapest bundled-fail=0 across sources).
Re-scan before submit: `PYTHONPATH=. .venv/bin/python reports/scripts/scan_unsigned_topk.py submission/overfit_nets`.

## Overfit ceiling ≈ 7249 (measured)
The lever is public leaky traps + a handful of udit/poby cheaper own-nets. 8000 needs structural mechanisms
we lack. task066 (5-Einsum) probed S17 = NOT a walk chain (5 independent contractions), structural floor.
