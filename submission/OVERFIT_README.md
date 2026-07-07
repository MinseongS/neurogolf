# Active 8000-mode submission (S17, 2026-07-06)

`submission/overfit_nets/` is the active 8000-mode artifact set. The upload archive must be named
repo-root `submission.zip`.

Gate = **bundled fail=0 + lower memory+params**. Fresh is diagnostic only unless the user explicitly
switches to safe/private mode.

## Confirmed LB (2026-07-06)
- **S18b = urad 7242.52 min-merge** (exp ~7253.6): 6 tasks where urad's public net is cheaper AND
  passes our bundled fail=0 (isolated): task355 2678->526 (+1.56), task264 4700->968 (+0.83),
  task197 2586->1160 (+0.77), task222 5096->1742 (+0.66), task236 208->96 (+0.63), task383 (+0.005).
  franksunp/poby 7240.26, jonathan 7242, seddiktrk surgeries = NOTHING cheaper (converged). Backups in scratch_mine/backup_orig.
- **S18 = 7249.18** (sub 54399767). +task243 walk-chain truncation (drop W2 slack, 5112->3816, +0.243) over the 7248.94 base.
- **Overfit v2 = 7248.76 public** (sub 54396589). Min-merge of {S16 overfit base, udit 7237.17, poby 7235.83}
  under bundled fail=0. Base overfit v1 = 7246.88 (sub 54396185); +1.886 from udit overlays.
- Safe hedge = **7245.33** (sub 54396297, fresh-gated). This is an opt-in backup mode, not the default.

## v2 overlays over the source/safe base (the paying ones)
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
To submit: create the repo-root `submission.zip`, then
`kaggle competitions submit -c neurogolf-2026 -f submission.zip -m "..."`.

## Rebuild
Min-merge script: `reports/scripts/overfit_minmerge.py` (per-task cheapest bundled-fail=0 across sources).
Measure active set: `PYTHONPATH=. .venv/bin/python reports/scripts/build_overfit_manifest.py`.
Re-scan before submit: `PYTHONPATH=. .venv/bin/python reports/scripts/scan_unsigned_topk.py submission/overfit_nets`.
Pack upload: `PYTHONPATH=. .venv/bin/python reports/scripts/pack_overfit_submission.py`.

## Overfit ceiling ≈ 7249 (measured)
The lever is public leaky traps + a handful of udit/poby cheaper own-nets. 8000 needs structural mechanisms
we lack. task066 (5-Einsum) probed S17 = NOT a walk chain (5 independent contractions), structural floor.
