# Active 8000-mode submission (S17, 2026-07-06)

`submission/overfit_nets/` is the active 8000-mode artifact set. The upload archive must be named
repo-root `submission.zip`.

Gate = **bundled fail=0 + lower memory+params**. Fresh is diagnostic only unless the user explicitly
switches to safe/private mode.

## Current Active (2026-07-08 S43)
- **Latest confirmed displayed best: 7274.89**, submission **54463756**.
  Latest submitted active candidate: **54463952** (pending at write time).
  Local active after task205 residual coordinate-tail fp16 recast: **7274.919292**,
  manifest **400/400**, unsigned TopK clean, packed as repo-root `submission.zip`.
- Batch2 regime cracks already present in active: task260/task075/task240/task159/task345/task109
  (candidates in `reports/candidates/taskNNN/regime.onnx`; see tasklogs). A later min-merge against
  those candidate dirs found 0 remaining wins, confirming active already includes them.
- Public tail added after batch2 from `boristown/neurogolf-chatgptloop` 11:51:
  task020 `1257 -> 1256`, task030 `2145 -> 2144`, task076 `12856 -> 12825`,
  task133 `21322 -> 21278`, task189 `894 -> 892`, task281 `1340 -> 1329`.
  Submitted as **54461084**, completed at displayed **7271.88** (same as 54461060; local cost improved
  without a new displayed LB tick).
- Follow-up active adds task295 regime crack: `1604 -> 393`, full active local
  **7274.210520**, isolated-all **400/400**, unsigned TopK clean. Submitted as **54461590**;
  completed at displayed **7274.65**.
- Follow-up active adds task377 output-coupled fp16 recast: `4567 -> 3967`
  (memory `3552 -> 2952`, params `1015` unchanged), full active local
  **7274.668531**, manifest **400/400**, unsigned TopK clean. Candidate:
  `reports/candidates/task377/task377_fp16_output_coupled.onnx`; backup:
  `reports/candidates/task377/adopt_backup_727465_rescan/task377.onnx`.
  Submitted as **54462993**, completed at displayed publicScore **7274.79**, with message
  `active 7274.668531 task377 fp16 output-coupled tensors cost 4567->3967 after task295 rescan fail=0 topk clean`.
- Follow-up active adds task205 final-Einsum fp16 input recast: `4652 -> 4372`
  (memory `4606 -> 4326`, params `46` unchanged), full active local
  **7274.730608**, manifest **400/400**, unsigned TopK clean. Candidate:
  `reports/candidates/task205/task205_fp16_final_einsum_inputs.onnx`; backup:
  `reports/candidates/task205/adopt_backup_727479/task205.onnx`.
  Submitted as **54463492**, completed at displayed publicScore **7274.85**, with message
  `active 7274.730608 task205 final-einsum fp16 inputs cost 4652->4372 after task377 fail=0 topk clean`.
- Follow-up active adds task355 final-output fp16 recast: `538 -> 518`
  (memory `496 -> 476`, params `42` unchanged), full active local
  **7274.768491**, manifest **400/400**, unsigned TopK clean. Candidate:
  `reports/candidates/task355/task355_fp16_output.onnx`; backup:
  `reports/candidates/task355/adopt_backup_727485/task355.onnx`.
  Submitted as **54463756**, completed at displayed publicScore **7274.89**, with message
  `active 7274.768491 task355 fp16 final output cost 538->518 after task205 fail=0 topk clean`.
- Follow-up active adds task205 residual coordinate-tail fp16 recast: `4372 -> 3760`
  (memory `4326 -> 3714`, params `46` unchanged), full active local
  **7274.919292**, manifest **400/400**, unsigned TopK clean. Candidate:
  `reports/candidates/task205/task205_fp16_coord_tail.onnx`; backup:
  `reports/candidates/task205/adopt_backup_727485_residual/task205.onnx`.
  Submitted as **54463952** (pending at write time), with message
  `active 7274.919292 task205 residual coord-tail fp16 cost 4372->3760 after task355 fail=0 topk clean`.

## Current Active (2026-07-08 S38)
- **Latest confirmed displayed best: 7264.67**, submission **54458829**.
  Local active: **7264.550544**.
  Adopted a lucifer 10:21 public-tail overlay for task157 only: bundled fail=0,
  memory `6595 -> 6591`, params unchanged `256`, total active cost **1062063**,
  isolated-all manifest `400/400`, unsigned TopK clean, packed as repo-root `submission.zip`.
  Backup before adoption:
  `reports/candidates/public_mine_20260708/lucifer_safe_boost_1021/adopt_backup_54458146/task157.onnx`.
  Re-mine scope: `prvsiyan/neurogolf-7250-24-w-visualizations` had no downloadable ONNX output via
  `kaggle kernels output`; `lucifer19/chimera-safe-boost-caddies` and rerun
  `llccqq624/neurogolf-public-notebook` were scanned with `mine_overfit_minmerge --margin 0`.
  Only task157 was adoptable.

## Current Active (2026-07-08 S37)
- **Latest confirmed displayed best before S38: 7264.54**, submissions **54457603** and **54458146**.
  Submission **54457603** message:
  `public tail: franksunp 7250.24 min-merge +0.0131 local (7 tasks: 36/20/281/189/264/319/44)`.
  Submission **54458146** message:
  `active 7264.415617 zero-compare tail: task044/281/319 cost -11 after 54457603, fail=0, topk clean`.
- Local active after the zero-compare tail is **7264.415616949520**, total cost **1062443**,
  400/400 by changed-task bundled rechecks + manifest row update, unsigned TopK clean,
  packed as repo-root `submission.zip`.
- Previous confirmed best was **7264.53**, submission **54454741**, message
  `reducesum spatial to einsum tail +0.0104 local`.
  Local active after that tail was **7264.411366**, 400/400 by unchanged manifest + 8 changed
  bundled rechecks, unsigned TopK clean.
- Previous confirmed best was **7264.52**, submissions **54452657** and **54453501**.
  Submission **54452657** message:
  `public tail min-merge 7249.95/plus/lucifer: +0.216 local, 65 tasks, local 7264.399214, scan-clean`.
  Latest submitted active after the 7249.98 refresh is **54453501**, also displayed as
  **7264.52**, with local active **7264.400942**, 400/400 bundled ok, unsigned TopK clean,
  packed as repo-root `submission.zip`.
- Source public dumps:
  `franksunp/7249-95-lb-compact-onnx-artifact-starter`,
  `llccqq624/neurogolf-plus-compact-task-union`, and
  `lucifer19/chimera-safe-boost-caddies`.
- Adopted 65 margin-0 public tail overlays. Largest local tails: task291 `+0.0299`,
  task189 `+0.0188`, task334 `+0.0153`, task342 `+0.0111`, task267 `+0.0105`,
  task010 `+0.0099`, task329 `+0.0085`, task354 `+0.0072`.
  Full report: `reports/candidates/public_tail_20260708_724995/ADOPTION_REPORT.md`.
- Backup before adoption:
  `reports/candidates/public_tail_20260708_724995/pre_tail_active_backup/`.
- Tooling note: `reports/scripts/build_overfit_manifest.py --isolated-all` was used because the
  long-lived ORT process can stall on pre-existing task376; task376 was not changed.
- Follow-up 7249.98 refresh adopted task268 and task019 for local `+0.001728`; see
  `reports/candidates/public_tail_20260708_724998/ADOPTION_REPORT.md`.
- Follow-up public-mechanism generalization adopted ReduceSum-spatial-to-Einsum overlays for
  task018/037/203/239/324/378/388/396 for local `+0.010424`; see
  `reports/candidates/reducesum_spatial_to_einsum_probe.py` and
  `reports/insight_registry.yaml` entry `spatial_reducesum_to_einsum_profile_tail`.

## Current Active (2026-07-08 S35)
- **Latest confirmed best: 7264.30**, submission **54451991**, message
  `salvage no-task101 tail: task044/157/035/091/268 local 7264.182707`.
  Local active after rebuild: **7264.182707**, 400/400 bundled ok, unsigned TopK scan clean,
  packed as repo-root `submission.zip`.
- Adopted tail over S34: task044 zero-compare bool-cast `4738 -> 4737`, task157 dynamic CSE
  `6852 -> 6851`, task035 dynamic CSE `1909 -> 1895`, task091 dynamic CSE `2898 -> 2894`,
  task268 dynamic CSE `3017 -> 3013`.
- Rejected: task101 dynamic CSE `13725 -> 13721` was Kaggle-falsified. Bundled-only micro-tail
  submission **54451532**:
  `active 7264.172930 micro tail dynamic-cse task101/157 + bool-cast task044`,
  completed at publicScore **7248.82**. Full dynamic-CSE sweep submission **54451744**:
  `active 7264.182998 dynamic-cse full sweep + bool-cast micro tail`,
  completed at publicScore **7248.83**.
- Backup dirs:
  `submission/overfit_nets/.micro_tail_backup_20260708/` and
  `submission/overfit_nets/.dynamic_cse_tail_backup_20260708/`.

## Current Active (2026-07-08 S34)
- **Latest confirmed best: 7264.29**, submission **54451017**, message
  `active 7264.170 public-optimizer dedupe task035/task340 (+0.0317 local) scan-clean`.
  Local active after rebuild: **7264.169951**, 400/400 bundled ok, unsigned TopK scan clean,
  packed as repo-root `submission.zip`.
- Delta over 54450447: task035 duplicate-initializer dedupe `cost 1970 -> 1909`
  and task340 `3405 -> 3404`; total local `+0.031748`. Backups are in
  `submission/overfit_nets/.dedupe_public_optimizer_backup/`.
- Previous confirmed best: **7264.26**, submission **54450447**, public re-mine of
  26 tasks from franksunp `7249-50-lb-compact-onnx-artifact-starter` and urad
  `best-7250-18`; local predicted `+4.4981`, realized `+4.51`.

## Current Active (2026-07-08 S30)
- **Local active = 7258.916667**, 400/400 bundled ok, unsigned TopK scan clean,
  packed as repo-root `submission.zip` from `submission/overfit_nets/`.
- Latest valid Kaggle submission: **54424285**, message
  `compiler MVP: task197 1201->904(+0.284) + task19 2793->2507(+0.108) + task324 -30par; local ~7254.77`,
  publicScore **7255.91**.
- Latest completed resubmit of the S27 scan-clean active: **54424480**, message
  `active 7255.802 compiler wins 197 019 324 scan-clean`, publicScore **7255.91**.
- Latest S28 submit: **54425941**, message
  `active 7255.812 task349 hpos w29 crop scan-clean`, publicScore **7255.92**.
- Pending S29 submit: **54426721**, message
  `active 7255.817 zero-compare cast peephole 11 tasks scan-clean`.
- Latest S30 submit: **54449124**, message
  `active 7258.917 task077 branch-einsum fold scan-clean`, publicScore **7259.02**.
- S27 active local deltas include task197 cost 1201 -> 904, task019 cost 2507 -> 2505,
  task324 params -30, plus the prior task354/208/148 dedupe wave.
- S28 active local delta: task349 h_pos QLinearConv width 30 -> 29 by producer pad crop
  and halo-consumer pad compensation, cost 15042 -> 14892 (+0.010022 local points).
- S29 active local deltas: zero-compare-to-bool-cast peephole on tasks
  009/044/077/134/224/233/234/308/355/363/397, total cost -11
  (+0.004843 local points).
- S30 active local delta: task077 branch-Einsum copy/edit epilogue fold,
  replacing `walk -> Greater -> Where(fill, yellow, input)` with a single
  `Einsum -> output`.  Bundled fail=0, task077 memory 0 params 2160.
  Current total includes prior local sweep state and lands at 7258.916667.
- S31 active local delta: byte-identical initializer/unused-initializer sweep
  adopted for tasks 138/285/390.  Bundled fail=0, unsigned TopK scan clean,
  `submission.zip` repacked.  Current local active total = 7258.917616.
- Rejected compiler oracle: signed INT8 TopK feeds were local-valid and scan-clean
  but Kaggle-rejected for the full bundle (**54418239**), two half groups
  (**54418729**, **54418747**), and all 11 single-task oracle submissions
  (**54418836**, **54418838**, **54418840**, **54418844**, **54418846**,
  **54418849**, **54418852**, **54418856**, **54418858**, **54418860**,
  **54418861**). Treat signed INT8 TopK as a hard negative filter.
- Rejected compiler oracle: task233 `ArgMax(uint8)` replacement for binary
  match-matrix TopK was local bundled fail=0 and scan-clean, but Kaggle
  submission **54423143** completed at publicScore **7240.45** (task233
  effectively zero). Treat ArgMax(uint8) binary-match replacement as a negative
  filter unless future Kaggle oracle disproves it.

## Confirmed LB (2026-07-06)
- **S23 local active = 7254.182586** (2026-07-07, scan-clean remerge pending):
  after S22, reran public min-merge against the scan-clean base and recovered
  task076 (+0.0051) and task285 (+0.0013) from boristown's clean public nets.
  Current active is 400/400 bundled fail=0, unsigned-TopK-clean, and packed as
  `submission.zip`.
- **S22 local active = 7254.176246** (2026-07-07, LB 7254.28):
  the S21 uint8-TopK local win was confirmed as a Kaggle submission ERROR, so
  task018/076/233/285 were reversed to fp16 TopK feeds.  Current active is
  400/400 bundled fail=0, `scan_unsigned_topk.py` clean, and `submission.zip`
  packed/submitted as `scan-clean active 7254.176 public-minmerge topk-fp16`.
- **S21 local active = 7254.378117** (2026-07-07, Kaggle ERROR due unsigned TopK): pulled fresh public
  outputs `uradkr/best-7242-98-score-urad-notebook-with-explain`,
  `hoangvux/neurogolf`, and `boristown/neurogolf-chatgptloop`; `mine_overfit_minmerge.py`
  adopted 32 bundled-fail=0 cheaper overlays for +0.3974 local points.  Largest wins:
  task377 +0.0944, task384 +0.0566, task355 +0.0542, task338 +0.0442, task205 +0.0381.
  `submission.zip` rebuilt.  `scan_unsigned_topk.py` reported pre-existing offenders
  task018/076/233/285; the S21 overlays did not add new offenders.
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
