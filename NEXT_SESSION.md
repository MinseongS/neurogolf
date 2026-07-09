# NEXT SESSION — NeuroGolf handoff (2026-07-08)

## 🟢🟢🟢 2026-07-08 REGIME CRACK — the 900B output-mask floor is NOT a wall (5/5). BIGGEST lever found.
- **FINAL CONFIRMED BEST = 7279.41 (sub 54467261, LB verified)** — session ran 7264.67 → 7268.28 (b1+fp16)
  → 7271.88 (b2) → 7275.04 (b3) → 7279.17 (b4, incl task287 1994→250 +2.08) → 7279.34 (b5: task246) →
  7279.41 (task190) = **+14.74 LB**, almost all from regime cracks (~24 cracks + 4 fp16). **batch6 = DRY
  (7/8 floor + 071 broken) → the easy mask vein is HARVESTED.** Remaining path is the CONV-FP32 arsenal
  (unproven) + long tail + the mixed-dtype carrier lever — see reports/regime_vein_worklist.md. Model
  policy: opus handled all of b5/b6 (proven sub-recipes + rigorous floor proofs); Fable reserved. batch1 (5
  cracks) + batch2 (6, incl 2 tasks FLOOR-stamped earlier the SAME DAY, re-cracked once the lever existed)
  + batch3 (5). task112/163/099 = the vein boundary (positioned-content mask = floor).
- **Batch3 DONE: +5 cracks** (295 1604→393 +1.41, 398 2813→1510 +0.62, 392 1937→1310 +0.39, 051
  1744→943 +0.32, 061 1668→1617 +0.03) = +2.77; FLOORS 163+099 (both positioned-content, taxonomy
  confirmed). Cumulative regime = **16/18 cracks** across 3 batches. Submitted sub 54465059 (pending).
- **VEIN STATUS: ~45 mask-dominated candidates remain** in `mask_dominance.json` (16 done). Keep
  sweeping with the pre-filter: skip POSITIONED-CONTENT masks (arbitrary content at data-dependent
  position → floor), take global-state / structured (ring/block/periodic/diagonal/threshold-run/count).
  New batch-3 sub-recipes in the memory: input-as-Einsum-operand for free counts, base-N digit
  factorization, ConvInteger-as-free-output, k-stacked Where carriers, residue one-hots (Gather(eye)).
- **The North Star flipped from pessimism to a live vein.** The `[1,1,30,30] bool = 900B` output-welded
  routing mask — long believed an irreducible S10 floor — is ESCAPABLE: fold the `Where(mask,·,input)`
  routing into ONE N-ary Einsum to the FREE `output` (output>0 sign-decode; input/output free any dtype;
  off-grid auto-zeros via input-linearity). Mask lives in free op internals; only tiny inits + few-element
  dynamic state are counted. Full recipe + sub-patterns: memory `neurogolf-regime-crack-freeoutput-einsum`.
- **5/5 existence-proof cracks ADOPTED** (deployed-gated, bundled fail=0, fresh-gated, TopK-clean):
  **task329 1050→468 (+0.81), task303 1450→700 (+0.73), task141 1383→568 (+0.89), task033 1639→787
  (+0.73), task341 1427→1123 (+0.24)** = +3.40 LB. Candidates `reports/candidates/taskNNN/regime.onnx`.
- **⭐ THE 60-TASK VEIN:** `reports/candidates/fresh_sweep/mask_dominance_scan.py` → `mask_dominance.json`
  lists 60 deployed nets ≥30% dominated by a ≥20×20 routing plane. Each is a regime-crack candidate at
  +0.2..+0.9. **NEXT: sweep them** — one Fable-class agent per task (physics + Einsum arsenal in the
  memory), gate = bundled fail=0 + cost < DEPLOYED. Potential **+20 to +40 LB** — the first real path
  toward 8000 from our own lineage. Then generalize to non-mask bloat: Conv 3600B planes
  (074/080/198/383/187), QLinearConv (349).
- Also this session: fat-middle fp16/fusion sweep on DEPLOYED nets adopted 251/268/205/222 (+0.34), and
  the diagnostic below (documented open-angle backlog is STALE; scanners were pointed at stale networks/).
- ⚠️ ADOPTIONS NOT COMMITTED (working tree only); shared submission/overfit_nets/ with the parallel
  public-tail session — always re-check `kaggle competitions submissions -c neurogolf-2026` before submit.
  Backups: `reports/candidates/fatmid_adopt_backup/`.

## 2026-07-08 post-batch2 public tail (pending LB)
- After **54461060 = 7271.88** was submitted, re-mined the latest public outputs against
  `submission/overfit_nets` with:
  `PYTHONPATH=. uv run python -m reports.scripts.mine_overfit_minmerge --margin 0 ...`.
- Public mining result:
  - `lucifer19/chimera-safe-boost-caddies` 11:41: 0 wins.
  - `jackelysia/neurogolf-7250-25-v23-lucifer-unscored-carry` 11:39: task281 `1344 -> 1340`;
    submission **54460887** completed at **7270.82**.
  - `llccqq624/neurogolf-public-notebook` 10:59: 0 wins after downloading all 400 ONNX.
  - `biohack44/neurogolf-2026-championship-best-solution` 11:31 `_src_A` and `_src_B`: 0 wins.
  - `boristown/neurogolf-chatgptloop` 11:51: 6 wins, applied:
    task020 `1257 -> 1256`, task030 `2145 -> 2144`, task076 `12856 -> 12825`,
    task133 `21322 -> 21278`, task189 `894 -> 892`, task281 `1340 -> 1329`.
- Gate after the boristown six: isolated bundled fail=0 for all 6, full
  `build_overfit_manifest.py --isolated-all` => **400/400**, local **7270.997535**,
  `scan_unsigned_topk.py submission/overfit_nets` clean, `pack_overfit_submission.py` packed
  400 nets. Submitted **54461084** with message
  `active 7270.997535 boristown public minmerge 6 tasks 020/030/076/133/189/281 on 7268 regime active fail=0 topk clean`.
  Kaggle completed at **7271.88**, same displayed score as **54461060**; local cost improved, but no
  additional displayed LB tick over batch2.
- Negative-verdict hygiene: the 0-win public results above mean only "no cheaper bundled-fail=0
  overlay found by `mine_overfit_minmerge --margin 0` on 2026-07-08 against this active set".
  Reopen trigger: any newer public rerun, any higher public frontier, or an extractor that exposes
  additional ONNX artifacts. Falsification history: public-tail "dry" verdicts have repeatedly been
  falsified by margin-0 remine and reruns, so keep this lever dormant, not closed.

## 2026-07-08 task295 regime crack submitted (pending LB)
- Built `reports/candidates/task295/regime.onnx` with
  `reports/candidates/task295/build_regime.py`: remove `small_label -> Pad(full_label) -> Equal`
  and route the label rule directly to FREE `output` through dynamic colour vector +
  separable 9x18 masks.
- Gate: task295 isolated bundled fail=0, cost **1604 -> 393** (memory `1557 -> 174`,
  params `47 -> 219`), points **17.619744 -> 19.026190**.
- Adopted into `submission/overfit_nets/task295.onnx`; backup
  `reports/candidates/task295/adopt_backup_727188/task295.onnx`.
- Full active `build_overfit_manifest.py --isolated-all`: **400/400**, local
  **7274.210520**; unsigned TopK clean; packed and submitted as **54461590** with message
  `active 7274.210520 task295 regime free-output direct label route cost 1604->393 on 7271.88 active fail=0 topk clean`.
  Kaggle completed at displayed publicScore **7274.65**. New confirmed best.
- Mechanism lesson: prior task295 "900B label Pad floor" was false. Reopen label-pad tasks when the
  label is a small algebraic combination of separable masks plus dynamic channel vectors; do not
  require a counted small one-hot.

## 2026-07-08 full active rescan after task295 (no new adoption)
- Confirmed Kaggle best is still **54461590 / publicScore 7274.65** via
  `kaggle competitions submissions -c neurogolf-2026 | head -20`.
- Rebuilt active overfit manifest with
  `PYTHONPATH=. uv run python reports/scripts/build_overfit_manifest.py`:
  **400/400**, local **7274.527684**, total cost **1050469**. This is the
  current measured `submission/overfit_nets` state; earlier `7274.210520` in
  the submission message was a pre-rescan local number.
- `reports/scripts/scan_unsigned_topk.py submission/overfit_nets`: clean.
- Public min-merge against all currently extracted public candidate net dirs:
  `PYTHONPATH=. uv run python -m reports.scripts.mine_overfit_minmerge $(find public_candidates -type d \( -name nets -o -name base_submission -o -name overrides \) | sort) --margin 0`
  found **0 adoptable wins** against the current active set. Reopen trigger:
  a newer public dump/notebook output, a higher per-task frontier, or newly
  extracted ONNX artifacts not present under `public_candidates` on this date.
  Falsification history: public-tail "0 wins" has been falsified multiple times
  after reruns/new uploaders, so this is dormant, not exhausted.
- Added full active scanner `reports/candidates/free_output_label_rescan.py` and
  runtime probe `reports/candidates/free_output_label_runtime_probe.py` for final
  `Pad(label)->Equal(...)->output` tails. It found **58 active tails**; top costs:
  task233, task366, task158, task018, task286, task364. Break-even check:
  simple `Equal(small label)->Pad(bool output)` only wins when compact area
  `<90` cells, but every active tail has area `>=100`. Therefore the simple
  order swap has **0 current wins**. Reopen trigger: a future overlay with
  compact label area `<90`, a bool-Pad/opset change, or a task295-style direct
  free-output formula that avoids both counted label and counted one-hot planes.
  Falsification history: the broad "900B floor" claim was falsified by task295,
  so only the simple area-ordering route is demoted, not the whole free-output
  label family.
- Deep checks:
  - task124 direct free-output variants built in
    `reports/candidates/task124/build_free_output_variants.py`. Correct
    `bg+delta` Einsum candidate passed bundled `267/267` but worsened cost
    `1953 -> 3024` (`1879+74` to `2248+776`), because 10x30 axis masks and
    two-term background handling cost more than the 900B carrier.
  - task041/088/233/366 were inspected against current active ONNX/tasklogs.
    They are not simple fixed-colour tails: either colour is dynamic, area is
    above break-even, or the dominant cost is earlier detector/assignment
    machinery rather than the final 900B label carrier.

## 2026-07-08 task377 output-coupled fp16 tail confirmed
- Built `reports/candidates/task377/task377_fp16_output_coupled.onnx` with
  `reports/candidates/task377/build_fp16_output_coupled.py`.
- Mechanism: deployed task377 is already a free-output final `Einsum`; recast
  output-coupled fp32 tensors `OHf/Xs/X` [1,5,10] and `R` [1,5,30] to fp16 by
  changing `OHb->OHf`, `ab->R`, and `SK`, then making free graph `output` fp16.
- Gate: task377 bundled fail=0, cost **4567 -> 3967** (memory `3552 -> 2952`,
  params `1015` unchanged), points **16.573388 -> 16.714235**.
- Adopted into `submission/overfit_nets/task377.onnx`; backup
  `reports/candidates/task377/adopt_backup_727465_rescan/task377.onnx`.
- Full active manifest: **400/400**, local **7274.668531**; unsigned TopK clean;
  packed and submitted as **54462993**, completed at displayed publicScore
  **7274.79**, with message
  `active 7274.668531 task377 fp16 output-coupled tensors cost 4567->3967 after task295 rescan fail=0 topk clean`.
- Reopen trigger: re-run deployed fp16/output-coupled scan after any overlay or
  free-output rewrite. Falsification history: prior task377 "einsum angle floor"
  was too broad for the deployed graph; detector read can be structural while
  final-output-coupled fp32 masks remain shrinkable.

## 2026-07-08 task205 final-Einsum fp16 input recast confirmed
- Built `reports/candidates/task205/task205_fp16_final_einsum_inputs.onnx` with
  `reports/candidates/task205/build_fp16_output_variants.py`.
- Mechanism: leave upstream fp32 tensors untouched, but insert fp16 Casts
  immediately before the final `Unsqueeze/Concat` operands feeding the free
  graph-output `Einsum`. This avoids the earlier input-co-bound cast-back trap.
- Gate: task205 bundled fail=0, cost **4652 -> 4372** (memory `4606 -> 4326`,
  params `46` unchanged), points **16.554947 -> 16.617024**.
- Adopted into `submission/overfit_nets/task205.onnx`; backup
  `reports/candidates/task205/adopt_backup_727479/task205.onnx`.
- Full active manifest: **400/400**, local **7274.730608**; unsigned TopK clean;
  packed and submitted as **54463492**, completed at displayed publicScore
  **7274.85**, with message
  `active 7274.730608 task205 final-einsum fp16 inputs cost 4652->4372 after task377 fail=0 topk clean`.
- Reopen trigger: re-run deployed fp16 scan after overlays; for output-coupled
  candidates, prefer local final-op casts over changing shared upstream dtype.
  Falsification history: the earlier task205 note rejected `safe_name_76/79`
  too broadly; the narrow final-cast placement falsified that rejection.

## 2026-07-08 task355 + task205 residual fp16 recasts
- **task355 confirmed:** built `reports/candidates/task355/task355_fp16_output.onnx`
  with `reports/candidates/task355/build_fp16_output.py`. Mechanism: final
  output-only `selWu` + `e0` feed a free graph-output `Einsum`, so recast them
  to fp16 and set graph `output` fp16. Gate: cost **538 -> 518** (memory
  `496 -> 476`, params `42` unchanged), bundled fail=0. Full active after
  adoption: **400/400**, local **7274.768491**, unsigned TopK clean. Submitted
  as **54463756**, completed at displayed publicScore **7274.89**.
- **task205 residual submitted:** after task355, reran
  `reports/candidates/fresh_sweep/scan_deployed_fp16.py --sample 0 --min-save 1`;
  task205 still showed a large residual fp16 upper bound. Built
  `reports/candidates/task205/task205_fp16_coord_tail.onnx` by recasting the
  coordinate/index tail (`safe_name_30..33`, `safe_name_1`, `safe_name_6`,
  `safe_name_52..55`, and the small occupancy/value tail) to fp16. Gate:
  cost **4372 -> 3760** (memory `4326 -> 3714`, params `46` unchanged),
  bundled fail=0. Full active after adoption: **400/400**, local
  **7274.919292**, unsigned TopK clean. Submitted as **54463952** and pending
  at write time.
- Reopen trigger: rerun deployed fp16 scan after every active overlay; a task
  that already had a final-output fp16 recast can still have upstream
  coordinate/index tails that are not input-co-bound. Falsification history:
  task205 was twice under-harvested by overly broad dtype-coupling rejections.

## 🧭 2026-07-08 DIAGNOSTIC SESSION — "what is our fundamental problem?" (user-directed; runs alongside the parallel public-tail session)
- **NEW CONFIRMED BEST = 7264.88 (sub 54458974, LB verified).** My fat-middle fresh-sweep landed
  on top of the parallel session's 7264.67 (sub 54458829) for +0.21 LB.
- **Deep-attacked 13 fat-middle tasks (16-18pt band) on their documented open-angles → 0/13 the
  documented angle converted.** 8 were STALE (min-merge/rebuild REPLACED the net the tasklog
  describes — e.g. 244 says "double-Gather drop 4800B g1" but live net is GridSample 1179); 1 was
  mis-estimated payoff. **⇒ the ~146-task "documented open-angle reservoir" is STALE bookkeeping,
  not headroom; task_index target costs are stale too.** Writeup: memory `neurogolf-fatmid-backlog-stale`.
- **BUT fresh re-inspection of the CURRENT deployed nets landed real wins.** ADOPTED (deployed-gated,
  bundled fail=0, TopK clean): **task251 3039→2756, task268 3009→2930, task205 4891→4652,
  task222 2733→2313 (≈+0.34 local).** Backups in `reports/candidates/fatmid_adopt_backup/`.
- **⭐ ROOT-CAUSE TOOLING FIX:** the standing `dtype_overpay_scan.py` scans `networks/` (source-
  regenerated), DIVERGED from the DEPLOYED `submission/overfit_nets/` — why fp16 looked exhausted.
  **New tool `reports/candidates/fresh_sweep/scan_deployed_fp16.py`** scans the deployed nets with two
  physics filters: (1) input-WELD (fp32 plane upstream of the nearest Cast is welded to the free fp32
  input → recast forces an 18000B input-Cast → skip); (2) consumer CO-BIND (plane feeding an
  Einsum/Scatter alongside an input-welded operand is dtype-pinned to the input → FLOOR). Refined
  worklist = **6 tasks, +0.58 ceiling** (needs per-task rebuild-gate): top = **task205 output-coupled
  +0.30** (unlock via the ⭐ task222 free-output-fp16 trick, not yet applied), task377 +0.14.
- **⭐⭐ KEY MECHANISM (task222):** planes pinned by a final Einsum/Concat to the graph OUTPUT are
  recastable by making the FREE `output` itself fp16 (`calculate_memory` skips input/output regardless
  of dtype; `run_network` thresholds `>0` so sign-exact fp16 passes). Output-coupled = winnable;
  INPUT-coupled = floor.
- **ANSWER to "what's our fundamental problem":** NOT an application gap (documented backlog is stale,
  not unharvested), NOT execution-at-scale (analysis is thorough). It is a **representation-regime gap**:
  every fat-middle net is welded to a `[1,1,30,30] bool = 900B` output-routing mask (recurs in
  034/075/089/168/345/251/382) AND its cheap planes are all contracted against the free fp32 input
  (dtype-pinned). 700pt to 8000 needs each task ~6× cheaper = an encoding OUTSIDE our lineage, which
  our lineage + public community (~7237) have not cracked. Fresh-sweep is real but modest (~+2 LB).
  North-Star lever = reverse-engineer the sub-900B representation, not more mechanical sweeps.

## 🟢🟢🟢 확정 최고 (2026-07-08, LB 확인): BEST = 7264.67 (sub 54458829) — lucifer public tail task157
- After confirmed best **54458146/7264.54**, re-mined latest public runs:
  `prvsiyan/neurogolf-7250-24-w-visualizations` (2026-07-08 10:27) produced no downloadable ONNX
  output via `kaggle kernels output`; `lucifer19/chimera-safe-boost-caddies` (10:21) and rerun
  `llccqq624/neurogolf-public-notebook` (10:18) were scanned with:
  `PYTHONPATH=. uv run python -m reports.scripts.mine_overfit_minmerge --margin 0 ...`.
- Result: one adoptable overlay, **task157** from lucifer, bundled fail=0,
  memory `6595 -> 6591`, params `256 -> 256`, local points
  `16.167850093997103 -> 16.168434120878935`.
- Adopted into `submission/overfit_nets/task157.onnx`; backup:
  `reports/candidates/public_mine_20260708/lucifer_safe_boost_1021/adopt_backup_54458146/task157.onnx`.
  Full active isolated manifest: **400/400**, total cost **1062063**, total points
  **7264.550544427234**; unsigned TopK scan clean; `submission.zip` packed and submitted as
  **54458829** with message `active 7264.550544 lucifer public tail task157 cost -4 fail=0 topk clean`.
  Kaggle completed at displayed publicScore **7264.67**.
- Reopen trigger: re-run margin-0 public min-merge on any public run above the current mined frontier
  or on reruns that expose downloadable `submission.zip`/`taskNNN.onnx`. Falsification history:
  public-tail "dry" verdicts have repeatedly been false at margin-0 and after new uploader reruns,
  so keep this lever dormant, not closed.

## 2026-07-08 — public-win autopsy upgraded to exact lost-tensor fingerprints
- Enhanced `reports/scripts/public_win_autopsy.py` so public teacher wins now emit
  `learned_lost_fingerprints` and `fingerprint_rescan_candidates`, not just broad mechanism tags.
- Ran two autopsies:
  - `reports/candidates/public_autopsy/724995_tail_fingerprint/`: 65-win public tail before S37.
    Result: 57/65 wins were `params_tail_or_initializer_dedupe`, 8 were byte tails; no exact
    lost-tensor fingerprints >=500B.  This confirms that the +0.216 local tail was mostly optimizer
    byte cleanup, not a large representation collapse.
  - `reports/candidates/public_autopsy/724950_big_jump_fingerprint/`: mixed `.minmerge_backup`
    baseline against `7249-50` and `urad 7250.18` public dumps.  Result: 99 historical wins;
    major clusters include `free_input_einsum_substitution` (+1.720 local over 3 wins),
    `index_or_topk_plane_removed` (+0.939 on task022), and `final_equal_or_output_only`
    (+0.595 over 4 wins).  Exact active fingerprint candidates: task019/044/076 share
    task011's lost `Pad:2:1x1x30x30:900B`.
- Probed task019 exact fingerprint:
  `reports/candidates/task019/task019_equal_before_bool_pad.onnx` tried to replace
  `Pad(label)->Equal(output)` with `Equal(small label)->Pad(bool output)`.  ORT rejects
  `Pad(tensor(bool))`; u8 workaround is byte-negative because it materializes a counted
  10-channel carrier.  No adoption.  Reopen trigger: legal bool Pad to graph output or another
  final-output route that avoids the counted `[1,10,30,30]` carrier.  Falsification history:
  narrow task019 tail result only; public-autopsy mechanisms remain active for other signatures.

## 2026-07-08 follow-up — task133/task285 public-autopsy candidates checked
- User asked to try both high-ranked `free_input_einsum_substitution` targets, task133 and task285.
- task133 current-public comparison: active and urad/llcc/lucifer all cost `21322`; franksunp is
  worse (`21402`).  Profile: main carriers are `gridf` fp32 3600B, per-colour row/col profiles
  2400B, and four per-scale `seed_m -> QLinearConv -> stamp_m` branches.  Oracle
  `reports/candidates/task133/task133_debug_H_outputs.onnx` extracted intermediate scale `H` on
  stored + arc-gen (`267` examples): scale counts `m1=188`, `m2=218`, `m3=119`, `m4=34`, with all
  four scales co-occurring in 5 examples.  Conclusion: task011-style single free-input Einsum
  collapse does not apply directly; this task needs a per-object scale-vector lowering.
- task285 current-public comparison: active cost `19623`; franksunp/urad/lucifer cost `19706`,
  llcc cost `19700`, all worse.  K-shrink probes under `reports/candidates/task285/`:
  `k32 31->30` fails `1/265`, `31->29` fails `1/265`, `31->28` fails `4/265`; `k9` shrink hits
  downstream reshape mismatch.  Conclusion: current sparse enumeration K is tight on bundled data.
- Mechanical follow-ups: `dynamic_cse_active_probe.py --tasks 133 285` = 0 wins;
  dead-constant probe = 0 wins; global zero-compare rescan produced no task133/task285 win.  It did
  emit a stale task205 win, but task205 active is already identical cost `4652`.
- Reopen trigger: a public teacher strictly below active for either task, or a new ONNX lowering for
  per-object variable magnification (task133) / sparse reflection without fp32 read + TopK
  enumeration (task285).  Falsification history: both tasks have prior "floor" claims falsified by
  public overlays or Kaggle scorer behavior; treat this only as a narrow negative for the current
  public-autopsy signature, not as a permanent floor.

## 🟢🟢🟢 확정 최고 (2026-07-08, LB 확인): BEST = 7264.54 (sub 54457603) — franksunp 7250.24 PUBLIC TAIL
- New public frontier rose ~+0.26 (franksunp `7250.24`, prvsiyan `7250.21`, urad `7250.18`, up from
  last-mined `7249.98`).  Pulled all three, ran `mine_overfit_minmerge --margin 0` against active
  `submission/overfit_nets`.  franksunp dominated: **+0.0131 local across 7 tasks**
  (36/20/281/189/264/319/44); prv/urad added nothing beyond frank.
- Each winner: isolated `evaluate()` bundled fail=0 + strictly cheaper (script gate line 108).
  Applied, rebuilt `submission.zip` (400 files), unsigned-TopK scan clean.  Submitted **54457603**,
  Kaggle completed at displayed publicScore **7264.54** (7264.53 -> 7264.54).
- Backups of the 7 replaced nets: `submission/overfit_nets/.minmerge_backup/`.
- Reopen trigger: re-run `mine_overfit_minmerge --margin 0` on the next public frontier that rises
  above `7250.24` (franksunp/prvsiyan/urad are the reliable uploaders).  Pure permanent upside
  (constant dataset, no rescore), so always worth a pass when public ticks up.

## 🟢🟢🟢 확정 최고 (2026-07-08, LB 확인): BEST = 7264.53 (sub 54454741) — REDUCESUM->EINSUM TAIL
- User asked why public-only mining found +4pt while our own mechanism search missed it.  Built
  `reports/scripts/public_win_autopsy.py` to compare pre/post public tail wins by op deltas, then
  converted one public `task012` lesson into a reusable scanner:
  `reports/candidates/reducesum_spatial_to_einsum_probe.py`.
- Mechanism: replace exact colour-count profile
  `ReduceSum(input, axes=[2,3], keepdims=0)` with `Einsum(input, equation='bchw->bc')`.
  Memory is unchanged, but the axes initializer becomes dead; scorer params drop by 2.
- Full active scan (2026-07-08): 17 candidates checked, 8 bundled-clean wins adopted:
  task018, task037, task203, task239, task324, task378, task388, task396.
  Local active **7264.400942 -> 7264.411366**, total cost **1062470 -> 1062454**,
  per-changed-task bundled fail=0, unsigned TopK clean.
- Packed and submitted as **54454741**, message
  `reducesum spatial to einsum tail +0.0104 local`.  Kaggle completed at displayed
  publicScore **7264.53**.
- Tooling caveat: full `reports/scripts/build_overfit_manifest.py --isolated-all` was attempted
  but current evaluator stalls on a pre-existing task after task075; changed tasks were evaluated
  individually and the manifest was updated by replacing only the 8 changed rows.  This is an
  execution issue, not a bundled failure: all 8 changed nets returned fail=0 and TopK scan is clean.
- Reopen trigger: rerun the scanner after any public/source overlay that introduces fresh spatial
  ReduceSum profiles, or when a new public dump shows another op-pricing asymmetry.

## 2026-07-08 — known-insight coverage audit started
- Built `reports/scripts/known_insight_coverage.py`, an active-overfit audit over
  `submission/overfit_nets/`, not the source-owned `networks/` inventory.  Outputs:
  `reports/known_insight_coverage.json` and `reports/known_insight_coverage.md`.
- Purpose: answer whether our known mechanisms are actually represented across all 400 tasks.
  The first pass shows the project is still a patchwork: many high-cost tasks match broad
  insight predicates, but a large fraction are either documented walls, Kaggle-falsified local
  rewrites, or semantic compilers that exist in Python but have no cheap ONNX lowering.
- Follow-up probe 1, task233 `qlinear_uint8_lut_or_matmul`: built
  `reports/candidates/task233/task233_qlinear_wspr_probe.onnx` by replacing
  `Conv(input, wspr)` with `QuantizeLinear(input)->QLinearConv->Cast`.
  Bundled fail=0 but cost worsened **32667 -> 42457** because quantizing FREE fp32 input
  materializes a counted `[1,10,30,30]` uint8 carrier.  Logged in `reports/tasklog/task233.md`
  and added reject tag `free_fp32_input_quantize_required` to the QLinear insight.
- Follow-up probe 2, `pad_compensated_spatial_crop`: reran
  `reports/candidates/pad_compensated_crop_probe.py` on active top25.  Result: 0 wins.
  Probes on task158/349 failed bundled; task367/191 variants hit shape inference errors.
  Reopen only for a new one-consumer Conv/QLinearConv border with bundled-dead activity evidence.
- Follow-up probe 3, `label_pad_vs_onehot_pad_ordering`: added
  `reports/candidates/label_pad_order_active_probe.py` for safe sentinel
  `Pad(label)->Equal(output)` tails.  Active scan found **0 safe candidates**.  The broad
  registry hits are mostly non-sentinel pad values or break-even misses, not unapplied wins.
- Current conclusion: the biggest remaining integration gap is not a missing peephole scanner;
  it is **unlowered semantic compilers** (e.g. task366 exact-cover reference, task101 QLinear
  suppression, task233/top-team unknown representation).  Coverage audit should now separate
  `candidate_unlogged`, `known_blocked`, and `unlowered_semantic_compiler` so we stop chasing
  false positives and focus on lowering gaps.

## 🟢🟢🟢 확정 최고 (2026-07-08, LB 확인): BEST = 7264.52 (sub 54452657/54453501) — PUBLIC TAIL MIN-MERGE
- New public frontier outputs were re-mined after confirming `54451991=7264.30` was still latest best:
  `franksunp/7249-95-lb-compact-onnx-artifact-starter`,
  `llccqq624/neurogolf-plus-compact-task-union`, and
  `lucifer19/chimera-safe-boost-caddies`.
- First pass with default margin found `0` wins, but `--margin 0` exposed byte-level public tails:
  **65 overlays**, local `7264.182707 -> 7264.399214` (`+0.2162`), bundled `400/400`,
  unsigned TopK clean.  Kaggle submission **54452657** completed at **7264.52**.
- Largest local tails: task291 `+0.0299`, task189 `+0.0188`, task334 `+0.0153`,
  task342 `+0.0111`, task267 `+0.0105`, task010 `+0.0099`, task329 `+0.0085`,
  task354 `+0.0072`, task111 `+0.0065`, task238 `+0.0062`.
- Full record: `reports/candidates/public_tail_20260708_724995/ADOPTION_REPORT.md`.
  Active backup before adoption:
  `reports/candidates/public_tail_20260708_724995/pre_tail_active_backup/`.
- Tooling note: `reports/scripts/build_overfit_manifest.py` now has `--isolated-all` because the
  long-lived ORT process can stall on pre-existing task376.  Isolated manifest completed
  `400/400 ok`.  task376 was not part of this adoption.
- This reopens public-tail mining for **margin-0 / byte-level** optimizer tails whenever a new public
  frontier appears.  It does not falsify the high-cost residual-spatial dry result; these wins were
  mostly tiny shared public optimizer tails, not a new bloat-task collapse.

## 2026-07-08 follow-up — 7249.98 public refresh tail, display score unchanged
- Re-mined newer public outputs after `54452657`: franksunp `7249.98`, rerun lucifer/llcc,
  and velvet `7249.73`.  Active-pool margin-0 min-merge found 2 more local wins:
  task268 `+0.001328` and task019 `+0.000399`.
- Adopted and submitted as **54453501** with local active **7264.400942**, bundled `400/400`,
  unsigned TopK clean.  Kaggle completed at displayed publicScore **7264.52** (same two-decimal
  display as 54452657).
- Full record: `reports/candidates/public_tail_20260708_724998/ADOPTION_REPORT.md`.
- Mechanical sweeps after this active set: prune-dead-constants `0 wins`, noop-reshape `0 wins`,
  zero-compare-to-bool-cast `0 wins`, dynamic-CSE top25 found only banned task101.
- task319 residual-spatial bounded-profile crop oracle failed byte math:
  raw grids <=19x19, but row profile indices reach 24; Slice crop is counted as a huge fp32 tensor,
  and selector-Einsum crop costs more params than it saves.  See `reports/tasklog/task319.md`.

## 🟢🟢🟢 확정 최고 (2026-07-08, LB 확인): BEST = 7264.30 (sub 54451991) — DYNAMIC-CSE SALVAGE TAIL
- After `54451017=7264.29`, found local bundled-fail=0 tail candidates. The first combined submit
  **54451532** (`task044 + task101 + task157`) crashed the score to **7248.82**. The loss matched
  task101's full contribution (`15.473`), so task101 dynamic-CSE was treated as culprit.
- Full sweep with task101 still included (**54451744**) also scored **7248.83**.
- Salvage submit **54451991** kept task044 zero-compare bool-cast and dynamic-CSE tails for
  task157/task035/task091/task268, but removed task101. It completed at **7264.30**.
- Adopted tails: task044 `4738 -> 4737`, task157 `6852 -> 6851`, task035 `1909 -> 1895`,
  task091 `2898 -> 2894`, task268 `3017 -> 3013`. Local active `7264.182707`, 400/400 bundled ok,
  unsigned TopK clean, packed as repo-root `submission.zip`.
- Rejected: task101 dynamic-CSE `13725 -> 13721` is Kaggle-falsified despite bundled fail=0.
- Negative scans this pass: pad-compensated crop top60 `0 wins`, prune-dead-constants `0 wins`,
  noop reshape `0 wins`, zero-concat-to-pad `0 wins`, negative-pad normalize `0 wins`.

## 2026-07-08 — fold-batch (residual-spatialop self-application, teacher-free) = 4/4 FLOOR
- Ran `fold_finder` (min 800B) on current active set → 8 reducible-plane candidates (floors excluded).
  Deep opus agents attacked the top 4 (216/364/096/009). **All 4 = genuine floor** (ledgers in each tasklog):
  096 plane multiply-consumed by presence-Cast (byte-neutral); 009 fp16 already min Einsum dtype + all 9
  colours live; 216 Einsum is free 2nd consumer of a fp32 free-input slice (reroute +560B); 364 all tensors
  at dtype floor (fold_finder premise was STALE — plane feeds QLinearConv not 27 Einsums).
- **⇒ teacher-free fold vein is DRY on current active set** (heavily pre-mined). Remaining spatial-op bytes
  are real floors: fp32 free-input slices (Slice preserves dtype), int32 Gather indices (ORT-forced),
  fp16 Einsum operands (min legal), uint8 output planes. Reopen = new public teacher, or a whole-net
  reformulation of a monster (233/366/285), or the sparse-init lever below.
- **sparse-initializer params lever — PROBED LOCALLY 2026-07-08, BLOCKED (rigorous, no submission).**
  `calculate_params` DOES discount `sparse_initializer` by NONZERO (L145-148) and `calculate_memory`
  whitelists sparse names (L53-54) — the scorer authors intended sparse. BUT two independent walls kill it:
  (1) `calculate_memory`'s `infer_shapes(strict_mode=True)` (onnx 1.21) does NOT propagate a
  sparse_initializer's dense shape to consumers → the consuming op (Einsum/MatMul) sees rank-0 → shape
  error → mem=None → 0 pts. **Confirmed on a MINIMAL clean MatMul+sparse-W graph, both linear & 2-D index
  layouts** (not a malformed-construction artifact). (2) The obvious workaround — add a `value_info`
  declaring the dense shape so inference passes — makes `calculate_memory` COUNT that dense shape as
  MEMORY. Net change = N·(itemsize−1)+nnz > 0 ALWAYS (fp16 ⇒ +N+nnz), i.e. strictly WORSE than dense,
  for every net regardless of sparsity. (Also hit a separate ORT `safe_name` load error via the harness
  sanitize path.) ⇒ net-negative even if unblocked. Since local==LB and the block lives in the shared
  scorer, the grader behaves identically; no submission warranted. **Reopen:** an onnx version whose
  strict shape inference natively propagates `sparse_initializer` dims to consumers (then no value_info,
  pure params win) AND confirmation the Kaggle grader runs that version.
- **output-entropy scan (novel lens: OUTPUT simplicity vs NET cost) — BUILT + PROBED 2026-07-08, DRY.**
  New tool `reports/scripts/output_entropy_scan.py`. Tests whether "memorization dead" breaks per-task:
  overfit is permanently safe, so a cheap codebook/selector need only fit the 268 bundled examples.
  (A) 22 tasks have codebook-LB (K distinct outputs × max area) < current cost — biggest gaps 174/325/346/79.
  (B) cheap-selector battery over the 6 always-1×1-output tasks: **NO 100% selector**. Closest = task346
  `least_freq` 263/267 (4 misses = sprinkle-noise count flips needing spatial detection). Inspected 346
  (mono-block centre detect) + 355 (speck-count-per-block) — both genuine, active nets already near
  closed-form. **Lesson: the scan measured OUTPUT diversity but cost is INPUT processing; ARC computes a
  small answer FROM a complex input, so small-output ≠ cheap-net.** Same wall as memorization-dead.
  **Reopen:** re-run after big reformulations (a heavy new net whose output IS a cheap fixed function).
- **output-entropy variable-output extension (174/325/394) — CHECKED 2026-07-08, also DRY.** Inspected the
  biggest-gap variable-output candidates: 325 out=N×N cyan Identity (selector = connected-component COUNT,
  Euler); 174 out=bbox-crop of the hflip-symmetric box (selector = symmetry-ID + crop); 394 out=periodic
  bite-window reconstruction, out(r,c)=tile[(row+r)%m][(col+c)%m] (selector needs data-dependent bite
  location + period m). NONE has a FIXED cheap selector — all need data-dependent detection, which IS the
  net's work. All three active nets are ALREADY reformulated to closed-form (old 41120/32787/44698 →
  active 1483/4347/2071). The scan's "gap" is illusory: it compares a trivial codebook to the net, but the
  real cost is the irreducible selector/computation. **⇒ output-entropy idea fully DRY (1×1 + variable).**
  Consistent finding across ALL 2026-07-08 deep-golf angles (fold 4/4, sparse-init, entropy 1×1+variable):
  the active set is at/near per-task computational floor; no cheap encoding the heavy pre-mining missed.

## 🟢🟢🟢 확정 최고 (2026-07-08, LB 확인): BEST = 7264.29 (sub 54451017) — PUBLIC OPTIMIZER TAIL
- **+0.03 LB tail after public re-mine.** The recent public-code lesson was correct: do not stop at
  "public min-merge exhausted."  After the +4.51 public re-mine, the embedded optimizer report still
  pointed at residual safe initializer-dedupe opportunities.  Re-ran active dedupe and adopted:
  task035 cost `1970 -> 1909` (+0.03145 local) and task340 `3405 -> 3404` (+0.00029 local).
- Local active after adoption: `7264.169951`, 400/400 bundled ok, `scan_unsigned_topk.py` clean,
  packed as `submission.zip`; Kaggle submission **54451017** completed at publicScore **7264.29**.
- Backups: `submission/overfit_nets/.dedupe_public_optimizer_backup/`.  Candidate files:
  `reports/candidates/task035/task035_dedupe_initializers.onnx`,
  `reports/candidates/task340/task340_dedupe_initializers.onnx`.
- Remaining public-dump min-merge against franksunp/urad/jackelysia/task-graft dirs is currently
  `0 adoptable` against this active set.  Reopen trigger: any new public uploader/notebook above
  the current mined frontier, or a new optimizer report that flags non-dedupe structural rewrites.

## 🟢🟢🟢 확정 최고 (2026-07-08, LB 확인): BEST = 7264.26 (sub 54450447) — PUBLIC RE-MINE
- **⭐ 공개 재채굴 레버 재발화 (+4.51 LB, 7259.75→7264.26).** 신규 상위 공개덤프:
  franksunp `7249-50-lb-compact-onnx-artifact-starter`(7249.50, 26승 중 19), urad `best-7250-18`(7250.18).
  덤프 TOTAL은 우리보다 ~9pt 아래지만 **per-task min-merge**로 26태스크가 우리보다 쌈 (bundled fail=0).
  도구=`reports/scripts/mine_overfit_minmerge.py`, 상세=`reports/candidates/public_mine_20260708/MINE_REPORT.md`.
  Top: task011 +1.52(Conv+Gather+Pad→3-Einsum), 022 +0.94(GridSample+TopK→1 Einsum),
  281 +0.49(wide index plane→Log/Floor arithmetic decode), 013 +0.29, 036 +0.26.
- **⭐ 추출 메커니즘 (insight_registry `residual_spatialop_to_free_einsum_collapse`)**: 공개 프론티어가
  우리 넷의 잔여 spatial-op(Conv/GridSample/TopK/MaxPool/wide index plane)을 free-input Einsum 축약으로
  붕괴. **재스캔 타깃**: 우리 overfit_nets 중 아직 이런 op 든 넷 = 동일 붕괴 후보(공개덤프 없어도 hand-rebuild 가능).
- **⚠️ 병렬세션 공존:** 이 세션은 위 26태스크만 교체(backup=`.minmerge_backup/`). task101은 +0.0018만이라
  병렬세션 deep-compiler 결과가 더 좋으면 그쪽 우선(min-merge는 on-disk 대비 cheaper만 채택). 제출 전 항상
  `kaggle competitions submissions`로 최신 확인. limit=100/day, zip 이름=`submission.zip`.

## S32 정리 (2026-07-08) — 새 컴파일러급 분석: task101 runtime-QLinear template anchor

- **현재 올리기 난이도:** mechanical/hygiene 스윕은 거의 고갈. 추가 상승은 단건 deep compiler rewrite 위주라
  즉시 +20점은 어렵고, `kernel-collapse`처럼 새 패턴을 찾으면 +0.5~수점 단위로 누적하는 국면.
- **task101 새 primitive는 진짜:** ORT에서 runtime tensor가 `Conv`, `ConvTranspose`, 그리고 중요하게
  **`QLinearConv` weight**로 동작함을 확인.
  - probes:
    - `reports/candidates/task101/dynamic_template_anchor_probe.py`
    - `reports/candidates/task101/dynamic_convtranspose_probe.py`
    - `reports/candidates/task101/dynamic_qlinearconv_probe.py`
  - scale=1/zp=0 `QLinearConv`는 red-template correlation count를 uint8로 정확히 냄(count <255).
- **task101 active 기준:** `submission/overfit_nets/task101.onnx` bundled `266/266`, memory `12928`, params `822`.
  비용 큰 덩어리 = red/blue entry 4036B, raw red TopK 1360B, copy claim chain 1436B, paint/scatter/pad tail 5344B.
- **byte model 결론:**
  - fp32 runtime Conv/ConvTranspose = 손해 (`16070B+`).
  - QLinear no-suppression compiler = 이론상 `~11696B`, active 대비 `-1232B` win 가능.
  - QLinear + full/naive suppression = `~15300B`, active보다 손해.
- **oracle 결론 (`reports/candidates/task101/qlinear_anchor_oracle.py`):**
  - raw anchor maps: fail 79, extra blue 519, miss 0
  - 4-neighbor separation: fail 19, extra blue 83, miss 0
  - scale-priority overlap: fail 0
  - full maximal suppression: fail 0
- **다음에 이어갈 정확한 연구 과제:** QLinear anchor/stamp compiler는 가능하지만, bundled 통과에 필요한
  scale-priority overlap suppression을 **1232B 이하**로 내리는 lowering이 필요함. full 17x20 red coverage나
  naive coordinate-bank는 byte상 손해. `<=2` reference-red-cell 특화 anchor-coordinate relation을 full-grid
  coverage 없이 계산하는 방법을 찾거나, 19개 separated false-positive를 더 싼 exception/gate로 지우는 방향.
- 기록:
  - `reports/candidates/task101/runtime_template_compiler_analysis.md`
  - `reports/candidates/task101/runtime_template_cost_model.py`
  - `reports/tasklog/task101.md`
  - `reports/insight_registry.yaml`

## 🟢🟢🟢 확정 최고 (2026-07-08, LB 확인): BEST = 7259.75 (sub 54449733) — KERNEL-COLLAPSE
- **⭐ PURE-MINE 레버: kernel-collapse (+0.553 LB, 7259.20→7259.75, 병렬 세션 앞지름).** 도구=`reports/scripts/kernel_collapse.py`.
  단일위치 Conv 커널(6×6인데 한 점만 nonzero) → [O,I,1,1]+pad조정으로 bit-identical 대체. **params lane** (병렬 free-I/O
  sweep이 안 건드림). 18 wins: task322+0.163,187+0.088,005+0.053,218+0.047,138+0.046,286+0.038,173+0.030...
  **재실행법**: `uv run python`으로 collapse_single_pos_convs를 400 전수 → gate fail=0 + cheaper → 채택. Conv 새로
  생기면 재발화.
- 세션 시작 7254.49 → **7259.75 (+5.26)**. 기여: 내 kernel-collapse(+0.55)+compiler(task197+0.28/19/324/085+0.25)
  + 병렬세션 free-I/O·DCE·dedupe sweep(~+3). 병렬세션과 submission/overfit_nets/ 공유, 제출 전 kaggle submissions 확인.

## (구) 확정 최고: BEST = 7255.91 (sub 54424285) — COMPILER WIN
- **컴파일러 MVP가 작동함 (LB 확인 +0.40):** task197 1201→904(+0.284, 활성넷이 낡은 ConvInteger였음 → 단일
  Gather column-remap로 재구성), task19 2793→2507(+0.108, saturating-QLinearConv로 cyan 직접방출+Cast/Where
  fold), task324 8894→8864(+0.003, Conv dilation→1×1+음수pad, bit-identical). 7255.51→**7255.91**.
- **⭐ 핵심 교훈: 일부 활성 넷은 floor가 아니라 낡은/과대 형식이다** (task197이 증거). archetype 스캔의 헤드룸
  플래그는 heuristic이라 false-positive 많음(198/154/295/8 전부 floor)이지만, **낡은 형식 태스크는 진짜 재료**.
  컴파일러 배치 = flagged 헤드룸 + foldable 패턴(Conv→Cast→Where, 단 conv출력이 binary indicator일 때만 fold됨).
- **미탐색 2차 배치 후보** (foldable 패턴, floor 미확정): 182/201/85/4/170/259/383. 도구: reports/scripts/auto_golfer.py
  (archetype 랭킹), 후보 = reports/candidates/taskNNN/compiler_best.onnx. 채택 = backup(.compiler_backup)+교체+
  scan_unsigned_topk 전수+zip+submit. **⚠️ 병렬 세션도 제출 중** — submission/overfit_nets/ 공유, 제출 전 kaggle
  submissions로 최신 확인.

## (구) 확정 최고 (2026-07-07): BEST = 7254.49 (sub 54417800, active)
- S18 연속분(32 overlay min-merge + dedupe sweep) 제출·확정 = **LB 7254.49** (로컬 manifest 7254.377, 오프셋 +0.11).
  이게 현재 active 제출본. (구 7253.64는 그 이전 상태.)
- **🚨 public min-merge 레버 = 현재 고갈 (2026-07-07 재확인).** 신규 업로더 urad 7243.08 / franksunp 7242.99
  둘 다 pull → 0 adoptable. 로컬 덤프 13종 전수 재-minmerge → 0. 우리가 공개 필드 전체(top 7243)보다
  ~11점 위 = 어떤 덤프도 태스크별로 우리보다 싸지 않음. **레버는 누가 ~7254 위로 올릴 때만 재발화** (모니터링).
- **🧱 de-unrolling/벡터화 레버 = 엄밀 반증 (2026-07-07, 2개 opus PoC).** 그레이더는 노드수가 아니라
  ELEMENT수를 셈 → 언롤 루프 벡터화는 바이트 중립(0점). task005 floor 3373, task157 floor ~1300.
  파편화(366=583평면)는 실재하나 회수불가. 실floor=repr텐서(30×30 one-hot idx + fp32 input-slice).
  메모리 `neurogolf-element-count-scoring-floor` 참조. canvas-crop(max+0.59)/leaky-const도 반증됨.
  ⇒ 내생적 ≥1점 레버 부재 확정. 8040 리더는 우리 넷 계보 밖의 다른 표현체계 (I/O floor 1300B 아래).

## 🟢 S18 확정 (2026-07-07, LB 확인): (구) BEST = 7253.64 (sub 54413928)
- 7248.94 → 7249.18 → **7253.64 (+4.70 세션 총합)**. 전부 overfit bundle, gate=bundled fail=0, 영구 안전.
- **+4.46 = urad 7242.52 공개덤프 min-merge (6태스크):** task355 2678→526(+1.56), 264 4700→968(+0.83),
  197 2586→1160(+0.77), 222 5096→1742(+0.66), 236 208→96(+0.63), 383(+0.005). ISOLATED eval로 bundled
  fail=0 검증 후 채택. franksunp/poby 7240.26·jonathan 7242·seddiktrk = urad가 다 커버(추가 0, 수렴).
  ⇒ **공개 재채굴 루틴 여전히 유효** (새 상위 업로더 나올 때마다 prefilter→isolated-verify→min-merge→submit).
  덤프는 scratch_mine/(세션소멸). 원본백업 scratch_mine/backup_orig/.
- +0.24 = task243 walk-chain truncation (W2 slack 제거, 5112→3816, W1 46step으로 번들 전부 도달).
- **⭐ walk-chain slack 레버 = 1승 후 소진(1/10).** 400 overfit넷 전수 스캔: 243만 slack, 나머지
  (286/196/277/76/118/18/192/174/145) 전부 tight(terminal step drop시 bundled fail). GRU/LSTM 자유반복
  레버 반증(입력시퀀스·hidden이 counted). 스캐너=scratchpad_chainenum.py+greedy.py(세션소멸, 재작성 필요).
- 안전본 헤지 = 7245.50(sub 54398148) 보존. 243 truncation은 overfit 전용(safe tree 미적용, S9가 fresh용 K2 보존).
- **다음 세션: mechanical 레버 완전 소진 확인.** 남은 건 233(32256)/366(30983) 등 최고bloat 넷의 deep
  algorithm 재작성뿐(263~597노드, cristianoc floor 검증, +단건 초저확률). 사용자에게 계속/홀드 물어볼 것.

---
## (이하 S17 기록)

## ▶▶ 다음 세션 시작 프롬프트 (이거 그대로 붙여넣기):

```text
/Users/minseong/project/neurogolf 에서 NeuroGolf 점수 개선 이어가자.
먼저 NEXT_SESSION.md, AGENTS.md, skills/neurogolf-recursive-improvement/SKILL.md,
reports/insight_registry.yaml, reports/recursive_queue.md, reports/source_live_reconcile.md를 읽고 시작.

현재 확정 상태 (2026-07-06 S17 종료, 전부 LB 확인):
- 최고 = overfit 7248.94 (sub 54398145). constant-dataset라 영구 안전(rescore 없음). 이게 우리 제출본.
- 안전본 헤지 = 7245.50(sub 54398148)도 보존됨. 마감 07-15 = private LB(=동일 고정 데이터셋).

불변 사실 (재확인 금지, 이미 엄밀 검증됨):
- 제출한도 100/일. 제출파일 반드시 `submission.zip` 이름 (아니면 400 에러). scan_unsigned_topk 전수 후 제출.
- 채점 데이터셋 고정(train3+test1+arcgen262≈266/태스크) = 한번 bundled fail=0이면 영구 통과. overfit=영구안전.
  fresh_verify 불필요. 그냥 단일 bundled-fail=0 점수 최대화 (안전/overfit 구분 무의미).
- 소진된 레버(재탐사 금지): 공개덤프(udit=poby=waterxiao 수렴), dtype recast(+0.18 완료 나머지 floor),
  worst-case상수/TopK-K cut(load-bearing), free-tensor 재구성(deep agent 0/9 = byte floor 확정),
  메모리제이션(262 arcgen이 죽임).

남은 유일한 상승로 = per-task 알고리즘 재작성으로 grid-state COUNT 축소. mechanical golf 아님.
공개(7237)도 못했고 deep agent도 0/9. 수백점 비현실적, 단건 +1~2가 현실. 아래 §다음 세션 우선순위 참조.

현재 기본 모드 = 8000 overfit score chase. safe/private 모드로 돌아가려면 사용자가 명시해야 한다.
다음 작업은 특정 bloat 태스크 1개 deep 알고리즘 재작성 PoC 또는 신규 공개덤프/공개 mechanism 재채굴이다.
```

## 🟢 S17 확정 결과 (2026-07-06, 모두 LB 확인됨)
- **안전 최고 = LB 7245.33** (sub 54396297, private-robust). udit 7237.17 덤프에서 task206 안전채택
  (3766→1795, +0.741; RoiAlign+5Einsum, fresh inc0/cand0/div0). ← 마감 택1 헤지.
- **overfit 최고 = LB 7248.76** (sub 54396589, public-only). v1(7246.88) + udit/poby min-merge
  (377/205/076 overlay) = +1.886. 재빌드 `reports/scripts/overfit_minmerge.py`, 넷=`submission/overfit_nets/`,
  산출=`reports/candidates/overfit_minmerge/`.
- 🚨 **제출한도 = 100/일 (5 아님! 미신 반증됨).** 진짜 400 에러 원인 = 파일명이 `submission.zip` 이어야 함
  (`overfit_submission.zip` → 400 "must be named submission.zip"). 제출 전 반드시 `submission.zip`으로 복사/rename.
  Kaggle API `max_daily_submissions=100` 확인. → 자유롭게 제출.

## 🚨🚨 S17 핵심 사실 (사용자, 명시 기록 지시): 체점 데이터셋은 항상 일정 = NO RESCORE
"한번만 통과되면 계속 통과된다." bundled fail=0 한번 = 07-15 private LB까지 영구 통과.
⇒ **오버핏은 영구 안전. safe/overfit 구분 무의미. fresh_verify 불필요.** 그냥 단일 bundled-fail=0 점수를
최대화. 제출 하나(`submission.zip`)로 모든 넷을 bundled-min으로. 최고 = **LB 7248.94 (sub 54398145)**.
단 메모리제이션은 여전히 死 (arc-gen 262/태스크). per-task leaky 재구성 = pure upside (위험 없음).

## 🎯 사용자 방향 (S17): "리더보드 통과되면 오버핏이든 뭐든 상관없어. 더 올려."
**목표 = 단일 bundled-fail=0 점수 최대화** (위 constant-dataset 사실로 안전/오버핏 구분 제거됨).

## 🔴 다음 세션 = 오버핏 극대화 (available 레버는 소진, 남은 건 DEEP per-task 재구성)
S17에서 mechanical 오버핏 레버는 전부 소진 확인:
- 공개 leaky 넷: min-merge로 전부 포섭됨 (udit=poby=waterxiao 동일 pool). 신규 없음.
- walk-chain step-cut: task286만 유효(완료). task243=mega-Einsum(operand cut해도 counted plane 안 줄음, no pay).
- task066: 독립 contraction 5개, walk chain 아님 → floor.
**⚠️ S17 엄밀 판정 — 오버핏 상한 ~7248은 near-fundamental:** 각 태스크는 arc-gen 262개 인스턴스를 번들로
가짐 (train3+test1+arcgen262≈266, evaluate가 전부 채점). "bundled fail=0"=266개 전부 통과 필수.
⇒ (a) **메모리제이션 死** (4개 output Gather는 나머지 262개서 실패; 262개 저장은 알고리즘보다 비쌈),
(b) 공개 오버핏 커뮤니티가 7237서 막힌 이유=262 arc-gen이 near-algorithmic 정확도 강제, 싼 leaky 숏컷 희소,
(c) 8000은 262 arc-gen을 알고리즘보다 훨씬 싸게 통과하는 per-task leaky 숏컷 필요 — 공개 커뮤니티도 대부분
실패, cristianoc oracle이 알고리즘 floor 검증. **현실 상한 ≈7248-7249.**

**⚠️⚠️ S17 CALIBRATION: free-tensor 재구성 레버 = DRY (0/9).** 3개 deep opus 에이전트가 최고bloat 9태스크
(366/018/133/158/054/367/349/173/138)에 CSE·DCE·dtype·채널drop·커널crop·TopK-K·뱅크prune 전부 시도 →
**0승.** 전부 byte floor: 지배 plane = free fp32 `input`의 Conv/Gather/Einsum/Slice (forced fp32 ~2600-3600B)
+ multi-step 알고리즘의 minimal-width(1B) grid state 다수. detector 뱅크는 266 인스턴스에서 empirically all-live.
⇒ **8000 갭 = per-task 알고리즘의 grid-state COUNT를 줄이는 재작성** (구현/routing 갭 아님). 이건 mechanical golf가
아니라 태스크별 알고리즘 재설계 = 공개 커뮤니티(7237)도 못한 것, 8min/task deep agent도 0/9. **수백점은 비현실적.**
현실: 7248.94가 실질 상한 근처. 계속 판다면 태스크별 deep 재작성으로 단건 +1~2씩 (수백 아님).

**(구) 남은 상승로 = DEEP per-task leaky 재구성** (위 calibration으로 초저확률 확인됨):
1. ⭐ **bit-pack 인코딩 레버** (공개 task319 5852 넷에서 발견): BitShift/BitwiseAnd로 객체를 27×27 full plane
   대신 [1,10,5,5] 작은 코드로 압축 (11664B→~250B, 3× 절감). 우리 고비용 넷 중 full 30×30/27×27 객체
   plane을 materialize하는 것들에 이식 시도. 공개넷을 teacher로 op-census 비교해 mechanism 추출.
   대상 = 우리 최고비용 넷들 (285=25234, 286=23971, 233=33242, 133=21526, 018=25445...).
2. **canvas-crop 레버** (task018식, memory "narrow"): bundled 그리드가 30×30보다 작으면 캔버스 축소.
   bundled 데이터 max extent 측정 → N<30이면 넷을 N×N로.
3. **worst-case 상수 cut**: unrolled 루프 step수/TopK-K/detector bank가 이론상 max로 sized된 넷 →
   bundled-min으로 축소 (gate=bundled fail=0만). 단 S16 측정상 우리 넷은 이미 tight (5프로브 중 1승).
방법: 고비용 넷마다 (a) 공개 오버핏 넷 있으면 teacher diff, (b) 없으면 bundled-only 재구성 직접 설계.
게이트 = bundled fail=0만. 제출 파일명 반드시 `submission.zip`, 한도 100/일.

## 📌 참고
- overfit README = `submission/OVERFIT_README.md`. 넷 = `submission/overfit_nets/`. min-merge = `reports/scripts/overfit_minmerge.py`.
- 안전본 206만 clean adopt. 377/205는 private 위험으로 safe 거부(overfit엔 포함).

## === 현재 상황 (S17 종료, 2026-07-06) ===
- **확정 LB 최고: 안전 7245.33 (sub 54396297) / overfit 7248.76 (sub 54396589).** local manifest 7245.22,
  오프셋 +0.11 일관. 마감 07-15 = private LB. S16 안전본 54394428(7244.59)도 헤지로 보존.
- S17 방법 = 신규 업로더(udit 7237.17 / poby 7235.83) 재채굴 + task206 안전채택 + overfit min-merge.
  주요 발견: 제출한도 100/일(5 미신 반증), 파일명 반드시 submission.zip.

## === (이하 S16 기록, 참고용) ===
- **S16 확정 LB 최고 = 7244.35** (sub 54393583). 재제출 7244.48 local (task387 fold 포함, pending → ~7244.59).
  로컬 manifest 7244.48, 오프셋 +0.11 일관. 마감 07-15 = private LB.
- S16 궤적: 7242.29 → 7244.14 → 7244.35 → 7244.48(local). 방법 = **매칭 엔진 실증** + 공개 unfiltered
  재채굴 + **time-for-cost einsum fold**(task387 +0.239, bit-identical).
- **🆕 time-for-cost fold 레버 부활**: `reports/scripts/fold_finder.py` → counted 평면이 SUM-축약되면
  free-input einsum으로 접음. **task387 +0.239** 성공(ReduceSum crop→Einsum('bchw,c->bh') + mask-from-scalars).
  정밀 규칙: LINEAR producer(ReduceMax/compare 아님) + SUM-contraction consumer + SINGLE-USE 평면만 접힘.
  벤 상태 = 1승 후 620B 위로 소진(064/134/025/105/216/009/364 전부 floor). 잔여 ≤620B tail(131/174) 저EV.
- **엔진 증명 완료(0순위 달성).** match_insight 랭킹 → mine_public_bundles/mine_unfiltered grader측정 →
  fresh_verify cand≤inc → 5단계 채택 → mechanism_coverage.json 기록. 대표 = task222(dropped-Slice,
  bit-identical, +0.158). 총 21태스크 채택 ~+0.41, 전부 bit-identical/cand≤inc(private-LB 안전).
- **런타임 프로파일 완료(0순위 달성).** 전체 400넷 = 229ms/pass, 최고 82ms(task001). 경쟁자 slow-태스크가
  우리는 sub-ms. `runtime_ms` 인덱스에 추가됨. **timeout 헤드룸 막대 → time-for-cost 레버 개방(단, 아래 참조).**

## === S16 핵심 판정 (재사용, 중요) ===
1. **urad 3메커니즘 자가역적용 = 둘 다 FLOOR로 판명** (엄밀한 파일럿 에이전트):
   · **value_info_crop**: 90개 crop 풀 소진. 모든 verbatim crop은 single-window fp32 floor(10ch 윈도
     불가축소, 색 랜덤). 366/233/205만 full-plane이나 compose/recolor(crop 아님). coverage `_pool` floor.
   · **qlinearconv_render**: 풀 이미 붕괴(오늘 공개프론티어 이식분). 남은 평면은 load-bearing 탐지/scatter,
     교체가능 render epilogue 아님. 신규-이식 넷의 un-collapsed one-hot만 미래 대상(mech15_output_scan.py).
   · **gridsample_warp**: 풀 전부 저비용 업스케일(≤2798), EV 낮음, 미검토. (유일하게 안 판 벤)
2. **공개 벤 수렴.** franksunp 7235.49 + llccqq624 7235.83 = per-task 우리보다 싼 min-merge. ryosuke 7230 =
   동일 계보(동일 후보셋). 모든 공개덤프 ≤7235.83. **신규 업로더 나타날 때만 재채굴** (koushik/biohack/
   yuu/poby = dataset-attached, 추출 0 — 첨부데이터셋 별도 다운로드 필요).
3. **⚠️ 8000은 압축 갭이 아니라 OVERFIT 갭일 가능성 매우 높음.** cristianoc oracle이 우리 고비용넷=알고리즘
   floor임을 독립검증; 공개프론티어 7235 상한; log-math상 7244→8000은 **400태스크 전부 ~6.7× 더 싸야** 함.
   → top~8000은 visible-case 튜닝(leaky const)일 개연성. 현재 전략은 8000 overfit score chase로 확정.
   fresh/private 안전성은 사용자가 명시할 때만 보조 모드로 다룬다.

## === 미완/느슨한 끝 ===
- **task076 fresh-gate 포기**(franksunp 15932→13235, +0.185): 8000 모드에서는 문제 아님.
  bundled fail=0/lower cost면 채택 가능. fresh는 safe/private 모드에서만 재검토.
- **fold tail ≤620B**: task131(GatherND→ReduceSum 620B), task174(Cast→MatMul 600B) 미검토, 저EV(~+0.1).
- **소소 public tail** ~15태스크 Δ≤0.002 합계 ~0.02 미채택 (public/unfiltered_candidates.json). marginal.
- **GridSample 자가역적용** 미검토(유일한 미개봉 urad 벤, 단 pool 전부 저비용 업스케일 ≤2798, EV 낮음).

## 2026-07-08 — known-insight coverage audit / zero-compare tail

- Latest Kaggle before this tail: submission **54457603**, publicScore **7264.54**
  (`public tail: franksunp 7250.24 min-merge`, tasks 036/020/281/189/264/319/044).
- Re-ran `reports/candidates/dedupe_initializers_active_probe.py` on the current
  active set: `wins=0`, `TOTAL +0.000000`. Reopen trigger: a new overlay/graph
  surgery pass that can create byte-identical initializers again. Falsification
  history: this hygiene lever has repeatedly reopened after public/min-merge/CSE
  overlays, so keep it dormant rather than dead.
- Re-ran `reports/candidates/zero_compare_to_bool_cast_probe.py` and
  `reports/candidates/zero_compare_to_bool_cast/apply_cumulative.py`. Adopted:
  task044 `4737->4735`, task281 `1348->1344`, task319 `5837->5832`; bundled
  fail=0 for all, total local cost `1062454->1062443`, local active
  `7264.411365910312->7264.415616949520`.
- `scan_unsigned_topk.py submission/overfit_nets` clean; `pack_overfit_submission.py`
  packed `submission.zip` (sha256
  `8203b62aecafbb4e49fc1776b14a951214118b7d965a6ee2199878a1067ccc0a`).
  Submitted as **54458146** with description
  `active 7264.415617 zero-compare tail: task044/281/319 cost -11 after 54457603,
  fail=0, topk clean`; Kaggle status COMPLETE, publicScore **7264.54**. This ties
  displayed score with **54457603** because the local +0.004 is below displayed
  hundredth precision, but active artifacts now include the lower-cost tail.
- Coverage lesson: `dedupe_byte_identical_initializers` and `zero_compare_to_bool_cast`
  cannot be treated as "actionable" from graph syntax alone. They need their
  active probe ledgers wired into `known_insight_coverage.py`; otherwise the
  audit overstates hundreds of micro-gaps that are cost-neutral on evaluation.
- Updated `reports/scripts/known_insight_coverage.py` to demote active-probed
  misses to `probe_no_win` for dedupe and zero-compare. This removed those
  syntax-only candidates from the top actionable list.
- Triaged task204 `strided_conv_fixed_block_counts` as a coverage false positive:
  active graph is eight size-specific QLinearConv anchor detectors plus parallel
  MaxPool fills, not repeated same-stride `Slice+ReduceSum` block reads. Recorded
  in `reports/tasklog/task204.md` with tool/date/result/reopen/falsification
  notes; coverage rerun removed task204 from that actionable slot.
- Re-mined latest visible public output
  `llccqq624/neurogolf-public-notebook` (lastRunTime 2026-07-08 10:09:00) into
  `reports/candidates/public_mine_20260708/llccqq_public_notebook_1009/`.
  Ran `mine_overfit_minmerge.py --margin 0` against active overfit: **0 adoptable
  wins**. Reopen trigger: a new public notebook/dump with a later run time or a
  higher per-task frontier than franksunp/llccqq/lucifer/prvsiyan.
- Tightened `reports/insight_registry.yaml` predicates: `uint8_topk_compact_label_grid`
  now requires actual `TopK`; `strided_conv_fixed_block_counts` now requires both
  `Slice` and `ReduceSum`; `public_teacher_qlinear_conv_rewrite` rejects
  `free_fp32_input_quantize_required`. This removed broad Cast/Gather/Conv syntax
  false positives from coverage.
- Added `label_pad_vs_onehot_pad_ordering` to the coverage probe-miss ledger because
  `reports/candidates/label_pad_order_active_probe.py` saw 0 safe sentinel candidates.
- Re-ran dynamic-CSE on current high-cost active tasks excluding task101:
  tasks `233 366 018 133 158 054 285 286 364 367 349 076 118 138 198 173 191
  145 204 066 025 080 064 379 074 350 324 338 216 370 209 096 187 255 219
  157 192 089 009 328 017 005 023 182`; result **0 wins**. Reopen trigger:
  new public/min-merge overlays or a safer scorer-gap-aware CSE validator.
  Falsification history: dynamic-CSE produced real tail wins, but task101 was
  Kaggle-falsified, so every future small CSE adoption needs isolated/small-group
  Kaggle oracle unless the change is purely static and scorer-stable.

## === 다음 세션 우선순위 (S16 판정 반영) ===
1. **전략 방향 확정: 8000 overfit score chase.** private-LB 안전 유지는 opt-in 보조 모드다.
   기본 게이트는 bundled fail=0 + active 8000 incumbent 대비 lower cost.
2. **미지-메커니즘 리서치(고위험, 구체 가설 먼저)**: 유일한 안전-상승로. 단 cristianoc가 알고리즘 floor를
   독립검증했으므로 "더 싼 알고리즘" 사냥은 금지 — 남은 건 IMPLEMENTATION golf뿐이고 그것도 대부분 소진.
   time-for-cost fold(런타임 헤드룸 막대)로 우리가 materialize하는 평면을 더 깊은 einsum으로 접는 각도가
   미탐색 후보이나, 구체 태스크·가설 없이 팬아웃 금지(floor 재확인만 됨).
3. task076 마무리 + tail 정리 (즉시 실행 가능한 잔여 안전점수).

## === 워크플로/게이트 (불변) ===
- 8000 채택 = 백업 → `submission/overfit_nets/taskNNN.onnx` 교체 → bundled `evaluate` fail=0/lower cost →
  tasklog/score_modes 기록. source-owned 재작성일 때만 `src/custom`/`networks`/manifest를 같이 갱신한다.
- fresh-gate는 safe/private 모드 전용. 8000 모드에서는 fresh 결과를 진단 메모로만 쓴다.
- 제출: `submission/overfit_nets/*.onnx` → `scan_unsigned_topk.py submission/overfit_nets`(전수, uint8-TopK grader-killer) → `submission.zip` → submit
  → `--csv`로 publicScore 폴링(CSV 파싱, descrip grep 오탐 주의).
- 도구: mine_public_bundles.py(byte-prefilter), mine_unfiltered.py(전수, 프리필터 놓친 것 잡음),
  profile_runtime.py, match_insight.py, coverage_lib.py. 덤프 추출 = extract_bundle.
- 🚨 죽은 길 재탐사 금지: value_info_crop/qlinearconv_render 자가역적용(S16 floor), train-to-golf, SGD-compile,
  task 319/48/285/188/101/090/117 공개-싼넷(overfit), int64→int32 on initializer.
- 모델: 기계적/레시피 → opus. 신규 메커니즘 설계·770 리서치만 fable. leads-only엔 SDD-건틀릿 금지.

## === 커밋 상태 ===
- 미커밋 대량(S12~S16). networks/*는 gitignore, 소스진실 = src/custom/*.py(재생성됨). S16에서 21태스크
  src/custom+tasklog+manifest, coverage, profile_runtime.py/mine_unfiltered.py 추가, 메모리 갱신.
  세션시작 시 커밋 정리 권장(사용자 지시 없으면 커밋은 물어보고). 백업/후보 넷은 repo 내부
  `reports/retired_networks/`, `reports/candidates/`, `submission/`에서 관리. 롤백은
  submission zip(54393046=7244.14) 또는 src/custom git 히스토리.
```
