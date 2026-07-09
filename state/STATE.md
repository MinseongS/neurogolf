# STATE - NeuroGolf live handoff (updated 2026-07-09, task367 pending LB)
> Replace this file at session end; do not append. History lives in git and state/submissions.md.

## Confirmed State
- Confirmed BEST LB: **7298.02** (sub **54493725**, complete).
- Current local manifest: **7297.9375** (400/400). This includes task367 plus a later parallel/adopted task107 fold-scan win.
- Latest submitted batch: **54493725** (`task367 v_main 10ch->6ch plus diagonal diffusion tap`, submitted 2026-07-09 12:24Z, local-at-submit **7297.9002**) -> publicScore **7298.02**.
- Unsubmitted local delta after sub 54493725: `task107` cost **3576 -> 3445** (**+0.0373**) from `candidates/task107/cand.onnx`; below the user's standalone threshold, batch with the next meaningful win.
- Latest completed submissions:
  - sub **54490915**: task080 fp32 colour-decode -> free-input Einsum per-read, publicScore **7297.79**.
  - sub **54491406**: task161 scalar carrier tail -> free-output Einsum, publicScore **7297.90**.
  - sub **54493725**: task367 predicate-bank prune + diagonal diffusion tap, publicScore **7298.02**.
- Verification after latest adoption: `uv run ng gate candidates/task367/cand.onnx --task 367` PASS, source build via `tools/live_to_exact_source.py` + `tools/rebuild_networks_from_source.py --tasks 367` PASS, `uv run ng verify --hash` -> **HASH-OK**, `uv run ng pack` completed.
- Deadline: 2026-07-15. Private is the fixed dataset; bundled fail=0 + cheaper than deployed remains the adoption gate.

## Adopted In Latest Score Batch
1. `task161` - cost **1808 -> 1620**, points **17.5000 -> 17.6098** (**+0.1098**)
   - Candidate: `candidates/task161/cand.onnx`
   - Source: `src/custom/task161.py`
   - Mechanism: delete uint8 scalar carriers `Rvalu/Cvalu/Rval/Cval/Rv/Cv`, full `code` plane, and final `Equal`. Keep `Rvalf/Cvalf` float row/col colour scalars, build tiny row/col banks, and write the one-hot output directly with one final `Einsum`.
   - Gate PASS; bundled + arc-gen **266/266**, fail=0.
   - Insight added: `scalar_carrier_tail_to_free_output_einsum`.
2. `task367` - cost **15890 -> 14082**, points **15.3266 -> 15.4473** (**+0.1207**)
   - Candidate: `candidates/task367/cand.onnx`
   - Source: `src/custom/task367.py` regenerated from the candidate ONNX and rebuild-verified.
   - Mechanism: prune the counted `v_main` QLinearConv predicate bank from 10 channels to 6 (`016789`), then add one diagonal tap to the downstream `Wc` diffusion kernel. The diagonal tap absorbs the single overfill failure found in the raw prune without restoring a full 400B channel.
   - Gate PASS; bundled + arc-gen **266/266**, fail=0.
   - Insight added: `predicate_bank_prune_compensated_by_diffusion_tap`.
3. `task107` - cost **3576 -> 3445**, points **16.8180 -> 16.8553** (**+0.0373**, parallel adoption detected after task367 submit)
   - Candidate: `candidates/task107/cand.onnx`
   - Mechanism: fold three fixed-position fp32 colour reads into one small dilated Conv label plane.
   - Below standalone submit threshold; do not submit by itself unless bundled with another meaningful win.

## Audit Ledger
- User threshold: avoid +0.0x byte cleanup; build only plausible **single-task +0.1**, preferably **+0.3+**, or reusable mechanisms.
- Dry/no-build outcomes are recorded in `state/levers.yaml` ledger entries dated **2026-07-09**:
  - `public-minmerge`: refreshed lucifer/yusuke/jonathann/prvsiyan/ryosuke public dumps after task161; 0 margin-0 adoptions.
  - `public-insight-generalize`: profile-product siblings, probe_then_build, axis-code/render fanouts rechecked against current deployed nets; no PASS candidate.
  - `dtype-overpay`: scoped 126/188/289/181/320/332/375/394 boundary gates all cost-worse.
  - `free-output-einsum-regime-crack`: stamp/crop scalar-carrier re-audit remains no-build at >=+0.1 except adopted task161.

## Near Misses / Reopen Triggers
- `task367`: raw prune near-miss is now resolved by the diagonal `Wc` tap and adopted. Do not repeat the same 6-channel subset sweep; the reusable follow-up is to scan other predicate-bank + diffusion-tail graphs for channel-prune candidates where a tiny downstream kernel change can repair sparse failures.
- `task138`: deleting both ray branches is only ~1058B = +0.0946 and still leaves no fill. Below user threshold.
- `task279/350/064`: crop/final-mask/direct-render attempts price above the deleted 900B carriers or fail bundled cases.
- `task054/118/285/133/233`: high-cost profile siblings have no current >=+0.1 deletion; existing candidates are tied, fail, or much more expensive.

## Active Veins
1. **New public frontier** - first priority if a new dump appears: `uv run ng mine-public --margin 0 <submission.zip...>`, then `public_autopsy` only for +0.1+ mechanism fingerprints.
2. **Predicate-bank prune + downstream repair** - task367 proved this pattern can turn a 1-fail aggressive bank prune into a +0.1 win without adding counted channels. Next targets should be QLinearConv bank tails with a small diffusion/union kernel, not generic dtype cleanup.
3. **High-cost true rewrite** - remaining path to 7300 is not tail cleanup; it needs a real replacement for the large detection/assignment cores in 018/233/366/133/285/349/101/076/118, or a new public teacher.

## Operational Guardrails
- Do not spend session time on +0.0x byte-tail cleanup unless explicitly requested.
- Do not retry cost-1 tail nets, 092-profile cohort repeats, dtype boundary casts, or 014/350/018 value-info crop without the reopen triggers above.
- Keep `onnx==1.21.0` and `onnxruntime==1.26.0`; no runtime upgrades without full 400/400 re-verify.
- Adopt only through `uv run ng gate` -> `uv run ng adopt`; submit through `uv run ng pack` -> `uv run ng submit`.

## Next Session Start
1. `uv run ng status`
2. `kaggle competitions submissions -c neurogolf-2026 | head -6`
3. Check new public dumps/notebooks first.
4. Do not submit task107 alone; it is only +0.0373 and should be batched with the next >=+0.1 win.
5. If no public frontier, scan for task367-style predicate-bank prune + downstream repair candidates, then high-cost true rewrites with explicit cost-split proof before building.
