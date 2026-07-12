# STATE - NeuroGolf live handoff (updated 2026-07-12; task233 dynamic correlation)
> Replace this file at session end; do not append. History lives in git, `state/tasks/`, and `state/levers.yaml`.

## Confirmed State
- Confirmed BEST LB before the current pending submission: **7316.30** (submission 54596069, MAIN v9).
- Local deployed MAIN remains **7317.9914** after the signed-polynomial wave; MAIN v10 submission 54597601 was pending at the last check.
- This session merged research and a validated task233 candidate into `main`; it did **not** adopt the candidate, pack, submit, or push.
- Competition strategy remains pure score maximization. Fresh validation is diagnostic for private-suite safety, not a hedge against a later redraw.

## Validated Candidate: task233 Dynamic Signed Correlation
- Candidate: `candidates/task233/cand_dynamic_corr.onnx`, reproducible with `candidates/task233/build_dynamic_corr.py`.
- Mechanism: derive the output location from a dynamic signed correlation instead of materializing the previous coordinate machinery.
- Cost **24703 -> 21816**; task points **14.885320 -> 15.009601** (**+0.124281**).
- `ng gate`: **266/266 pass, 0 fail**. Fresh A/B: **3600 draws, 0 divergence, 0 regression**.
- Status: source, tests, design, task ledger, and insight registry are merged. The deployed model is unchanged because this closeout requested merge-only.

## Global High-Upside Searches Closed This Session
- Exhaustive rectangular Pool attribute sweep across all 400 tasks: kernel H/W 1..30, every fitting dilation, and every asymmetric padding preserving 30x30. **0/400 hits**. AveragePool/LpPool share the same positive-support geometry and are covered by this negative result.
- Cost<=1 scalar/single-node sweep across all 400 tasks: CumSum (8 modes), ReverseSequence (H/W), and Trilu (upper/lower, k=-29..29). **0/400 hits**.
- Public artifact poll checked the latest 75 authoritative Kaggle kernels: no new or updated artifacts.
- Reopen conditions and evidence are recorded in `state/levers.yaml`; reusable scanners and tests are under `candidates/attr_pool_sweep/`.

## Next Session Start
1. Run `uv run ng status`, then check the latest Kaggle submissions before any pack or submit.
2. Decide whether to adopt task233 through the mandatory `ng adopt` gate. Its measured gain is +0.124281 points and all current validation is clean.
3. Continue with a qualitatively different cross-task mechanism; do not repeat the exhausted Pool or scalar-op enumerations unless their ledgered reopen conditions fire.
4. Poll public artifacts again before starting a long build, and rescan all 400 tasks whenever a genuinely new reusable mechanism is registered.

## Operational Guardrails
- Adoption only through `ng adopt`; submission only through `ng pack` then `ng submit`.
- Keep all candidate and scratch artifacts under `candidates/`.
- Preserve evaluator pins: onnx==1.21.0 and onnxruntime==1.26.0; do not upgrade without a 400/400 revalidation.
- Use FREE input/output tensors aggressively, but require bundled fail=0 and a cheaper model for overfit adoption.
- Before submission, check `kaggle competitions submissions` because parallel sessions may submit independently.
