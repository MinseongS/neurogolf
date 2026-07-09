# NeuroGolf Score Modes

## Default: 8000 Overfit Mode

Objective: maximize the single Kaggle LB submission toward 8000.

Active artifact set:

- `submission/overfit_nets/taskNNN.onnx`
- repo-root `submission.zip` for upload
- `submission/OVERFIT_README.md` for rebuild notes
- `reports/overfit_manifest.json` and `reports/overfit_manifest.md` for measured active-set state

Gate:

- bundled `src.harness.evaluate` fail=0
- lower `memory + params` than the active 8000-mode incumbent for that task
- `scan_unsigned_topk.py submission/overfit_nets` before upload

Fresh generator verification is diagnostic only. It can be recorded, but it does not block an
8000-mode improvement unless the user explicitly switches to safe/private mode.

Public ONNX files under `public_candidates/` may be used directly as 8000-mode overlays when they
pass the gate. Mine their mechanisms into `reports/insight_registry.yaml` or tasklogs when the idea
can transfer to more tasks.

## 8000-Mode Iteration Loop

Use iteration freely when it lowers counted cost. The allowed trade is more search time, more ONNX
internal work, and more submissions in exchange for lower `memory + params`.

1. Pick a target from `reports/overfit_manifest.md` high-cost tasks, public-teacher deltas, or
   `reports/recursive_queue.md`.
2. Build many candidates under the task's scratch area. Optimize for bundled fail=0 and lower cost,
   not fresh robustness.
3. Exploit free tensors aggressively:
   - route final work directly into graph `output`;
   - keep graph `input` as the source for heavy reads instead of materializing full-canvas planes;
   - move work inside one large op when possible, especially `Einsum`, `Conv`, `QLinearConv`,
     `Gather`, `ScatterND`, `GridSample`, or final `Equal`;
   - delete counted intermediate carriers even if the replacement is slower.
4. Measure every promising candidate with `src.harness.evaluate`.
5. If bundled fail=0 and cost improves, copy it to `submission/overfit_nets/taskNNN.onnx`.
6. Refresh `reports/overfit_manifest.*`, run the TopK scan, pack `submission.zip`, and submit.
7. Because the daily limit is 100 and Kaggle keeps the best submission, submit aggressively when the
   local bundled gate says the score should improve.

## File And Artifact Locations

Final active 8000-mode artifacts:

- `submission/overfit_nets/taskNNN.onnx` — exact ONNX files that should be submitted.
- `submission.zip` — repo-root upload archive, generated from `submission/overfit_nets/`.
- `reports/overfit_manifest.json` / `.md` — measured active 8000-mode score state.
- `submission/OVERFIT_README.md` — submit/rebuild notes for the active overfit set.

Source-owned reusable implementations:

- `src/custom/taskNNN.py` — only when the mechanism is worth owning as source.
- `networks/taskNNN.onnx` — source/safe deployed artifact, not the default 8000 submission set.
- `reports/manifest.json` — source/safe manifest, not the active overfit score.

Temporary and candidate work:

- `reports/candidates/taskNNN/` — task-specific candidate scripts, notes, generated candidate ONNX.
- `reports/candidates/overfit_minmerge/` — min-merge output from public/bundled sources.
- `public_candidates/<source>/` — downloaded or extracted public teacher/submission ONNX files.
- `reports/retired_networks/` — backups before replacing active task nets.
- `reports/tasklog/taskNNN.md` — reasoning, failures, measurements, and adopted result.

Do not create project work products in `/tmp`, external agent worktrees, or hidden session memory.

Refresh command:

```bash
PYTHONPATH=. uv run python reports/scripts/build_overfit_manifest.py
PYTHONPATH=. uv run python reports/scripts/scan_unsigned_topk.py submission/overfit_nets
PYTHONPATH=. uv run python reports/scripts/pack_overfit_submission.py
```

## Source-Owned / Safe Mode

This mode is opt-in only.

Active artifact set:

- `src/custom/taskNNN.py`
- `networks/taskNNN.onnx`
- `reports/manifest.json`
- `reports/source_live_reconcile.md`

Gate:

- bundled fail=0
- lower `memory + params`
- fresh candidate fail <= incumbent fail
- `reports/source_live_reconcile.md` must cover `001-400` and report `mismatches: 0`

Use this mode only when the user asks for private robustness, clean adoption, or source/live parity.
