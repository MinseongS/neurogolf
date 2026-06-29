---
name: neurogolf-recursive-improvement
description: Use when working on NeuroGolf score improvement in this repository. Enforces source-owned control, global insight propagation, public-teacher discipline, and verified score improvement.
---

# NeuroGolf Recursive Improvement

Use this skill for NeuroGolf score work in `/Users/minseong/project/neurogolf`.

## Current operating model

- Keep design control in `src/custom/taskNNN.py`.
- `networks/taskNNN.onnx` is a local deployed artifact, ignored by git, not the source of truth.
- Public ONNX files are teachers only. Never blind-import public models as final ownership.
- A discovery is useful only if it is recorded in `reports/insight_registry.yaml` or a tasklog and then rescanned across all 400 tasks.

## Required loop

1. From repo root, refresh the global view:

   ```bash
   PYTHONPATH=. .venv/bin/python reports/scripts/build_layer_inventory.py
   PYTHONPATH=. .venv/bin/python reports/scripts/find_insight_candidates.py
   ```

2. Read:

   - `reports/ACTIVE_RESEARCH_STATE.md`
   - `reports/HIGH_SCORE_FRONTIER.md`
   - `reports/source_live_reconcile.md`
   - `reports/recursive_queue.md`
   - `reports/insight_registry.yaml`

3. Pick the highest-ROI candidate that is not blocked by a known wall reason.
4. Deep-dive one task:

   - inspect `src/custom/taskNNN.py`
   - inspect `networks/taskNNN.onnx`
   - read `reports/tasklog/taskNNN.md` if present
   - build only source-controlled candidates
   - verify with stored and fresh generation in a fresh process

5. If a reusable mechanism is found, update `reports/insight_registry.yaml`.
6. Rerun the global scripts and check whether the insight applies to other tasks.
7. Adopt/submit only after normal local/fresh/adopt gates pass.

## Public teacher loop

When using public high-score models:

1. Put public ONNX files under `public_candidates/` with task id in path or filename.
2. Run:

   ```bash
   PYTHONPATH=. .venv/bin/python reports/scripts/public_teacher_scan.py
   ```

3. Read the generated teacher report.
4. For promising candidates, extract structure for inspection only:

   ```bash
   PYTHONPATH=. .venv/bin/python reports/scripts/public_teacher_extract.py N public_candidates/.../taskNNN.onnx
   ```

5. Convert any useful idea into source-owned code or an insight registry entry.
6. Do not overwrite `networks/` or `src/custom/` directly from public ONNX.

## Hard rules

- Do not optimize by average score only; classify high-score mechanisms and transfer them.
- Do not spend long cycles on low-score wall tasks unless there is a concrete new mechanism.
- Do not claim score improvement without verification output.
- Do not break source/live parity unless adopting a verified better source-owned build.
- If `networks/` is missing or stale, rebuild it from source with
  `PYTHONPATH=. .venv/bin/python reports/scripts/rebuild_networks_from_source.py`.
- Preserve the fact that Kaggle submissions are effectively abundant for this project: 100/day.
