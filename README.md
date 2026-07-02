# NeuroGolf 2026

Repository for the Kaggle NeuroGolf 2026 competition.

Current operating state is intentionally compact. Do not use old sweep/floorbreak
handoff docs as authority.

## Source of truth

- `NEXT_SESSION.md` — copyable next-session prompt.
- `AGENTS.md` — repository-local agent instructions.
- `skills/neurogolf-recursive-improvement/SKILL.md` — project-local workflow skill.
- `reports/USER_REVIEW_WORKFLOW.md` — human-in-the-loop task review workflow.
- `reports/ACTIVE_RESEARCH_STATE.md` — current research state and next direction.
- `reports/HIGH_SCORE_FRONTIER.md` — high-score mechanism frontier.
- `reports/source_live_reconcile.md` — source/live parity status.
- `reports/insight_registry.yaml` — reusable mechanisms.
- `reports/recursive_queue.md` — current global queue.
- `src/custom/taskNNN.py` — source-owned builders for all 400 tasks.
- `networks/taskNNN.onnx` — local deployed ONNX artifacts, ignored by git.

## Core rules

- Public ONNX files are teacher artifacts only. Never blind-import them as final ownership.
- Keep design control in `src/custom/taskNNN.py`.
- Record reusable discoveries in `reports/insight_registry.yaml` or tasklogs, then rescan all 400 tasks.
- Verify stored and fresh behavior before claiming score improvement.
- Kaggle submissions are effectively abundant for this project: 100/day.

## Standard refresh

```bash
PYTHONPATH=. .venv/bin/python reports/scripts/build_layer_inventory.py
PYTHONPATH=. .venv/bin/python reports/scripts/find_insight_candidates.py
PYTHONPATH=. .venv/bin/python reports/scripts/source_live_reconcile.py
```

## Rebuild local ONNX artifacts

If `networks/` is absent or stale:

```bash
PYTHONPATH=. .venv/bin/python reports/scripts/rebuild_networks_from_source.py
PYTHONPATH=. .venv/bin/python reports/scripts/source_live_reconcile.py
```
