# NeuroGolf Agent Instructions

This repository is the working state for the Kaggle NeuroGolf 2026 effort.

## Mandatory Local Skill

Before doing NeuroGolf score work, read and follow:

- `skills/neurogolf-recursive-improvement/SKILL.md`

The repo-local skill is authoritative. Do not rely on an external copy under
`~/.codex/skills` unless the repo-local skill is missing.

## Source Of Truth

- `src/custom/taskNNN.py` owns each task's build logic.
- `networks/taskNNN.onnx` is a local ignored build/deploy artifact.
- `reports/manifest.json` records current local score state.
- `reports/source_live_reconcile.md` must remain at `mismatches: 0` after adoption.
- `reports/tasklog/taskNNN.md` stores task-specific reasoning, failed probes, and useful rules.
- `reports/insight_registry.yaml` stores reusable mechanisms.
- `reports/recursive_queue.md` is the global candidate queue.
- `reports/USER_REVIEW_WORKFLOW.md` describes the current human-in-the-loop task review mode.

## Operating Rules

- Public ONNX files are teacher artifacts only. Never blind-import them as final improvements.
- Prefer source-owned semantic rewrites over graph surgery.
- Do not claim score improvement without stored verification; use fresh generator verification when available.
- Record failures as assets in tasklogs, especially when they rule out an entire mechanism family.
- After a successful or meaningful failed mechanism probe, update tasklog/registry as appropriate and rerun the global scripts:

```bash
PYTHONPATH=. .venv/bin/python reports/scripts/build_layer_inventory.py
PYTHONPATH=. .venv/bin/python reports/scripts/find_insight_candidates.py
PYTHONPATH=. .venv/bin/python reports/scripts/source_live_reconcile.py
```

## Current Collaboration Mode

The user wants to inspect tasks one by one and challenge the semantic analysis.
For each selected task, present the task in human-readable terms first:

- input/output examples and visual rule hypothesis;
- current ONNX mechanism and cost;
- whether the current analysis is verified, uncertain, or contradicted by visible data;
- concrete options for reducing memory or params.

When the user proposes a rule, test it as a Python oracle before building ONNX.
Only then build source-owned ONNX candidates.
