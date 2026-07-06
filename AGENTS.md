# NeuroGolf Agent Instructions

This repository is the working state for the Kaggle NeuroGolf 2026 effort.

## North star (2026-07-06 directive)

- **Aim for 8000** (LB top ≈ 7982–8013). Treat every "floor / exhausted / ceiling" verdict as a
  hypothesis to attack, not a stop sign — floors here have been repeatedly falsified.
- **Exploit the FREE `input`/`output` tensors aggressively** — they cost zero and op internals are
  uncounted; route all work through them so no counted intermediate plane survives.
- **Design only to "just passes the LB"** (bundled fail=0 + smaller mem/params + fresh ~98%+); don't
  over-engineer robustness past what scores.
- **Submit very freely** — 100/day, Kaggle keeps your BEST.
- **Mine public INSIGHT, not just nets — then generalize.** Use public Kaggle discussions, code
  notebooks, and net dumps; when a public solution beats ours, reverse-engineer WHY (the mechanism),
  record it in the playbook + insight registry, and apply it across every one of our 400 tasks that
  shares the pattern. A borrowed net = +1 task; a generalized mechanism = +N.

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
- Use the repository-local `arc-gen/` tree as the authoritative generator source.
  Do not follow stale `/tmp/arc-gen` or external `.codex` worktree paths when a
  local generator exists.
- Keep scratch candidates and temporary ONNX files inside this repository, e.g.
  under `reports/candidates/`; do not create project work products in `/tmp` or
  `.codex/worktrees`.
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
