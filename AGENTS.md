# NeuroGolf Agent Instructions

This repository is the working state for the Kaggle NeuroGolf 2026 effort.

## North star (2026-07-06 directive)

- **Aim for 8000** (LB top ≈ 7982–8013). Treat every "floor / exhausted / ceiling" verdict as a
  hypothesis to attack, not a stop sign — floors here have been repeatedly falsified.
- **Exploit the FREE `input`/`output` tensors aggressively** — they cost zero and op internals are
  uncounted; route all work through them so no counted intermediate plane survives.
- **8000 mode is the default.** Design only to "just passes the LB": bundled fail=0 + smaller
  mem/params. Fresh verification is diagnostic only and must not block a bundled-fail=0 score win
  unless the user explicitly switches to safe/private mode.
- **Submit very freely** — 100/day, Kaggle keeps your BEST.
- **Mine public INSIGHT, not just nets — then generalize.** Use public Kaggle discussions, code
  notebooks, and net dumps; when a public solution beats ours, reverse-engineer WHY (the mechanism),
  record it in the playbook + insight registry, and apply it across every one of our 400 tasks that
  shares the pattern. A borrowed net = +1 task; a generalized mechanism = +N.

## Mandatory Local Skill

Before doing NeuroGolf score work, read and follow:

- `skills/neurogolf-recursive-improvement/SKILL.md`

The repo-local skill is authoritative. Do not rely on external skill memory or
copies under `~/.codex/skills` for this project.

## Source Of Truth

- `pyproject.toml` and `uv.lock` own Python dependency resolution. Use `uv sync --dev`
  to restore the local `.venv/`, and run project commands with `uv run`.
- `src/custom/taskNNN.py` owns each task's build logic.
- `networks/taskNNN.onnx` is a local ignored build/deploy artifact.
- `submission/overfit_nets/` owns the active 8000-mode submission artifact set.
- `reports/score_modes.md` defines the current mode; default is 8000 overfit mode.
- `reports/manifest.json` records the source-owned/safe local score state, not necessarily the active 8000 submission score.
- `reports/source_live_reconcile.md` is required for source-owned/safe adoption; it is not the active overfit submission manifest.
- `reports/tasklog/taskNNN.md` stores task-specific reasoning, failed probes, and useful rules.
- `reports/insight_registry.yaml` stores reusable mechanisms.
- `reports/recursive_queue.md` is the global candidate queue.
- `reports/USER_REVIEW_WORKFLOW.md` describes the current human-in-the-loop task review mode.

## Operating Rules

- Public ONNX files are both teacher artifacts and valid 8000-mode overlays. A public/bundled ONNX
  can go into `submission/overfit_nets/` when it is cheaper and bundled-fail=0; still mine the
  mechanism and record it when it generalizes.
- Prefer source-owned semantic rewrites for reusable mechanisms, but do not let source ownership
  block an 8000-mode submission overlay.
- Use the repository-local `arc-gen/` tree as the authoritative generator source.
  Do not follow stale `/tmp/arc-gen` or external worktree paths.
- Keep scratch candidates and temporary ONNX files inside this repository, e.g.
  under `reports/candidates/`; do not create project work products in `/tmp` or
  external agent worktrees.
- Do not claim score improvement without stored bundled verification; fresh generator verification is optional diagnostic context in 8000 mode.
- Record failures as assets in tasklogs, especially when they rule out an entire mechanism family.
- After a successful or meaningful failed mechanism probe, update tasklog/registry as appropriate and rerun the global scripts. Run source/live reconcile only when source-owned files or `networks/` changed:

```bash
PYTHONPATH=. uv run python reports/scripts/build_layer_inventory.py
PYTHONPATH=. uv run python reports/scripts/find_insight_candidates.py
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
