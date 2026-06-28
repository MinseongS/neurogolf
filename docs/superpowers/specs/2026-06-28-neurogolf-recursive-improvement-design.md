# NeuroGolf recursive improvement system design

## Problem

The repository now has source builders for all 400 tasks, but score work is still
too task-local.  A discovered mechanism should become a reusable insight that is
immediately checked against all structurally related tasks.  The workflow must
separate global pattern search from per-task deep work while keeping discoveries
in one shared registry.

## Design

Add a lightweight recursive improvement loop:

1. Build a global layer inventory from `networks/taskNNN.onnx`,
   `reports/manifest.json`, `reports/tasklog/*.md`, and `src/custom/taskNNN.py`.
2. Store reusable mechanisms in `reports/insight_registry.yaml`.  The file is
   JSON-compatible YAML so scripts can parse it without extra dependencies.
3. Generate `reports/recursive_queue.md` and
   `reports/insight_applications.json` by matching insight predicates against the
   global inventory.
4. When one task produces a new mechanism, add it to the registry, rerun the
   candidate finder, and apply/verify it across all matching tasks.
5. Use a Codex skill so future sessions follow this loop by default instead of
   regressing to one-off task work.

## Components

- `reports/scripts/build_layer_inventory.py`
  - Creates `reports/global_layer_inventory.json`.
  - Records ops, node counts, params, manifest score, method, source builder
    class, tasklog wall/open-angle tags, and pattern tags.
- `reports/insight_registry.yaml`
  - Central list of active mechanisms and matching predicates.
- `reports/scripts/find_insight_candidates.py`
  - Reads the inventory and registry, ranks tasks per insight, and writes a
    queue plus machine-readable applications.
- `reports/scripts/apply_insight.py`
  - Safe runner/skeleton for one insight/task pair.  It verifies the task source
    builder and tells whether an automated transformer exists.
- `~/.codex/skills/neurogolf-recursive-improvement/SKILL.md`
  - Forces future score work through global inventory → insight registry →
    queue → deep task → registry update → global rescan.

## Acceptance criteria

- Inventory indexes all 400 tasks and explicitly reports which tasks have no
  source-controlled `build()`.
- Candidate finder emits a non-empty queue from initial insights.
- Skill exists locally and documents the recursive loop.
- Existing dirty user files are not modified.
- No network/manifest adoption occurs in this setup step.

## Legacy policy

Do not delete tasklogs, ledgers, or submission logs until their information is
ingested into the new inventory/registry.  It is safe to remove generated caches
such as `__pycache__` and stale scratch logs after the new loop is verified, but
that cleanup should be a separate, explicit commit.
