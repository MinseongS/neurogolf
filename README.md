# NeuroGolf 2026

Repository for the Kaggle NeuroGolf 2026 competition.

Current operating state is intentionally compact. Do not use old sweep/floorbreak
handoff docs as authority.

## North star (2026-07-06 directive)

- **Aim for 8000** (LB top ≈ 7982–8013), not "our base's ceiling". Every "floor / exhausted"
  verdict is a hypothesis to attack — floors here have been repeatedly falsified (e.g. task133
  "definitive floor" broken by an external net).
- **Exploit the FREE `input`/`output` tensors aggressively.** They cost zero and op internals are
  uncounted — route all work through them (contract against the free input, write directly into the
  free output, collapse spatial ops into one Einsum) so no counted intermediate plane survives.
- **Design only to "just passes the LB".** Gate = bundled fail=0 + smaller mem/params + fresh ~98%+.
  Don't over-engineer robustness past what scores.
- **Submit very freely.** 100/day, Kaggle keeps your BEST — a bad submit never costs standing.
- **Mine public INSIGHT, not just nets — then generalize.** Use public Kaggle discussions, code
  notebooks, and net dumps. When a public solution beats ours on a task, reverse-engineer WHY it is
  cheaper (the mechanism), record it in the playbook + insight registry, then apply it across every
  one of our 400 tasks that shares the pattern. A borrowed net = +1 task; a generalized mechanism = +N.

## Source of truth

- `NEXT_SESSION.md` — **THE live handoff / next-session prompt (authoritative; current best + queue).**
- `AGENTS.md` — repository-local agent instructions.
- `reports/REBUILD_PLAYBOOK.md` — proven mechanisms + reject-checks + gates.
- `reports/scripts/mine_public_bundles.py` — public-bundle mining routine (the S15 engine).
- `reports/source_live_reconcile.md` — source/live parity status.
- `reports/insight_registry.yaml` — reusable mechanisms.
- `reports/recursive_queue.md` — current global queue.
- `skills/neurogolf-recursive-improvement/SKILL.md` — project-local workflow skill.
- `reports/USER_REVIEW_WORKFLOW.md` — human-in-the-loop task review workflow.
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
