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
- **8000 mode is the default.** Gate = bundled fail=0 + smaller mem/params. Fresh verification is
  diagnostic only unless the user explicitly asks for safe/private mode. Don't over-engineer
  robustness past what scores.
- **Submit very freely.** 100/day, Kaggle keeps your BEST — a bad submit never costs standing.
- **Mine public INSIGHT, not just nets — then generalize.** Use public Kaggle discussions, code
  notebooks, and net dumps. When a public solution beats ours on a task, reverse-engineer WHY it is
  cheaper (the mechanism), record it in the playbook + insight registry, then apply it across every
  one of our 400 tasks that shares the pattern. A borrowed net = +1 task; a generalized mechanism = +N.

## Source of truth

- `NEXT_SESSION.md` — **THE live handoff / next-session prompt (authoritative; current best + queue).**
- `reports/score_modes.md` — active score mode definitions; default is 8000 overfit mode.
- `submission/overfit_nets/` — active 8000-mode submission artifact set.
- `reports/overfit_manifest.md` — measured active 8000-mode score state.
- `reports/scripts/pack_overfit_submission.py` — packs `submission/overfit_nets/` into repo-root `submission.zip`.
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

## Environment

Python dependencies are managed with `uv`. The local `.venv/` directory is a
reproducible cache, not source of truth.

```bash
uv sync --dev
```

Use `uv run` for project commands:

```bash
PYTHONPATH=. uv run python reports/scripts/build_layer_inventory.py
PYTHONPATH=. uv run streamlit run tools/onnx_viewer.py
```

## Core rules

- Public ONNX files are valid 8000-mode overlays when they are cheaper and bundled-fail=0; also mine their mechanisms for transfer.
- Keep reusable design control in `src/custom/taskNNN.py` when practical, but `submission/overfit_nets/` is the active 8000 submission set.
- Record reusable discoveries in `reports/insight_registry.yaml` or tasklogs, then rescan all 400 tasks.
- Verify bundled fail=0 and measured cost before claiming score improvement; fresh is optional diagnostic context in 8000 mode.
- Kaggle submissions are effectively abundant for this project: 100/day.

## Standard refresh

```bash
PYTHONPATH=. uv run python reports/scripts/build_layer_inventory.py
PYTHONPATH=. uv run python reports/scripts/find_insight_candidates.py
```

Run `reports/scripts/source_live_reconcile.py` only when source-owned files or `networks/` changed.

## Rebuild local ONNX artifacts

If `networks/` is absent or stale:

```bash
PYTHONPATH=. uv run python reports/scripts/rebuild_networks_from_source.py
PYTHONPATH=. uv run python reports/scripts/source_live_reconcile.py
```
