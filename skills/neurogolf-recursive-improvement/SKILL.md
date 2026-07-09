---
name: neurogolf-recursive-improvement
description: Use when working on NeuroGolf score improvement in this repository. Enforces 8000-mode score chasing, repo-local artifacts, global insight propagation, and bundled-fail=0 verification.
---

# NeuroGolf Recursive Improvement

Use this skill for NeuroGolf score work in `/Users/minseong/project/neurogolf`.

## Current operating model: 8000 mode

- The objective is **single-submission public/LB score maximization toward 8000**.
- Python dependencies are owned by `pyproject.toml` + `uv.lock`; restore the
  local environment with `uv sync --dev` and run commands with `uv run`.
- Primary gate: **bundled fail=0 + lower `memory + params` than the active 8000-mode incumbent**.
- Fresh verification is diagnostic only in 8000 mode. It must not block a bundled-fail=0 score win unless the user explicitly switches to safe/private mode.
- `submission/overfit_nets/taskNNN.onnx` is the active 8000-mode submission artifact set.
- `reports/manifest.json` and `networks/taskNNN.onnx` track the source-owned/safe local tree unless explicitly regenerated for overfit mode.
- Keep reusable design control in `src/custom/taskNNN.py` when practical, but direct public/bundled ONNX overlays are allowed for 8000-mode submissions if they pass bundled fail=0 and improve cost.
- A discovery is useful only if it is recorded in `reports/insight_registry.yaml` or a tasklog and then rescanned across all 400 tasks.

## Negative-verdict recording rule (epistemic hygiene — MANDATORY)

This project has a documented failure pattern: it declares levers "exhausted / floored / ceiling /
unreachable" too fast, records that as durable truth, and stops rescanning. The same claims
("public min-merge exhausted", "we are at the byte floor") were falsified repeatedly (S15, S18,
2026-07-08 each added points a prior "exhausted" verdict said were impossible). A "floor" almost
always means "not found with tool X at time T against our own current net", NOT "does not exist".

Therefore, **never write a bare negative verdict.** Every "exhausted / floor / dry / no lever / ceiling"
claim (in memory, tasklog, or NEXT_SESSION) MUST carry all four fields, or it is not allowed:

1. **What was actually run** — the concrete search (tool, task sample size, e.g. "static-byte scanner + 9 deep agents on 9 tasks").
2. **Tool + date** — so staleness is visible.
3. **Reopen trigger** — the specific new fact that revives it (new uploader above last-mined ceiling, a new op-collapse scanner, an independent-minimum oracle result). No trigger ⇒ do not record the verdict.
4. **Falsification history** — has this verdict-type been proven wrong before? If yes, lower confidence in the new claim and say so.

Additional rules:
- **Floor must be measured against an INDEPENDENT minimum** (cristianoc algorithm floor, an op-collapse
  oracle, repr-tensor byte math), never against "our current net looks reasonable" (self-referential floor).
- A lever is **dormant, never dead.** "1 win then exhausted" is banned; replace with "1 win → full-400
  rescan → demote to dormant WITH reopen trigger".
- When a negative verdict is later falsified, **update its LEDGER** (do not silently overwrite) so the
  falsification-history field keeps growing — that meta-signal is how we learn to distrust our own "exhausted" calls.

## Required loop

1. From repo root, refresh the global view:

   ```bash
   PYTHONPATH=. uv run python reports/scripts/build_layer_inventory.py
   PYTHONPATH=. uv run python reports/scripts/find_insight_candidates.py
   ```

2. Read:

   - `NEXT_SESSION.md`
   - `reports/score_modes.md`
   - `submission/OVERFIT_README.md`
   - `reports/source_live_reconcile.md`
   - `reports/recursive_queue.md`
   - `reports/insight_registry.yaml`

3. Pick the highest-ROI candidate that is not blocked by a known wall reason.
4. Deep-dive one task:

   - inspect `src/custom/taskNNN.py`
   - inspect `networks/taskNNN.onnx`
   - read `reports/tasklog/taskNNN.md` if present
   - build or collect candidates under repo-local paths only
   - verify bundled fail=0 and measured cost with `src.harness.evaluate`
   - run fresh generation only as a diagnostic note, not as an 8000-mode blocker

5. If a reusable mechanism is found, update `reports/insight_registry.yaml`.
6. Rerun the global scripts and check whether the insight applies to other tasks. Run source/live reconcile only for source-owned changes.
7. Adopt into the 8000-mode submission set after bundled fail=0, lower cost, TopK scan, and a repo-local record of the result.

## Public model loop

When using public high-score models:

1. Put public ONNX files under `public_candidates/` with task id in path or filename.
2. Run:

   ```bash
   PYTHONPATH=. uv run python reports/scripts/public_teacher_scan.py
   ```

3. Read the generated teacher report.
4. For promising candidates, extract structure for inspection:

   ```bash
   PYTHONPATH=. uv run python reports/scripts/public_teacher_extract.py N public_candidates/.../taskNNN.onnx
   ```

5. If the public model is cheaper and bundled-fail=0, it may be copied into `submission/overfit_nets/` as an 8000-mode overlay.
6. Convert reusable ideas into source-owned code or an insight registry entry when the mechanism generalizes.
7. Do not overwrite `networks/` or `src/custom/` directly from public ONNX unless intentionally adopting a source-owned rebuild.

## Hard rules

- Do not optimize by average score only; measure the exact bundled pass/fail and cost.
- Do not spend long cycles on low-score wall tasks unless there is a concrete new mechanism or a public/bundled overlay to test.
- Do not claim score improvement without stored bundled verification output.
- Do not confuse safe/source parity with active 8000 submission state; state which mode a result belongs to.
- If `networks/` is missing or stale, rebuild it from source with
  `PYTHONPATH=. uv run python reports/scripts/rebuild_networks_from_source.py`.
- Preserve the fact that Kaggle submissions are effectively abundant for this project: 100/day.
