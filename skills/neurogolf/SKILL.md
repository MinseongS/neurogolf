---
name: neurogolf
description: Use when starting or continuing NeuroGolf score work in this repository (any session that runs `ng scan/gate/adopt/pack/submit`, works a lever from state/levers.yaml, or touches submission/overfit_nets/). Trigger — session start on this repo, "NeuroGolf 세션 시작", "레버 작업", "다음 태스크 채굴".
---

# NeuroGolf — lever-engine operating manual

NeuroGolf score work in `/Users/minseong/project/neurogolf` runs through the `ng` CLI
(`uv run ng ...`) and the lever registry in `state/levers.yaml`. This skill is the loop;
`AGENTS.md` is the one-paragraph pointer to it. Read this before doing any score work.

Objective: maximize LB total = Σ over 400 tasks of `max(1, 25 − ln(mem+params))`, toward 8000.
Current state and numbers live in `state/STATE.md`, not here — this file is process, not status.

## 1. 세션 시작

1. `uv run ng status` — deployed net count (expect 400/400), manifest total points, git
   dirty-check, and the head of `state/STATE.md`.
2. Read `state/STATE.md` in full — the live handoff (BEST LB, active veins, invariants).
3. Open `state/levers.yaml` and pick a `status: live` lever (never `dormant` unless its
   `reopen` trigger just fired — see §3). Each lever entry names its `scanner`, `recipe`
   (a `playbook/*.md` file), `agent_class` (`opus`|`fable`), and `expected_yield`.

## 2. 표준 루프

1. **Scan.** `uv run ng scan <lever>` (optionally `--tasks N N N` to restrict) prints a
   worklist. `uv run ng queue` shows the standing cross-lever queue.
2. **Fan out.** Take the worklist's top N candidates and dispatch one agent per task via
   the `Agent` tool. Respect `agent_class` from the lever entry:
   - `opus` — verified recipe, mechanical application. Use `subagent_type: "opus"`-class
     agent (model override `opus`) when the recipe in `playbook/<file>.md` already covers
     the pattern.
   - `fable` — novel mechanism / regime crack, needs judgment. Use the default/heavier
     model (no override, or `fable` where supported).
   Each agent's prompt MUST include: the recipe file path (`playbook/<lever's recipe>.md`),
   the task ledger (`state/tasks/NNN.md`), and the deployed net path
   (`submission/overfit_nets/taskNNN.onnx`). Agents write candidates only under
   `candidates/taskNNN/` (repo-local scratch, gitignored) — never edit
   `submission/overfit_nets/` directly.
3. **Gate.** For each candidate: `uv run ng gate candidates/taskNNN/cand.onnx --task NNN`.
   PASS means bundled fail=0 and cost below the deployed incumbent; only a PASS may proceed.
   Never adopt a candidate that gate rejected — see §5 (gate 우회 금지).
4. **Adopt.** `uv run ng adopt candidates/taskNNN/cand.onnx --task NNN --note "<mechanism>"`.
   This re-gates, backs up the old net to `submission/.backups/`, replaces
   `submission/overfit_nets/taskNNN.onnx`, updates `state/manifest.json`, and **auto-stamps**
   `state/tasks/taskNNN.md` with an `## ADOPTED` block (cost delta, points, source, note) —
   no manual tasklog write needed for a win.
5. **Pack + submit, per batch (not per task).** After a batch of adoptions:
   `uv run ng pack` (builds `submission.zip`), then `uv run ng submit -m "<summary>"`.
   Log the result in `state/submissions.md`.
6. **Public-insight deep lane (mandatory pairing).** Whenever a `public-minmerge` adoption
   lands, immediately run the `public-insight-generalize` lever in the same session
   (`playbook/public-insight.md`, scanner `public_autopsy`): reverse-engineer the op-delta
   fingerprint, reproduce it as source in `src/custom/taskNNN.py`, register it in
   `state/insights.yaml`, then fan out to the fingerprint's `rescan_candidates` across the
   400 tasks. A min-merge byte-adoption without this follow-up pairing is an incomplete win —
   do not treat the min-merge alone as the end of the lever.
7. **Record.**
   - Win (gate PASS → adopted): already stamped by `ng adopt` (step 4) — nothing more to do
     beyond noting it in `state/submissions.md` at pack/submit time.
   - Dry run / no candidate beat the incumbent: record a `state/levers.yaml` ledger entry
     under the lever with the 4 mandatory fields from §3 (date/ran/verdict/reopen). Do not
     leave a negative result unrecorded, and do not record it anywhere except the ledger.

## 3. 에피스테믹 룰 (mandatory — carried over from the retired session skill)

This project has a documented failure pattern: declaring levers "exhausted / floored /
ceiling / unreachable" too fast, then stopping rescans. Claims like "public min-merge
exhausted" and "at the byte floor" were each falsified later (S15, S18, 2026-07-08, task011
net-surgery +1.52) and cost real LB by staying unexamined. A "floor" verdict almost always
means "not found by tool X at time T against our current net", not "does not exist".

**Never write a bare negative verdict.** Every exhausted/floor/dry/no-lever/ceiling claim —
in `state/levers.yaml`, a tasklog, or `state/STATE.md` — MUST carry all four fields, or it is
not allowed:

1. **What was actually run** — concrete tool + scope (e.g. "mask_dominance scanner + 8
   opus agents on batch6").
2. **Tool + date** — so staleness is visible.
3. **Reopen trigger** — the specific new fact that revives it (new uploader above the last
   mined ceiling, a new op-collapse scanner, an independent-minimum oracle result, a new
   sub-recipe). No trigger ⇒ do not record the verdict at all.
4. **Falsification history** — has this verdict *type* been proven wrong before? If yes,
   lower confidence in the new claim and say so explicitly.

Additional rules:

- **Floor must be measured against an INDEPENDENT minimum** (cristianoc algorithm floor, an
  op-collapse oracle, representation-tensor byte math) — never against "our current net looks
  reasonable" (a self-referential floor).
- **A lever is dormant, never dead.** `state/levers.yaml`'s `status` field only takes
  `live`/`dormant`. "1 win then exhausted" is banned; the correct move is "1 win → full-400
  rescan → demote to `dormant` WITH a reopen trigger" (see the `dormant` entries in
  `state/levers.yaml` for the pattern).
- When a negative verdict is later falsified, **append a new ledger entry** (do not silently
  overwrite the old one) so the falsification-history field keeps growing — that meta-signal
  is how the project learns to distrust its own "exhausted" calls.

## 4. 세션 종료

1. Replace `state/STATE.md` in place with only what is currently true — BEST LB, active vein
   list, invariants, next-session start procedure. **Append is forbidden**; history lives in
   git and `state/submissions.md`, not in STATE.md. If a claim in the old STATE.md is now
   stale, delete it rather than annotate it.
2. Commit. Stage the specific files touched (levers.yaml, STATE.md, submissions.md,
   tasklogs, source changes) — never `git add -A` blindly. Trailer:
   `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## 5. 안전 레일

- **게이트 우회 금지.** Every adoption goes through `ng gate` → `ng adopt`. Never copy a
  candidate onnx into `submission/overfit_nets/` by hand.
- **`submission.zip`.** `ng pack` always produces exactly this filename; Kaggle requires it.
- **Parallel sessions.** Before `ng submit`, check `kaggle competitions submissions -c
  neurogolf-2026` to avoid a duplicate/overlapping submit from another concurrent session.
- **100/day limit.** Kaggle submissions are capped at 100/day — batch adoptions before
  `ng submit` rather than submitting per task.
- **Grading-environment pin — never upgrade.** `pyproject.toml` pins `onnx==1.21.0` +
  `onnxruntime==1.26.0`. Do not bump either without a full 400/400 re-verify
  (`uv run ng verify`): onnx 1.22's strict shape-infer rejects negative-pad Conv nets, ORT
  ≥1.27 rejects a MaxUnpool net (task347), and ORT ≤1.23 lacks fp16 Max/ConvInteger CPU
  kernels that some adopted nets depend on.
- **Knife-edge nets → isolated eval only.** Single-Conv nets that rely on ORT weight-aliasing
  (220/230/294-style) can silently misscore in a batched 400-net eval loop (~54pt
  undercount observed). Always score a suspect net with `uv run ng score NNN` (isolated
  process), not a bulk sweep, before trusting a result on those nets.
