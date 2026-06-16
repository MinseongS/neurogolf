# RESUME — restart the autonomous sweep in a fresh session

## Step 0 — environment check (do this FIRST; the loop breaks without it)
```
ls /tmp/arc-gen/tasks | wc -l        # must be ~901. If MISSING (reboot clears /tmp):
tar xzf ~/.neurogolf-arc-gen.tar.gz -C /tmp     # restore the generators (no git remote exists for them)
ls .venv/bin/python                  # repo venv (onnx/onnxruntime/numpy)
ls /opt/homebrew/Caskroom/miniconda/base/bin/kaggle   # kaggle CLI
PYTHONPATH=. .venv/bin/python reports/lb_status.py    # prints current stored / confirmed LB / gap / PROJECTED LB
```
`reports/arc_mapping.json` hardcodes absolute `/tmp/arc-gen/...` paths, so the restore MUST land there.

## Step 1 — paste this CANONICAL loop prompt (reads live state from files; no stale counters)
```
/loop Autonomous NeuroGolf sweep (read reports/SWEEP_SYSTEM.md, STRATEGY_7500.md, BUILD_PROMPT.md first).
Each iteration: (0) `PYTHONPATH=. .venv/bin/python reports/lb_status.py` — note stored/proj-LB/gap (gap ~61.3
stable; if it grows after a submit, wins aren't translating). (1) check completed build agents; adopt via
`python -m src.adopt N` ONLY if generalizes AND beats current by >=+0.3 stored (else mark skip-marginal +
log insight); ignore duplicate notifications for already-adopted tasks; salvage a died-agent's leftover file
through the same gate (reject if worse). (2) mem_profile adopted nets; if mem>>tier floor queue a re-attack.
(3) update reports/tasklog/taskNNN.md + reports/sweep_ledger.json. (4) git commit each win AND `git push origin main` (remote = github.com/MinseongS/neurogolf, already configured); graduate reusable
insights into reports/BUILD_PROMPT.md. (5) read reports/lb_anchor.json for the last submit anchor; if a
submission is PENDING poll `kaggle competitions submissions -c neurogolf-2026`, and on COMPLETE record
reports/submission_log.md + rewrite lb_anchor.json (stored_at_submit, lb, time). (6) when current stored
exceeds lb_anchor.stored_at_submit by ~>=8 (≈5 wins), pack() + kaggle submit + poll + re-anchor. (7) dispatch
next 2-3 build agents (short prompt: "Read reports/BUILD_PROMPT.md and follow it INCLUDING anti-stall; TASK
N=<n> P=<pts>" + the entry-hypothesis sketch from reports/retriage_build_queue.json) — PICK next targets =
highest-gain entries in reports/retriage_build_queue.json whose sweep_ledger status is NOT done/confirmed-
infeasible (re-triage est is OPTIMISTIC & ~25% false-positive — trust adopt), then sweep_wave_queue.json
(21,42), then refill from sweep_ledger lowest-points non-bail pending. Skip confirmed-infeasible. GUARDRAILS:
Kaggle submit authorized every ~5 wins (never lose standing, best kept); auto-commit; max 2-3 concurrent
agents (if agents repeatedly STALL at 600s = env degraded → back off, retry ONE later, STOP after 4
consecutive failures and report); skill-creator only if a judgment-heavy procedure recurs (don't force).
Stop when the retriage_build_queue feasible set is exhausted + no productive work, or user interrupts.
kaggle CLI: /opt/homebrew/Caskroom/miniconda/base/bin/kaggle.
```

## State is fully in the repo (committed) + project memory — source of truth, not the prompt:
- reports/sweep_ledger.json — 1→400 status (done / retriage-feasible / pending-retry / confirmed-infeasible)
- reports/retriage_build_queue.json — ~42 remaining feasible targets ranked by est gain
- reports/lb_anchor.json — last confirmed submit (re-anchored each submit); lb_status.py projects from it
- reports/submission_log.md — LB history (6384.61→6393.20→6400.24→6406.72→6409.40→6419.29, all proj ±0.01)
- reports/{SWEEP_SYSTEM,STRATEGY_7500,BUILD_PROMPT}.md — system, 7500 ceiling analysis, build protocol
- reports/tasklog/taskNNN.md — per-task insight logs

## Honest status at pause (2026-06-16)
Confirmed LB **6419.29** (session start 6384.61, +34.68). Projected LB 6419.70. 7500 verdict: re-encoding +
the 50-feasible reservoir tops ~6800-6900; the 72 genuine-infeasible (flood/connectivity/correspondence) are
the scorer-capped wall — 7500 is NOT reachable by known methods. Paused after 5 consecutive agent stalls
(systemic env degradation). ~42 feasible reservoir tasks remain → resume restarts there.
