# NEXT SESSION — NeuroGolf sweep handoff (2026-06-18 ~15:30)

## State at handoff
- **Confirmed LB 6578.01** (#22, 20th proj-exact). **#23 PENDING** (proj 6585.60) — confirm first.
- Stored 6615.15, **gap 28.96 == walls 219+255** (the ONLY two 0-score tasks; everything else translates 1:1).
- Session: 6492.58 → proj 6585.60 = **+93.0**, ~97 wins. All state committed + pushed (github.com/MinseongS/neurogolf).
- +0.59 unsubmitted (straggler re-adopts 9/182 after #23) rolls into the next submit.

## STEP 0 (do first, fresh session)
```
ls /tmp/arc-gen/tasks | wc -l        # ~901; if missing: tar xzf ~/.neurogolf-arc-gen.tar.gz -C /tmp
PYTHONPATH=. .venv/bin/python reports/lb_status.py
/opt/homebrew/Caskroom/miniconda/base/bin/kaggle competitions submissions -c neurogolf-2026   # confirm #23 (proj 6585.60)
```
On #23 COMPLETE: set `reports/lb_anchor.json` pending=false + add submission_log row 23 with the real score.

## STEP 1 — paste this loop prompt
```
/loop Autonomous NeuroGolf sweep (read reports/RESUME.md, BUILD_PROMPT.md, SWEEP_SYSTEM.md first).
STATE: confirmed LB 6578.01; #23 PENDING proj 6585.60; gap 28.96==walls 219+255 (only 0-score tasks).
Each iteration: (0) lb_status; read lb_anchor.json, if PENDING poll kaggle + record submission_log + set
pending=false. (1) check completed build agents; run `python -m src.adopt N` SEPARATELY and READ its
verdict BEFORE committing — ADOPT only on "ADOPTED"; on REJECT (deployed net already better / fails stored)
rm the un-adopted src/custom/taskNNN.py + mark ledger done-or-skip; re-adopt if an agent's later report
beats its adopted checkpoint. GAP-CLOSER: if adopt prints "current: generalizes=False, real=0.00" the base
scored ~0 on LB -> submit immediately to lock its full stored as real LB. sweep_ledger.json is a LIST.
(2) update reports/sweep_ledger.json (mem/pts from manifest) + git commit each win AND git push origin main
(stage ONLY files that exist; agents often skip tasklog); graduate reusable levers into BUILD_PROMPT.md.
(3) submit when stored exceeds lb_anchor.stored_at_submit by ~>=8 OR a gap-closer is adopted; pack via
`python -c "from src.pipeline import pack; pack()"`, kaggle submit, poll, re-anchor (projections exact, gap
28.96). (4) TARGETING — rank by MANIFEST points NOT ledger: pick lowest-points tasks with NO
src/custom/taskNNN.py, skip walls{219,255,209,118,2,90,157,366,251,18,101}. SWEET SPOT = mid-manifest
14-16.5 bloated gen-imports (big wins +0.5..2.2). Lowest-manifest 11-13 are MOSTLY walls (agents bail
INFEASIBLE; fine). 16.x imports near-optimal/marginal. (5) keep ~8-10 subagents in flight; short agent
prompt: "Read reports/BUILD_PROMPT.md, follow it INCLUDING anti-stall; run `src.show N --gen` first;
checkpoint src/custom/taskNNN.py early; scratch to /tmp ONLY; TASK N=<n> (bloated import, headroom); beat
current by >=+0.3 stored AND generalize isolated fresh 200/200; INFEASIBLE fast if a genuine wall (variable-
count components / flood connectivity / multi-object correspondence / non-deterministic)". GUARDRAILS:
auto-commit; max ~10 concurrent; if agents repeatedly STALL at 600s (infra-level streaming hang) probe ONE
and if it also stalls FALL BACK to building nets MYSELF in the main loop (recipe in RESUME.md — never stalls);
watch GLITCHED agents (return their own instructions, 0 tool_uses, ~6s) -> re-dispatch; clean repo-root *.py
scratch + src/custom/task*.py.* backups when idle. kaggle CLI: /opt/homebrew/Caskroom/miniconda/base/bin/kaggle.
Stop when the mid-manifest pool is exhausted + no productive work, or user interrupts.
```

## Source of truth (committed)
- reports/manifest.json — current deployed net per task (RANK TARGETS BY ITS `points`, not the ledger)
- reports/lb_anchor.json — last submit anchor (projection re-calibrates from it)
- reports/submission_log.md — 23-submission LB history, all proj-exact (±0.01)
- reports/sweep_ledger.json (LIST) — per-task status; reports/BUILD_PROMPT.md — levers + protocol
- reports/RESUME.md — full handoff; project memory neurogolf-gap-closers — gap-closer method + discriminator

## Practical ceiling
gap 28.96 (219+255) is unrecoverable. Remaining runway = mid-manifest sweet spot (~38 untouched 14-16.5
tasks, ~+0.5/win avg with occasional +1-2) + a few low-manifest solvables (like 191 +2.24). Once mined out,
the practical ceiling is reached (avg ~17/task on solvables; 7500 not reachable per STRATEGY_7500.md).

## UPDATE (handoff continued, ~15:40)
- #23 CONFIRMED **6585.61** (proj 6585.60, 0.01). That is the live confirmed LB.
- **+1.79 unsubmitted** sitting on top: task14 (+1.20, adopt showed current real=0.00 → MIGHT be a gap-closer
  worth ~+15 real, or a false-neg like task105 — the next submit reveals which) + 9/182 stragglers.
  → NEXT SESSION: this already exceeds nothing-special, but SUBMIT EARLY to (a) lock +1.79 and (b) resolve
  whether task14 is a gap-closer (if LB jumps ~+15 over proj, it was). anchor.stored_at_submit=6614.56.
- task80 confirmed-infeasible (data-dependent-period linegrid, runtime-stride wall). 349/387/17/44/185 left
  pending (agents killed mid-run or rejected). Mid-manifest pool ~36 untouched remain.
