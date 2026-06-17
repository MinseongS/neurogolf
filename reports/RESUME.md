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

## TWO BIG REOPENED RESERVOIRS (discovered 2026-06-16 afternoon) — work these, NOT just the curated queue:
1. **UNTRIAGED PENDING POOL**: `sweep_ledger.json` has ~270 low-score `pending`/`uncertain` tasks the
   re-triage never examined. These have REAL HEADROOM (just under-golfed public nets), NOT all walls.
   Proven: 088 13.85→15.53, 070 13.90→16.25, 238 13.93→15.34, 011 14.12→15.94, 037 14.12→14.84,
   328 14.28→14.94 — ~6/8 probes were wins. PROBE lowest-points-first WITH the EARLY FEASIBILITY CHECK
   (bail fast on variable-size flood / global-argmax walls). Refill source = step 7 tertiary.
2. **GAP-CLOSING — CONCLUDED A DEAD END (2026-06-16, verified)**: all 3 gap giants are genuine walls —
   219 (information bottleneck, ~85% ceiling), 255 (connectivity wall), 209 (non-deterministic, ~97% ceiling).
   The 61.27 stored−LB gap is STRUCTURAL/unrecoverable: the overcounted base nets overfit tasks no function
   can solve. Do NOT spend more agents re-encoding gap-attribution tasks (the partial ones 118/2/90/157/etc.
   are near-optimal too). The PENDING POOL (#1) is the only productive reservoir. [Original thesis below kept
   for record — it was wrong about closeability:]
   ~~**⭐ GAP-CLOSING on NON-GENERALIZING base nets** (highest value): the 61.27 stored−LB gap is mostly a
   handful of base nets with HIGH stored but fresh-rate 0.00 → they score ~0 on the real LB. `src/adopt.py`
   ALREADY counts the current net as 0 pts when it fails fresh, so ANY generalizing custom you build for
   them is adopted and raises LB by ~its full score (even at low stored). See `reports/lb_status.md` gap
   attribution table. TOP TARGETS: **219 (stored 15.00, fresh 0.00 → ~+15 LB)**, **255 (13.95, 0.00 →
   ~+14 LB)**, then 209/118/2/90/157/366/251/18/101 (partial, fresh 0.88-0.97, +0.3-1.7 each). These were
   mislabeled "confirmed-infeasible" by judging +0.3 against the INFLATED stored — IGNORE that label and
   build a generalizing exact encoding. This alone is worth ~+25 LB from 219+255.~~ (NOTE: closeability disproven — all 3 walls)

## ▶▶ RESUME HERE (handoff 2026-06-18 ~04:40 — STOPPED on ENV DEGRADATION, not done)
Confirmed LB **6559.24** at last confirmed submit (#19, proj-exact 17th). Session 6492.58→6559.24 = **+66.66**.
Stored 6588.93, **gap 28.96 == EXACTLY the two walls 219(15.00)+255(13.95)** — gap is now fully structural.
19 submissions, all projections exact. **48 pending-pool wins + 2 GAP-CLOSERS (274,332)** adopted this session.
+0.73 unsubmitted (task176 salvage) — not worth a lone submit; rolls into next batch.

⛔ WHY STOPPED: 5 consecutive build agents stalled at the 600s watchdog (78,176,292,109 + probe 78).
Env degraded (likely transient/machine-load). NOT out of work — pending pool still has headroom (floor ~16.3,
wins ~+0.4 steady). RESUME = just re-run the loop; if agents stall again immediately, wait longer / fewer agents.

🔑 BIGGEST FINDING THIS SESSION (see project memory neurogolf-gap-closers — UPDATED): the "gap is structural
dead end" conclusion was WRONG. Non-gen base nets that score ~0 on real LB but are EXACTLY solvable are
gap-closers worth ~+16 each. Found 2 (274 conv1x59, 332 conv1x59), both proj-exact +16. DISCRIMINATOR:
non-gen base method (conv1x59/our-own) → solvable gap-closer; "gen:" IMPORT → almost always a WALL
(23/2/209/157 all confirmed walls). Reservoir now EXHAUSTED (gap==219+255 only). adopt.py reveals hidden
gap-closers incidentally: when `src.adopt N` prints "current: generalizes=False, real=0.00", that base was
a gap-closer — SUBMIT immediately to lock its full ~stored as real LB.

NEXT TARGETS: lowest-points pending in sweep_ledger (78,176-done,292,109 stalled-retry; then 217,312,197,
60,256,269...). Skip walls 219/255/209/118/2/90/157/366/251/18/101. Tell agents to CHECKPOINT build file
early + write scratch to /tmp (they clobber each other's root build.py and lose work on watchdog kill).

[prior handoff 2026-06-17: LB 6486.73→6492.58, 30 wins — superseded.]

THE PLAYBOOK FOR NEXT SESSION (do exactly this):
1. Run the canonical loop, 5-6 concurrent agents. PRIMARY reservoir = UNTRIAGED PENDING POOL: probe
   lowest-points `pending` tasks in sweep_ledger.json, EARLY FEASIBILITY CHECK to bail fast.
   NOTE: hit rate has DROPPED this session (~50% vs ~85% early) — the easy sub-15 tasks are mined out, the
   pool floor is now ~15.4+ pts and wins average ~+0.4. More skip-marginal/at-floor results; that's expected.
   `PYTHONPATH=. .venv/bin/python -c "import json;led=json.load(open('reports/sweep_ledger.json'));
   c=sorted([(e['points'],e['task']) for e in (led.values() if isinstance(led,dict) else led) if isinstance(e,dict)
   and e.get('status') not in ('done','confirmed-infeasible','skip-marginal','pending-retry') and e.get('points')];print(c[:20])"`
   gives next targets (skip gap tasks 219/255/209/118/2/90/157/366/251/18/101 — DEAD). pending-retry: 8, 242-done.
2. Adopt gate (`src.adopt N`) is generalization-aware and never lets score drop. Submit every ~5 wins / +8 stored.
   IMPORTANT adopt-gate naming: candidates MUST be zero-padded `src/custom/taskNNN.py` (e.g. task055.py not task55.py)
   AND `src.adopt` writes networks/taskNNN.onnx + manifest.json + truegen.json — COMMIT ALL of those, not just the .py.
3. Floor knowledge in BUILD_PROMPT. Levers graduated this session: sentinel row/col Gather (off-grid auto-handled),
   Pad-small-plane-as-output, MatMul rank-broadcast channel-contract, 2-weight signature boundaries, dual-mirror
   D2-symmetry hole recovery, min-area-cover candidate match, data-dependent period via Gather(c%p)+ArgMin.
4. IN-FLIGHT AT HANDOFF: agents on **237, 178** were marginal (logged skip-marginal). No salvage needed.
5. Agents drop scratch in repo root despite /tmp instructions — `find . -maxdepth 1 -name '*.py' -delete` before each
   commit; `git` only adds explicit paths so it's safe. Tell agents again to use /tmp + zero-padded filenames.

⚠️ GAP-CLOSER HUNT WAS A FALSE ALARM (2026-06-17): a single-process fresh_pass scan falsely flagged 220/230/282/317
as non-generalizing. ROOT CAUSE: arc-gen generators share module-level state → single-process scans pollute.
Cross-task fresh scans MUST use isolated processes (`python -m src.genverify`, Pool maxtasksperchild=1) or trust
the per-task `src.adopt` gate. There are NO big buildable gap-closers — only walls 219/255 + 9 small overcounts.
See reports/gap_closers.md + project memory neurogolf-gap-closers.

Verdict on ceilings: gap-closing is a CONFIRMED dead end (61.27 gap is structural). The pending pool is the
remaining runway and is now thinning (floor ~15.4+); once mined out, the practical ceiling is reached. 7500 is
NOT reachable (avg 18.75/task vs current ~16.3 + structural gap + ~14 BAIL-class floor on unsolvable tasks).
