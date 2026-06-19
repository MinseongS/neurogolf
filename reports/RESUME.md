# RESUME — restart the autonomous sweep in a fresh session

## ▶▶▶▶▶▶ RESUME HERE (handoff 2026-06-19 LATE — 🚀 LB 6667.42 → **7121.23**, ABOVE public crowd 7113.80)
**Confirmed LB 7121.23** (stored 7152.05, 400/400). Progression today: 6667.42 → 7107.01 (7k-harvest keep-best of
sajayr) → 7113.80 (pure kojimar audited blend) → **7121.23 (merge_E)**. The winning recipe + key lessons:
- PUBLIC NETS ARE THE LEVER NOW. Best public blend = `kojimar/neurogolf-7113-80-minimal-onnx-assets-v1` dataset
  (base_submission.zip + overrides.zip; overrides = full 400-net 7113.80 solution). Pull via `kaggle datasets
  download`. Other sources: sajayr/neurogolf-7k (7092 raw), octaviograu 6154, jsrdcht 6029, konbu17 blends.
- **EMPIRICAL RANKING (submitted & confirmed):** pure-sajayr-where-valid=7092.28 < our keep-best=7107.01 <
  pure-kojimar=7113.80 < **merge_E=7121.23**. Lesson: our keep-best+fresh-gate is SOUND (beat pure-sajayr); the
  crowd's 7113.80 is kojimar's audited blend (NOT sajayr's raw).
- **merge_E = THE WINNING METHOD (reports/merge_E.py):** base = best public blend (kojimar); OVERLAY our net ONLY
  where ours scores strictly higher on our examples AND passes fresh-200 (fresh-pass ⇒ generalizes ⇒ LB-safe);
  fall back to ours where kojimar's net fails our examples. 14 overlays + 5 fallbacks → +7.43 over pure-kojimar.
- **Fresh-gate is load-bearing** (reports/merge_7k_fresh.py): rejects overfit public nets that pass stored
  examples but fail held-out (caught sajayr's ~4, kojimar's wall-overrides). NEVER blind-merge public nets.
- Tools: `reports/compare_7k.py` (score any external onnx dir vs our manifest), `reports/onnx_inspect.py`
  (structure dump, `--theirs` reads /tmp dirs), `src.merge_external` (built-in keep-best, but NO fresh-gate).
**LB CONTEXT (2026-06-19 07:13):** top = 7843 / 7832 / 7800 / 7737 / 7687 / 7651 / 7634 / 7600 (top-14 all ≥7500).
Public-blend CEILING = 7113.80 (kojimar; no newer one yet). So the **7121→7800 gap (~720) is PURELY original
per-task golf** — the top teams hand-built it over months; public can't reach it. 400 tasks × ~+1.8/task avg needed.

**⚠️ STRATEGY REFINEMENT (verified by inspecting kojimar nets): GENERIC plane-elim is NEAR-EXHAUSTED.** kojimar's
mid-range nets already use OUR general levers (profile-sum, free-output routing, fp16/uint8 tails, small
intermediates) — independently discovered, same competition meta. So blanket re-golf has a LOWER hit-rate now.
Two kinds of our insight: (A) general golf levers = mostly already baked into kojimar's nets; (B) task-specific
cheaper RULES (documented in tasklog/ledger) = still differentiated. The 720-pt gap to 7800 is mostly (B) +
wall-cracking (new algorithms), NOT generic byte-shaving.

**▶ NEXT SESSION = AUTONOMOUS LOOP (just paste the loop prompt; it reads this + BUILD_PROMPT + HARD_WALLS):**
STEP 0: `ls /tmp/arc-gen/tasks|wc -l` (~901; restore if missing); `PYTHONPATH=. .venv/bin/python reports/lb_status.py`
(confirmed 7121.23, nothing pending). Then FIRST check for a newer/higher PUBLIC blend (free +N):
`kaggle datasets list -s neurogolf --sort-by updated | head` and `kaggle competitions leaderboard -c neurogolf-2026 --show | head`;
if a higher public blend dataset exists, download it, point `reports/merge_E.py` KOJI dir at it, re-run, submit.
THEN the re-golf engine, per iteration:
1. **Targets are pre-ranked in `reports/regolf_queue.json`** (244 non-wall tasks, lowest-pts-first = most headroom;
   40 true walls already excluded). Take the top untried N.
2. For each, `python -m reports.onnx_inspect <task>` (byte-accurate now) → classify dominant plane: REMOVABLE
   carrier (mask/union/candidate/extra colour plane / oversized output-routable) = WINNABLE; FORCED fp32 entry
   (10→1 colour Conv, single-ch fp32 Slice, flood/correspondence stack) = FLOOR, skip. This is the session's
   proven discriminator (don't waste agents on forced floors).
3. Dispatch the 8-wide plane-elim agent fleet (BUILD_PROMPT levers) on the removable-carrier targets. Build a NEW
   `src/custom/taskNNN.py` that BEATS the current (kojimar) net. Prefer (B): is there a fundamentally CHEAPER RULE
   than kojimar's approach (check the generator + our tasklog)? That's where real points are.
4. Each fresh-200-verified win → `python -m src.adopt N` (keep-best vs current kojimar net) → accumulates in manifest.
5. **Submit via merge_E recipe** when enough stored gain: keep current public base, overlay our fresh-verified wins
   (`reports/merge_E.py` already encodes this). ~5 submissions/day; transient 400 on submit → retry 60s via
   `submission/submission.zip`. Re-anchor `reports/lb_anchor.json` + log `reports/submission_log.md` each confirmed submit.
6. Mark tried tasks (won/at-floor) in regolf_queue.json / ledger so the loop doesn't re-attempt floors.
Realistic: +30-80 LB per re-golf wave (7121→~7180); 7500+ is a multi-session grind on (B)-type rule breakthroughs.

## ▶▶▶▶▶ (superseded) RESUME (2026-06-19 — 7k-HARVEST: LB 6667.42 → 7107.01, ABOVE sajayr 7015)
**Confirmed LB 7107.01** (400/400 solved, stored 7135.87, gap 28.86). Two things happened this session:
1. **Plane-elim re-golf wave (14 wins, +~6 stored)** before the harvest: 080/218/362/351/161/192/238/341/131/013/
   275/340/093/264/222 etc. New levers graduated to BUILD_PROMPT (nested-Where union-carrier, profile-Conv,
   pure-copy route, uint8 index-tail, occupancy-only collapse, combined-max-via-Conv-bias, BitwiseOr=uint8-max).
2. **🚀 7k-HARVEST (the big jump):** user flagged public notebook kaggle.com/code/sajayr/neurogolf-7015 (LB 7015).
   The NOTEBOOK is just a zip packager — NO scoring exploit (re-confirmed). The actual nets are in the PUBLIC
   DATASET `sajayr/neurogolf-7k` (395 onnx). Pulled via `kaggle datasets download sajayr/neurogolf-7k`. Scored
   all 395 vs ours (reports/compare_7k.py → compare_7k.json: 317 beat ours, raw +436). Adopted via
   `reports/merge_7k_fresh.py` = merge_external + a FRESH-200 GENERALIZATION GATE (critical — sajayr admits ~4-5
   fail private; our gate REJECTED 4 overfits, protecting real LB). Result: **313 adopted, keep-best kept OUR
   better models** (our re-golfs 284/165/387 correctly lost to their better nets). method tag `ext:sajayr7k`.
   We ended ABOVE sajayr's 7015 by ~92 because of the kept-better-models.
**▶ NEXT SESSION = PUSH PAST 7107 toward 7800.** The user's plan: sajayr's nets are ALSO just onnx → analyze their
   structure (`python -m reports.onnx_inspect <task> ` reads networks/, `--theirs` reads /tmp/ng7k/extracted) and
   RE-GOLF FROM THEM with our plane-elim levers (we beat "already-optimized" nets all session). Dispatch the same
   8-wide re-golf agent fleet but baseline = the now-adopted ext:sajayr7k nets; target the highest-mem ones first
   (inspect to find removable carriers vs forced fp32 entries — the session's hit/floor discriminator). Also the
   13 remaining sub-100% gap tasks (~28.86 pts: 219/255/157/2/319/366/118/233 — mostly true walls; 157 NOW has a
   working generalizing net from sajayr at 15.35). STEP 0: lb_status; nothing pending (7107.01 confirmed, anchor set).

## ▶▶▶▶ (superseded) RESUME HERE (handoff 2026-06-19 LATE — STRUCTURAL session; NEXT SESSION = OPTION 2 HARD-TASK CAMPAIGN)
Confirmed LB **6662.12** (#36). This session pivoted to a STRUCTURAL investigation after the user noted the top
of the leaderboard broke **7800** (we are ~6662 → a ~1140-pt gap that is NOT a ceiling). Session wins (proj LB
**6663.09**, +0.97 unsubmitted at handoff, 5 wins): 243 BFS-flood +0.20, 096 matched-filter-fp16 +0.23, 367
gather-free-corner-gate +0.32, 004 entry-collapse +0.34, 213 plane-elim +0.62. STEP 0 next session: lb_status;
if stored ≥ anchor(6691.36)+8 OR uncommitted plane-free wins exist, pack()+submit+poll+re-anchor.

⭐⭐ **STRUCTURAL FINDINGS (the real product — all EMPIRICALLY VERIFIED, graduated to BUILD_PROMPT 🔴 CORRECTION block):**
1. **NO SCORING EXPLOIT EXISTS.** Investigated Kaggle discussion 692827 ("Issues in onnx-tool") via authenticated
   browser. The Expand/broadcast onnx_tool trick (`Sqrt(Expand)` cheap) is DEAD: the official scorer
   `data/neurogolf_utils.py` (our harness mirrors it byte-for-byte) is TRACE-BASED — reads ACTUAL runtime shape
   from the ORT profiler, max(static,runtime). Direct test: Expand `temp` plane counts FULL 36000B (25.0→14.5,
   WORSE). onnx_tool 1.0.1 (the "fixed" ver) also counts it full; constant-folding doesn't change it. 36 proj-exact
   submissions confirm local harness == real LB. Other thread bugs (neg-step Slice off-by-one, ConstantOfShape
   zero-collapse, Constants-not-counted) were all PATCHED by the host's metric update. DO NOT chase scoring exploits.
2. **Scorer honors DECLARED dtype** (fp32=4B/fp16=2B/uint8/bool=1B per elem) — the old "3600B floor is universal /
   ORT upcasts to fp32" belief is FALSE. But we already narrow planes in practice, so only ~+6 there.
3. **Params = only 2.7% of total cost.** Sparse-initializer params exploit BLOCKED by check_model for ALL ops.
4. **The gap is PLANE-FREE REFORMULATION** (route full-grid result into the FREE "output", keep only scalar/vector
   intermediates). A net with even ONE 30×30 plane caps ~16.8-18.0; leader avg ~19.5 needs nets with NO full plane.
   The ONLY zero-cost full-grid result is naming the producing op's output "output" (e.g. `Sqrt(input)->output`=mem0=25.0).

## ▶▶▶▶ OPTION 2 — HARD-TASK + DEEP-REFORMULATION CAMPAIGN (what the user wants THIS-coming session)
The user chose: option-1 (plane-free harvest) finished last session; **this session do option-2.** Goal: attack the
~1140 gap for real. Realistic estimate **+150-300** (to ~6800-6950); reaching 7800 likely needs a technique not yet
found (the leaders had months) — flag if you find it. Priority order:
1. **HARD TAIL (~50 nets at 13-15 pts = flood/correspondence/walls):** use HARD_WALLS.md bounded-iteration unrolling,
   BUT apply the FLOOD-AT-FLOOR FAST-BAIL law first (BUILD_PROMPT, task286: compute floor 25−ln(2·D·2·Wk²); if the
   deployed net is already a MaxPool+Min unroll at size cap it's at floor). The CRACKABLE ones are mislabeled-closed-form
   (this session cracked 96/243/367 from this set; 367 v2 gather-free-carry was the model). Confirmed TRUE walls to SKIP:
   219/255/209/233/173/285/77/66/118/319/366/157. Re-attack candidates with unrolling/closed-form escapes per HARD_WALLS §5.
2. **RE-EXAMINE the "arbitrary-colour-copy needs 3600B fp32 entry plane" floor verdict (FLOOR_RESEARCH.md):** it likely
   OVER-CLAIMS. Distinguish FIXED recolor (10→10 colour permutation = a channel Gather/1×1 Conv routed to output = mem~0,
   like task016=22.7) from POSITION-DEPENDENT recolor (needs the plane). Many "at floor" single-plane nets may be fixed
   recolors mis-encoded with a full plane → big wins. This is the highest-EV untested reframe.
3. **Remaining plane-free scout targets (est ~4× high, so temper):** 355, 312, 82, 161, 132, 297, 362, 204, 256, 350,
   218, 80, 84-sib-263, 351. Full ranked table: search this file / sweep_ledger notes (scout ran 2026-06-19).
4. **Deeper competitor research:** read MORE of the Kaggle discussion/notebooks (browser cookies now set up — `browse`
   binary at ~/.claude/skills/gstack/browse/dist/browse, `goto` then `text`; ka_sessionid imported). The CompressARC
   paper (arxiv 2512.06104) frames ARC as weight-bit code-golf — possibly relevant. Look for any disclosed hard-task technique.
OPERATIONAL: same loop — `python -m src.adopt N` (ADOPTED-only), commit+push each win, submit at stored ≥ anchor+8.
Scout-estimate caveat: plane-free "est_gain" runs ~4× HIGH (213 est +3.7 → actual +0.62). The real win = whatever fully fuses.

## ▶▶▶ RESUME HERE (handoff 2026-06-19 — skip-list + skip-marginal floor-break session)
Confirmed LB **6658.65** (#34, proj-exact +14.47). Session 6635.63→6644.18(#33)→6658.65(#34) = **+23.02 confirmed,
27 wins / 8 walls / 1 reject**. **#35 PENDING at handoff (proj 6661.39, +2.74**: 6 skip-marginal floor-breaks) —
NEXT SESSION STEP 0: poll kaggle to confirm #35, set lb_anchor pending=false + add submission_log row. Anchor
stored_at_submit=6691.36; gap stable 29.97.

⭐ **THREE BIG LESSONS this session (all graduated to BUILD_PROMPT.md / project memory neurogolf-hard-walls):**
1. **The gap-region "skip-list walls" (219/255/209/118/2/90/157/366/251/18/101) with BLANK notes are NOT all walls.**
   251 (+2.06 hole-fill, bounded-unroll) and 090 (+1.62 max-empty-rect, suffix-min MaxPool) were FALSE-POSITIVES.
   Re-probe any blank-note wall with the hard protocol. (CONFIRMED true walls now documented: 018 info-bottleneck,
   101 chaining-placements-need-runtime-SE, plus 046/319/366/118/187/076.)
2. **"skip-marginal"/"at-floor" verdicts are ~40% wrong once new levers exist.** This session's new levers overturned:
   194 (+0.47 GridSample→invert-to-source-index-Gather), 365 (+0.65 global-argmax→two-forward-prefix-scans),
   032 (+0.63 colour-0==bg + crop-conv-on-free-input), 330 (+0.66 ScatterND-histogram per-component count),
   069/169 (component label/count via gray-gated multi-res sum-conv), 183 (+2.38 GatherND batch_dims=2).
3. **18.19 mem-0-Conv[10,10,3,3] cluster is GENUINE hard floor** (120/283/147/015 all confirmed): grouped-conv escape
   gated by SPAN (need g|10 with g≥|out_ch−in_ch|+1 per coupled pair; high-index non-copy ch reading ch0 = floor) AND
   CONTIGUITY (coupling component must be contiguous in channel order, else permutation Gather costs 9000B). Also:
   sparse_initializer Conv-weight shrink PERMANENTLY blocked (check_model(full_check) rejects sparse_tensor(float)).

🧱 **PRODUCTIVE RESERVOIR NOW EXHAUSTED.** Remaining no-custom skip-marginal are documented tight floors: 18.19 cluster
{98,171,294} (mem-0 conv erosion/frame, GENERALIZE), tiny fixed-crop fp32-slice floors {87,140,135,326} (160-360B
irreducible), 19.09+ near-optimal, 21.6 group {53,113,116,164} (do-not-resweep). 'pending' pool all ≥21.6 near-optimal.
True walls remaining (skip): 219,255,209,233,173,285,077,005,054 + the documented ones above. Practical ceiling reached;
the only runway is a deep custom effort on a genuine wall (research-grade), not the re-probe engine.

## ▶▶▶ PRIOR HANDOFF (2026-06-18 ~23:45 — pivot to HARD walls)
Confirmed LB **6635.63** (#32, +7.01). Session 6620.24→6628.62(#31)→6635.63(#32) = **+15.39, 30 wins / 11 walls**.
Anchor stored_at_submit=6665.61, current stored 6665.61 (nothing banked). gap 29.98 (grew +1.03: one wave-2 net
over-stored — find it among {58,48,208,265,85,29,333,162,134,382,178,30,355,117,80,110}).
**The easy reservoir is mined out**: 14–16pt blank-note "infeasible" false-positives are ~all done. ~6 remain
(44 canary in flight, 185,319,174,196,46 — re-dispatch; got cut off by an API session limit). Finish those, then:
**▶ THE NEXT PHASE IS HARD WALLS — read `reports/HARD_WALLS.md` FIRST.** The user explicitly wants the hard tasks
worked with research / long thinking / diverse methods (NOT fast-bail). Master key = BOUNDED-ITERATION UNROLLING
(flood=unrolled dilation D≈30, connected-components=label propagation, all expressible without Loop/Scan since
grid ≤30×30; proven on task48). PRIORITY = gap-closer tasks (157,319,366,118 — cracking one = DIRECT +LB, not
just +stored). Hard-wall agents get a LONG leash, ≥3 attack angles, may produce big nets (mem is fine: 25−ln(m+p)).
Loop prompt below still works; for the hard campaign, swap the agent prompt to the HARD_WALLS.md §3 protocol.

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

## ▶▶ RESUME HERE (handoff 2026-06-18 ~19:50 — evening session)
Confirmed LB **6618.58** (#29, proj-exact 25th). **#30 PENDING at handoff (proj 6620.24** — session-final: 73/149/352).
Evening session 6586.75→6618.58 confirmed (proj 6620.24) = **~+33.5**, **~49 golf wins**, 7 submissions (#24-#30) all proj-exact.
gap stable **28.96 == walls 219(15.00)+255(13.95)** (structural, unrecoverable). STEP 0 NEXT SESSION: poll kaggle to
confirm #30, set lb_anchor pending=false + submission_log row 30.

🧱 **THE PRODUCTIVE POOL IS EXHAUSTED (verified this session).** The 14-19.0 bloated/decomposable ext-import golf
pool is fully mined to near-optimal. Remaining ~17 fresh tasks are all **params≤40 near-optimal** (179/241=25.00
perfect; 16/276/309/337=22.70; 21.60-group params30; 307=21.31; 326/258 mem-floor) — every one re-confirmed
AT-FLOOR by a final probe batch (0 wins / 13 at-floor across the conv-group + params30-40 group). Plus the
mem0/900-conv group {127,282,317,331,230,258} = mem-0-single-Conv genuine-neighbour walls. **Do NOT re-sweep
these.** Only future runway: the 219+255 walls (confirmed infeasible) — i.e. none. Practical ceiling reached.

⭐ **GAP-CLOSER FINAL VERDICT (017+151+230 ALL FALSE this session):** `src.adopt real=0.00` and agent "fresh=0.00"
claims do NOT predict Kaggle — arc-gen fresh generators are harder/different than Kaggle held-out. ONLY a
post-submit LB-jump-above-proj confirms a gap-closer (274/332 were the only real ones, ever). ALWAYS `src.adopt`-gate
before trusting; bundle any test with golf wins so a false-positive nets ~0; Kaggle keeps BEST so standing never lost.
truegen [t,0.0,18.19] flags are STALE — re-verify the live net (317/282/230 all re-verified generalizing 200/200).

🔑 **NEW LEVERS this session (all in BUILD_PROMPT.md):** uint8 whole-pipeline (out>0 ⇒ output dtype irrelevant,
Equal/Not/And/Or/Cast/Concat/Pad/Gather/Where accept uint8; Mul/Add/Sub/ReduceSum reject); COUNT→FIXED-PATTERN
(tiny one-hot gated by Greater(schedule,cnt), Pad into free output, ~100B, hit 20+); COUNT-RANK iterative ArgMax;
scalar-recovery (size²=total/3 crop, side=√pixelcount for filled squares); DILATED-CONV (dense conv at fixed stride
→ small dilated kernel); dwconv-height-trim (offset-span brute-search); k-way-classifier (positional-Conv fingerprint
+ Equal-against-bank); GROUPED-CONV sub-floor escape (block-localised coupling → group=2, mem-0); Where-broadcast
colour-router; flip-via-Gather (neg-index-wrap free off-grid clamp); stacked-halves AND/NOR; bool-Pad opset13.
AT-FLOOR walls: mem-0 single/group-Conv genuine-neighbour-predicate emitting subtractive bg ch0 (params=elements,
fp16/sparse don't help — sparse-conv DEAD); fp32 small-region crop ~360B (10ch·3x3, uint8 cast only adds); full-canvas
GridSample params~1800; fixed-factor Kronecker grouped-ConvTranspose K·s²; full-width single-Gather copy params=output-dim.

[superseded morning handoff 2026-06-18 ~15:30: confirmed LB 6578.01 (#22), #23 pending proj 6585.60, +93 morning session.]

🔑 TARGETING (the big lesson, late this session): rank candidates by **MANIFEST points**, NOT stale ledger points.
`man=json.load(open('reports/manifest.json'))['tasks']`; pick lowest-points tasks with NO src/custom/taskNNN.py,
skip walls{219,255,209,118,2,90,157,366,251,18,101}. SWEET SPOT = **mid-manifest 14-16.5 bloated gen-imports**
(big wins: 191+2.24, 253+1.62, 325+1.55, 192+1.18, 338+1.08, 396+0.48...). LOWEST-manifest 11-13 are MOSTLY
WALLS (158/286/133/96 confirmed-infeasible: multi-object scatter / unbounded flood / correspondence / matched-
filter) though occasionally a +2 win (191). 16.x imports are near-optimal (marginal). ~38 mid-manifest left:
9-done,182-done,89-done,338-done,325-done; next 80,14,44,185,387,17,319,253-done... regen the list each session.

⚙️ OPERATIONAL NOTES: infra recovered (was down 7-stall streak mid-session; if it recurs, probe 1 agent then
fall back to MAIN-LOOP hand-building — recipe below — which never stalls). Run `python -m src.adopt N`
SEPARATELY and read its verdict BEFORE committing — adopt REJECTs when the deployed net already beats the
candidate (many "pending"-by-ledger tasks already have good ext nets, e.g. 81/146/122); on REJECT rm the
un-adopted src/custom file. Agents leave scratch in repo ROOT (build*.py/ref*.py/verify*.py) and task*.py.*
backups in src/custom — `find . -maxdepth 1 -name '*.py' -delete` + `find src/custom -name 'task*.py.*' -delete`
when idle. Agents often skip tasklog — stage only files that exist. Re-adopt if an agent's later report beats
its adopted checkpoint. Scaled to ~10 concurrent agents fine.

🛠️ HAND-BUILD RECIPE (main-loop fallback when subagent infra stalls): FIXED-SIZE simple recolor/fill/geometry
(small active region). `src.show N --gen`; build a small color-index plane L (per-row/col color via channel-
contract Σ k·input[:,k]; data-dep magnify via task159 two-gather), Cast uint8, Pad to 30x30 with SENTINEL 99
(off-grid MUST be all-False — convert_to_numpy leaves off-grid all-zero, in-grid bg=ch0=1), Equal(L,arange)
→BOOL output. Verify `solve_custom(load_task(N),task_num=N)`+`evaluate`; points=25−ln(mem+params). Proven on
task060/292/078. Avoid variable-size tasks needing a full in-grid channel read (too memory-heavy by hand).

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
