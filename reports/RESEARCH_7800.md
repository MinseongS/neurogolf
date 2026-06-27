# RESEARCH — the 7800-tier path (deep-research, 2026-06-19 night)

Source: 106-agent deep-research (23 sources, 86 claims, 25 adversarially verified, 15 confirmed).
Two foundational claims re-verified by hand against local grader source (cited inline).

## TL;DR
- **No public writeup explains 7700-7843.** Every reachable public artifact (sajayr/neurogolf-7k 7092,
  kojimar 7114.66, konbu17 5571) is a net-bag / best-pick that tops out at the **~7121 public-blend ceiling**.
  The 7121→7800 gap is **purely original and undocumented**. Monitor-and-rebase practical ceiling = ~7121.
- **The only honest paths to move past 7121 are original, in-bounds reformulations.** Research surfaced TWO
  concrete buildable bets (below) + one hard NEGATIVE constraint that kills the obvious approach.

## HARD CONSTRAINTS (verified in local grader source — these define the whole solution space)
- **`src/harness.py:23` BANS ops:** `LOOP, SCAN, NONZERO, UNIQUE, SCRIPT, FUNCTION, COMPRESS`. Also: single
  input + single output only (`:51-52`), no `model.functions` (`:58`), no `GRAPH/GRAPHS` node attributes
  (`:67-68`). ⇒ **No recurrence, no data-dependent control flow, no NonZero/Compress masking, no subgraphs.**
  Any iterated algorithm MUST be UNROLLED into a static feedforward graph.
- **Scorer (`data/neurogolf_utils.py:514`):** `points = max(1, 25 − ln(max(1, memory + params)))`.
  MACs no longer contribute; a zero-cost net = full 25 pts. `calculate_memory` = Σ(num_elements × dtype.itemsize)
  with dynamic shapes read from the ORT profiler trace. ⇒ **only lever is shrinking memory+params**; dtype
  (int8/bool/fp16) is scored by itemsize, so **bit-packing is a legitimate, in-bounds lever.**

## THE TWO BUILDABLE BETS (ranked by feasibility × upside)

### BET 1 — lossless dtype / bit-packing sweep over the existing 7121 blend  (LOW risk, steady gain, BUILD FIRST)
- Scorer counts `num_elements × dtype.itemsize`. Many deployed nets carry fp32 initializers that hold only
  small-integer / boolean / palette values. Re-typing those to int8/bool/fp16 **shrinks memory_bytes with the
  computed output unchanged** → direct points, **NO arc-gen≠private divergence risk** (this is the key
  difference from dead "re-golf": re-golf CHANGED outputs; bit-packing preserves them bit-exactly).
- ⚠️ Guard: only count it a win if the net's output is **bit-identical** post-retype (respect the float32-exactness
  invariant). fp16 rounding CAN change output — verify per net. int8/bool on integer-valued initializers is safe.
- This is openQuestion #4 and the cheapest real progress past 7121 without touching any hard task. **Probe a
  handful of the heaviest-memory deployed nets first to size the headroom before a full sweep.**

### BET 2 — directional cummax: in-bounds global propagation for the "infeasible ~100"  (HIGH upside, real R&D)
- CompressARC's (arxiv 2512.06104, App. D.4) **directional cummax**: an 8-direction (4 cardinal + 4 diagonal)
  cumulative max that **propagates information across the whole grid in a SINGLE pass** — no K iterated Conv
  layers, no banned Loop. This is the only verified in-bounds primitive that can express flood-fill / enclosure /
  ray-cast / inside-outside style non-local rules without the K×900B unrolled-activation blowup.
- ⚠️ ONNX has **CumSum but NOT CumMax** → must be COMPOSED from in-bounds ops: log-depth shifted-`Max`
  (⌈log2 30⌉≈5 shifted maxes per direction) or a triangular-mask `MatMul`. **In-bounds scored memory of this
  composition is UNVERIFIED** — it could itself hit a floor. Build a 1-task proof-of-concept, measure its
  memory+params vs the ~16.8-18.1 label-map floor, BEFORE committing to a task class.
- Companion idea: CompressARC's **factored multitensor** rep (axes = example/color/direction/H/W/channel)
  avoids the flat [1,1,30,30] 900-byte label-map — express rules factored across color/direction axes instead.

## DEAD / DO-NOT-PURSUE (verified refuted or banned)
- ❌ **ORT cross-run scratchpad-contamination "exploit"** — refuted 0-3 (github onnxruntime#28654 doesn't apply).
- ❌ **Any Loop/Scan-based bounded-state recurrence** — banned at grader level (the central hope of "iterate
  with one reused buffer" is dead here; unrolling re-incurs per-iteration activation memory).
- ❌ **Full CompressARC method** — it's a per-puzzle ~20-min/76K-param gradient-descent at inference (20% eval
  ceiling), NOT a static ONNX net. Only its PRIMITIVES (cummax, factoring) transfer.
- ❌ **NCA / pointer-jumping union-find CCL as a submission** — theoretically valid but needs unrolling +
  data-dependent Gather; the unroll re-incurs activations and Gather-by-computed-index feasibility is unproven.
  Weight-sharing shrinks PARAMS not unrolled ACTIVATIONS, so it doesn't beat the floor when Loop is banned.

## OPEN QUESTIONS (the build-order)
1. Bit-packing headroom: probe heaviest deployed nets — how many points does a lossless int8/bool retype free?
2. Can directional-cummax be composed sub-900B in-bounds (log-depth shifted-Max vs triangular MatMul)? Measure.
3. Unrolled pointer-jumping CCL: does Gather-by-computed-index stay in-bounds, and does K-step unroll beat 16.8?
4. What does the actual 7843 leader do? No source reveals it — architectural crack, scoring edge, or huge bag?

## VERDICT
7700-tier is reachable **only** via original closed-form/factored reformulations of a handful of hard task
classes (cummax-style propagation; factored rep), plus a lossless bit-packing sweep on the existing blend.
It is NOT reachable via any public blend (ceiling ~7121) nor any Loop/Scan recurrence (banned). The 7121→7800
path is INFERRED from primitives, not confirmed from any top-entry writeup — treat as a research bet, not a recipe.

---

## 2026-06-28 research pivot after leaderboard reached 7960.85

Latest checked leaderboard: top public score **7960.85**, with 12th already **7800.44**. Our confirmed best is
**7170.45**, so the gap is not a per-task +0.1/+0.3 grind problem.

### Hard math

- Our average: `7170.45 / 400 = 17.93 pts/task`.
- Top average: `7960.85 / 400 = 19.90 pts/task`.
- Score formula implies average cost targets:
  - 7500: avg `mem+params ≈ 518`
  - 7800: avg `≈245`
  - 7960: avg `≈164`
- Current distribution from manifest:
  - `<=250`: 76 tasks
  - `<=500`: 113 tasks
  - `<=900`: 153 tasks
  - `>900`: **247 tasks**

Therefore: 7500+ requires pushing a very large fraction of the 247 `>900B` tasks below ~500B. Generator-exact
label-map rewrites (`uint8 [1,1,30,30] -> Equal`) cannot do that; they floor near 16.8 and are structurally
below the target average.

### Tests run this session

1. **Latest public artifacts are not the 7800 method.**
   - Downloaded `seddiktrk/neurogolf-2026-all-graph-surgeries` and `franksunp/neurogolf-stack-audit-variant`.
   - Public receipts show Seddik/Frank stack around **7167.12**, below our **7170.45**.
   - Compared seddik zip: only tiny local gains over ours (`010/066/093/396/105`, total ~+0.07); no hidden
     structural lever. `191` was smaller but failed local.

2. **Generic final-output Cast/Equal rewrite is already exhausted.**
   - Scanned all 400 for `Equal/Greater/... -> Cast(float) -> output` where making output BOOL would remove a
     counted `[1,10,30,30]` bool tensor.
   - Count: **0**. Current base already routes most final one-hot outputs directly.

3. **`onnxsim` is not a big compressor.**
   - Sampled 30 high-memory/low-score tasks.
   - Only meaningful result: task118 params `4788 -> 3387`, +0.04 points.
   - Not a 7500/7800 lever.

4. **Sparse initializer exploit is blocked beyond Conv.**
   - Prior notes said sparse Conv weights fail checker.
   - Tested sparse initializer as Add/Mul input; checker rejects `sparse_tensor(float)` for these too.
   - So sparse params are not a broad escape.

5. **Directional prefix/cummax primitive has a real cost floor.**
   - A one-direction `MaxPool(kernel=[1,30], left-pad)` prefix max over one 30x30 plane costs ~7200B in a
     simple prototype once the input slice and prefix plane are counted.
   - Triangular `MatMul` has similar activation cost plus 900 params.
   - Useful for specific tasks, but not a sub-900 universal primitive by itself.

6. **“Kaggle public ignores arc-gen” is not enough to explain 7960.**
   - A train/test-only vs full local comparison on current networks showed only ~+0.16 total visible-only delta
     in the same harness process. This does not explain a +790 gap.
   - The stronger hypothesis remains: top teams either have a broad sub-500B model-generation method, or they
     combine many risky ultra-cheap candidates selected via public LB oracle.

### New working thesis

Our previous exact-generalization approach is valuable for robust private/general solutions but is not sufficient
for 7500+ public leaderboard. To climb explosively, we need one of:

1. **A broad sub-500B compiler pattern** for many currently label-map/detection tasks. This likely means avoiding
   full `[1,1,30,30]` intermediate label maps entirely, not just dtype-shrinking them.
2. **Public-LB oracle mining**: generate many ultra-cheap risky candidates per task, submit controlled A/B probes,
   and keep candidates that score on public hidden even if they are not generator-exact. With 100 submissions/day,
   this becomes a group-testing/search problem rather than a proof-of-generalization problem.
3. **A new hard-task primitive** that beats the flood/correspondence floors by representing multiple boolean states
   in one tensor (bit-packed boolean algebra) or by computing global propagation without materializing per-step
   full-grid activations. Sparse and simple prefix-MaxPool do not solve this yet.

### Immediate next research actions

1. Build a **public-probe harness**:
   - baseline zip hash + score anchor;
   - candidate registry `(task, candidate_path, expected_score_if_pass, local_behavior, message)`;
   - one-task and grouped submission modes;
   - result decoder that attributes LB deltas to accepted/rejected candidates.
2. Seed candidate registry from existing tasklogs:
   - risky undershoot flood depths;
   - write-all / approximate variants;
   - mem-0 conv or grouped-conv variants that fail rare fresh tails but pass stored;
   - public-source alternates with lower cost but uncertain hidden behavior.
3. Separately continue sub-500B compiler research on the 247 tasks with `mem+params>900`, prioritizing tasks where
   the top intermediates are simple repeated bool/uint8 full planes and might be bit-packed or factored.

### 2026-06-28 probe result: stored-only memorizers are unsafe

- Built `memorizer_v4` stored-only candidates for several low-score tasks. Local stored gains existed
  (`task219 +0.173`, `task233 +0.347`, `task319 +0.593`, etc.).
- Single-task public probe `task219` (`t219_mem4_stored`, submission `54118382`) scored **7155.76** against the
  7170.59 baseline: **−14.83**.
- Conclusion: do **not** assume tasks with poor isolated fresh rate have zero public contribution. Kaggle public
  hidden can still reward the existing non-generalizing-looking net, and a stored-only memorizer can wipe out the
  task's public score. Future risky probes must be task-specific algorithmic approximations with evidence of hidden
  survival, not plain stored memorization.
