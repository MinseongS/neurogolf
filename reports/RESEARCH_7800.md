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
