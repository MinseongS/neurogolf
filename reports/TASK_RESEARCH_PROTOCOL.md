# Task Research Protocol

This repo should improve one task at a time, but every task attempt must produce
global knowledge.  `task001` is the model: a real gain came from decomposing the
rule into scorer semantics, channel factorization, spatial routing, and rejected
ONNX/scorer edges.

## Session Shape

Use one session for one primary task unless the task is killed early by a known
wall.  Each session has four phases.

0. **Distrust the prior**
   - Treat old tasklogs, old wall labels, old fresh notes, and old mechanism
     conclusions as hypotheses, not truth.
   - Re-check the visible stored examples and the current source/live graph.
   - If old notes conflict with visible data, say so explicitly and update the
     tasklog.  Do not defend the old conclusion.
   - A task may only be skipped as a wall if the current session can reproduce
     the wall reason or the tasklog contains a concrete, still-valid proof.

1. **Read and explain**
   - Read `src/custom/taskNNN.py`, `reports/tasklog/taskNNN.md`, `manifest.json`,
     and the current ONNX profile via `reports/scripts/measure_task.py N`.
   - State the rule in human terms before proposing ONNX.
   - Mark confidence: verified, uncertain, or contradicted.

2. **Cost anatomy**
   - Split current cost into named components: entry colour read, label plane,
     full mask, scan stack, selector params, channel factors, output carrier.
   - Identify the dominant cost and the reason it exists.
   - Decide the highest possible tier:
     - `S`: mem0 / tiny params / direct output.
     - `A`: no full-canvas intermediates except final output.
     - `B`: one unavoidable full plane, but reduced dtype/area.
     - `FLOOR`: full entry/label plane is semantically required.
     - `WALL`: information loss, assignment ambiguity, or nonlocal generator edge.

3. **Mechanism tests**
   - First test the semantic rule as a Python oracle when the rule is uncertain.
   - Then test one mechanism at a time.  Each mechanism needs a proof test and a
     kill condition before implementation.
   - Prefer source-owned builders in `src/custom/taskNNN.py`.  Do not patch
     `networks/` directly except through adoption/rebuild.

4. **Adopt or record**
   - A win must pass stored eval and fresh/adopt when available.
   - A failure is still useful if it rules out a mechanism family.  Record it in
     the tasklog.
   - Promote reusable lessons to `reports/insight_registry.yaml`, then rerun:

```bash
PYTHONPATH=. .venv/bin/python reports/scripts/build_layer_inventory.py
PYTHONPATH=. .venv/bin/python reports/scripts/find_insight_candidates.py
PYTHONPATH=. .venv/bin/python reports/scripts/source_live_reconcile.py
```

## Task001 Lessons To Transfer

Task001 improved because the work moved from "implement the obvious rule" to
"exploit what the scorer actually checks".

- Output is thresholded with `> 0`; exact numeric one-hot values are not needed.
- Boolean products can sometimes be replaced by signed scores.
- Full carriers are usually worse than direct final-output equations.
- Dense algebra can be fit only on reachable generator states, not all theoretical
  colour pairs.
- Sparse initializer shortcuts are not valid under this scorer unless proven by
  official shape inference.
- When mem is already 0, params are the whole game.  Factorization, shared
  selectors, and sign-rank searches matter.

## Global Mechanism Families

Start every task by checking these families in order:

1. **Direct output threshold algebra**
   - Best for pairwise AND, fixed-coordinate products, small symbolic rules.
   - Ask whether `>0` scoring lets a mask/label carrier disappear.

2. **Input one-hot direct routing**
   - Best when output copies cells from input with crop/period/remap.
   - Avoids colour-index planes entirely.

3. **Final Equal / overlay-only output**
   - Best when only one overlay class changes.
   - Prefer `Where(mask, onehot_color, input)` over label plane + final Equal.

4. **QLinear / uint8 local routing**
   - Best for integer count, local stencil, small LUT, exact Conv replacement.
   - Reject if saturation, scale, or dtype support changes semantics.

5. **Bounded crop before scan**
   - Best for flood/connectivity when generator bounds are smaller than 30x30.
   - Needs generator maximum proof, not just sample luck.

6. **Template/component matching**
   - Best for small deterministic sprites.
   - Must prove component extraction and assignment edge cases.

## Initial Queue

Use this queue for the next detailed sessions:

| order | task | current | reason |
|---:|---:|---|---|
| 1 | 264 | 16.2915, mem 5348, params 706 | NEAR_18 shortlist; claimed +1.84 matched-filter glyph/template route |
| 2 | 270 | 16.9709, mem 2742, params 327 | NEAR_18 shortlist; scalar pull-back may delete directional MatMuls |
| 3 | 354 | 16.9683, mem 2998, params 79 | NEAR_18 shortlist; CumSum reset/run-boundary hypothesis |
| 4 | 086 | 16.9493, mem 2946, params 190 | NEAR_18 shortlist; morphology + final Equal possibility |
| 5 | 037 | 16.5349, mem 4614, params 132 | NEAR_18 shortlist; direction-separable diagonal Conv hypothesis |
| 6 | 368 | 16.2637, mem 6022, params 203 | NEAR_18 shortlist; rectangle offset/run-length chain |
| 7 | 193 | 15.3610, mem 15284, params 68 | possible manifest regression vs old ext score; audit before more work |
| 8 | 338 | 15.7121, mem 10140, params 666 | QLinear/local routing family still has visible cost |

Do not retry hard walls without a new primitive or external teacher:
`002 018 044 077 090 096 118 157 209 219 255 319 361 366`, plus `008` and
`286` as partial walls.

## Codex Turn Contract

For each task, ask Codex for this exact deliverable:

```text
Research taskNNN deeply.  First explain the visible rule and current cost
anatomy.  Then test at most 2 mechanism hypotheses from TASK_RESEARCH_PROTOCOL.
Do not adopt unless stored and fresh/adopt gates pass.  Record every meaningful
failure or win in reports/tasklog/taskNNN.md, promote reusable insights, and end
with the next concrete experiment.
```

## Anti-Shallow Rules

These rules exist because broad sweeps tend to produce shallow conclusions.

- Do not handle more than one primary task in a deep-dive session.
- Task agents must not run whole-project scripts.  Full reconcile, inventory,
  insight scans, rebuild-all, pack, and submission are main-session actions.
- Do not move to the next task just because two quick attempts failed.
- Do not summarize a task as "floor" without naming the exact tensor/op/semantic
  requirement that creates the floor.
- Do not trust old `confirmed-infeasible`, `wall`, or `fresh failed` notes unless
  the current session can reproduce the reason or the note contains a concrete
  counterexample/mechanism proof.
- Do not build ONNX before the semantic rule is clear enough to write a Python
  oracle, unless the task is purely graph-golfing an already verified rule.
- Do not end a session with only "no improvement"; end with one of:
  - adopted improvement;
  - verified wall/floor proof;
  - contradicted prior log and new semantic hypothesis;
  - exact next experiment with expected byte/param payoff.

## Deep-Dive Definition Of Done

A task deep dive is complete only when the tasklog contains all of the following:

1. Human-readable rule and confidence label.
2. Current score/memory/params/method.
3. Cost anatomy table with the dominant cost identified.
4. At least two mechanism hypotheses considered, or one decisive hypothesis with
   a proof that it dominates all others.
5. Stored verification result for any candidate graph.
6. Fresh/adopt result when a generator or adopt gate is available.
7. Floor/wall reason if no improvement lands.
8. One reusable insight or an explicit statement that no reusable mechanism was
   found.

## Human Role

The user does not have to manually solve every task.  The user's highest-leverage
role is to challenge semantic assumptions:

- "Show me the examples and your rule."
- "Why is that tensor necessary?"
- "Can the background be implicit?"
- "Are you using the fact that outputs are thresholded?"
- "Is this old wall note actually reproduced?"
- "What reachable generator states did you fit against?"

The agent must answer these with evidence from data, source, ONNX cost, or a
small oracle/probe.  If it cannot, the task remains `uncertain`, not `wall`.
