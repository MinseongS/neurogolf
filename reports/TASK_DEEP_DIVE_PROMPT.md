# Task Deep-Dive Prompt

Use this when assigning one task to Codex.  Replace `NNN`.

```text
In /Users/minseong/project/neurogolf, deep-dive taskNNN only.

Mandatory operating rules:
- Read skills/neurogolf-recursive-improvement/SKILL.md.
- Read reports/TASK_RESEARCH_PROTOCOL.md.
- Treat existing reports/tasklog/taskNNN.md as a hypothesis, not truth.
- Do not work on any other task except when checking whether a discovered
  mechanism transfers.
- Do not stop after a shallow scan.
- Do not run whole-project scripts.  Do not rebuild all networks.  Do not run
  `source_live_reconcile.py`, `build_layer_inventory.py`, `find_insight_candidates.py`,
  `rebuild_networks_from_source.py`, `pipeline.pack`, or submission commands.
  Those are reserved for the main session.
- Only measure and verify taskNNN.

Deliverables:
1. Explain the task in human-readable terms from stored examples:
   input/output examples, shapes, colours, rule hypothesis, edge cases.
2. Inspect current source and live ONNX:
   src/custom/taskNNN.py, networks/taskNNN.onnx, manifest score/mem/params.
3. Build a cost anatomy table:
   which tensors/params dominate, why they exist, and what semantic job they do.
4. Challenge prior notes:
   identify any old tasklog claims that are unverified, contradicted, or still
   valid with evidence.
5. Test the semantic rule as a Python oracle if the rule is uncertain.
6. Pick at most two serious mechanism hypotheses from TASK_RESEARCH_PROTOCOL.
   For each, define:
   - expected byte/param payoff;
   - proof test;
   - kill condition.
7. Implement only source-owned ONNX candidates in src/custom/taskNNN.py or a
   temporary local candidate path.
8. Verify with stored eval and fresh/adopt when available.
9. Update reports/tasklog/taskNNN.md with:
   - what changed;
   - failed probes and why they failed;
   - floor/wall proof if no win;
   - next exact experiment.
10. If a reusable mechanism is found, write a proposed registry entry or patch,
    but do not run global rescans.  The main session will merge the insight and
    run global scripts.

End state must be one of:
- adopted improvement;
- verified no-adopt but useful mechanism failure;
- prior tasklog contradicted with a new semantic hypothesis;
- blocked because a specific missing generator/source artifact prevents proof.
```

## Scope Boundary

Task agents own exactly one task.  Main session owns project-wide state.

Task agent may:

- inspect `src/custom/taskNNN.py`;
- inspect `networks/taskNNN.onnx`;
- inspect and update `reports/tasklog/taskNNN.md`;
- run `reports/scripts/measure_task.py NNN`;
- run stored eval for taskNNN;
- run fresh/adopt checks for taskNNN when available;
- create a source-owned candidate for taskNNN;
- propose an insight registry change.

Task agent must not:

- rebuild all networks;
- run source/live reconcile for all tasks;
- run global layer inventory or insight candidate scans;
- edit unrelated `src/custom/taskMMM.py`;
- edit many tasklogs;
- pack or submit;
- rescan public candidates;
- claim transfer to all tasks without main-session review.

Main session may:

- assign tasks;
- review task-agent claims;
- accept or reject shallow reports;
- merge/adopt final changes;
- update `reports/insight_registry.yaml`;
- run global scripts;
- run full source/live reconcile;
- pack and submit.

## Short Version

```text
Deep-dive taskNNN only.  Do not trust old tasklog claims.  Explain the rule,
profile current cost, challenge the prior wall/floor, test at most two serious
mechanisms, verify stored/fresh, update tasklog, and stop with either an adopted
win or a concrete proof/next experiment.  Do not run whole-project scripts or
touch unrelated tasks.
```
