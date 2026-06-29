# Active NeuroGolf research state

Last consolidated: 2026-06-29 KST.

## Source-control base

- All 400 tasks are under source-owned control through `src/custom/taskNNN.py`.
- Latest source/live reconciliation reports `mismatches: 0`.
- Exact-preserve builders are scaffolding, not research wins. Future improvements should replace exact builders with semantic rewrites.

Authoritative files:

- `reports/source_live_reconcile.md`
- `reports/source_live_reconcile.json`
- `reports/scripts/live_to_exact_source.py`

## Current score reality

- Local manifest total is around `7178.33`.
- The target remains 7500+, but small +0.01/+0.02 changes are not the path.
- Daily submission limit is 100, so Kaggle can be used as an A/B probe when a candidate is reconstructible and clearly labeled.

## Main strategic conclusion

The important gap is not ordinary low-score grinding. The useful direction is:

1. catalogue mechanisms used by 20+ point tasks;
2. identify 15-18 point tasks whose semantics can be rewritten into those mechanisms;
3. implement source-owned semantic rewrites;
4. record the mechanism and rescan all 400 tasks.

See `reports/HIGH_SCORE_FRONTIER.md`.

## Confirmed high-score frontier

- 20+ points generally requires `mem + params <= ~148`.
- 23+ points is single-digit cost.
- Any full `[30,30]` uint8/BOOL plane costs 900 and caps score below 20.
- Therefore a 7500/8000 path needs direct-output, scalar-count, single-op, grouped/dilated local, or tiny-fixed-pattern mechanisms; not full-canvas label maps.

## Public model policy

`public_candidates/` is local-only and ignored by git.

Public models can be used to extract ideas, but not as final authority. If a public model is better:

1. scan it as a teacher;
2. extract structure;
3. convert the insight into source-owned code;
4. verify stored/fresh;
5. record the insight.

## Recent negative/partial findings worth keeping

- `task092`: semantically simple sticks, but crossing/vertical priority requires a scalar label carrier; direct sparse output not solved.
- `task366`: foreground/dot-mask insight is valid, but safe non-background mode extraction is the blocker.
- `task191`: 8-orientation grouping rejected; stamp masks can split into dynamic footprints.
- `task054`: sparse edit stream saved bytes but failed because duplicate inactive `ScatterElements` overwrite semantics matter.
- `task233`: semantic breakthrough candidate exists, but `iter_bounded_span` still fails fresh; component/template assignment edge remains.
- `task285`: random per-instance color/shape mapping appears genuinely wall-like.

Keep detailed notes in `reports/tasklog/` and promote only reusable mechanisms to `reports/insight_registry.yaml`.

## Next work queue

1. Build a high-score mechanism catalogue from all tasks scoring 20+.
2. Map each mechanism to operators, cost, and semantic preconditions.
3. Search 15-18 point tasks for matching semantics.
4. Try one candidate at a time as source-owned rewrite.
5. After every success or hard failure, update tasklog/registry and rerun the global scripts.
