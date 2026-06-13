# NeuroGolf 2026 — Plan to 7000+

_Updated 2026-06-14. Real Kaggle LB: **6300.89** (400/400). Deadline 2026-07-15._

## The one thing that matters (learned the hard way)

**Kaggle scores against FRESHLY GENERATED arc-gen instances, not the examples we
downloaded.** Our local `evaluate` only checks stored examples, so:
- memorizer (exact-match lookup) nets score ~0 on Kaggle → useless.
- a conv/net that overfits the patterns it saw scores 0 → useless.
- ONLY nets implementing the TRUE rule (verified on fresh instances) earn points.

Every adoption goes through `python -m src.adopt N`, which gates on
`evaluate`-ok AND `genverify.fresh_pass` (fresh instances). The local "stored"
manifest number is vanity; **the only truth is the real LB and `fresh_pass`.**

## Where the points are (current 400 tasks)

| band (stored) | tasks | note |
|---|---:|---|
| <14 | 55 | mostly public/memorizer base; some real-0 hidden here |
| 14–16 | 174 | **prime custom-solve targets** |
| 16–18 | 99 | decent public/custom; push the clean ones higher |
| 18–20 | 55 | mostly good |
| 20+ | 17 | near-optimal |

- ~142 sub-16 tasks have clean (non-infeasible) geometric rules → custom-solvable.
- Rough ceiling if all 142 reach ~17.5: **~6700**; pushing the 16–18 band too gets
  toward **~6900–7000**. 7000 is at the edge of feasibility via custom solving —
  the LB top (~7710) is reached this exact way (per-task hand-crafted nets).

## Phases

**Phase 0 — reliability sweep (in progress).** Re-run `recover_merge` with
process isolation to undo any non-generalizing public nets that displaced
generalizing alternatives (e.g. task204 was a public net scoring real 0 while our
custom generalized at 13.90). One-time correctness recovery.

**Phase 1 — custom-solve the ~142 sub-16 geometric tasks (the bulk).**
Agent waves, 4 at a time, fresh-gated `adopt`. Each net +2–5 real. Skip the
infeasible classes immediately. Lowest first (most headroom). Target: ~6700–6800.

**Phase 2 — push the 16–18 band.** Where a public net is borderline or the rule
is a clean transform, replace with an exact custom squeezed to 19–21. Target: toward 7000.

**Phase 3 — borderline-risk cleanup.** Replace nets that fail fresh 1–5%
(Kaggle-zero risk) with exact rules even at equal nominal score (robustness).
Known: 23 157 76 2 209 118 233.

**Ongoing:** re-verify + resubmit every ~30–50 real points to confirm and lock LB.

## Infeasible classes (skip — memorizer/public is the ceiling)

- flood-fill / connectivity / enclosed-region (187 251 286 338)
- multi-object reconstruction / shape-correspondence selection (96 319)
- output-grid-size not recoverable from input content (358)
- dense random-pixel scatter (255)

## Throughput & logistics

- 4 agents/wave; >4 burns the shared session quota (repeatedly killed runs).
- Session limits (5h rolling + daily) interrupt often; resume waves when they lift.
- ~20–40 successful solves/day realistic → 142 targets ≈ 1–2 weeks of waves.
  Deadline 7/15 leaves ample margin.
- Each agent: read SOLVING.md → `src.show N --gen` → build → `fresh_pass` 200/200
  → `src.adopt N`. Tools: Gather-index, runtime Conv/ConvTranspose weights,
  channel-perm MatMul, outer-product assembly, signed channel, ConstantOfShape,
  int8/bool intermediates, fuse into free `output`.

## Tooling (all in src/)

- `genverify.py` — fresh-instance verification (batch uses Pool maxtasksperchild=1).
- `adopt.py` — generalization-gated single-task adoption. **The only safe adopt path.**
- `recover_merge.py` — rebuild submission from best generalizing candidate per task.
- `truegen.py` — reliable per-task real-score audit.
- `show.py --gen` — ground-truth generator. `pipeline.py --pack` — build submission.zip.
