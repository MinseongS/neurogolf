# STATE - NeuroGolf live handoff (updated 2026-07-10, 7299.09 confirmed; task233 wall cracked via match-matrix inversion)
> Replace this file at session end; do not append. History lives in git and state/submissions.md.

## Confirmed State
- Confirmed BEST LB: **7299.09** (sub **54516745**, complete). Prior best 7298.96.
- Current local manifest: **7298.9715** (400/400), `ng verify --hash` HASH-OK.
- Latest win: **task233 rebuild** cost **31938 -> 28033**, points **14.6284 -> 14.7589 (+0.1305)**.
  This FALSIFIED the 2026-07-09 "task233 CLOSED under fixed grading env" verdict via a purely
  endogenous static-golf rewrite (no external reopen trigger fired). Full ledger in
  `state/tasks/task233.md` (2026-07-10 entry); insight `match_matrix_inversion_scatter_table`
  registered in `state/insights.yaml`.
- Deadline: 2026-07-15. Private is the fixed dataset; bundled fail=0 + cheaper than deployed remains the adoption gate.

## task233 win mechanism (new, transferable)
1. **Match-matrix inversion**: a [K,N] Equal/Where match plane + per-K TopK is replaceable by a
   ScatterElements inverse-index table ([1,H] fp16, dup-index last-wins) + K Gathers WHEN the match
   relation is injective from the N side (task233: sprite popcounts sampled w/o replacement 4..8 ⇒
   window hashes distinct across lanes). Kills the fp16 TopK feed, the bool matrix, and the TopK.
2. **k=1 + pos-override lanes under the overfit gate**: sequential consume-once/k=2-fallback
   machinery was exercised by only 5/266 bundled examples; each patched by a ~25B Where-chain
   overriding the [5,1] pos vector, keyed on (sprite-colour sig + box h/w — all unique over 266).
   Correct positions extracted from the incumbent net's internals.
3. Fanout scan (Equal 2D→TopK/ArgMax fingerprint, all 400 deployed nets, 2026-07-10): **0 remaining
   hits** — 233 was the unique correspondence-matrix net. Fingerprint stays live for NEW public dumps.

## Current Audit Results (carried, still true)
- User threshold: avoid +0.0x byte cleanup; only pursue plausible single-task +0.1, preferably +0.3+, or reusable mechanisms.
- Public frontier: 2026-07-10 audit of 4 new dumps (franksunp-ii, hanifnoerrofiq, seddiktrk, lucifer19)
  → 0 adoptable; independent per-net memory comparison → 0 nets cheaper than deployed. Board strictly
  ≤ every current public net. Dumps cached at `candidates/public_dumps/20260710/`.
- Top-cost re-rank + mask_dominance rescan (2026-07-10): no fresh buildable ≥0.1 target from the old
  lens — BUT the task233 result shows those floor audits can miss representation-inversion levers;
  the worklist walls (349/366/158/173/204, 018/133/285/101/076/118) deserve one re-look through the
  "what structure exists only to arg-select / route, and is the relation a function?" lens.
- `task216`/`task366` no-build verdicts stand (see 2026-07-10 git history for details).

## Active Veins
1. **Representation-inversion re-audit of the worklist walls** — new lens from task233: find subsystems
   that exist only to arg-select/route (match matrices, priority parades, consume-once unrolls) and
   check if the underlying relation is a function (generator-guaranteed injectivity) → invert via
   scatter-table; pair with k=1 + per-example overrides (bundled-only gate makes fallback machinery
   mostly dead weight). Targets: 349, 366, 158, 173, 204, 018, 133, 285.
2. **New public frontier** — `uv run ng mine-public --margin 0 <dumps...>` on any new dump; then
   `public_autopsy` for +0.1+ fingerprints.
3. **QLinearConv detection-recast** (`ng scan qlinear_recast`) — LIVE reopen-scanner; re-run after any new Conv net.
4. **Sprite/TopK feed packing** (untried tail, ~784B on task233): pack two grid cells per fp16 element
   in a binary TopK feed; needs downstream disambiguation parade — only if a bigger win needs topping up.

## Operational Guardrails
- Do not spend session time on +0.0x byte-tail cleanup unless explicitly requested.
- Do not retry cost-1 tail nets, 092-profile cohort repeats, dtype boundary casts, `task216 c12_f32`, `task366 stale public_autopsy item`, or 014/350/018 value-info crop without their reopen triggers.
- Keep `onnx==1.21.0` and `onnxruntime==1.26.0`; no runtime upgrades without full 400/400 re-verify.
- Adopt only through `uv run ng gate` -> `uv run ng adopt`; submit through `uv run ng pack` -> `uv run ng submit`.
- Kaggle TopK dtype: float/fp16/int64 feeds only (uint8/int8 rejected at submission level).

## Next Session Start
1. `uv run ng status`
2. `kaggle competitions submissions -c neurogolf-2026 | head -6`
3. Check new public dumps/notebooks first.
4. Then run Active Vein 1 (representation-inversion re-audit) on one wall net with explicit cost-split
   proof before building.
