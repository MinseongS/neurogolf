---
deployed_cost: 12034
logged_costs_match: exact
migrated: 2026-07-09
---

# task054 — 264363fd

## Current live exact

`memory=6772`, `params=5262`, `cost=12034`, `points=15.604509`.

The deployed graph is the 2026-07-14 bounded relational renderer. It detects the
reference motif and box/seed relations, renders all required lines and motif stamps in
one signed relation, preserves arbitrary source pixels, and resolves the fixed three-box
case by exact cancellation. Isolated Kaggle submission 54689861 scored 15.60 before
adoption; the official gate then passed 266/266 with fail=0.

## Dominant memory

- 3600B colour-index Conv output.
- Many 900B full-canvas uint8/bool edit planes:
  `label_u8`, `bg_mask`, `other/seeds`, `h_line`, `line_with_seeds`,
  `line_target`, `cleared_label`, `filled_label`, `output_label`.
- 1024B/960B int64 sparse fill/index tensors.

## 2026-06-29 sparse-edit-stream probe

Hypothesis: replace the full-canvas line cascade

`h_line_u8 -> line_with_seeds_u8 -> line_target_b -> output_label`

with sparse row/column edits applied directly to `filled_label`.

Graph-surgery candidate:

- Gather current rows from `filled_label` at `h_rows_4`.
- Use `h_updates_4_30 > h_seed_rows` to build sparse horizontal line updates.
- `ScatterND` those rows into `filled_label`.
- Gather current vertical columns and scatter vertical line updates.

Result:

- Best attempted candidate: `memory=26397`, `params=238`, but failed stored
  `244/266` (`22` arc-gen failures).  The raw saving was only `480B`.
- Variant trying to mimic inactive vertical overwrite by gathering from original
  `filled_label` failed much worse (`137/266`).

Reason:

The incumbent relies on `ScatterElements` duplicate/overwrite semantics for
inactive vertical slots.  Inactive vertical updates can intentionally erase a
horizontal line candidate before the final `line_with > seeds` comparison.
Applying sparse label edits directly loses this subtle mask-space behavior.

Conclusion:

The broad mechanism is still plausible only when sparse edits are monotonic
overwrites.  It is not safe when the intermediate mask uses inactive duplicate
scatter writes as part of the logic.  For task054, a successful rewrite must
either preserve the exact mask-space overwrite semantics or avoid generating
duplicate inactive vertical indices at the source.

## S8 (2026-07-02) — sparse-edit chain + free-input profiles (+0.234) ADOPTED
15×900B mask planes + 1920B CumSum stream → 4-plane sparse-edit chain: seeds via free-input
einsum row profiles ('bchw,c->h' + col-weighted) with motif 3×3 zone masked (seeds Chebyshev≥4
from centre); ScatterND ref-wipe → ring → row → col. KEY: line cells written as 255 with
reduction='max' → duplicate scatter writes = idempotent union (kills the 2026-06-29 overwrite
failure mode); final Equal matches 255-adjusted colour table. 20335+336 vs 25885+238 → +0.234.
Fresh 2500+5000+1500 div 0. TRAP: train[2] = fixed 3-box validation example outside the random
generator's 2-box path — box-agnostic segment spans required.

## S8 (2026-07-02, late) — select_last_index idiom ×4 (+0.014) ADOPTED, div 0

## S10 (2026-07-03) — scout re-confirm: FLOOR (label-read floor + irreducible edit chain)
Counted 20091 = labf 3600 (per-cell colour read, structural floor, measured 7 ways) +
fidx i64 1024 (ScatterND requires int64) + 4× uint8 edit boards 3600 (S8 golf result;
stages consumed sequentially, reduction='max' union semantics block fusion) + CumSum
segment planes. Seeds already free-input einsum profiles. Only sub-300B micro-levers left.

## S11 (2026-07-03) — signed-priority overlay (playbook 15) scout: KILL — output preserves the arbitrary input (per-cell read floor 3600B) and the line overlays are data-dependent 2-D interval fills = mechanism-15's own ~3000B band floor; incumbent's 4x900B edit boards land at the same cost. S8 FLOOR stands.

## 2026-07-07 — 8000-mode initializer dedupe micro-overlay ADOPTED

Built `reports/candidates/task054/task054_dedupe.onnx` from the active overfit net by
deduplicating six identical scalar initializers and rewiring their node inputs to the
first identical initializer.  No graph behavior changed.

Bundled gate:

- incumbent: `pass=266`, `fail=0`, `memory=20091`, `params=288`, `cost=20379`,
  `points=15.077739762140588`
- candidate: `pass=266`, `fail=0`, `memory=20091`, `params=282`, `cost=20373`,
  `points=15.078034226218099`

Adopted directly into `submission/overfit_nets/task054.onnx`.  This is only a
params micro-golf; the S8/S10/S11 structural floor assessment remains unchanged.

## 2026-07-07 — bundled dynamic-CSE active overlay ADOPTED

Built `reports/candidates/task054/task054_dynamic_cse_greedy.onnx` with
`reports/candidates/dynamic_cse_active_probe.py`.  The probe exposes
intermediate tensors over the bundled set, hashes their runtime values, and
rewires a later tensor to an earlier tensor only when bundled signature and
static shape/dtype are identical.

Main duplicate families removed: small scan row/column scalar/vector carriers
such as `sr*`, `sc*`, `mr*`, `mc*`, plus a few one-element flag/value aliases.
The full edit-chain floor remains; this only removes repeated intermediates in
the current active graph.

Bundled gate: fail=0.  Cost: 20373 -> 20186 (memory 20091 -> 19904, params 282
unchanged).  Active overlay updated in `submission/overfit_nets/task054.onnx`;
backup at `reports/candidates/task054/task054_pre_dynamic_cse.onnx`.

Follow-up pruning after the CSE rewrite removed now-dead initializers `TWO_I32`
and `C28_64` via `reports/candidates/task054/task054_prune_dead_constants.onnx`.
Bundled gate remained fail=0.  Cost: 20186 -> 20184 (memory 19904 unchanged,
params 282 -> 280).


## 20260709 — NO-WIN 재개 레저 (free-output-einsum fanout)
092-fanout(opus 딥 2-pass, 20260709): NO-WIN(단 룰 완전 크랙). oracle4.py = 검증된 정답 룰(266/266 bundled + 3000/3000 fresh, einsum-foldable primitive만; candidates/task054/에 보존). 배포넷(20131)은 이미 optimal compact-coordinate sparse-edit(ArgMax box-bounds→Gather/Scatter, full mask 미생성). 어떤 full-canvas einsum도 fp32 detection plane(label+box+seed+motif+segmented lines ≥18000B) 때문에 26k~37k > 20131. priority-fold는 anti-optimal(carrier는 이미 900B uint8 Equal tail). Reopen(공통): mixed-dtype Einsum escape(fp16 carrier + fp32 free-input co-bind) — ORT uniform-T가 현재 차단; 이게 풀리면 이 클래스 fp32-detection floor가 fp16으로 반토막. 또는 새 공개 덤프. public <19852 / generator가 box row/col 비공유 증명 시. ⭐재사용 primitive: 2-level(col-then-row) cumsum segmentation이 Loop/Scan 없이 stacked/transposed/overlapping 축정렬 rect 재구성. ⭐HEURISTIC: free-output/signed einsum fold는 counted mass가 avoidable label/priority CARRIER일 때만 이득 — fp32 spatial DETECTION이면 fold가 더 나쁨. 시도 전 detection-vs-carrier split을 측정하라.

## ADOPTED 20260714T151024Z
- cost: 20131 -> 12034 (points 15.6045)
- source: candidates/task054/relational_renderer.onnx
- note: LB probe 54689861 confirmed 15.60; relational renderer bundled 266/266; replaces cost20131 with cost12034
- sha256: a90259a9050599856df3bd714f3eff8758aa0ad1061af973379e857ca439956f
- validation: official bundled 266/266, fresh 40/40, pinned ORT execution complete
