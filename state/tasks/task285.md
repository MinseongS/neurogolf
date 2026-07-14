---
deployed_cost: 19623
logged_costs_match: match
migrated: 2026-07-09
---

# Task 285

## 2026-06-29 — angle-creature reconstruction wall check

- Baseline: `pts=14.848481180641883`, `mem=25080`, `params=550`, stored `265/265`.
- Generator semantics: 1..3 continuous creatures, each 4..8 cells in a 5x5 local frame. Four angle placements are generated around `(brow,bcol)` with one visible angle plus the anchor cell across angles; output reconstructs all legal colored angle placements.
- Public candidates checked (`biohack_mix`, `boristown`, `lucifer`, `urad`) are exactly equivalent to the live/source model: `pts=14.848481`, `mem=25080`, `pass=265/265`.
- Fresh generator sampling over 20k cases found:
  - max grid shape `(30,30)`;
  - max input nonzero cells `33`;
  - max output nonzero cells `96`;
  - max added cells `63`.
- The current `topk_cells_k=33` therefore matches the observed generator ceiling for visible input cells and is not obvious slack.
- Main cost floor is the full-grid one-hot-to-color-index path plus 30x30 masks:
  - `color_f [1,1,30,30] fp32 = 3600`;
  - `color_idx_flat_f [900] fp16 = 1800`;
  - full 900-cell `color_idx`, `scatter_grid`, `inside_mask`, `final_grid`, and output indexing tensors.
- Alternative one-hot color indexing via `ArgMax` would introduce an int64 30x30 intermediate, which is worse under current scoring. `Mul`/`ReduceMax` over 10 channels also materializes a 10x larger grid. No source-owned semantic rewrite was adopted.

Insight: sparse creature tasks can still be pinned by full-grid color-index conversion when pivots can occur anywhere in 30x30 and the visible-cell TopK is already tight.

## 2026-06-29 — uint8 TopK over compact label grid adopted

- Previous source/live: `pts=14.848481180641883`, `mem=25080`, `params=550`.
- Rewrite: removed `Cast(color_idx_flat -> color_idx_flat_f fp16)` before the first
  `TopK`; ONNX Runtime accepts `TopK` directly on the compact uint8 label vector.
- Stored eval: `265/265`, `mem=23247`, `params=550`, `pts=14.922685198705624`.
- Important verification nuance: both incumbent and candidate have rare failures
  against the fresh generator beyond the 120-sample adopt gate, so truth-fresh pass
  is not the right equivalence criterion for this local trim.
- Side-by-side incumbent-vs-candidate fresh audit: `20000/20000` identical outputs.

Insight: if a sparse pipeline uses `TopK` only to enumerate nonzero cells from a
compact uint8 label grid, an intermediate fp16 cast may be unnecessary. Verify
operator dtype support and side-by-side incumbent equivalence, because tie ordering
and rare generator failures can make truth-fresh gates misleading.

## 2026-06-30 — re-fit re-attack under reframe → FLOOR (mechanism-bound)

Re-examined with fresh eyes + arc-gen generator (`/tmp/arc-gen/tasks/task_b775ac94.py`).
**Rule (clean derivation):** each sprite = a `continuous_creature` (4–8 cells, always
rooted at (0,0)) drawn in 4 reflected quadrants around a 2×2 pivot at `(brow,bcol)`.
Input shows ONE quadrant fully (the `shows[idx]` angle) + the 2×2 pivot block (the 4
angle (0,0) anchors, one color each; an angle may be dropped → color 0). Output =
every visible cell reflected across the pivot's H/V mid-axes (`r↔2brow+1-r`,
`c↔2bcol+1-c`), each of the 4 copies colored by its destination-quadrant pivot color.

**Essential variables = the FULL spatial config of up to 3 sprites:** each sprite's
pivot (brow,bcol) ANYWHERE in a 12..30 grid, its shape (connected component), 4 pivot
colors, and shown orientation. This is irreducibly per-cell spatial info — NOT
reducible to counts/profiles/extents. So `color_f` 3600B fp32 read is mandatory (FULL_READ).

**Why the existing 25080 is at the mechanism floor (not the min_stat 4500 "floor"):**
min_stat's 4500 = 3600 read + 900 output carrier, and is BLIND to the reconstruction
working memory, which is genuinely irreducible here:
- Label pipeline `color_f(3600)→color_idx 4D uint8(900)→color_idx_flat(900)=5400`:
  Conv must emit 4D fp32; reshaping fp32 to flat costs a 2nd 3600 plane, so 4D-uint8
  then flatten is the cheapest path. Floor.
- Cell/pivot enumeration `TopK` feed `color_idx_flat_f` fp16 [900]=1800: TopK forbids
  uint8 INPUT (Kaggle grader-killer, playbook §2); int16/int32 are ≥ fp16. 1800 floor.
  Enumeration is unavoidable — pivots can be anywhere, must scan all cells.
- Pivot-detection `cand_marker/cand_source` [4,33,3] i32 indices (1584×2): 33 = proven
  max visible cells (3 sprites × 11 = (8−1 creature)+4 pivot); 4 orientations × 3
  neighbor offsets needed to detect the 2×2 multi-color pivot signature + creature
  attachment; Gather indices must be i32 (no int16). Must run on all 33 (pivots
  unknown a priori). A dense alternative (3 shifted 30×30 planes = 2700 + multi-color
  bool planes + still a fp16 enumeration TopK) measures ≥ the sparse machinery — no win.
- Output tail `color_flat_out(900)→scatter_grid(900)→final_grid(900)`: genuine 2-D
  scatter (≤96 cells anywhere) needs a grid² carrier (playbook §1); the clear=10
  sentinel + inside_mask are needed so off-grid→all-false. Fuses to no fewer planes.

**Tried/ruled out (no strict drop):** flatten-before-cast (adds 3600 plane); int16
indices (illegal for Gather); int8/int16 TopK (= fp16 or grader-killer); merge the two
[4,33,3] gathers (same bytes + extra split planes); drop color_idx 4D (forces 3600
flat). Only micro-lever found = dedup duplicate inits `marker_flat_offsets`(4,3) /
`cand_marker_flat_offsets`(4,1,3) → −12 params (Δpts ≈ +0.0005), not worth the risk.

**Verdict: FLOOR.** Incumbent (mem=25080, params=550, pts=14.85) is at the
reconstruction-mechanism floor; the min_stat +1.74 headroom is an unreachable
detection-only lower bound that ignores the irreducible 3-sprite reconstruction memory.
File left unmodified (clean).

## S8 (2026-07-02) — matrix-sweep verdict: priced FLOOR (block-4 opus agent). Do not re-attempt without a new mechanism.

## S11 (2026-07-03) — signed-priority overlay (playbook 15) scout: KILL — output = arbitrary connected-blob reflection into 4 quadrants (per-cell 2-D scatter, not separable); cost = 3600B FULL_READ + enumeration TopK + orientation-scan + connectivity MaxPool + 2-D scatter carrier. No deletable carrier.

## 2026-07-06 — 3-way parallel mechanism assault → FLOOR re-confirmed (empirical)

User pushed hard ("무조건 가능"). Ran THREE genuinely-distinct attacks in parallel, grader as arbiter. All three independently confirm 24684 is a real floor — not incompetence, but ONNX scoring counting every intermediate with no buffer reuse.

- **A · lean-enumeration** (attack fp16 dup + [4,33,3] indices + dup 900-planes): NO WIN. `33` cell budget empirically tight (max nonzero input cells = **exactly 33** over 40k fresh samples, hit 17×). Merging the two [4,33,3] index planes is byte-conserving (3168 either way). bool "≥2-of-3" algebra already optimal — ReduceSum rejects uint8; int32/fp16 reduction ([4,33,3]=1584/792B) costs MORE than the 1-byte bool planes (132B). Every tensor at minimal legal dtype.
- **B · dense flip-shift reflection** (eliminate enumeration entirely): built a CORRECT, fresh-gated alternative (bundled 265/265, 2000/2000 fresh bit-identical) — lands at **99952 bytes = 4.0× incumbent** (13.48 pts). Decisive finding: the dense multi-pivot detector = ~95 full-grid 900B planes = **86566B**, vs the enumeration's [4,33,3] machinery = **~6300B** (~13× cheaper). **The 33-cell enumeration IS the memory optimization, not the bloat.** Also: the reflect/scatter back-end is only ~6906B — it was never the bottleneck.
- **C · read attack** (halve the 3600 fp32 read via fp16/MatMul): NO WIN. **Reshape of the fp32 free input is never free** — value_info counts full declared bytes (measured: `Reshape(input)->[10,900] fp32` = exactly 36000B; no view escape hatch in calculate_memory's max(declared,trace)). Kills flatten-then-MatMul. Read-chain floor = **7200B fixed**: fp16-unification is an exact wash (TopK feed saves 1800 but plane+flat each double 900→1800).

**Verdict: 24684 FLOOR, now proven by 3 independent mechanisms (one a fully-working correct alternative).** Root cause = grader counts every intermediate, no in-place reuse → sparse 33-cell enumeration beats any dense full-grid path. The "clever math trick" the user wanted already lives in the incumbent (sparse enumeration = 13× cheaper than dense). Do not re-grind. Same class: 286 (sibling), 233, 366.

## S19 adoption (2026-07-07) — uint8 TopK feeds on active overlay (+0.0587)

- Candidate: `reports/candidates/task285/task285_uint8_topk_feeds.onnx` generated by
  `reports/candidates/task285/uint8_topk_feeds.py`.
- Current active overlay already beat the source-floor net, but still had three
  fp16 TopK feeds: `gf[900]`, `scoref[31]`, and `mf16[3,45]`, all cast from uint8
  score/label tensors.
- Rewrite: cast those TopK feeds to uint8 instead of fp16; change downstream
  `Greater` checks on TopK values from fp16 zero to existing uint8 zero.
- Bundled gate: incumbent 265/265 fail=0, candidate 265/265 fail=0.
- Cost: 19767 -> 18640 (memory 19347 -> 18220, params 420 unchanged).
- Active overlay updated: `submission/overfit_nets/task285.onnx`.

## S21 submission hygiene (2026-07-07) — reverse uint8 TopK feeds for Kaggle

- The S19 uint8 TopK feeds are local-ORT-valid but Kaggle rejects unsigned TopK
  inputs at submission level.
- Candidate: `reports/candidates/task285/task285_scan_clean_reverse_topk_fp16.onnx`.
- Rewrite: restore `gf`, `scoref`, and `mf16` TopK feeds/value outputs to fp16.
- Bundled gate: 265/265 fail=0.
- Cost: 18640 -> 19768 (memory 18220 -> 19347, params 420 -> 421).
- Active overlay updated to restore submit-ready unsigned-TopK-clean status.

## S22 public clean re-merge (2026-07-07) — recover part of S21 loss

- Source: `reports/candidates/public_mine_20260707/boristown_nets/task285.onnx`.
- After reversing uint8 TopK feeds, the boristown public net became cheaper than
  the scan-clean active net and is itself unsigned-TopK-clean.
- Bundled gate: fail=0.
- Cost: 19768 -> 19743 (memory 19347 -> 19323, params 421 -> 420).
- Active overlay updated via `mine_overfit_minmerge.py --apply`.

## S23 local-only REJECTED (2026-07-07) — signed INT8 TopK feeds (+0.0587)

Candidate: `reports/candidates/task285/task285_int8_topk_greedy.onnx`.
Recast `gf`, `scoref`, and `mf16` TopK feeds to signed INT8.  This is the
scan-clean replacement for the prior unsigned compact TopK feed.

Bundled gate: fail=0.  Cost: 19743 -> 18617 (memory 19323 -> 18196, params
420 -> 421).

## 2026-07-08 S31 — unused initializer drop ADOPTED

Candidate from `reports/candidates/dedupe_initializers_sweep.py`.

Dropped one unused initializer from the scan-clean active overlay.  Bundled gate
remained fail=0.  Cost `19627 -> 19623` (params -4); active overlay updated in
`submission/overfit_nets/task285.onnx`.

## 2026-07-08 — public-autopsy free-Einsum follow-up, K shrink refuted

Tool/date: after `public_win_autopsy.py` learned that the +4.51 public jump came from
`free_input_einsum_substitution`, `index_or_topk_plane_removed`, and
`final_equal_or_output_only`, task285 was rechecked because it remains high on the broad
wide/spatial/index candidate lists.

Current-public comparison:

- active: fail=0, memory `19203`, params `420`, cost `19623`, points `15.115542373`;
- franksunp 7249-50 / urad 7250.18 / lucifer 10:21: cost `19706` (worse);
- llccqq 10:18: cost `19700` (worse).

Current active profile confirms the remaining cost is the sparse enumeration mechanism:
`Conv(input)->cf` fp32 3600B, `Cast/Reshape` compact grid carriers 900B each,
`gf` fp16 TopK feed 1800B, padded gather vector 1024B, and final 900-cell scatter/output
tail.  This is not the same shape as task011's free-input Einsum collapse; the graph uses
enumeration precisely to avoid dense full-grid reflection machinery.

Probe: `reports/candidates/task285/task285_shrink_k32_minus{1,2,3}.onnx` reduced the current
cell TopK initializer `k32` from `31` to `30/29/28`.  Results:

- `31 -> 30`: bundled `264 pass / 1 fail`, memory `19105`, params `420`;
- `31 -> 29`: bundled `264 pass / 1 fail`, memory `19007`, params `420`;
- `31 -> 28`: bundled `261 pass / 4 fail`, memory `18909`, params `420`.

Shrinking `k9` then hits fixed downstream reshape assumptions (`{3,8,3} -> {81}`) before it
can be a valid cheaper graph.  No overlay adopted.

Mechanical follow-ups: `dynamic_cse_active_probe.py --tasks 133 285`, dead-constant probe, and
the global zero-compare rescan found no task285 win.  Reopen trigger: a public teacher strictly
below active, or a new representation that avoids both the full fp32 color read and the sparse
TopK enumeration without materializing dense pivot/reflection planes.  Falsification history:
uint8/signed TopK local wins were Kaggle-falsified by unsigned/signed TopK submission behavior,
so any future TopK dtype or K shrink must be isolated before adoption.


## 20260709 — NO-WIN 재개 레저 (free-output-einsum fanout)
092-fanout(opus 딥, 20260709): NO-WIN. data-dependent 2D reflection(shape→data-dependent pivot); free-output einsum엔 fold-coupling 텐서 ~18000 params 필요 > 19623 예산(=더 나쁨). 4회 독립 floor 확인 재확증. TopK feed 모두 fp16-clean 유지. Reopen(공통): mixed-dtype Einsum escape(fp16 carrier + fp32 free-input co-bind) — ORT uniform-T가 현재 차단; 이게 풀리면 이 클래스 fp32-detection floor가 fp16으로 반토막. 또는 새 공개 덤프. mixed-dtype면 fp16 shift carrier로 30×5 fold 가능(sibling 286/233/366 배치).

## 2026-07-10 representation-inversion re-audit (task233 lens, opus agent) — NO BUILD
- Ran: full graph dump + per-node byte map (onnx shape-inference / scorer trace), arg-select op trace, generator read, bundled-usage probes.
- Verdict: **NO-ARGSELECT-SUBSYSTEM**. 3 TopKs are enumeration/ranking (k=31 visible cells, k=3 pivot score, k=9 creature cells), not key-match arg-selects; eq [8,31] is a neighbor-consistency feature, not a correspondence matrix — no precomputed key set exists (pivot position is the thing being detected). Generator regenerates on overlap/adjacency (lines 46-55) so the consume-once fallback case structurally cannot occur; k widths bundled-tight (31->30 fails).
- Tool+date: opus triage agent, onnx 1.21.0 / ort 1.26.0, 2026-07-10.
- Reopen triggers: new public net < 19623; mixed-dtype Conv halving cf 3600B (hard-closed); a reflection-reconstruction avoiding both fp32 read and full-grid scatter carrier (attack B dense alt measured 99952B = 4.0x, sparse is optimum).
- Falsification history: this is the systematic 233-lens sweep prescribed by STATE.md Active Vein 1 after the 2026-07-10 task233 win falsified its own 07-09 CLOSED verdict; lens applied and did not fire here.

## ADOPTED 20260711T042922Z
- cost: 19623 -> 18674 (points 15.1651)
- source: dumps/poby7722_7263/nets/task285.onnx
- note: min-merge from nets

## 2026-07-14 — affine-reflection compiler: exact formula, current contraction ERROR

- Derived a pivot-relative affine reflection formula and built
  `candidates/task285/affine_reflection.onnx`: two variadic Einsums, 5625 params,
  predicted scorer cost `9225` versus deployed `18674` (potential `+0.7052`).
- The mathematical compiler matched the generator rule on `5000/5000` numpy cases.
- Pinned ORT could not finish the first inference with graph optimizations disabled;
  the original single-Einsum contraction requested an estimated 291.6 TB temporary,
  and the staged two-Einsum pivot contraction also timed out locally.
- Authoritative single-task submission `54688350` (candidate SHA256
  `0098d582534fdc634474f37815c2cabb2a00f9ddc14de0050714369f52fe1276`)
  returned `ERROR`. No deployment change.

This rejects only the present contraction plan. The affine-reflection identity remains
a live mechanism if the pivot score can be computed by a bounded-memory staged graph.
