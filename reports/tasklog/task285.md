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
