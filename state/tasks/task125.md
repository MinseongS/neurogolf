---
deployed_cost: 3463
logged_costs_match: no — ledger Attempts below describe the OLD ~8775 net; deployed net was
  re-golfed to 3463 (u8-MaxPool enclosure + code-plane + free Equal). Real structure documented in
  the 2026-07-11 audit section.
migrated: 2026-07-09
---

# task125 — 543a7ed5

**Rule:** 15×15 grid, background cyan(8). Several non-overlapping (separated by ≥2) solid pink(6)
rectangles, each with a smaller rectangular HOLE punched out of its interior (the hole shows as
background cyan inside the pink rect). OUTPUT, per rectangle: pink cells stay pink(6); the interior
hole becomes yellow(4); a 1-cell green(3) outline is drawn around the rectangle's bounding box.
Closed-form (verified 0/800): pink=in==6; enc = 4-directional prefix/suffix-OR of pink all true
(input has only pink & cyan, so first-non-cyan==first-pink ⇒ enclosed cyan == hole); region=enc∨pink;
green = dilate3×3(region) ∧ ¬region; out = 8; green→3; enc∧¬pink→4; pink→6.
**Current (prior):** 15.50 pts.
**Target tier:** A — closed-form bbox/enclosure, no flood-fill, no connectivity, output routed into FREE bool output.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 4-dir triangular-MatMul prefix/suffix-OR; hole=enc∧¬pink; green=maxpool−region; L=arith; Equal | A | 12825 | 484 | 15.50 | — | works, at floor |
| 2 | enc via product of 2 col/row sums; drop cyan slice (use ¬pink) | A | 11925 | 484 | 15.574 | — | trim |
| 3 | keep masks BOOL (225 vs 450); index via Where priority chain (3 Where) | A | 10350 | 484 | 15.71 | — | trim |
| 4 | priority-chain folds hole=enc & green=dilation (pink/enc override later → no ¬-masking) | A | 9450 | 484 | 15.796 | — | dropped 2 bool ANDs + 2 NOT |
| 5 | Cast index plane to uint8 BEFORE the 30×30 Pad (uint8 pad 900 vs fp16 1800) | A | 8775 | 484 | 15.867 | 500/500 | adopted-as-best |

## Best achieved
15.867 @ mem 8775 params 484 — adopted? N (per instructions). Beats prior 15.50? Y (+0.367). Fresh 500/500.

## Irreducible-floor analysis
Dominant intermediates: pink_f32 slice (900B fp32, Slice preserves input dtype), the L30 padded
index plane (uint8 30×30 = 900B, the one full-canvas plane the Equal one-hot must broadcast against),
and ~11 fp16 15×15 working planes (450B: pink-f16 for the 4 matmuls, the 4 directional sums, 2 row/col
products, region/dilation, 3 Where index stages). The fp16 15×15 planes are the active-canvas floor
(225 elems × 2B); the 4 matmuls are intrinsic to the 4-direction enclosure test. uint8 pad already
halves the 30×30 carrier; can't go lower than 900 there (bool can't be padded).

## OPEN ANGLES (re-attack backlog)
- Collapse the 3-stage Where index chain to 2 (encb⊂dilb): build inner=Where(encb,4,3),
  mid=Where(dilb,inner,8), L=Where(pinkb,6,mid) — same 3; a genuine 2-Where needs a packed
  pink/hole inner color routed by region. ~+0.05 if it lands.
- Pack the two row matmuls into one (pink @ [SU|SL]) — but the split+multiply re-adds planes; no net.

## 2026-07-11 AUDIT — deployed net is 3463 (ledger above was stale), July arsenal = NO WIN

**Real deployed structure** (13 nodes, 7 inits, params=29, all intermediates u8):
Slice input[:,6,1:14,1:14]→`p` fp32 676B (DETECTION, pink read) → Cast→`p_u8` 169B →
4 directional MaxPool (kernel 6, one-sided pad = bounded reach-6 dilation) L/R/U/D each 169B (676B) →
Min→`bounded` 169B (region = pink∧enclosed, per-rect 4-way AND — NOT global row/col profile) →
MaxPool3×3→`dilated15` 225B (green ring) ; `bp13`=bounded+p_u8 169B → Pad(+1)→`bp15` 225B →
Add(dilated15,bp15)→`color15` 225B (additive code 0/1/2/3 = bg/green/hole/pink) →
Pad(fill=4, →30×30)→`color30` 900B (label carrier, fill=4 is a never-matched don't-care sentinel for
the outside-15×15 region) → Equal(channel_ids[10,1,1])→ **free** output 9000B.
channel_ids=[5,6,7,1,2,8,3,9,0,10] is the inverse code→channel map (code0→ch8 cyan, 1→ch3 green,
2→ch4 yellow, 3→ch6 pink). Byte map sums to 3434 mem + 29 params = **3463**. Bundled 265/265 pass.

**Detection/carrier split:** DETECTION = `p` fp32 676B (irreducible pink read; input is fp32, Slice
preserves dtype; Slice-then-Cast=845B is the cheapest u8 materialization — Cast-then-Slice would
count a 9000B input cast). CARRIER = everything else, but all carriers are already u8 (1B/cell) and
minimal-shape (13×13 active, 15×15 ring stage, one 30×30 label). No fp32 dying carrier exists.

**Per-mechanism verdicts (tool=manual byte-model + onnx dump, 2026-07-11):**
- free-output N-ary Einsum — NO. Routing (region/ring) is fully data-dependent 2-D POSITION =
  positioned-content FLOOR. The 900B `color30` is an Equal DECODE discriminator (dense 4-colour recolor),
  not a low-rank Where routing mask (taxonomy pt2: dense-recolor floor 124/354/390).
- signed-einsum separable routing — NO. Output is per-rectangle enclosure+hole+ring at data-dependent
  positions, not separable axis-aligned rect fills; cost is DETECTION not priority-carrier (233-kill reject).
- s8port fp32 tail-fold — NO. Only fp32 plane is `p`, which dies into a MaxPool (nonlinear), not a
  linear contraction; fp32-coupling gate needs a linear dying carrier ≥~480B — absent.
- fp16 recast — NEGATIVE (would worsen). All intermediates already u8 (1B) < fp16 (2B).
- QLinearConv scale1/zp0 — NO. 4-way directional AND (enclosure) isn't a linear sum-threshold; a
  4-out-channel Conv reproduces the 676B but adds weight params → strictly worse.
- kernel-collapse / walk-chain slack / TopK refit / scatter-inverse / dynamic-kernel-stamp — N/A
  (no large Conv, no TopK, no scatter, no template match). Reach=6 lives inside MaxPool = free; trimming
  it saves 0 bytes.
- label-pad vs onehot-pad ordering — current Pad(label 900)→Equal(free) is optimal; Equal at 15×15
  then Pad = 10ch×225 = 2250B > 900B.
- Where(mask,onehot,input) overlay (output=input+edits) — NO. Two edit classes (yellow/green) need a
  counted [1,10,30,30] edit-onehot (9000B) or nested Where (9000B bridge) → worse than 900B label.
- crop-tighten — DEAD. Pink bbox = rows/cols 1..13 across all 265 bundled; crop already minimal.

**NEGATIVE LEDGER (4-field):**
1. ran: full onnx byte-dump + detection/carrier split + all 12 arsenal mechanisms hand-costed against
   the byte model; bundled re-verified 265/265; pink-bbox extent measured (rows/cols 1..13, minimal crop).
2. tool+date: manual ONNX shape-infer dump + arc-gen bundled eval, 2026-07-11.
3. reopen-trigger: (a) a public/top-team task125 dump beats 3463; (b) a new primitive that reads the
   fp32 pink channel as u8 in one <845B op (mixed-dtype Slice-cast, or free-input→u8 detector); (c) a
   proof the 4-way enclosure AND collapses to a single sum-threshold Conv (would kill 3 of the 5 detection
   planes); (d) grader change making outside-grid (rows≥15) cells scored (would break the fill=4 sentinel
   and force a different, possibly foldable, tail).
4. falsification-history: this AUDIT itself falsifies the pre-2026-07-11 ledger which claimed floor at
   15.867 pts / mem 8775 — the deployed net was silently re-golfed to 3463 by a later session and the
   Attempts table + OPEN ANGLES below are obsolete (they target the retired 8775-net epilogue).

## 2026-07-11 — rect-segmentation-primitive re-check (ci_triage builder) → DRY, CONFIRMS FLOOR
ran: generator-derived semantics (task_543a7ed5.py L54-62): pink(6) rects each with a punched hole
  (input hole shows as bg cyan); output = pink stays pink, interior hole→yellow(4), 1-cell green(3)
  ring around each rect bbox. This IS interval/band/fill rect-segmentation and the deployed 3463 net
  ALREADY implements the primitive optimally (4 directional MaxPool bounded-reach dilation = per-rect
  enclosure AND; Pad ring; additive 0/1/2/3 code; free Equal decode). Checked whether a 2-level cumsum
  formulation of the enclosure beats the deployed directional-MaxPool formulation.
tool+date: generator read + existing 2026-07-11 full-arsenal audit (this file, above), 2026-07-11.
verdict: NO cheaper build. A cumsum-based enclosure produces the same masks as the deployed directional
  MaxPool (reach-6 dilation lives inside a free op), so it saves 0 carrier bytes. Dominant cost is
  DETECTION — the 676B fp32 pink read `p` (Slice preserves fp32 input dtype) — which is orthogonal to
  the segmentation method and irreducible under onnx-1.21 op-types (see full audit above). Output is
  positioned-content dense-recolor (Equal decode), not a foldable low-rank Where mask. DID NOT BUILD.
reopen: identical to the full-audit reopen triggers above (public dump < 3463; <845B fp32→u8 pink read;
  4-way-AND→single-Conv collapse; grader scores outside-15 cells).
falsification history: consistent with the 2026-07-11 full-arsenal audit (FLOOR at 3463); the pre-07-11
  ledger's 15.867/mem-8775 floor claim was already falsified by the silently-regolfed 3463 deployed net.

## INSIGHT (transferable)
⭐ Multi-component "find each rectangle + fill its hole + outline its bbox" is NOT a flood-fill /
connectivity wall when the input has only TWO colors (shape + bg): an interior hole = a bg cell with
shape-pixels in ALL FOUR directions = `aL∧aR∧aU∧aD` where each is a strict-triangular prefix/suffix-OR
MatMul (the task070 bbox idiom applied per-direction, NOT globally — so it handles many separated
boxes without merging them). And in a Where PRIORITY chain you can DROP the ¬-masking of every
lower-priority mask (set hole:=enclosed, green:=full-dilation) because the higher-priority Where
(pink, then hole) overwrites the over-marked cells — this killed 2 ANDs + 2 NOTs (1.5kB) for free.
