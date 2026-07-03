# task234 — 98cf29f8 (slide the line-coloured block to its partner)

**Rule:** two solid rectangular blocks of distinct colours connected by a 1-wide
straight line; the block whose colour == the line colour slides along the line
until adjacent to the other block, and the line is removed.  Output = the two
solid rectangles (one stationary, one moved) on background.

## 2026-06-30 — signed-Einsum routing (ADOPTED, +0.28)

The 4 prior golf agents all marked this "floor-bound: 900B [1,1,30,30] colour
label feeding the final Equal + arbitrary recolor blocks plane-free routing."
That was WRONG.

The output is exactly TWO solid colour rectangles on a background, i.e. a sum of
**separable rank-1 terms** in (channel, row, col).  The background channel (0) is
NOT a blocker: emit it as a third term whose coefficient is +1 on ch0 over the
whole in-grid box, and give each rectangle term a -1 on ch0 so a rectangle cell
nets ch0 = 0 (off) and only its colour channel is positive.  All three terms go
through ONE `Einsum('tnk,tr,tc->nkrc', weight[3,1,10], rsel[3,30], csel[3,30])`
whose output IS the free graph output.

This deletes the entire render path: `color_grid_padded` (900), the six
`[1,1,20,20]` mask/label planes (2400), and the final Equal.  The kept scalar
chain (object detection, armed/line-direction, moved positions) is reused as-is;
the new tail builds tiny `[3,30]`/`[3,10]` selector/weight tensors.

**Result: 16.448 pts @ mem 5113, params 66 — 266/266** (was 16.167 @ 6809).

**Verification:** stored 266/266; rebuilt the prior render graph and compared
OLD vs NEW on 1571 random in-domain instances — **0 cases of old-right/new-wrong**
(the 16 raw diffs are all out-of-domain malformed grids where my generator made
non-rectangular shapes; both graphs disagree with the rule there).  Source-owned,
`networks/task234.onnx` rebuilt, parity confirmed.  Fresh-gen N/A locally.

**Remaining floor (5113):** the two `[10,30]` fp32 per-colour bbox profiles
(`row_any_all`/`col_any_all`, 1200 each = 2400) now dominate — they are the
fp32 `ReduceMax(input)` occupancy reads (fp32 inherited from input), genuinely
needed to find both objects' bboxes.  Plus ~700 selector/weight planes.

⭐ **TRANSFERABLE (overturns the "background forces a 900B label" belief):** any
output that is a union of a few solid axis-aligned colour rectangles on
background can be emitted by a SINGLE `Einsum` of per-term (colour-weight,
row-selector, col-selector) straight into the free output, INCLUDING the
background channel, via a signed ch0 coefficient (+1 whole-grid term, -1 per
rectangle).  No colour-index label plane, no per-cell render.  Scan for tasks
whose output decomposes into <=~4 solid rectangles.

## S8 (2026-07-02) — rect-recipe conversion ADOPTED, div 0
per-colour bbox from einsum count profiles + Sign/ArgMax; row/col_any_all planes dropped; signed-einsum routing untouched; 5179→3808, +0.307. Fresh: agent uncached 2500 div0 + my uncached 400 div0.

## S11 (2026-07-03) — ADOPTED: fp16 recast of the input-free einsum island (+0.1715)
16.755 → 16.927 (3808B → 3208B). dtype_overpay_scan flagged 1387B; realized 600B because
only the FINAL routing einsum ('tnk,tr,tc->nkrc') has no free-input operand — its three
operands + upstream weight path (onehot_f/minus_bg/weight310/weight, values {-1,0,1})
recast jointly to fp16. The g0/g1/rp*/cp* detection planes are dtype-bound (co-operand =
fp32 free input) and stay fp32. ⭐TRANSFERABLE: recast whole einsum ISLANDS (connected
subgraphs with no free-input operand), not individual tensors. Gates: bundled fail=0,
fresh 2000 divergence 0 (bit-identical). Backup: reports/retired_networks/task234_pre_s11_recast.onnx.
