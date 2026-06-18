# task350 — dbc1a6ce

**Rule:** Grid is height×width (width 8..24, height in [width-2,width+2] ⇒ height≤26, width≤24),
placed top-left of the 30×30 canvas (rest is background 0). Random blue(1) pixels scattered. In the
OUTPUT, for every pair of blue pixels sharing a ROW the cells strictly between them are filled
cyan(8) (unless already blue); likewise for pairs sharing a COLUMN. Net: per row the closed span
[min blue col, max blue col] becomes blue-or-cyan; same per column; blue endpoints stay blue. A cell
is cyan iff it is NOT blue AND lies in some row-span OR some col-span. (label confirmed-infeasible was
a FALSE-POSITIVE — the task is closed-form.)

**Current:** 15.105 pts, ext:kojimar6275, mem 19800, params 30
**Target tier:** A — closed-form per-row/per-col span fill via directional prefix/suffix-OR, no
flood-fill; 10-ch expansion routed into the FREE Where output.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | full 30×30 fp16 triangular-MatMul prefix/suffix-OR (4 matmuls), span∧bg→Where(cyan) | A | 24300 | 1826 | 14.83 | 200/200 | exact but bloated |
| 2 | product-combine (Mul), notblue instead of bg slice, 26-crop | A | 18024 | 1382 | 15.13 | 200/200 | better |
| 3 | non-square 26×24 crop + Transpose to build Uc/Ur (save params) | A | 19280 | 1282 | 15.07 | 200/200 | Transpose ADDS plane → worse |
| 4 | direct-init both triangulars (params cheaper than the transpose plane) | A | 16776 | 2534 | 15.13 | 200/200 | mem down |
| 5 | ⭐ replace 4 triangular MatMuls with 4 fp16 MaxPool prefix/suffix-OR (params→30) | A | 16776 | 30 | 15.27 | 500/500 | BEST |

## Best achieved
15.27 @ mem 16776 params 30 — adopted? N (not self-adopted). Beats prior 15.105? Y by **+0.165
(MARGINAL, < +0.3 threshold)**.

## Irreducible-floor analysis
The rule is genuinely NON-separable (each row has its own [min,max] col span, each col its own
[min,max] row span) so it requires FOUR full-canvas directional scan planes (leftOR/rightOR/upOR/
downOR), which is the floor. Memory breakdown (26×24 active canvas):
- blue_f32 fp32 slice = 2496B (Slice preserves input fp32 dtype — irreducible entry plane)
- B fp16 cast = 1248B (MaxPool needs float; fp16 halves vs fp32 scan planes)
- 4 MaxPool OR planes (fp16, 1248 each) = 4992B — the irreducible core (4 independent directions)
- combine (2 products + booleanize, OR/AND) ≈ 3744B; notblue/fill_s ≈ 1248B
- pad-to-30 tail: fill_u8(624)+fill30(900u8)+fill(900 bool) = 2424B (Where needs a 30×30 BOOL cond;
  Pad rejects bool and Where rejects uint8 cond ⇒ the u8-Pad→bool-Cast pair is forced)
Total 16776. To reach +0.3 (15.405) needs mem+params ≤ ~14728 — a ~2078B cut ≈ deleting 1.7 full
planes, which the 4-directional-scan structure does not admit. uint8 MaxPool is rejected by ORT
(invalid type), so fp16 is the scan-plane floor.

## OPEN ANGLES (re-attack backlog)
- 4→2 scan planes: derive suffix-OR from prefix via row/col total. Tried analytically (CumSum
  prefix-sum + total−prefix, or weighted-index ReduceMax bounds): every variant still needs 2 full
  planes per axis (a product/diff plane or a second cumsum), so it ties the 4-MaxPool floor. No win
  found — would need a single op that yields min AND max bound simultaneously.
- Eliminate the 2424B pad tail: only possible if Where could broadcast a 26×24 cond against the
  30×30 input (it cannot) or accept a uint8 cond (it cannot). Structurally blocked.

## INSIGHT (transferable)
⭐ DIRECTIONAL PREFIX/SUFFIX-OR = fp16 MaxPool with a FULL-LENGTH 1-D kernel + ONE-SIDED pad, NOT a
triangular MatMul. `MaxPool(B, kernel=[1,W], pads=[0,W-1,0,0])` = running-max-from-left (prefix-OR);
swap the pad side for suffix-OR; `[H,1]` kernel for the vertical axis. Identical plane size to the
triangular-MatMul idiom (task070) but ZERO params (the matmul's two [W,W] triangular initializers
cost ~2500 params and Transposing to share them ADDS a materialized matrix plane to MEMORY). Works
under ORT_DISABLE_ALL on fp16 (uint8 MaxPool is rejected). Use this for any per-row/per-col span /
bbox-as-mask where params matter. ⚠️ Non-separable per-line spans need all 4 directional planes —
this is a genuine ~16.8KB structural floor (≈15.27 pts), MARGINAL over a near-floor public net.
