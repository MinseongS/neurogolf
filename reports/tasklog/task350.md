# task350 — dbc1a6ce

**Rule:** Grid is height×width (width 8..24, height in [width-2,width+2] ⇒ height≤26, width≤24),
placed top-left of the 30×30 canvas (rest is background 0). Random blue(1) pixels scattered. In the
OUTPUT, for every pair of blue pixels sharing a ROW the cells strictly between them are filled
cyan(8) (unless already blue); likewise for pairs sharing a COLUMN. Net: per row the closed span
[min blue col, max blue col] becomes blue-or-cyan; same per column; blue endpoints stay blue. A cell
is cyan iff it is NOT blue AND lies in some row-span OR some col-span. (label confirmed-infeasible was
a FALSE-POSITIVE — the task is closed-form.)

**Current (2026-06-21):** 15.63 pts, uint8 directional-pool net, mem 11700, params 11.
(Prior log entries below measured an OLDER fp16 net at 15.105/19800 — the deployed net has since been
upgraded to all-uint8 MaxPools: 11700B/15.63. The "uint8 MaxPool rejected by ORT" claim in the floor
section below is STALE — uint8 MaxPool DOES run under ORT_DISABLE_ALL at opset≥12 and is the current net.)
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
Deployed uint8 net: 15.63 @ mem 11700 params 11 — already optimal. Cannot beat by +0.3 (INFEASIBLE).
(Historical: fp16 self-build reached 15.27 @ 16776/30, since superseded by the uint8 deployed net.)

## 2026-06-21 re-probe vs uint8 deployed net (11700B / 15.63)
Measured mem breakdown of the deployed net: ONE fp32 plane 3600B (Gather of blue channel — Gather/
Slice/Conv all inherit fp32) + Cast→uint8 900B + 4 directional uint8 MaxPool planes 3600B + 2 Min +
1 Max + 1 Greater-bool 3600B = 11700B. To beat +0.3 needs mem+par ≤ 8676B (cut ~3035B). The only
3000B item is the fp32 blue-extraction bridge, and it is IRREDUCIBLE: no single ONNX op maps the free
fp32 input to a single-channel uint8 plane — Cast(input) keeps all 10 channels (9000B), Gather/Slice/
Conv/MaxPool inherit fp32, ReduceMax-over-channels rejects uint8 out, ArgMax→int64. The downstream
algorithm is already minimal (4 non-separable directional planes + Min/Min/Max/Greater). Absolute
theoretical floor 3600+900+3600+2700+11 ≈ 10811B → 15.71 pts, only +0.08 over deployed. VERDICT:
INFEASIBLE — at structural floor.

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
