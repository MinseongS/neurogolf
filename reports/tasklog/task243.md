# task243 — 9edfc990

**Rule:** size in [12,18]; grid filled with random colours (~50% black). `common.blue()` =
colour index **1** (NOT 2). Output copies the input, then every blue(1) cell floods blue into all
4-connected cells currently 0 (black), propagating. Net: every black cell 4-connected through black
to a blue seed becomes blue; unreachable black stays black; non-black colours unchanged. Genuine
4-connected flood; the iterated dilation is structural (no separable/count collapse). Grid gen-bounded
<=18x18, top-left anchored.

**Current (deployed):** 13.86 pts, mem 68616, params 102 (gen:vyank6322, uint8 MaxPool flood, opset 18).

## ⚠️ CORRECTION TO PRIOR SCOUT LOG (the 14.30 "adopted" figure is NON-GENERALIZING)
A prior scout reached 14.30 @ D=28 and labelled it adopted/1000-fresh-pass. That is WRONG: it only
sampled depths present in the 265 official cases (max 28). The TRUE worst-case BFS geodesic over
**200000 fresh** instances is **38** (frac D>33 = 6e-5, frac D>30 = 1.6e-4); rare near-empty grids let a
single corner seed snake the whole grid. Re-measured on fresh 20000:
- D=28 (scout's "adopted"): **7/20000 fail** (0.035% silent leak) — does NOT generalize.
- D=33: 2/20000 fail (0.01% leak).
- D>=38: 0 fail.

## This session — robust build (fp16 cross-Conv flood, deliverable)
- Slice bg(ch0)+blue(ch1) to 18x18 (fp32), Cast fp16, `passable = bg+blue`.
- reach0 = blue. Per round: `count = Conv_cross(reach,[[0,1,0],[1,1,1],[0,1,0]])`; `reach = Min(passable,count)`.
- FINAL round gates `Min(bg16,count)` -> output IS flooded-bg mask (fuses reach∧bg).
- Pad fp16 18x18->30x30, `Greater(.,0.5)`->bool, `Where(mask, blue_onehot, input)` (recolour in FREE output).
- N_ROUNDS=38 (covers worst-case geodesic 38). Opset 11.

| version | D | mem | params | pts | fresh | verdict |
|---|---|---|---|---|---|---|
| **fp16 robust (deliverable)** | 38 | **56484** | 47 | **14.057** | 500/500 | +0.20 MARGINAL, FULLY general |
| uint8 MaxPool robust | 38 | 61812 | 44 | 13.967 | 500/500 | worse (4 planes/step vs 2) |
| fp16 undershoot | 33 | 50004 | 47 | 14.179 | leaks 2/20000 | +0.32 but NOT general |
| uint8 undershoot (scout) | 28 | 48852 | 44 | 14.203 | leaks 7/20000 | +0.34 but NOT general |

## fp16 Conv vs uint8 MaxPool: equivalent at the flood floor
- **fp16 cross-Conv = 2 planes/round** (count 648B + reach 648B) = 1296B/round.
- **uint8 cross-MaxPool = 4 planes/round** (pv,ph 324B each + Max + Min) = 1296B/round. IDENTICAL bytes.
- uint8 Conv is INVALID_GRAPH even at opset 18 (no uint8 type constraint for Conv), so a uint8 flood is
  FORCED onto MaxPool's 4-plane cross. fp16 Conv does the whole cross in ONE op -> fewer total tensors and
  less overhead -> fp16 wins (56484 vs 61812). The BUILD_PROMPT "uint8 1B/plane beats fp16" intuition does
  NOT apply to a 4-connected flood because the cross can't be a single uint8 op.

## Computed flood floor (the wall)
Per round = 1296B (2 fp16 18x18 or 4 uint8 18x18). 4-conn forbids a single MaxPool (3x3 box = 8-conn, leaks
through 1-thick walls); 1-cell walls force radius-1. Budget for +0.3 (14.16) = exp(25-14.16) = 51021B. Bare
flood planes alone at robust D=39 = 50544B, already over budget before ANY overhead. **Full-robustness +0.3
is mathematically impossible.** Crossover D for 14.16 is ~33, below the worst-case geodesic 38, so "clearing
+0.3" forces undershooting D and silently mis-handling the rare high-D tail (task286 non-generalizing pattern).

## Best general achievement: 14.06 (+0.20) — MARGINAL
Beats deployed 13.86 by +0.20 with EQUAL-OR-BETTER generalization (cropping to 18x18 + final-round fusion +
dropping the colf-Conv pack). Does NOT clear the +0.3 bar.

## OPEN ANGLES (all dead)
- 1-plane-per-round: no ONNX op does masked cross-dilation atomically; uint8 Conv blocked; can't drop count or dil.
- Fewer rounds: D=38 is the true worst-case geodesic; a larger kernel leaks 1-cell walls; no distance-doubling with walls.
- Lattice downsample (task80 lever): bg is random-scattered, not a parity lattice — no clean sub-canvas.
- Data-dependent crop <18x18: symbolic-dim trap (calculate_memory -> None).
- Bool-Concat pad-back: saves ~936B mem but +576 params (pad inits) — net wash.

## INSIGHT (transferable)
⭐ The "uint8 1B/plane beats fp16 2B" lever does NOT help a 4-connected flood: uint8 Conv is INVALID_GRAPH
(even opset 18), so a uint8 cross-dilate must use 4 MaxPool/Max/Min planes = exactly the same 1296B/round as
a 2-plane fp16 cross-Conv. The flood floor `D·1296B + overhead` is dtype-agnostic. For task243 (Wk=18, D=38)
the floor lands at 14.06 — the smaller Wk and D give a HIGHER floor than task286 (Wk=25, D=59 -> 13.10), enough
to beat the deployed 13.86 by +0.20 via cropping+fusion, but the +0.3 crossover D (=33) sits BELOW the true
worst-case geodesic (=38), so reaching +0.3 is only possible by silently leaking the high-D tail (the scout's
14.30 @ D=28 is exactly such a non-generalizing net). Verdict: MARGINAL (+0.20, fully general).
