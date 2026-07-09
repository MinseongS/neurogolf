# task034 regime-crack attempt — FLOOR (measured negative)

**Ran (2026-07-09, opus):** `build_regime.py` → `regime.onnx`. Free-output Einsum fold of the
`Where(mask30, paint, input)` epilogue: `bkHW,hH,wW,bshw,skc,sc->bcHW` with s-axis decomposition
G[0]=1−mask9 / G[1]=mask9, Dc[0]=I / Dc[1]=ones, P[0]=1 / P[1]=paint, Pr/Pc = 9→30 placement.
Math verified exact: `input·(1−mask)+paint·mask`.

**Result:** fail=0, 267 pass (FUNCTIONAL) but **mem 2130 + params 994 = 3124 vs deployed 2261 → LOSS −863.**

**Why it loses (POSITIONED-CONTENT vein → FLOOR per taxonomy):**
1. Free `input` is fp32 ⇒ every Einsum operand must be fp32 (4B). The mask stack G[2,9,9] fp32 = 648B
   ≈ the 900B bool mask it replaces — no memory win (2130 vs 2100, actually slightly worse).
2. Gating the FREE 30×30 input inside the Einsum needs placement matrices Pr[9,30]+Pc[9,30]=540 params
   + colour-map Dc[2,10,10]=200 params. The deployed padded-bool-mask + Where path pays **zero** extra
   params (mask is 1B bool) and reuses free input directly. Params blow 161→994.
3. The task is POSITIONED-CONTENT: a fixed diagonal-band staircase motif at a **data-dependent 2-D
   position** (seed/red-corner location varies per example: rows4-5 / rows1-2 / rows2-3). The deployed
   QLinearConv exploits translation-equivariance to encode variable position for FREE inside the op;
   an Einsum/polynomial reformulation must re-encode placement explicitly (expensive). This is exactly
   the taxonomy's FLOOR category (mask = content ⊛ data-dependent position), boundary confirmed by
   task112/163/099 floors.

**Deployed net is near-optimal** (already regime-worked: staircase drawn in QLinearConv internals,
only plane is the 900B bool `mask30` pad, which enables free-input Where at 0 param cost). The 900B
bool pad is the CHEAPEST encoding here — placement-matrix params + fp32 operands price strictly higher.

**Reopen trigger (per epistemic rule — negative is dormant, not dead):** revisit iff (a) a way is
found to gate free 30×30 fp32 input without per-axis placement-matrix params (e.g. a sub-30 canvas
crop that stays free), OR (b) the seed position becomes derivable as ≤~4-element global state cheap
enough that a shifted-polynomial band undercuts the QLinearConv (would move it into GLOBAL-STATE vein),
OR (c) mask30 bool pad ceases to be 1B/element. Tool: build_regime.py in this dir (re-measure vs
DEPLOYED, not src).
