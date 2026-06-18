# task366 — e6721834

**Rule:** Input is two equal halves (stacked vertically or side-by-side; horiz/foreside random). Each
half has a uniform background (two distinct `backs`). The FORE half contains 2–3 solid `forecolor`
rectangles (size 2–7 each side, random positions) with colored dots punched inside; box `idx` carries
`idx+1` dots, each box's dot color random. The NON-FORE half contains, for each box present there, ONLY
the dots (same relative arrangement & color as the matching fore box), placed at an INDEPENDENT random
position. The output = the non-fore half's background with each box RECONSTRUCTED: match each non-fore dot
cluster to the fore box whose dot stencil (color + relative geometry) coincides, then stamp the full
forecolor rectangle + dots there. (Box presence on the non-fore side is partial: idx0 always present,
idx≥1 dropped ~1/3 of the time.) Input can be up to 30×34 → such instances are dropped by the harness
(>30 returns None); ~95% fit.

**Current:** 13.50 pts, gen:vyank6322, mem 97119, params 1451 — FAILS fresh (38/40 genverify), so a
generalizing net would be a direct gap-closer (adopt scores the failing current ~0).
**Target tier:** detection/correspondence — in-context template matching, hardest class.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | python ref: dot-COUNT key + anchor translate | — | — | — | — | 245/479 | rule incomplete (count not unique key) |
| 2 | python ref: (count,colorset) key + pattern-equality align | — | — | — | — | 586/950 | nonfore dots NOT adjacent → CC grouping fails |
| 3 | python ref: full-cell-CC templates + sliding dot-stencil CORRELATION | — | — | — | — | 885/958 | best; fails on fore-box adjacency CC-merge + false-positive placements |
| 4 | python ref: forecolor-only-CC templates | — | — | — | — | 736/1426 | worse (adjacent fills merge) |

## Best achieved
No ONNX net built. Best PYTHON reference (scipy CC + correlation) ≈92% — not even exact in Python.

## Irreducible-floor analysis
This is multi-object in-context template matching with VARIABLE count (2–3), VARIABLE size (2–7 per axis),
RANDOM positions, and a per-instance template SET that must be learned from the fore half and applied to
the non-fore half. A correct solver requires, in sequence: (1) data-dependent axis split (vertical vs
horizontal) — even a purity/2×2-block heuristic is ~2% ambiguous when both H and W are even; (2)
connected-component labeling to isolate fore boxes (unrollable flood, ~30 rounds × 900B, but fore boxes
can ABUT → CC merges them, breaking template extraction); (3) data-dependent extraction of up to 3
arbitrary-size template rectangles; (4) a sliding 2D correlation of each VARIABLE-SIZE template's dot
stencil against the non-fore half (a data-dependent-kernel Conv per box) to find placements, with
false-positive resolution; (5) data-dependent stamping of each variable-size rect. None of (2)–(5) is
expressible without Loop at reasonable memory, the variable kernel size defeats a fixed Conv, and the
reference algorithm itself is not exact. Fresh 200/200 is unreachable.

## OPEN ANGLES (exhausted-but-listed)
- Per-template fixed-max-size (7×7) Conv-correlation bank with masked stencils, one Conv per fore box —
  blocked by needing to first ISOLATE/extract each fore box (CC) before you have a kernel, and by
  false-positive placements + fore-box adjacency. Would still need unrolled CC and is not exact.
- Bounded enumeration of all (axis, fore-side, box-set) — combinatorial, not closed-form.

## INSIGHT (transferable)
⭐ task366 is a GENUINE template-matching WALL, NOT a blank-note false-positive. The matching key is the
full dot STENCIL (color + relative geometry), NOT dot-count (counts collide: idxs like [0,0,1,1] give two
2-dot boxes) and NOT (count,colorset) (still ~0.3% collide). Non-fore dots of one box are SPREAD across the
box footprint and non-adjacent, so connected-components does NOT group them — grouping requires sliding the
fore template (the box size is only known from the matched template → circular without correlation). Best
scipy reference solver ≈92%, capped by fore-box adjacency merging templates under CC and false-positive
template placements. Diagnostic for this wall class: "reconstruct objects in region B from per-instance
templates discovered in region A, with variable count/size/position" ⇒ in-context matching wall.
