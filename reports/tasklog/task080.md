# task080 — 39e1d7f9

**Rule:** A `size`×`size` bitmap (size 5–10, only ≤30-px renders survive → sizes
5,6,7,9,10 with spacing 4,4,3,2,2) is rendered as a LINEGRID: each bitmap cell is a
solid `spacing×spacing` block, blocks separated by single `linecolor` grid lines
(period p=spacing+1). The bitmap holds `npix=(size-1)//2` "centre" cells, all colour
c0=colors[0]. Exactly ONE centre (the template) is decorated: 4 orthogonal block-
neighbours = c1, and (if 3 colours) 4 diagonal block-neighbours = c2. The OUTPUT
decorates EVERY centre the same way. Closed form (verified 0 fail / 15k+ fresh):
V=colour-index plane; lc=colour with the largest single-row pixel count (excl bg);
p=first full lc-line row +1; c0=stamp colour with the largest bbox span (centres
scattered, c1/c2 clustered at the one template); M=(V==c0); E=OR of M shifted by
(±p,0),(0,±p); Cn=OR of M shifted by (±p,±p); c1=max V over E, c2=max V over Cn;
paint c1 on E, c2 on Cn (corner over edge), keep V (centres+lines), sentinel outside
the aw×aw active grid. Shifts are zero-fill via pad-to-31 + clamped Gather; period p
(data-dependent) drives the geometry with no fixed kernel.

**Current:** 14.358 pts, ext:biohack_new, mem 41638, params 217 (generalizes 200/200)
**Target tier:** B (variable-geometry stamp/dilation; output copies a fixed decoration
onto a data-dependent set of centres) — not S (data-dependent period + colour routing).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | full fp32 masks + fp32 gather shifts | B | 201171 | 176 | 12.79 | — | correct but huge |
| 2 | uint8 pad+gather shifts, span-c0 | B | 77608 | 168 | 13.74 | — | shifts→uint8 |
| 3 | masked-max c1/c2 (drop stamp filter), simplified compose | B | 59608 | 168 | 14.00 | — | |
| 4 | bool Gather (Cast Mpad→bool once, drop 8 per-shift Greaters) | B | 53369 | 167 | 14.11 | 200/200 | BEST — still < stored 14.358 |

## Best achieved
14.11 @ mem 53369 params 167 — adopted? **N**. Beats prior 14.358? **NO** (−0.25, and
far below the +0.3 bar). Generalizes 200/200 fresh (isolated, by-file-path gen load).

## Irreducible-floor analysis
The closed form needs, irreducibly: **6 fp32 [1,1,30,30] planes** (21600B) — V (Conv
colour index, needed for lc/p/c0/M), v_E & v_C (masked-V for c1/c2 max), L1/L2 (paint
edges/corners), L (in-grid sentinel mask, applied LAST because arms of edge-centres
extend off-grid and would otherwise overwrite the sentinel). Plus the **8-direction
dilation** of M = 8 bool gathers + 6 OR + 3 row-shift planes (~16000B), and the
**per-colour span recovery for c0** (~8400B in [1,10,30,1]/[1,10,1,30] fp32). Total
~53000B. Every fp32 plane is single-channel 3600B (the documented hard floor — fp16
upcasts, dtype tricks add bytes). The dilation period p is data-dependent so no fixed
Conv/MaxPool can replace the 8 gathers. The stored 41638B net is ~12000B leaner than
this closed form, so a cheaper encoding exists that I did not find.

## OPEN ANGLES (re-attack backlog)
- **Cut span (~8400B):** c0 needs max(rowspan,colspan) over channels — single-axis
  span, min-count, min-density, row/col-presence ALL fail genverify (tested 8–12k).
  A cheaper EXACT c0 detector would directly recover ~8000B → ~14.5. The "lonely-block"
  detector (c0 = stamp colour with a block having no stamp neighbour at ±p) is also
  exact but costs ~10000B (4 stamp-shifts + masked-max) — worse than span.
- **Cut dilation:** the 8 directional gathers seem minimal for an 8-neighbour data-
  dependent-period dilation; a "compress-out-the-lines then shift-by-1" remap (Gather
  to delete line rows/cols, dilate by ±1, scatter back) might collapse the 8 gathers to
  ~2 but needs a data-dependent compaction index — untried, plausible ~−8000B.
- **c1/c2 without v_E/v_C (−7200B):** bbox-corner presence (c2 at its bbox corner, c1
  not) gets 588/4223 wrong (edge-clipping distorts bboxes); row/col-share-with-c0 gets
  ~2000/4223. A robust cheap c1-vs-c2 separator would remove 2 fp32 planes.
- The stored ext net at 41638B proves ~14.36 is reachable ~12000B leaner — reverse-
  engineering its structure is the most promising path.

## INSIGHT (transferable)
- ⭐ **c0 = max-bbox-span colour** is an exact, flood-free way to pick the SCATTERED
  colour out of a mix where the other colours sit in one compact cluster (centres vs a
  single decorated template). Needs BOTH axes' span (single-axis, count, density all
  fail). Same family as task036's min-span "find the clustered object", inverted.
- **Zero-fill data-dependent shift idiom:** Pad to N+1 (zero border at index N), then
  Gather with `idx = where(arange±p out-of-range, N, arange±p)` — clean zero-fill shift
  by a runtime scalar p, no symbolic-dim trap, no fixed kernel. Clamped/border-repeat
  Gather pollutes the off-grid padding and fails; zero-fill is mandatory.
- **bool Gather beats uint8-gather+Greater:** Pad rejects bool, so Pad in uint8 then
  Cast→bool ONCE; subsequent Gathers emit bool directly and OR with no per-shift
  threshold — removed 8 Greater planes (~7200B). uint8 Add/Or/Max are all rejected by
  ORT (only Greater works on uint8), so combine shifted masks in bool, not uint8.
- The colour-plane→`Equal(L,chan)`→free-BOOL-output route is far cheaper than building
  the 10-channel output via bool Or of shifted one-hot channels (each [1,10,30,30] term
  is 9000B); keep all paint logic on the single-channel L plane.
- Off-grid cells (input all-zero) must map to a NON-channel sentinel (−1), not bg(0):
  the target leaves off-grid all-False, but Equal(0,chan0) would set channel 0.
