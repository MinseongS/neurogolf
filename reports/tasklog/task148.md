# task148 — 673ef223

**Rule:** Grid H∈[16,24], W∈[8,12], colours only {0,2,8}. TWO vertical red(2) "portals", each a
run of length 4..6: one capping col 0 (left), one capping col W-1 (right); equal lengths, tops
offset by delta=second-first (≥6, source run above dest). A few cyan(8) "markers" sit in the SOURCE
portal's rows (the run whose cap is `srccol`). For each marker at row rs, col c: (a) the source row
gets a cyan(8) beam from the cap inward up to the marker with a yellow(4) at col c (cap stays red,
beyond the marker unchanged); (b) the aligned DEST row rd=rs+delta in the OTHER portal gets a FULL
cyan beam across all in-grid cols, dest cap stays red. flip mirrors columns so the source portal can
be on either side; detected from which red run contains the marker rows.

**Current:** 15.00 pts, ext:kojimar6275, mem 21676, params 353
**Target tier:** B (label-map on a small WORK canvas) — output COPIES input colours + closed-form
overrides; the only long-range coupling (dest = source-marker rows shifted by data-dependent delta)
collapses to ONE Gather(axis=row) on a [1,1,24,1] vector, so no full coupling plane is needed.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | label-map, full Conv colour-index + 2 full 30×30 planes | B | 24598 | 85 | 14.89 | 200/200 | worse than floor |
| 2 | H/W from 1-D profiles (kill 2nd full plane), fp16 working planes, uint8 V | B | 15042 | 160 | 15.37 | 200/200 | beats +0.3 |
| 3 | drop Conv (input is {0,2,8}: slice red/cyan channels), positional red base | B | 13258 | 176 | 15.49 | 200/200 | |
| 4 | drop col_ingrid (final ingrid-Where cleans off-grid), lo/hi src-beam, drop iscyan fp16 | B | 11098 | 169 | 15.67 | 200/200 + 500/500 stress | **BEST** |

## Best achieved
15.67 @ mem 11098 params 169 — adopted? (build-agent: file only). Beats prior 15.00? **YES, +0.67**.

## Irreducible-floor analysis
Dominant remaining intermediates: cyan-channel slice `cyf24` [1,1,24,12] f32 = 1152B (the one
unavoidable fp32 entry plane — Slice preserves input f32 dtype; cyan markers can sit anywhere in
the 24×12 region so it can't shrink), the final Pad carrier L [1,1,30,30] uint8 = 900B (output
carrier, required), `cyancol_idx` fp16 576B (marker-col detection), `redwin` 480B (red last-col
window cols 7..11), and ~17 working bool masks at 288B each. These 288B masks are the bulk; further
merging is possible but yields <0.1 pts.

## OPEN ANGLES (re-attack backlog)
- Merge several 288B bool masks into the nested Where chain (defer row⊗col ANDs) — ~2-3 planes
  removable → ~+0.1 pts. Marginal.
- The delta-shift Gather + dstfill could in principle be fused with the dest-beam Where to avoid the
  hasmark_f / dstfill_g intermediates.

## INSIGHT (transferable)
- ⭐ A "confirmed-infeasible / BLANK-note" entry can be a pure FALSE POSITIVE: this was a clean
  closed-form label-map task with ~7 pts of headroom mislabelled infeasible. Always re-triage.
- ⭐ When the generator uses only a FEW colours (here {0,2,8}), SKIP the 3600B 10→1 colour-index Conv
  entirely — slice just those colour CHANNELS of the free input to the active region. Saved the
  single largest plane.
- ⭐ Long-range row-to-row coupling by a DATA-DEPENDENT vertical shift = ONE Gather(axis=2) of a
  [1,1,H,1] per-row indicator with idx=clip(rowramp−delta,0,H−1); no shift matrix, no coupling plane.
- ⭐ Off-grid cleanup is FREE if a final `Where(ingrid, L, sentinel)` runs LAST — every override may
  freely over-paint off-grid cells, so drop all per-override in-grid ANDs.
- Orientation (flip) handled with ZERO duplicate planes: select per-row scalars lo/hi via f32
  `Where(scalar_bool, A, B)`; for BOOL operands use (s∧A)∨(¬s∧B) since ORT Where rejects bool operands.
