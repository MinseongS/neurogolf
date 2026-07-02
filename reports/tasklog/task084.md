# task84 — 3bd67248

**Rule (cristianoc oracle):** square grid; col 0 holds a colour, rest background. Output overlays
colour-4 on the bottom row (cols 1..W-1) and colour-2 on the anti-diagonal cells (A-1-c, c). Overlay
positions depend ONLY on grid size; overlaid cells are always old-colour 0. Routed via a single
ScatterElements onto the FREE output.

## S5 win — dedup replicated table (LANDED +0.060)
**Before:** mem 1797, params 241, total 2038. The `row_offsets` init was (1,5,2,21)=210 params but the
5 channels were IDENTICAL copies — pure redundancy.
**Change:** store once as (1,1,2,21)=42 params, broadcast the channel axis via a small ones-plane scaled
by last_row (=A-1). Channel dim stays 5 (must address channel 4). c=0 offsets (-22,+9) preserved (route
masked col-0 writes into padding so the coloured col 0 is never wiped).
**After: mem 1837 (+40), params 83 (−158), total 1920, pts 17.44.** evaluate fail 0/175;
`fresh_verify 84 "" 1500` fail 0. ⭐ TRANSFERABLE: a param table replicated identically across the
channel axis → store 1 channel + broadcast (params are element-count, so dedup beats the small mem add).
