# task085 — 3bdb4ada

**Rule:** The grid holds several "punchcards": each a solid 3-row × `wide`-col rectangle of
one colour (`wide` is always ODD; bands placed at rows r, r+=randint(3,4), so no two bands
share a row → at most ONE horizontal run per row). The OUTPUT copies the input EXACTLY, except
in the MIDDLE row of each punchcard, where only cells at EVEN offset from the punchcard's left
edge are kept (c%2==0); odd-offset cells are erased to background.
**Current (stored):** 15.31 pts, gen, mem 14880, params 1201, 265/265.
**Target tier:** A (separable-ish closed form). Output = input with a mask zeroed → routed into
the FREE `input` Where, NO colour-index/label output plane. Not S (cells are removed, not copied
verbatim everywhere). Mask is the AND of two independent 1-D-prefix predicates.

## Key structure
- Since exactly ONE run per row, horizontal prefix-count from col 0 at an occupied cell =
  (within-run offset + 1). even prefix ⟺ odd offset ⟺ REMOVE. Prefix via fp16 triu MatMul
  `occ @ L[30,30]`, then `mod 2 == 0`.
- Middle row detected by vertical prefix-count `U[16,16] @ occ` mod 3 == 2 (each 3-tall band
  contributes a multiple of 3 to any column it fully covers; within a band T=1,M=2,B=0 mod3;
  bg cells = 0 mod 3). vpre==2 mod3 fires ONLY at occupied middle-row cells, so it SUBSUMES the
  occupancy gate (this sweep's improvement: dropped the separate `occ_b` AND `oe` planes).
- Height ≤ 16 (generator bound) → all per-cell prefix/mod planes run on a 16×30 fp16 canvas
  (960 B) instead of 30×30 (1800 B); the final mask is padded back to 30 rows (u8, Pad rejects
  bool) for the free `Where(removed, e0, input)`.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | stored: occ_b & even_b & mid_b, 16-row fp16 MatMul prefixes | A | 14880 | 1201 | 15.31 | — | baseline (==P) |
| 1 | drop occ_b + oe (mid_b subsumes occupancy gate) | A | **13920** | 1201 | **15.38** | 200/200 | best; only +0.065 over P |

## Best achieved
**15.38 pts @ mem 13920, params 1201 — 265/265 stored, fresh 200/200.** Beats stored P=15.31 by
**+0.065 → MARGINAL (< +0.3).** Adopt-recommend: **N**.

## Irreducible-floor analysis
Dominant intermediates (mem 13920):
- **occ32 3600 B fp32 [1,1,30,30]** — the 1×1 Conv occupancy plane (Σ ch1-9). Documented
  entry floor: any 10→1 reduction of the FREE fp32 input is a fp32 30×30 plane; fp16/uint8
  tricks only ADD a plane. Irreducible.
- **occ16f 1800 B fp16 [1,1,30,30]** — cast occ32→fp16 before slicing to 16 rows. Tested
  slice-first-then-cast (occ32→fp32 16×30 = 1920 B, then fp16) = 1920 > 1800, so cast-first is
  optimal. Irreducible second entry plane.
- prefix planes hpre/hmod/vpre/vmod 4×960 B = 3840 B fp16 16×30 — the two prefix MatMuls + two
  Mods. fp16 needed (fp32 doubles each to 1920); CumSum would kill the 1156 matrix params but
  forces fp32 prefix planes (+~3000 B mem net), a LOSS.
- tail rem16(480)+remU(900 u8)+removed(900 bool) = 2280 B — pad 16→30 rows for the 30×30 Where
  condition; Pad rejects bool so the u8→pad→bool double is forced.
- params 1201 = L[30,30]=900 + U[16,16]=256 + conv 10 + scalars. Width can be 30 so L can't
  shrink; U is already height-tight. Matrix-sharing (one 30×30 triu, slice+transpose for U)
  saves 256 params but adds >256 B of slice/transpose intermediates — net loss.

Floor ≈ 3600 + 1800 + 3840 + 2×480 + 2280 + smalls ≈ 13920 → 15.38. The entry (5400) is half
the budget and irreducible; the rest is two prefix predicates + a 30×30 mask. Beating +0.3
(→15.61, mem+params ≤ ~11910) is not reachable without removing the fp32 entry plane, which is
the proven floor.

## OPEN ANGLES (re-attack backlog)
- Replace the fp32 entry occupancy with something cheaper: no known path — every channel-axis
  reduction of the fp32 input is a 30×30 fp32 plane (3600). This is the binding constraint.
- Fuse the two prefix predicates: I want only `(hpre mod2==0) AND (vpre mod3==2)`; both Mods are
  separate ops/tensors. No linear/single-op route to mod-2-of-a-sum exists, so the 4 prefix
  planes stay. ~0 pt upside.

## INSIGHT (transferable)
A mod-K vertical/horizontal prefix-count predicate can SUBSUME a separate occupancy gate when
the residue value that flags the target (here vpre≡2 mod 3 = band middle row) is UNREACHABLE by
background cells (bg cumulative count is always ≡0 mod the band height) — drop the `occ>0` AND
plane and the intermediate `And` it feeds (−960 B here). Check the residue-vs-bg collision before
adding an occupancy gate to any prefix-parity/phase task.
