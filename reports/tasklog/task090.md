# task090 — 3eda0437

## 2026-06-29 scan/local-stencil screen

Current source score: 16.963750 @ mem 3009 params 82.

Rule: find the unique largest all-black rectangle in a height 2..5, width 20..30
grid and paint that rectangle pink.

Despite the very large node count, the source is memory-compact: it keeps a
5x30 black slice (600 B), a full-canvas paint mask (900 B), and many 1D/scalar
run-length values.  The unrolled scan is ugly but mostly scalar/row-vector state.

Ran `reports/scripts/conv_fit.py 90`; k=1, k=3, and k=5 all failed on channel 0
over 300 fresh training examples.  No rewrite adopted.  The blocker is semantic:
selecting the largest all-black rectangle is a global argmax over candidate
rectangles, not a translation-invariant local stencil.

## 2026-07-01 re-adjudication (borrowed-net pass)

Independent re-measure: mem=3009 params=82 pts=16.964. Per-tensor: 3 tensors
>=100B sum 1650 (black_top 600 fp32 read = Slice ch0 rows0:5 [1,1,5,30];
black_u8_4d 150 cast; paint_mask 900 carrier = And(row_strip,col_strip)
[1,1,30,30]); remaining 1359B over 593 sub-100B tensors (run-length scalars +
ten [1,1,1,30] interval column-run strips).

Floor proof: reading 150 empty-cell flags costs 600B fp32 (detection floor,
150x4) + 150B uint8 cast. The 900B paint_mask is irreducible: output = input
(arbitrary static) with one solid rect recoloured pink, so it MUST be routed
Where(mask, pink, input) and the separable row⊗col mask has to be materialised at
[1,1,30,30] to combine with the 30x30 input — no Einsum/strip route can rebuild
the arbitrary input, and a per-row [1,10,5,30] route costs 1500B > 900B. So
1650B/3009 (55%) is at the proven floor. The residual ~1359B is the
largest-empty-rectangle search (global argmax over 10 row-intervals; conv_fit
already refuted a local-stencil rewrite); shaving it is <0.2 pt and re-fit-prone.

Incumbent generalises cleanly (fail=0/800 fresh). VERDICT: FLOOR.

## S9 (2026-07-03) — kojimar teacher REJECTED (fresh 15/2500 fails, delta only +0.006)
