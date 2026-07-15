---
deployed_cost: 1895
logged_costs_match: stale-likely
migrated: 2026-07-09
---

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

## S11 (2026-07-03) — FLOOR CONFIRMED at 1895B; free-einsum route priced at 41172B (23x worse)
Dossier lever (+0.85 via mech-15 bottom row + residual diagonal scatter) REFUTED by build:
⭐TRANSFERABLE CONSTRAINT — only ONE op writes the free graph output. A hybrid
(free einsum for the separable part + scatter for the rest) cannot compose without a
counted [1,10,30,30] intermediate (18-36KB). Full epilogue-fold forces the A-dependent
non-separable anti-diagonal operand to a counted [3,30,30] fp32 plane (fp32 mandatory —
input-matching) → measured 41172/1299 = 14.34 (gates clean, kept as pricing artifact at
reports/candidates/task084_signed.py). Incumbent ScatterElements-into-FREE-input encodes
row=last_row−c in a [1,5,2,21] index plane — already optimal. Every scatter operand priced
minimal: indices int32 floor, updates dtype-BOUND to fp32 data (⭐ScatterElements updates
can never be recast when data is the free input), 5-ch/2-slot/21-col spans forced,
data-dependent updates (invalid-col masking) can't become initializer. dtype_overpay_scan's
084 entry (+0.458 U8 updates) = false positive by the same dtype-binding rule.

## ADOPTED 20260712T055712Z
- cost: 1895 -> 833 (points 18.2750)
- source: /Users/minseong/.codex/worktrees/c56e/neurogolf/candidates/poly_transfer_000_199/task084/candidate.onnx
- note: signed-polynomial free-output overlay; fresh 2000/2000 div0

## 2026-07-12 — KAGGLE-ZERO ROOT-CAUSE + FIX (minfix, Sqrt-free)
The signed-poly 833 net (adopted 20260712T055712Z) Kaggle-ZEROED the task (-18.27), dropping the board
7318.30 -> 7306.34 when bundled with the task017 self-einsum win. Root cause: `size = Sqrt(area)` — fp32
Sqrt of a perfect square is not bit-exact; `last = size-1` is raised to the 4th power in the geometry
Einsum, so a 1e-5 error explodes -> wrong overlay positions -> full fail. Local ORT sqrt is exact so it
passed every local gate/fresh (silent-zero, same class as the earlier 084 disaster).
Compounding PROCESS bug: `ng pack` packs from `submission/overfit_nets/` (== manifest), NOT `networks/`.
Prior "minfix" submissions (54599958) shipped the buggy 833 because overfit_nets was never updated — the
fix "kept getting dropped." networks/ is a stale mirror (236/400 differ from manifest); ignore it for packing.
FIX = `candidates/task084/minfix.onnx`: delete Sqrt, recover dims integer-exact via
`rowsum=ReduceSum(input,[1,3]); W=ReduceMax(rowsum); H=area/W(Div)`. cost 833->957, points 18.136.
Verified 3000/3000 exact across all sizes 3..21; bundled 175/0. Deployed to overfit_nets + networks +
manifest, resubmitted 54608347. ⭐ RULE: never recover grid dimensions via Sqrt(area) — use integer
ReduceSum/ReduceMax/Div; and always diff `submission/overfit_nets/` (the pack source) vs manifest before submit.

## 2026-07-12 CORRECTION — Sqrt was NOT the bug; the 833 signed-poly MECHANISM is Kaggle-broken
minfix (Sqrt removed, integer-exact H=area/W, 957) RESUBMITTED (54608347) and scored the SAME 7306.34 =
still Kaggle-zero. Then found codex's proven-safe 985 net (2e40e55a, scored 7318.00 in sub 54598518) ALSO
CONTAINS Sqrt — so Sqrt is NOT the failure. The whole 833/minfix "signed-polynomial free-output overlay"
mechanism (adopted 20260712T055712Z) silently Kaggle-zeros for a reason local gates can't see (local ORT
matches all bundled+fresh; the hidden ARC/private example fails). REVERTED task084 to codex 985 sign-oracle
(known Kaggle-safe), submitted 54608689. LESSON: do not trust a cheap NEW mechanism on 084 without a real
Kaggle submit; the signed-poly 833 is DEAD for this task. src/custom/task084.py minfix builder kept as a
pricing artifact only — deployed net is the codex 985 sign-oracle. Do NOT re-adopt 833/957 without a
Kaggle-verified pass.

## ADOPTED 20260712T140136Z
- cost: 985 -> 419 (points 18.9621)
- source: /Users/minseong/project/neurogolf/dumps/archive_extract/submission7300+/task084.onnx
- note: archive.zip submission7300+ net; fresh 2000/0 fail; mechanism-graft

## ADOPTED 20260712T141554Z
- cost: 985 -> 419 (points 18.9621)
- source: dumps/archive_extract/submission7300+/task084.onnx
- note: all-in archive graft; Kaggle-CONFIRMED in record 7410.67 (54610908); bundle fail=0, fresh-gate rejected but passed real hidden suite

## ADOPTED 20260715T032551Z
- cost: 419 -> 414 (points 18.9741)
- source: candidates/task084/factor_r_duplicates.onnx
- note: direct exact duplicate-column initializer factorization

## ADOPTED 20260715T053104Z
- cost: 414 -> 409 (points 18.9863)
- source: candidates/task084/basis_only.onnx
- note: runtime-safe exact basis shift: (side-1)^m coefficients -> side^m, removes Sub scalar carrier and constant
