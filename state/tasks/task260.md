---
deployed_cost: 734
logged_costs_match: match
migrated: 2026-07-09
---

# task260 — a78176bb

**Rule:** INPUT (always 10x10): one solid main diagonal `r-c == diag` in a single
colour (never gray), plus gray triangle(s) hanging off the diagonal; each corner's
gray fills a contiguous band of diagonals from `diag±1` out to the corner's
`row-col` extreme. OUTPUT paints the same colour on `r-c == diag` PLUS one parallel
echo per occupied side: gray ABOVE diag → echo at `(max gray-above)+2`; gray BELOW →
echo at `(min gray-below)-2`. No gray in output. Random instances have one corner;
the hand-authored train/test cases straddle the diagonal (two corners → two echoes).
**Current:** 15.58 pts (public net)
**Target tier:** A (separable diagonal masks + scalar offsets; no flood-fill, no per-cell colour plane)

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 10-ch crop + Σk·input colf30, fp32 L plane | A | 11324 | 142 | 15.65 | — | works but marginal |
| 2 | drop 10-ch crop: cval from per-channel COUNTS, ch0/ch5 slices for masks, fp16 L | A | 5496 | 160 | 16.36 | — | big win |
| 3 | + fp16 D / floors → all 10x10 mask work fp16 | A | 4886 | 160 | 16.47 | 200/200 | ADOPTED |

## Best achieved
16.47 @ mem 4886 params 160 — beats prior 15.58 by **+0.89**. Fresh isolated 200/200.

## Irreducible-floor analysis
Dominant intermediate is the single fp16 30x30 colour-index plane `L` (1800B) feeding
the final `Equal(L, arange)` bool output — that 30x30 carrier is the structural floor
for any one-hot output. Everything upstream is on the 10x10 active crop. The colour
scalar is recovered from a 40B per-channel ReduceSum (no colour-index plane needed),
and the diagonal offsets are scalars from ReduceMax/Min over a fp16 `D=r-c` plane.

## OPEN ANGLES (re-attack backlog)
- The ~5 fp32 10x10 helper planes (bgch/graych slices 400B each) could in principle be
  bool — but ORT Slice preserves input fp32 dtype, so they cost 400B; converting to a
  channel-presence bool via Equal adds an op for ~300B. Marginal, likely not worth it.
- L could conceivably be built directly as a bool diagonal mask broadcast into the FREE
  output with the colour one-hot (Where(mask, colour_onehot[1,10,1,1], 0)) avoiding the
  1800B L plane entirely — would push toward Tier-S-ish; untried (colour is dynamic so
  the one-hot must be assembled from cval, an extra reduction).

## INSIGHT (transferable)
⭐ "Set of full parallel diagonals in one colour" is closed-form Tier-A: a diagonal
`r-c==k` is a single `Equal(D, k_scalar)` against a constant `D=r-c` plane; the offsets
are SCALARS recovered by ReduceMax/Min of `D` masked to the relevant cells (with an
out-of-[-9,9] floor that AUTO-GATES an absent side — no branch). ⭐ Recover the output
COLOUR from `ReduceSum(input,[2,3])` per-channel COUNTS (40B) + an `okidx` channel mask,
never a 3600B Σk·input colour plane — works whenever the output colour is a single
input colour identified by "not bg, not the marker colour".

## S16 (2026-07-06) — public bit-identical golf (franksunp, unfiltered re-mine) ADOPTED
Engine public-mine loop (byte-prefilter relaxed → found this). fresh_verify 1500 = 0/0/0 (bit-identical).
Cost drop (dead-init/redundant-node), private-LB safe. Manifest updated. Backup in scratchpad.

## 2026-07-07 — dtype-overpay ReduceSum feed probe KILL

`dtype_overpay_scan.py` flagged the two small `Greater -> Cast(fp32) -> ReduceSum`
feeds (`safe_name_36`, `safe_name_37`) as narrow integer carriers.  Two candidates
were stored under `reports/candidates/task260/`:

- `task260_uint8_reducesum_feeds.onnx`: load failure because ORT rejects uint8
  input to `ReduceSum`.
- `task260_fp16_reducesum_feeds.onnx`: load failure from ONNX Runtime precision
  cast/type mismatch around the downstream fp32 path.

Conclusion: this is a dtype-scan false positive unless the whole surrounding
reduction path is rebuilt.  Do not direct-recast these ReduceSum feeds.

## 2026-07-08 — ⭐ REGIME CRACK ADOPTED (batch 2, 900B mask → free-output Einsum)
- 1573→734 (+0.76). Union of K dynamic parallel diagonals = 1-Π(x-k_i)^2 as 2K bilinear factors, int32 Einsum; affine bg routing A·F+B via t-axis; sentinel-k gates absent branches.
- Bundled fail=0, fresh-gated, unsigned-TopK clean, deployed-gated. Candidate: reports/candidates/task260/regime.onnx (builder build_regime.py). Backup: reports/candidates/fatmid_adopt_backup/task260.onnx.bak.
- ⭐ See memory neurogolf-regime-crack-freeoutput-einsum (vein taxonomy + sub-recipes). Vein list: reports/candidates/fresh_sweep/mask_dominance.json.
