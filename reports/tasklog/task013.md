# task013 — 0a938d79 (alternating periodic stripes from two seeds)

**Rule:** Grid W×H (W∈20..30, H∈6..12). Two seed pixels at columns `start` and
`start+sep+1` (period p=sep+1∈2..6), each in row 0 or H-1, colours c0,c1∈1..9.
Output paints FULL vertical stripes at cols start, start+p, start+2p,… (<W),
alternating c0,c1,c0,…  If `xpose`, the whole grid (in & out) is transposed →
stripes become full rows. Orientation recoverable: xpose=1 IFF both seed columns
lie in {0,W-1}. Closed-form, fully separable: recover period-axis colour vector
pvec[30] from two no-pad colour-weighted profile Convs, alternate by (t-first)%2p,
gate to the in-grid rect, route the 10-ch expansion into the FREE bool output.

**Current (prior):** 15.77 pts, closed-form separable net, mem 10101, params 65
(ledger 'memory' was STALE — real scored mem was 10101, not the ~1800 in docstring).
**Target tier:** A (separable row⊗col into free output; the orientation swap forces
one fp16 combine plane, see floor analysis).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | prior net (3 full fp16/bool planes + 2×1200B ReduceSum) | A | 10101 | 65 | 15.77 | — | baseline |
| 1 | ROW/COL-SUM-PROFILE-AS-ONE-CONV (kill perch_row/col 1200B planes) | A | 8991 | 656 | 15.83 | — | partial |
| 2 | uint8 (255-sentinel) for the 3 full planes | A | 6021 | 656 | 16.19 | — | partial |
| 3 | single fp16-Max combine plane (orientation folded into vectors) | A | 5841 | 655 | 16.22 | — | partial |
| 4 | fp16 the whole position/colour recovery vector chain | A | 5115 | 657 | 16.34 | — | partial |
| 5 | drop fp32-cast-for-reduce (ReduceSum/Max/Min accept fp16 now) | A | 4379 | 655 | 16.48 | 200/200, 300/300 | ADOPT |

## Best achieved
16.48 @ mem 4379 params 655 — beats prior 15.77 by **+0.70** (≥+0.3 ✓).
Stored 267/267, isolated fresh 500/500 (200+300 batches).

## Irreducible-floor analysis
Dominant intermediate: **L16, the single fp16 [1,1,30,30] combine plane (1800B)**.
The output is `Equal(L16, arange_ch)`; L16 must be a full-grid colour-index plane.
It cannot drop to uint8 (900B): the combine of "colour on the period axis" with
"in-grid gate on the cross axis" needs a uint8 `Max`/`Add`, both ORT-unsupported,
and the orientation (xpose) SWAPS the row/col roles so a single uint8 `Where`
(whose 3 args have fixed axis roles) cannot serve both orientations — a `Where`
per orientation + a select = THREE uint8 planes (2700B) > one fp16 Max (1800B).
Remaining 4×120B fp32 are the two profile Convs + two ReduceMax occupancy outputs
(born fp32, immediately Cast→fp16). ~28×60B fp16 [30] recovery vectors + 13×30B
bool make up the rest; each is already minimal-dtype. Params 655 = the two
[1,10,30,1]/[1,10,1,30] profile kernels (600 elems, must span 30 rows for arbitrary
grid heights; replacing with ReduceSum reintroduces 2×1200B planes = net worse).

## OPEN ANGLES (re-attack backlog)
- uint8 single combine plane (900B): would need a uint8 Max OR an orientation-free
  formulation. A Transpose-of-one-canonical-plane route still costs 3 planes. If a
  future ORT build adds uint8 Max, the combine drops 1800→900 (+~0.13).
- Smaller profile kernel: xpose=1 grids are up to 30 tall so the 30-row kernel is
  required; no saving available.

## INSIGHT (transferable)
⭐ When orientation (xpose) swaps the row/col roles of a separable row⊗col output,
the cross-axis gate + period-axis colour can be combined into ONE plane with a
broadcast `Max` of two perpendicular [30] vectors (off-grid→200 sentinel on EITHER
axis, in-grid→max(colour,0)=colour), selecting each vector's CONTENT (not axis) by
`Where(xpose,…)`. This collapses the usual 3-plane "build L0, build L1, select"
into a SINGLE fp16 combine plane. The combine is pinned at fp16 (1800B) because ORT
has no uint8 Max/Add and the role-swap blocks a single uint8 `Where`.
⭐ ReduceSum/ReduceMax/ReduceMin ACCEPT fp16 under ORT_DISABLE_ALL on the current
build (the "reduce ops reject fp16" gotcha is STALE) — so the whole integer
position/colour recovery chain runs in fp16 with NO fp32 bridge casts, halving every
[30] working vector (120→60B). Only Conv and the input-ReduceMax outputs are forced
fp32 (born from the fp32 input); Cast them to fp16 immediately.
