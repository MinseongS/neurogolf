---
deployed_cost: 9095
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task379 — ecdecbb3

**Rule:** 1-2 full-width horizontal CYAN(8) lines; each column has ≤1 RED(2) dot.
Each dot shoots a ray (painting red along its column) toward the NEAREST line
above AND the NEAREST line below it (a line strictly between blocks the farther
one). Where a ray reaches a line at (L,c): paint the inclusive column segment
[dot..L] red, stamp a 3×3 CYAN box centred at (L,c), then set the box centre RED.
Paint priority: cyan-lines < ray-red < box-cyan < centre-red. `xpose` flips the
whole figure (lines become vertical).
**Current (public):** 13.82 pts, mem ~70k.
**Target tier:** B (closed-form masks; per-cell reconstruction routed into the
free BOOL output) — full Tier S impossible (output colours are fixed but the
geometry is a per-column ray + stamp, not a pure copy/permutation of input cells).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | dual-branch fp32 closed-form | B | 140k | 53 | 13.15 | — | correct, too big |
| 2 | single branch (select inputs by xpose flag) | B | 92.8k | 53 | 13.56 | — | correct |
| 3 | + fp16 whole pipeline | B | 67.2k | 54 | 13.88 | — | +0.06 |
| 4 | + CROP-TO-ACTIVE 20×20 + pad-back | B | 41.7k | 61 | 14.36 | — | +0.54 |
| 5 | + uint8 nested-Where label (no fp32 L planes) | B | 38.5k | 64 | 14.44 | — | |
| 6 | + combined-channel Slice + bool in-grid | B | 29.3k | 68 | 14.71 | — | |
| 7 | + ArgMax dotrow (no rred plane) | B | 28.7k | 68 | 14.73 | — | |
| 8 | + scalar line-rows (no dval/uval full planes) | B | 26.5k | 68 | 14.81 | 200/200 | prior best |
| 9 | PROFILE rewrite: dot/line via no-pad collapse Convs (NO red/cyan full plane) | B | 11385 | 1851 | 15.51 | — | breaks 11.4k floor |
| 10 | + merge count into dot conv via (r+1) offset (6→4 convs) | B | 10865 | 1252 | 15.60 | — | |
| 11 | + per-line SEPARABLE box (row-band ⊗ widened-reach-col), drop fp16 MaxPool | B | 10633 | 1252 | 15.62 | — | no full fp16 plane |
| 12 | + band-pack 2 profiles/conv (val=(dot+1)+64·cyancnt), 4→2 convs | B | 11189 | 654 | 15.62 | — | params 1252→654 |
| 13 | + box rows = {L-1,L+1} ONLY → centre red falls out of line<ray<box (drop 4 planes) | B | 9469 | 653 | **15.78** | 3000/3000 | **BEST — beats deployed 15.66 by +0.12** |

## Best achieved
**15.7775 @ mem 9469 params 653 (total 10122)** — beats deployed kojimar 15.658
(mem 10134 + params 1274 = 11408) by **+0.12 EXACT**. fresh 3000/3000 + isolated 200/200.

## Irreducible-floor analysis
Dominant survivors (all 1600 B = fp32 20×20): 3 entry colour slices (red/cyan/bg,
free input → slice is counted), the fp16 casts of the red/cyan masks needed for
the float ReduceSum (cyan-per-row line detect) / ReduceMax (red-per-col presence),
and the box dilation (bool→fp16 cast + 3×3 MaxPool — MaxPool needs float). The
orientation transpose/select are now uint8 (400 B). ReduceSum/ReduceMax reject
uint8/bool so a colour count needs ≥1 fp16 full plane per axis — that's the
remaining structural cost, plus the unavoidable 3 entry slices.

## KEY BREAKTHROUGHS (this session, 26.5k→9.5k)
1. **PROFILE-not-plane**: never slice red/cyan to a full plane. Recover dot row
   and cyan-line via no-pad collapse Convs (kernel [1,10,30,1]/[1,10,1,30]) that
   emit 1-D [1,1,1,30]/[1,1,30,1] vectors. Kills all the fp32 entry slices.
2. **BAND-PACK 2 profiles per conv**: one collapse-direction conv carries BOTH
   dot position (weight r+1 on red) AND cyan count (weight 64 on cyan) — decode
   by floor/mod 64 (ints<2048 fp16-exact). 4 convs → 2 ⇒ params 1252→654.
3. **SEPARABLE per-line box** kills the fp16 3×3 MaxPool: box = (row-band ⊗
   widened-reach-col), widen the 1-D reach profile with a tiny 1×3 MaxPool, AND
   the per-row band — all bool planes, zero full fp16 plane.
4. ⭐ **BOX ROWS = {L-1, L+1} ONLY (drop the line row L)**: the line row is
   already cyan from lineB and the ray already reds (L, dotcol). With box never
   covering row L, the priority `line(8) < ray(2) < box(8)` reproduces the RED
   box centre for FREE — eliminates the entire centre layer (cen_min/cen_max/bcB
   + 1 compose Where = 4 full planes, ~1.7k). This was the win that crossed the
   floor.

## OPEN ANGLES (untried, total now 10122)
- The ~3.4k of tiny 1-D decode planes (Div/Floor/Slice/Where chains, 120B each)
  could likely shave a few hundred B by selecting the PACKED value first then
  decoding once, instead of decoding both orientations then selecting.
- Ray still costs 4 full bool planes (lt_lo/gt_hi/out_r/rayB); a single
  between-test would help but the empty-range (no-reach) sentinel blocks the
  product trick.

## INSIGHT (transferable)
⭐ A "ray + iterative stop-on-cyan + 3×3 stamp" generator that LOOKS like a flood
is fully CLOSED-FORM: the stop-on-cyan reduces to "reach the NEAREST line in each
direction" → per-column scalar `Ldown=min line>dot`, `Lup=max line<dot`. With
≤2 lines these come from a tiny `Where(lineB, rowidx, ±BIG)` ReduceMin/Max over
the **[1,1,WK,1] line vector** (NO WK×WK candidate plane) plus a 2-level Where
chain on the [1,1,1,WK] dot-row vector. Ray = row-vs-{dot,L} range masks; box =
3×3 MaxPool of the line-intersection mask; compose by nested **uint8** Where
(priority order) → no fp32 colour-value plane. xpose handled by uint8
transpose+select of the masks once (uint8 Where works; bool Where does NOT).
⭐ ArgMax works on uint8 (ReduceSum/ReduceMax do not) — use it for "row index of
the unique marker per column" to avoid a full coord×mask product plane.

## ADOPTED 20260709T041318Z
- cost: 9095 -> 7733 (points 16.0467)
- source: candidates/public_dumps/20260709/neurogolf-7266-72-w-visualizations/nets/task379.onnx
- note: min-merge from nets

## ADOPTED 20260709T084638Z
- cost: 7733 -> 7599 (points 16.0642)
- source: candidates/task379/agent_ray_collapse/cyan_threshold_f32_width.onnx
- note: tail cleanup: threshold cyan profiles before fp16 cast/crop and keep width scalar fp32; bundled pass and 1500/1500 fresh diff=0

## NEGATIVE-VERDICT AUDIT 2026-07-11 (full July-arsenal re-audit, opus)
**cost model confirmed:** cost 7599 = memory 7446 (node-output BYTES) + params 153 (element COUNT, not bytes). `ng gate` on deployed copy = pass 266/fail 0.
**byte map:** DETECTION floor (fp32 einsum/reduce reading free input) = e1,e2,e3,e5,rMax,cMax = 720B + red-pos casts/slices ~440B — irreducible (einsum must be fp32 to consume free fp32 input; both orientations computed because horB is runtime). CARRIER = 11×[20,20] planes (ge,le,TRAIL,box0,box1,BOX,base,m_trail,m_box,m_box_T,MAP = 4400B) + padded[30,30] 900B. Compose/construction is at structural floor: 11 planes is the minimum for 4 priority levels (bg/line<trail<box) + build masks (trail needs ge,le; box needs box0,box1 as Where-conditions so cannot fold the &) + orientation (m_box_T,MAP=800B).
**what ran (per-mechanism verdict, all LOSE or N/A):**
- free-output N-ary Einsum / signed-rect routing: BLOCKED. TRAIL (ray) = (row in per-column [lo,hiBot]) is AND-of-halfplanes / non-separable per-column range — explicitly outside the one-linear-contraction+single->0 budget. Forces a materialized 2-D operand; at output-res 30x30 that operand is 900B and multiple complement operands (notBOX,notLINE) needed -> MORE expensive than current 20x20-crop+nested-Where. LOSE.
- s8port tail fold: dying tail (base/m_trail/m_box/MAP/padded) is uint8/bool, NOT fp32 -> fp32-coupling gate fails, fold makes it WORSE. N/A.
- BOX einsum-fold ((rb0 x DIL0)|(rb1 x DIL1) rank-2): einsum output must be fp16 -> [20,20]fp16=800B vs current bool planes 400B; +cast/greater. LOSE (fp16 penalty on small plane).
- scatter-inverse(233)/dynamic-kernel-stamp/QLinearConv/kernel-collapse/TopK-refit/walk-chain-slack: no TopK, no Conv, no propagation chain, no match matrix — N/A.
- fp16 recast: fp16 planes already fp16; fp32 entry einsums can't be recast (free fp32 input). No win.
- tighter crop: bundled input dims max exactly 20x20 (grids 12..20, placed at origin); fixed [0:20] crop already tight. No win.
- delete a branch: orientation split is 264 horizontal / **2 vertical** in bundled -> dual branch + transpose (800B) cannot be dropped without failing the 2 vertical examples.
- orientation-elimination (build 2-D planes in true frame, skip m_box_T+MAP=800B): needs ~20 extra 1-D orientation-selects/transposes (~600B) -> net ~+200B (~+0.026 pts), high complexity/bug risk, plausibly net-negative. Not built (below directive's 0.X bar; marginal).
- param reduction: 153 params all structurally needed (ones30/coordsP length-30 pinned to 30-wide free-input contraction; colorvec[10] pins 10 output channels; coord ramps needed).
**tool+date:** onnx dump + static byte/param map + `ng gate` + bundled orientation/size scan, opus agent, 2026-07-11.
**result:** NO win found >= gate (cost<7599). Deployed net is the smallest known: all 20+ public dumps are >=8108 static (ours 8016 static / 7599 grader). Net is at its mechanism floor for the ray+stamp closed-form family.
**reopen trigger:** (1) new public dump for task379 with mem<7446; (2) a separable/linear reformulation of the per-column ray range that escapes AND-of-halfplanes (would unlock free-output einsum, ~-2000..-2900B); (3) a mechanism that supports the 2 vertical bundled cases without a full 20x20 transpose pair; (4) cost-model change making params byte-weighted (would reopen dtype/param levers).
**falsification history:** none yet — first full-arsenal audit of this net. Prior June "irreducible-floor analysis" (self-referential) is superseded but its DURABLE physics (fp32 detection floor from free-input einsum, 900B pad optimal for 20x20 via label-pad<onehot-pad) re-confirmed here.
