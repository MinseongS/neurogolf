# HARD-WALL ATTACK PLAYBOOK (start here for the hard-task campaign)

Written 2026-06-18 after the blank-note false-positive sweep (LB 6620.24 → 6635.63, +15.39, 30 wins).
The easy reservoir (14–16pt blank-note "infeasible" labels that were really just bloated) is nearly mined out.
This doc is the strategy for the NEXT phase: deliberately cracking the GENUINE-hard tasks with diverse,
research-backed, long-thinking methods — NOT fast-bail.

## 0. Reframe: "wall" has meant "agent gave up in <10 min"
That bail bar was tuned for THROUGHPUT during the sweep. It is WRONG for a hard-task campaign. The truly-
impossible set is small. Everything else is "expensive but EXPRESSIBLE." Stop fast-bailing; invest effort.

Genuinely impossible (do NOT retry — confirmed): random per-instance colour↔shape BIJECTION needing a global
dictionary with no positional key (233, 173, 285); within-component gap > between-component gap so no radius
separates groups (77); non-deterministic generators (209). 219/255 are info-bottleneck/connectivity walls.

## 1. THE MASTER KEY — bounded-iteration unrolling
The op set bans Loop/Scan/NonZero, which is WHY iteration "feels" impossible. But the grid is ≤30×30, so every
iterative algorithm has a BOUNDED step count D and unrolls into a fixed DAG:
- **Flood-fill / reachability**: `S_{k+1} = Greater(Conv3x3(S_k, ones), 0) AND passable`, unrolled D times.
  D = Manhattan-diameter upper bound (≤58 for 30×30; tighter from the generator's size cap). PROVEN: task48
  deployed a 12-round BFS. Cost ≈ D × 900B uint8 planes; D≈30 → ~27KB → 25−ln(27000) ≈ 14.8 pts. This BEATS
  most flood tasks' public nets (60k–190k bloat = 12.8–13.5 pts) AND generalizes exactly.
- **Connected-component labeling**: init label = flat index (r·30+c) on foreground, 0 on bg; iterate
  `L = min over passable 3x3 neighbours` (min-pool via −MaxPool(−L) masked), unroll D times → each component
  collapses to its min index. Then recolor/measure by label.
- **Component SIZE / property**: after labeling, one-hot the label and ReduceSum → per-component pixel count
  broadcast; threshold/compare. (task369 did the bounded-small case via local degree; general case = label+count.)
- **Distance transform / nearest-marker**: iterated min-plus conv (add 1 each ring), unrolled — gives geodesic
  distance for variable-length arms (relevant to maze task 66).

## 2. Per-wall-class technique catalog
| Wall class | Crackable when | Technique |
|---|---|---|
| flood / reachability | connectivity unambiguous | unrolled dilation D≈size-cap (task48) |
| recolor-by-component-size | labeling works | unrolled label-prop + one-hot scatter-count |
| variable-count ≤K objects | K≤5 & objects separable by a cheap key (bbox/colour/position) | peel-off K slots: TopK/ArgMax, per-slot mask, sum |
| symmetry-centre detection | candidate centre set is small | score every candidate by orbit-consistency Conv bank → ArgMax (retry 361 with CHEAPER candidate scoring; it lost only on bloat) |
| matched-filter / template | template recoverable from a fixed locus | grouped-Conv correlation bank + count-gate (task143) |
| multi-object correspondence | binding is POSITIONAL/deterministic (not a random bijection) | per-object bbox extract + fixed pairing |
| hole-fill / interior | enclosure decidable by flood-from-edge | unrolled flood from grid border = background; interior = unreached; HANDLE edge-clipped boxes with an off-grid sentinel ring (task367 failed only on clip + line-loops — solve clip explicitly) |

## 3. Diverse-method protocol for hard-wall agents (next session dispatch)
1. LONG leash — do NOT bail at 10 min. Require ≥3 distinct attack angles before writing INFEASIBLE.
2. First: characterize the generator EXACTLY — bounds (max objects, max size, grid cap), determinism, what is
   data-dependent vs fixed. Read /tmp/arc-gen/tasks/task_<arcid>.py fully.
3. Try in order: (a) closed-form/separable, (b) BOUNDED-ITERATION UNROLLING (the master key), (c) candidate
   enumeration, (d) PARTIAL: if a sub-component is exact and only the rest is the wall, build the partial and
   measure isolated fresh — `src.adopt` takes ANY generalizing gain, so a partial that beats the current real
   score still wins.
4. Mem is NOT the enemy: points = 25 − ln(mem+params). An exact 30-round unrolled flood (~30KB, 14.8 pts) BEATS
   a non-generalizing 100KB public net (13.0 pts) — and if the public net fails Kaggle held-out, it's a GAP-CLOSER
   that raises REAL LB by ~its full score.
5. Research hook: if stuck, use deep-research / web search on the specific ARC task id (src.show prints it) or on
   "ONNX connectivity/connected-components without Loop", "label propagation as convolution", "distance transform
   min-plus". Diverse published techniques exist. Spend the thinking time the user asked for.

## 4. PRIORITY — gap-closer tasks (direct LB gain, not just +stored)
gap ≈ 30 pts = tasks whose deployed net scores LESS on Kaggle than its stored points (fails fresh held-out).
`reports/lb_status.py` attribution: **219, 255, 157, 2, 319, 366, 118, 233** (+ one new wave-2 over-stored net —
#32 landed proj −1.03; find & note which of {58,48,208,265,85,29,333,162,134,382,178,30,355,117,80,110} it is).
For a gap task, a GENERALIZING net adopted via `src.adopt` raises REAL LB by ~its full score even at low stored
(adopt scores the failing current as ~0). 219/255 = confirmed true walls. RETRY WITH UNROLLING: 157, 319, 366,
118 — an exact generalizing net here is a DIRECT +LB, the highest-value outcome available.

## 5. Re-attack target list (high-bloat, re-classified — use unrolling, NOT fast-bail)
- **66** (maze/staple path): bounded BFS between terminal pairs + distance transform — SAME shape as task48 which
  WORKED. The bend is set by cyan corner markers; identify them, then geodesic-connect. Strong retry candidate.
- **367** (gray-box interior fill over line clutter): unrolled flood-from-edge = background; interior = unreached.
  The two failures were (a) edge-clipped boxes (interior touches border) → pad an off-grid sentinel ring so the
  border flood can't leak in through the clip; (b) line-formed loops → require the enclosure to be a filled
  RECTANGLE (4 straight gray sides) before filling. Solve both explicitly.
- **286** (variable-size flood connectivity): unrolled dilation to the size cap.
- **96** (clipped-symmetric sprite recovery): candidate reflection-axis enumeration + orbit max.
- **158** (multi-object): check if the object→target binding is POSITIONAL (then enumerable) vs random (then wall).
- **77** (re-confirm): agent measured within-box gap can exceed between-box gap → likely a TRUE wall; verify once.
- TRUE walls, skip: 233, 173, 285, 5, 54, 219, 255, 209.

## 6. Operational
- Same loop machinery: `src.adopt N` gates (accepts any generalizing gain); commit+push each win; submit at
  stored ≥ anchor+8; re-anchor; VERIFY `git ls-files src/custom/taskNNN.py` (the 247 add silently failed once).
- Prefer ROBUST exact rules: arc-gen-fresh 200/200 is NOT a full Kaggle guarantee (#32 lost ~1pt). When a net
  passes fresh but you suspect edge cases, widen the fresh check to 1000+ and reason about generator corner cases.
- Concurrency: agents share ONE API quota; it can hit a session limit (resets at the plan boundary). On limit,
  dispatch ONE canary first; if it also limits, back off 1800s+ rather than burning turns.
- Levers already graduated into BUILD_PROMPT.md: RE-PROBE WAVE LEVERS, STALE DTYPE CLAIMS, DATA-DEP PERIOD/LATTICE.
