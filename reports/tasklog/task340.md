# task340 — d687bc17

**Rule:** H×W rectangle (H,W∈[10,20]) anchored at (0,0); 4 solid one-colour DISTINCT walls
(top=tc row0, bottom=bc row H-1, left=lc col0, right=rc col W-1). Interior scattered single
pixels; each interior pixel of colour v shoots to its matching wall, landing just inside it:
v==tc→(1,c), v==bc→(H-2,c), v==lc→(r,1), v==rc→(r,W-2); non-matching colours vanish; walls
kept; interior otherwise cleared. Verified exactly (0/266). The real generator places interior
pixels at distance ≥2 from every wall (never on the inner ring) ⇒ NO cross/same-colour routing
collisions, so an ADDITIVE single index plane is exact.
**Current (prior deployed):** 15.69 pts, ext:kojimar7113 (crowd net, not re-golfed by us).
**Target tier:** A — separable row/col routing into a single index plane → FREE bool output.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | leftover (8-line packed MatMul, fp32 Conv counts + edge-MatMul machinery) | A | 11932 | 773 | 15.55 | — | below P |
| 1 | single packed MatMul Acol[30,10]@Brow[10,30] + wall colours from per-ch counts | A | 9304 | 163 | 15.844 | 500/500 | beats P +0.15 |
| 2 | reuse occupancy for off-grid sentinel | A | 9244 | 163 | 15.851 | 500/500 | best |

## Best achieved
15.851 @ mem 9244 params 163. Beats prior 15.69 by **+0.16 → MARGINAL (< +0.3)**.

## Irreducible-floor analysis
Dominant intermediates (all forced by an fp32 10-channel input):
- `colcount` [1,10,1,30] **1200 fp32** + `rowcount` [1,10,30,1] **1200 fp32** — per-channel
  column/row pixel counts. Needed for BOTH the 4 wall colours (border-line slices) AND the
  interior-presence test (`count>1` cancels the wall's own +1). ReduceSum/Conv/MatMul all emit
  fp32 from the fp32 input; casting to fp16 only ADDS a plane (fp32 producer still counts);
  cropping to the ≤20 active region adds a Slice plane on top of the full one. ⇒ 2400B hard floor.
- `og` index plane [1,1,30,30] **1800 fp16** — the one full-canvas plane; MatMul emits fp16,
  Equal needs ≥fp16, a uint8 cast adds a plane. ⇒ 1800B floor.
- `Acol`[1,1,30,10] + `Brow`[1,1,10,30] **600+600 fp16** — the MatMul's two operands; the 10
  pre-Concat line vectors duplicate ~1200B more (operand build is unavoidable).
Sum ≈ 7800; remaining ~1450 is line-construction small tensors (selectors, interior masks,
presence, off-grid sentinel, wall-colour argmax). Reaching +0.3 needs mem+params ≤ 8184; the
count floor (2400) + og (1800) + MatMul operands (1200) + build vectors leave no room. The
deployed kojimar net (~11050B) IS beaten on stored, just not by the +0.3 bar.

## OPEN ANGLES (re-attack backlog)
- Eliminate ONE count plane: needs per-row AND per-column wall-colour presence from a single
  reduction — appears impossible (orthogonal 2-D reductions). If an fp16-output count op ever
  lands under ORT_DISABLE_ALL the 2400→1200 clears +0.3 immediately.
- Fuse Acol/Brow operand build via batched Equal/Mul (~300-400B) — measured ~byte-neutral.

## INSIGHT (transferable)
- ⭐ "Shoot interior pixels to their matching wall" = SEPARABLE per-direction routing, NOT a
  detection wall. Per-wall presence (does this column/row carry an interior pixel of the wall
  colour) = per-channel column/row COUNT with `count>1`: the wall line itself contributes exactly
  +1 to every interior column/row, so the threshold cancels it with NO interior masking and NO
  variable-row slice. Wall COLOURS come from the SAME count planes via tiny border-line slices
  (rowcount@row0/@Hidx, colcount@col0/@Widx) + per-channel argmax — no extra 10-ch slice planes.
- ⭐ ONE packed outer-product MatMul `Acol[1,1,30,K] @ Brow[1,1,K,30]` assembles K disjoint
  row/col lines AND folds an off-grid `+10` sentinel into the same plane, so the whole sparse
  output (4 walls + 4 routed lines) is a single fp16 index plane → Equal → FREE bool output.
  Off-grid index 0 would falsely fire channel-0; the +10 sentinel is load-bearing.
- ⚠️ A self-written fresh generator MUST replicate the real generator's placement constraints
  (here: interior pixels ≥2 from walls). A naive generator created impossible inner-ring
  collisions and produced phantom "failures"; the net is exact on the real distribution (500/500).
  Always read the constraint off the STORED data (min pixel-to-wall distance) before trusting a
  fresh-gen fail.
