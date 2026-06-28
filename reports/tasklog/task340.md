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
| 2 | reuse occupancy for off-grid sentinel | A | 9244 | 163 | 15.851 | 500/500 | best (prior) |
| 3 | drop 2 off-grid sentinels → ONE in-grid +1 term (rowocc⊗colocc) + chan=colour+1 | A | 8884 | 102 | 15.897 | — | inner-dim 10→9, params 163→102 |
| 4 | interior c<=W-2 as Less(c,Widx) (drop Sub planes); value vecs via Where(mask,colour,0) | A | 8404 | 101 | 15.952 | — | kills fp16 mask casts |
| 5 | wall colour = Sum k*(count>0) in fp16 (no argmax); H=Sum(rowocc) (drop ramp*occ Mul); Where-fold | A | **8016** | **102** | **15.998** | 3000/3000 | **NEW BEST, beats +0.307** |
| 6 | QLinearMatMul packed uint8 outer-product + onnxsim | A | **5860** | **279** | **16.278** | 1000/1000 base, 500/500 sim | **ADOPTED** |

## Best achieved
**16.278 @ mem 5860 params 279.** Beats prior live 16.029 (`custom:task340+onnxsim`) by **+0.249**.
The uint8 QLinearMatMul source passed 1000/1000 fresh; the adopted onnxsim file passed 500/500 fresh.

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

## PLANE-ELIMINATION WAVE (2026-06-21, +0.307 — supersedes the "no room for +0.3" verdict)
The prior floor analysis double-counted: the real core is og 1800 + 2 counts 2400 + 2 concats
1080 = 5280, NOT 7800. The ~3000B TAIL was the fat. Cuts that landed (9244→8016):
- ⭐ OFF-GRID SENTINEL → IN-GRID +1: replace the two `+10*(off-grid)` MatMul sentinel columns
  (k8,k9 + roff/coff/row_off/col_off/ones plumbing) with ONE in-grid term rowocc16⊗colocc16
  and shift the Equal target ramp to colour+1. In-grid bg→og=1→ch0; colour k→og=k+1→ch k;
  off-grid→og=0→matches no channel (ramp starts at 1)→all-false. MatMul inner-dim 10→9,
  concats 600→540, params 163→102. (−360 mem, −61 params.)
- ⭐ WALL COLOUR = Sum_k k*(count_k>0) in fp16 — a solid wall line holds ONLY its colour
  (interior pixels ≥2 from walls), so NO ReduceMax/Equal argmax: one Greater→Where(chramp16,0)
  →ReduceSum, all fp16 ([1,10,1,1] 40B→20B). Killed ~8 planes of the wall_color helper.
- ⭐ SOLID-RECT EXTENT = COUNT not argmax: grid rows 0..H-1 all occupied (solid origin rect),
  so H=Sum(rowocc16), Hidx=H-1 — drops the rowocc*ramp Mul planes (rowx/colx, 120B).
- value vectors via `Where(mask_bool, colour, 0)` (not Cast(mask)+Mul) → no fp16 mask planes.
- interior upper bound `c<=W-2 == Less(c, Widx)` → drops the Sub-vector planes.

## OPEN ANGLES (remaining)
- og 1800 (fp16 index, MatMul→Equal floor) + 2 counts 2400 (fp32 ReduceSum, orthogonal 2-D
  reductions can't merge) + 2 concats 1080 = 5280B hard core. Remaining ~2700 tail is genuine
  selector/value/presence vectors. A further win needs a count op that emits fp16 (ReduceSum
  rejects narrow dtypes) or a non-MatMul index assembly.
- 2026-06-28 update: the fp16 `og`/concat floor was false. The packed outer-product terms are
  small non-negative integers, so `QLinearMatMul` with uint8 operands and scale=1/zp=0 preserves
  the additive index exactly while cutting `og` 1800→900 and A/B operands 540+540→270+270.
  Same mechanism as task055; use it whenever packed MatMul is pure integer label assembly.

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
