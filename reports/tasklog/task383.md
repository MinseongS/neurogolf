# task383 — f1cefba8

**Rule:** One axis-aligned box (2-px OUTER ring colour C0, SOLID inner block colour C1) sits on a bg-0 canvas. A few "barnacle" markers are single C1 pixels placed ON the C0 inner-ring line (rows brow+1/brow+tall-2, cols bcol+1/bcol+wide-2). Each marker projects a STRIPE perpendicular to the ring it sits on: a top/bottom-ring marker -> a full COLUMN stripe through its column; a left/right-ring marker -> a full ROW stripe through its row. Output = clean box PLUS, for each stripe line: INSIDE the box the crossed col/row flips to C0; OUTSIDE the box the crossed col/row is painted C1 along the full grid extent (perpendicular direction only). The construction is fully SEPARABLE into 1-D row/col vectors.
**Current:** 14.37 pts (prior public net), mem unknown
**Target tier:** A — separable row/col routed into a free bool output; no flood-fill, no 2-D component labelling. Tier S is blocked because output colours C0/C1 vary per instance (a fixed Conv can't route arbitrary colours), so a recovered-scalar colour-index plane + Equal one-hot is the minimal admissible form.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | separable 1-D vectors, fp32 planes, Mul-reduce colf | A | 90368 | 24 | 13.59 | 266/266 | correct but heavy |
| 2 | 1x1 Conv for colf (kill [1,10,30,30]) | A | 54368 | 24 | 14.10 | - | better |
| 3 | cast colf->fp16, axis-reduce ingrid, fp16 downstream | A | 39364 | 24 | 14.42 | - | better |
| 4 | compact label (min 30x30 plane count) | A | 34864 | 24 | 14.54 | 266/266 | better |
| 5 | Where-fuse mask+select (drop Cast/Mul planes) | A | 29464 | 25 | 14.71 | 266/266 | beats target |
| 6 | drop nonbg gate in C0 recovery (bg colf=0<C0) | A | 27664 | 25 | 14.77 | 500/500 | FINAL |

## Best achieved
14.771 @ mem 27664 params 25 — adopted? N (left at src/custom/task383.py for caller). Beats prior 14.37? Y (+0.40).

## Irreducible-floor analysis
Dominant intermediate is the one fp32 colour-index entry plane `colf32` [1,1,30,30] = 3600B from the 1x1 Conv (the 10->1 reduction must emit fp32; ORT upcasts). It is immediately cast to a single fp16 `colf` (1800B) and ALL downstream full-canvas ops run in fp16 (task377 lever). Remaining mem is ~9 fp16 [1,1,30,30] planes (1800B each: colf, colf_inner, colf_c0, c1ring_r/c, out_ig, out_val, in_val, L) and ~6 bool planes (900B each). These are the minimum needed for: colour recovery (2 masked-max planes), stripe detection (2 masked-max planes), and the 4-stage label Where chain. Cannot drop below the colf32 fp32 entry without casting the whole input to fp16 (18000B, worse).

## OPEN ANGLES (re-attack backlog)
- Fuse the two colour-recovery masked planes (colf_inner, colf_c0) — both reduce to a [1,1,1,1] scalar; a single packed accumulation (e.g. additive band encoding of "inner vs ring colour") could collapse them to one plane (~ -1800B -> ~14.85).
- The stripe-detection c1ring_r/c1ring_c could potentially share one masked plane if ring-row and ring-col detection were packed into one ReduceMax over a combined ring mask with a magnitude band, but the two reductions are along different axes so a single plane is non-trivial.
- in_val/out_val/L Where chain (3 fp16 planes) might fold via an additive colour-index arithmetic (rowcode + colcode + box term) routed into ONE final Equal, but the label is piecewise (not rank-1 separable), so this needs a banded encoding — uncertain payoff.

## INSIGHT (transferable)
"Markers as pattern-breakers projecting perpendicular stripes" decoder: a single anomalous pixel ON a structural ring encodes a full line perpendicular to that ring — detect via (colour==C1) reduced over a ring-line mask, NOT as a 2-D correspondence problem. The whole task is separable into 1-D ibr/ibc/ringrow/ringcol/SC/SR vectors and only materializes ~4 unavoidable 30x30 Where planes for the piecewise {bg,C0,C1,off-grid} label. ⭐ Reusable micro-lever: for a masked argmax/max over a 30x30 plane, `Where(mask_bool, colf, sentinel)` then ReduceMax FUSES the mask+select into ONE plane vs Cast(mask)->Mul->ReduceMax (saves a full fp16 plane each, ~1800B per recovery). Also: when excluding one colour for a max, you often don't need a separate "nonbg" gate — bg/off-grid cells carry value 0 which loses to any real colour >=1 in the ReduceMax.
