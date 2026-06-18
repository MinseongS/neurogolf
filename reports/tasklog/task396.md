# task396 — fcb5c309

**Rule:** A 12-18 square grid holds 2-3 hollow rectangular boxes (1px outline in `colors[0]`,
black interior) plus scattered single-pixel "static" in `colors[1]` (some static lands inside
the boxes). `wides`/`talls` are each sorted DESCENDING, so box 0 is the LARGEST box (max width
AND max height). Output is a `tall0 x wide0` grid = box 0's region, but EVERY non-black cell of
that region (outline + interior static) is painted with the STATIC colour `colors[1]`; black
interior stays black. Exactly: `out[r][c] = c1 if input[brow0+r][bcol0+c]!=0 else 0`.
**Current:** 14.009 pts, gen:thbdh6332, mem 58199, params 1115 (bloated import)
**Target tier:** A — closed-form crop + recolour; no flood-fill / connectivity / global-argmax.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | per-channel adjacency (fp32) + run-convs (fp32 30x30) | A | 333664 | 229 | 12.28 | — | works but bloated |
| 2 | adjacency fp16 on 18x18 slice | A | 180944 | 235 | 12.89 | — | better |
| 3 | run-convs fp16 on 18x18 | A | 94976 | 211 | 13.54 | — | better |
| 4 | grouped-conv adjacency | A | 88856 | 248 | 13.60 | — | marginal |
| 5 | **colf-only equal-pair run-length (drop per-channel)** | A | 38465 | 189 | 14.44 | 200/200 | adopt-worthy |
| 6 | fp16 label/output carrier | A | **36407** | 190 | **14.49** | 200/200 | BEST |

## Best achieved
14.49 @ mem 36407 params 190 — beats prior 14.009 by +0.48 (>= +0.3). Fresh 200/200 (x2 runs).

## Irreducible-floor analysis
Dominant intermediates: `colf30` [1,1,30,30] fp32 = 3600B (the one mandatory 10->1 colour-index
entry plane; ORT forces fp32 here) and `L30` [1,1,30,30] fp16 = 1800B (the output carrier, padded
30x30 so the final `Equal(L,arange)` can route the 10-ch expansion into the FREE bool output).
Everything else lives on a 18x18 fp16 working canvas (~600B planes). The key lever was realising
the box geometry needs NO per-channel [1,10,..] planes: run-length runs entirely on the single
colf plane via same-colour equal-PAIR detection (`Equal(colf,shift) AND colf>0`), so a horizontal
run of length L = L-1 consecutive pairs; the global max pair-run = box 0's edge (largest box).

## OPEN ANGLES (re-attack backlog)
- `colf18f` (1296B fp32 slice) is redundant with `colf30`; a fused Slice-then-Cast still
  materialises it. A direct fp16 1x1 Conv on an 18x18 input slice could drop the fp32 18x18 copy.
- The six per-k run-conv presence planes (~600B each x ~24 tensors) could collapse to ~2 planes
  via a single cumulative/doubling run-length instead of per-k convs (would shave ~8-10kB).
- `Vr` [1,1,8,30] (960B) is the row-gathered colf before the col-gather; gathering cols first on
  the 18-canvas (then rows) would be smaller.

## INSIGHT (transferable)
⭐ "max same-colour run length" and "which colour forms the boxes" both come from the SINGLE colf
colour-index plane via same-colour EQUAL-PAIR maps — `eqh = Equal(colf[:,:,:,:-1], colf[:,:,:,1:])
AND colf>0`, run-of-length-L = L-1 consecutive eqh pairs (per-k 1xk valid conv == k, summed). No
per-channel [1,10,H,W] adjacency plane is ever needed. The "largest box" is just the global
max-pair-run, and the box colour falls out by reading colf at the winning run's top-left cell —
so a multi-box "pick the biggest & recolour" task is closed-form tier-A, NOT a correspondence wall.
