# task193 — 7f4411dc

**Rule:** The grid holds several solid rectangular boxes (`color`, each ≥2×2 since
wides/talls∈[2,5]) plus isolated single "static" pixels of the same colour (the
generator runs `remove_neighbors` so no two static pixels are adjacent, and refuses
a static pixel that would connect two shapes or touch a box border more than once).
OUTPUT = INPUT with every static pixel removed → only the boxes survive. The exact
local rule: keep cell (r,c) iff it belongs to at least one fully-filled 2×2 square
(box cells always do; a 1×1 static pixel never can, even when it abuts a box on one
side). Verified exact over 3000 brute-force fresh instances.
**Current (prior):** ~14.11 pts, tier B label-map
**Target tier:** B+ (single-cell-local but non-separable: the 2×2 membership couples a
cell with its orthogonal+diagonal neighbours → not a row⊗col separable A; not a pure
per-cell-of-input S). Achievable well above the B floor via 2 cheap 2×2 convs.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | has-≥1-orthogonal-neighbour | — | — | — | — | — | WRONG 1417/2000 (static can abut a box) |
| 2 | part-of-filled-2×2 (brute) | — | — | — | — | 0 fail/3000 | exact rule found |
| 3 | 2 convs, f32 removed-plane (10ch) | B+ | 56700 | 34 | 14.05 | — | works, 10ch intermediate too big |
| 4 | single-Where free output (keep OR offgrid) | B+ | 22500 | 33 | 14.98 | — | removed 10ch intermediate |
| 5 | pack ch0+colour into ONE 1×1 conv plane g | B+ | 17100 | 32 | 15.25 | — | killed 2nd f32 reduce |
| 6 | 20×20 crop of conv planes | B+ | 16100 | 47 | 15.31 | — | smaller canvas |
| 7 | crop g→20×20, all planes 20×20, pad selcond w/True | B+ | 13400 | 47 | **15.49** | **500/500** | FINAL |

## Best achieved
15.49 pts @ mem 13400 params 47 — adopted? N (orchestrator gates). Beats prior ~14.11
by **+1.38**. Generalizes: stored 266/266, isolated fresh 200/200 and 500/500.

## Irreducible-floor analysis
Dominant intermediate = `g30` [1,1,30,30] f32 = 3600 B — the channel-collapse of the
f32 input (1×1 Conv packing [50,1..9] → bg=50 / colour=k / off-grid=0). It is f32 and
full-30×30 because ORT Conv inherits the input dtype and spatial extent; reading the
input at 20×20 would need a [1,10,20,20] f32 slice (16000 B), far worse. Everything
downstream is cropped to the 20×20 active region (size∈[7,20]) so the two 2×2 detection
convs and all bool planes are 400–800 B; the only 30×30 tensors are g30 and the final
select-cond (needed for the FREE output Where). So 13400 is at the practical floor for
"one f32 read of the input + two 2×2 convs".

## OPEN ANGLES (re-attack backlog)
- Fuse the two 2×2 convs into ONE 3×3 conv + threshold: blocked — "in a filled 2×2"
  is an OR of 4 corner AND-triples, not linearly separable by a single weighted-sum
  threshold (a 2×3 colour strip would over-count). Would need a magnitude-band trick
  that survives a static pixel abutting a box; untried, ~ -1600 B if it works.
- Read the input at 20×20 via a cheaper op than Slice (e.g. a strided/region conv) to
  drop g30 from 3600→1600; no opset-11 op does this without materialising the slice.

## INSIGHT (transferable)
⭐ "Remove the noise pixels, keep the boxes" = the **part-of-a-filled-2×2** predicate,
computed as TWO 2×2 sum-convs: conv1 (pad bottom/right) counts each 2×2 block →
`==4` is a full block; conv2 (pad top/left) dilates the full-block map back over the
4 covering cells. Exact for any "isolated single pixels vs ≥2×2 solid shapes" task,
no flood-fill. Pairs with the single-Where free-output idiom by folding off-grid into
the keep condition (`selcond = keep OR offgrid`) so the removed branch only ever needs
a constant [1,10,1,1] background one-hot — no 10-channel delta plane. And one 1×1 Conv
with weight [50,1,2,…,9] packs background / colour-value / off-grid into a single f32
plane read by thresholds, eliminating a second f32 channel-reduction.
