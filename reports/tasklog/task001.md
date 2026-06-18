# task001 — 007bbfb7 (fractal self-tiling of a 3x3 sprite)

**Rule:** A 3x3 grid S has 2..8 same-coloured on-cells (one random colour 1..9).
The 9x9 OUTPUT renders the shape with copies of itself = the Kronecker product
`kron(S,S)*colour`:  `output[3i+r,3j+c]=colour iff S[i,j] AND S[r,c]` (else bg 0).
Input sprite sits at top-left rows 0..2 cols 0..2; output 9x9 at top-left.
Strictly EASIER than task195 (no upscale, no random offset, no fixed colour).
**Current:** prior 16.83. This session: **17.68 pts, custom label-map (occ slice +
colour argmax + kron + Equal), mem 1448, params 62, fresh 500/500.**
**Target tier:** B (label map + final Equal). Tier S/A blocked: output cell value
is the 2-factor index map `S[u//3,v//3] AND S[u%3,v%3]` (kron), NOT a row⊗col
separable rectangle, so no separable bool-output Tier-A; colour is data-dependent
(any 1..9) so no fixed-Conv Tier-S route.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | occ=1-ch0 slice(3x3) + colour=ArgMax(masked cnt) + kron via two [9,9] flat macro/micro Gathers + Where→9x9 uint8 → Pad 30x30 → Equal | B | 1511 | 211 | 17.55 | 200/200 | correct; macro/micro [9,9] int64 maps = 162 params |
| 2 | kron via four [9] index vectors (row-Gather then col-Gather of 3x3 S, axis2/axis3), drop Reshape | B | **1448** | **62** | **17.68** | **500/500** | BEST: params 211→62 |

## Best achieved
**17.68 @ mem 1448 params 62 — fresh 500/500 (isolated, file-path generator).**
Beats prior 16.83 by **+0.85**. Adopted? N (build-only per brief).

## Irreducible-floor analysis
One intermediate dominates: **L [1,1,30,30] uint8 = 900** of 1448 — the Pad output
driving the final Equal. The Equal must span the full 30x30 output footprint and
uint8 is already the smallest dtype, so this is the canonical label-map floor.
Everything else is ≤81 B (9x9 bool kron factors / uint8 label, [1,1,9,3] gathers,
3x3 occ slice, [1,10] colour-count vector). Ceiling if L were the only cost:
`25-ln(900+62)≈18.13`.

## OPEN ANGLES (re-attack backlog)
- **Drop the 900 L-plane**: output footprint is only the top-left 9x9, but ORT
  **Pad rejects bool** (so can't Equal at 9x9 → [1,10,9,9] bool then Pad to 30x30),
  and Concat/ScatterND assembly of the 10-ch 30x30 output from a 9x9 block costs
  ≥900 in carrier/zero tensors. No clean sub-900 final found (same wall as task195).
- Shave the ~548 of sub-900 intermediates further (e.g. fuse the two kron factors)
  — marginal (~+0.05), the 900 dominates.

## INSIGHT (transferable)
⭐ **kron via four [9] row/col index vectors beats two [9,9] flat macro/micro maps**
on PARAMS: `kron(S,S)[u,v]=S[u//3,v//3] AND S[u%3,v%3]` builds as Gather(S, div,
axis=2)→Gather(·, div, axis=3) for the macro factor and the same with `mod` for the
micro factor (div=[0,0,0,1,1,1,2,2,2], mod=[0,1,2,0,1,2,0,1,2]) — 36 index params
vs 162 for the flat [9,9] maps, same tiny [1,1,9,9] bool intermediates. Retrofit
into task195 (would cut its 243 params). Keeping the factors 4-D ([1,1,9,9]) lets
the final Where/Pad skip a Reshape.
⭐ When the sprite is at a FIXED corner (no offset like task195), occupancy is just
`1 - channel0` over the corner slice (one channel set per cell ⇒ ch0=1 ⇔ bg) — no
bounding-box ReduceMin recovery needed. Colour (data-dependent 1..9) is one scalar:
`ArgMax(ReduceSum(input,[2,3]) · ch0-mask)` — mask ch0 or background steals it.
