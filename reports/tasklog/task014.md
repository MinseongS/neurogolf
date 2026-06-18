# task014 — 0b148d64

**Rule:** Grid split into 4 quadrants by a thick all-background cross. Random pixels (0.9 density) per quadrant are coloured; the one special quadrant gets `color_list[0]`, the others `color_list[1]`. Generator only `break`s when the rarest FOREGROUND colour == color_list[0], so the special quadrant is the colour with the FEWEST pixels in the grid. Output = grid cropped to the bbox of that rarest colour (cells inside are rarest-colour or background 0; nothing else lands in one quadrant).
**Current:** 14.43 pts, ext:kojimar6275, mem 38399, params 650
**Target tier:** B (label-map + final Equal) — crop is data-dependent + colours are arbitrary input colours, so no pure-copy Tier-S and no separable row⊗col routing.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | task036 crop idiom, min-COUNT colour, WORK=20 f32 | B | 19551 | 163 | 15.11 | 200/200 | beats +0.68 |
| 2 | fp16 occupancy+window, WORK=19 | B | 16392 | 161 | 15.29 | 200/200 | fp16 occ planes recast to f32 by ORT (no help) |
| 3 | select rarest channel FIRST → bbox from single [1,1,30,30] plane (kills 6×[1,10,30,1] occ planes) | B | 11564 | 157 | 15.63 | 200/200 | big win |
| 4 | f32 window (drop fp16 cast; PrecisionFreeCast cancelled fp16 benefit) | B | 11626 | 156 | 15.63 | 300/300 | adopted |

## Best achieved
15.63 @ mem 11626 params 156 — beats prior 14.43 by +1.20. Fresh 300/300.

## Irreducible-floor analysis
Dominant intermediates: `bplane32` [1,1,30,30] f32 = 3600 (the channel-select plane; the 3600B fp32 entry-plane floor — data-dependent row/col gather forces materialising the full chosen-colour plane before cropping), then `Vr` [1,1,19,30] f32 = 2280 (full-width row-gather intermediate) and `Vs` [1,1,19,19] f32 = 1444 (cropped window). fp16 on the window does NOT help: ORT inserts a PrecisionFreeCast back to f32 for the `Greater(Vs,·)` compare, so the fp16 path (cast 1800 + recast 1444) costs more than staying f32.

## OPEN ANGLES (re-attack backlog)
- Avoid the full `bplane32`: the crop only needs WORK×WORK. But the row/col gather indices are data-dependent so the chosen-colour plane must exist at full 30×30 before cropping — structural 3600B floor.
- `Vr` 2280: gather cols-then-rows gives the same [1,1,30,WORK] intermediate; no saving.
- A 1×1 runtime-one-hot Conv to select the channel is the same size as the Gather; no win.

## INSIGHT (transferable)
⭐ For crop/bbox tasks on ONE selected colour: select the rarest/target CHANNEL **before** computing the bbox — then occupancy profiles are [1,1,30,1] (120B) instead of per-channel [1,10,30,1] (1200B). Dropping the 10-ch occupancy stage was the single biggest win here (16392→11564). And confirming the guide: fp16 on full-canvas planes feeding a `Greater`/compare is silently re-cast to f32 by ORT (PrecisionFreeCast) — measure before trusting fp16.
