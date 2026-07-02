# Grid bound analysis (randint inclusive both ends; common.randint = random.randint)

Binding = max over train+test bundled, stored `arc-gen` split (~262 ex, graded), and fresh generator (8000 + 2500 samples, agreed).
Output-size notes: 029/205 output is small (zoom/crop) but INPUT plane drives net cost; 014/138 output is a crop.

| task | arcid | static code bound (H,W) | src line | sampled max (N=8000, in / out) | train+test | arc-gen split | BINDING | verdict |
|------|-------|-------------------------|----------|-------------------------------|-----------|--------------|---------|---------|
| task187 | 7b6016b9 | 25,25 | `width, height = common.randint(20, 25), common.randint(20, 25)` | in 25x25 / out 25x25 | 25 | 25 | 25 | CROPPABLE 25 (save 5) |
| task029 | 1c786137 | in 25,25 | `width, height = common.randint(10, 25), common.randint(10, 25)` | in 25x25 / out 23x23 | 23 | 25 | 25 | CROPPABLE 25 (input plane) |
| task243 | 9edfc990 | 18,18 | `size, colors = common.randint(12, 18), []` | in 18x18 / out 18x18 | 16 | 18 | 18 | CROPPABLE 18 (save 12) |
| task198 | 83302e8f | 29,29 | `minisize=randint(3,5); size=randint(8,10)-minisize` -> linegrid size*(minisize+1)-1 | in 29x29 / out 29x29 | 29 | 29 | 29 | CROPPABLE 29 (save 1, marginal) |
| task080 | 39e1d7f9 | 31,31 (!) | `size=randint(5,10)`; linegrid size*(spacing+1)-1, spacing=6-(size-1)//2; size=8 -> 31 | in 31x31 / out 31x31 | 29 | 31 | 31 | NOT CROPPABLE (exceeds 30 cap; 48 stored 31x31 ex) |
| task205 | 8731374e | 30,30 | `width, height = common.randint(15, 30), common.randint(15, 30)` | in 30x30 / out 10x10 | 27 | 30 | 30 | NOT CROPPABLE (input reaches full 30x30) |
| task173 | 72322fa7 | 25,25 | `width, height = common.randint(10, 25), common.randint(10, 25)` | in 25x25 / out 25x25 | 22 | 25 | 25 | CROPPABLE 25 (save 5) |
| task138 | 5daaa586 | H26,W25 | `width = common.randint(10, 25)` / `height = width + common.randint(-1, 1)` | in 26x25 / out 24x23 | 22 | 26 | 26 (H26,W25) | CROPPABLE 26 (rect 26x25; save 4) |
| task014 | 0b148d64 | 25,25 | `width, height = common.randint(15, 25), common.randint(15, 25)` | in 25x25 / out 18x18 | 21 | 25 | 25 | CROPPABLE 25 (save 5) |
| task077 | 36fdfd69 | H20,W21 | `height = common.randint(15, 20)` / `width = height + common.randint(-1, 1)` | in 20x21 / out 20x21 | 18 | 21 | 21 (H20,W21) | CROPPABLE 21 (rect 20x21; save 9) |
| task193 | 7f4411dc | 20,20 | `size, color = common.randint(7, 20), common.random_color()` | in 20x20 / out 20x20 | 17 | 20 | 20 | CROPPABLE 20 (save 10) |
| task192 | 7e0986d6 | 20,20 | `width, height = common.randint(10, 20), common.randint(10, 20)` | in 20x20 / out 20x20 | 17 | 20 | 20 | CROPPABLE 20 (save 10) |
| task222 | 91714a58 | 16,16 | `def generate(..., size=16)`; grid is size x size | in 16x16 / out 16x16 | 16 | 16 | 16 | CROPPABLE 16 (save 14) |

## Conditional-branch flags (could produce larger than a naive sample)
- task080: size=8 branch yields 31x31 (>30). PRESENT in stored eval (48/262). Hard non-crop; net must already handle 31.
- task138: height = width + randint(-1,1) makes H exceed W by 1 (H up to 26 while W<=25). Crop must be rectangular or square-26.
- task077: width = height + randint(-1,1) makes W exceed H by 1 (W up to 21, H<=20).
- task198/task080: dims come from create_linegrid magnification (size*(spacing+1)-1), not a direct randint -- non-obvious; verified by formula + sampling.
