# task021 — 1190e5a7

## 2026-06-29 output-mask screen

Current source score: 17.935241 @ mem 1072 params 98.

Rule: an input grid is divided by grid lines; output is a small rectangle whose
height/width are the counts of row/column cells, filled with the main colour.

The graph is close to the final-output frontier: it computes the main colour and
two scalar output dimensions, builds a full [1,1,30,30] bool `cell_in_out` mask
(900 B), and uses `Where(mask, main4, zero) -> output` so the 10-channel rectangle
is the free graph output.

`conv_fit.py 21` failed for k=1/3/5.  A smaller 7x7 mask route is not obviously
better: once the random main colour is applied, it either creates a counted
[1,10,7,7] one-hot plane or a 30x30 uint8 label before final `Equal`, roughly
matching or exceeding the current 900 B full mask.  No rewrite adopted.

## S9 (2026-07-03) — kojimar teacher REPAIRED then ADOPTED (+0.465)
Teacher drops the [1,1,30,30] bool mask (900B): computes the two dims via free-input
einsums, builds small H×W mask, Pads to 30. RAW teacher had height cap 6 (range_h6
[0..5]) but generator rows∈[2,7] → 8/761 fresh fails (bundled max H=6 hid it — classic
bundled-overfit). Repair: range extended to 7 (+78B mem, +1 param), pad fixed.
mem 1072→675, params 98→60. Gates: stored fail=0; fresh 2929 valid/10000 (oversize grids
rejected): inc 0 / repaired 0. Backup reports/retired_networks/task021_pre_s9.onnx.
⚠ raw base_submission/task021.onnx is private-LB-fragile — never adopt it directly.
