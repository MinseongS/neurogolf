# task256 — a65b410d

## 2026-06-29 compact-output screen

Current source score: 17.011118 @ mem 2898 params 50.

Rule: a red horizontal segment defines a triangular green/blue fill inside a
variable width/height top-left grid.  Width and height are independent random
extensions beyond the triangle.

The current graph pays for `black` [1,1,13,13] fp32 = 676 B to recover the true
in-grid background, then builds compact four-channel output (`bg`, `blue`, `red`,
`green`) as [1,4,13,13] bool = 676 B and pads it directly to the free output.

No rewrite adopted.  A full label-map route would need at least a 13x13 label plus
a 30x30 padded label before `Equal` (~1069 B), larger than the compact one-hot.
The black/in-grid slice is not removable from the red segment alone because the
generator chooses width and height independently; red geometry does not identify
the full rectangular grid boundary.
