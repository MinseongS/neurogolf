# task156 — 694f12f3

## 2026-06-29 compact-bitwise screen

Current source score: 17.536063 @ mem 1697 params 47.

Rule: two yellow rectangles have their interiors recoloured according to which
rectangle is smaller/taller, with optional vertical flip.

The source already uses a compact label program: channel-4 slice, 3x3
`AveragePool` to detect rectangle interiors, scalar row-sum tests to determine
orientation, then a 5-channel 10x10 one-hot (`out5`, 500 B) padded directly to the
free output.

No rewrite adopted.  A full uint8 label-map route would need the 10x10 label plus
a 30x30 padded label before final `Equal`, larger than the current 5-channel
compact one-hot.  The 3x3 interior detector is the semantic floor for distinguishing
border yellow from interior recolour cells.
