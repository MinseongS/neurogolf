# task295 — bbc9ae5d

## 2026-06-29 final-expansion screen

Current source score: 17.619744 @ mem 1557 params 47.

Dominant tensor is `full_label` [30,30] uint8 = 900 B, followed by the 9x18
working label/mask planes.  This is already the cheaper final-output route for
this rule: replacing the label plane with a compact one-hot before padding would
materialize roughly 9x18x10 bool cells, larger than the single uint8 label plane.

No rewrite adopted.
