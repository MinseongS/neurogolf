# task054 — 264363fd

## Current live/source-owned exact

`memory=26877`, `params=238`, `points=14.792158`.

The graph is exact-preserve/source-owned.  It detects a reference star motif,
removes the reference motif, finds up to four seed stars inside large boxes,
draws horizontal/vertical guide lines through each seed across its containing
box, stamps the motif at each seed, and emits final one-hot with `Equal`.

## Dominant memory

- 3600B colour-index Conv output.
- Many 900B full-canvas uint8/bool edit planes:
  `label_u8`, `bg_mask`, `other/seeds`, `h_line`, `line_with_seeds`,
  `line_target`, `cleared_label`, `filled_label`, `output_label`.
- 1024B/960B int64 sparse fill/index tensors.

## 2026-06-29 sparse-edit-stream probe

Hypothesis: replace the full-canvas line cascade

`h_line_u8 -> line_with_seeds_u8 -> line_target_b -> output_label`

with sparse row/column edits applied directly to `filled_label`.

Graph-surgery candidate:

- Gather current rows from `filled_label` at `h_rows_4`.
- Use `h_updates_4_30 > h_seed_rows` to build sparse horizontal line updates.
- `ScatterND` those rows into `filled_label`.
- Gather current vertical columns and scatter vertical line updates.

Result:

- Best attempted candidate: `memory=26397`, `params=238`, but failed stored
  `244/266` (`22` arc-gen failures).  The raw saving was only `480B`.
- Variant trying to mimic inactive vertical overwrite by gathering from original
  `filled_label` failed much worse (`137/266`).

Reason:

The incumbent relies on `ScatterElements` duplicate/overwrite semantics for
inactive vertical slots.  Inactive vertical updates can intentionally erase a
horizontal line candidate before the final `line_with > seeds` comparison.
Applying sparse label edits directly loses this subtle mask-space behavior.

Conclusion:

The broad mechanism is still plausible only when sparse edits are monotonic
overwrites.  It is not safe when the intermediate mask uses inactive duplicate
scatter writes as part of the logic.  For task054, a successful rewrite must
either preserve the exact mask-space overwrite semantics or avoid generating
duplicate inactive vertical indices at the source.
