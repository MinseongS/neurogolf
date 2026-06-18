# task133 — 57aa92db

**Rule:** A per-instance random 3×3 connected "creature" template (one cell tagged
"pixel0") is stamped 2–4 times. Each stamp has its own random magnification m∈{1..4}
(stamp 0 always m=1 = the "master"), random position, and a distinct random color
(all ≠ pcolor). In the INPUT, the master is drawn FULLY (creature at m=1, its pixel0
cell drawn in pcolor), while every other sprite shows only TWO magnified blocks: its
pixel0 block (drawn pcolor) and one adjacent "show" template cell (drawn in the sprite
color). The OUTPUT fully reconstructs every sprite: the whole magnified template in the
sprite color, with the pixel0 block recolored to pcolor.
**Current:** 12.84 pts, gen-import (gen:thbdh6332), mem 191249, params 206.
**Target tier:** none admissible — reconstruction/correspondence wall.

## Feasibility analysis (why INFEASIBLE)
A full numpy reference solver was built and is exact, but ONLY by using explicit
non-local connectivity:
1. **pcolor** = argmax(bbox_area / cell_count) over present colors — verified 0/300
   against true generator pcolor. (This single piece IS ONNX-friendly.)
2. **master** = the one non-square sprite (creature, not a solid block).
3. **template** recovered from master color cells ∪ its pcolor pixel0 cell.
4. **per-sprite anchor**: each non-master sprite shows a `show` block of side m AND
   a separate pcolor pixel0 block; they must be MATCHED by spatial adjacency (the
   pcolor channel holds 2–4 differently-sized pixel0 blocks, one per sprite, and
   nothing but proximity distinguishes which belongs to which sprite). The `show`
   index is random, so the anchor is NOT derivable from the sprite channel alone.
5. **stamp**: per-sprite variable-magnification Kronecker upscale of the recovered
   arbitrary template at a data-dependent anchor, with pixel0-block recoloring.

Steps 2–5 each require operations with no compact ONNX form:
- variable NUMBER of sprite components (2–4) ⇒ no fixed-slot unrolling without
  data-dependent component assignment (ArgMax/sort/NonZero over components — BANNED);
- cross-channel pixel0↔sprite matching is a connectivity/correspondence grouping
  (flood-fill / component labeling — BANNED, and no separable surrogate exists);
- per-channel RUNTIME-magnification Kronecker of a per-instance ARBITRARY template
  needs a data-dependent Gather-index plane PER sprite channel (~3600B fp32 each),
  and even that presupposes the (already-blocked) anchor.

This is the BUILD_PROMPT-flagged wall: variable-size/variable-count components +
cross-component spatial correspondence. The public net's 191KB is just a bloated
gen-import; there is no compact exact construction to undercut it meaningfully.

## OPEN ANGLES (exhausted)
- Per-channel processing avoids sprite ENUMERATION but NOT the pcolor↔sprite
  spatial matching (pcolor blocks are mutually indistinguishable within the channel).
- No bound on sprite count or magnification makes any fixed-slot unroll exact.

## INSIGHT (transferable)
⭐ pcolor / "signature marker color stamped once per object" is recoverable as
argmax(bbox_area / pixel_count) — the marker's cells are scattered across all objects
so its bbox is huge relative to its count, while each object color is a tight block.
But "reconstruct N variable-scale Kronecker stamps of a per-instance template, each
anchored by matching a same-colored marker block to its object" is a genuine
correspondence wall (no separable/banded/closed-form escape; matching needs banned
connectivity ops).
