# task46 — variable-width snake re-pack (BAIL)

**Rule:** Height-3 grids. Generator builds a horizontal snake of 3-4 colored segments
(width 2-4 each) that turn vertically. INPUT shifts each segment idx RIGHT by idx cols
(gray separator cols inserted), applies a per-segment data-dependent vertical roll, and
recolors inter-segment junction pixels to gray(5). OUTPUT re-packs segments contiguously,
undoes vertical offsets, restores junction colors. Output width data-dependent.

**Target tier:** none — BAIL at EARLY FEASIBILITY CHECK.

**Verdict: CONFIRMED-INFEASIBLE (BAIL-class wall).** Three independent wall signals:
1. Connectivity-based segmentation — gray junctions co-occur with content in the same
   column, #empty_cols != #segments, so segment recovery needs tracing the connected
   turning path (flood/connectivity floor).
2. Per-segment data-dependent vertical roll (94% of instances) — piecewise-constant-per-
   segment row shift on variable-width blocks = data-dependent GatherND.
3. Data-dependent column compaction — output width + col mapping data-dependent.

Current P=14.64 (mem+params 31506). Beating +0.3 needs ~23300 (~25% cut); transform
needs 3600B fp32 entry plane + segment-label plane + roll gather + compaction gather,
none collapsible. Existing ~30640-mem solver is near structural floor.

**INSIGHT:** "Re-pack variable-width path segments with per-segment data-dependent
vertical realignment" = canonical GatherND wall. Tell = shift amounts constant-within-
segment but data-dependent across segments AND boundaries needing a traced connected path
(gray junctions co-located with content defeat any column-profile segmentation). Both
signals ⇒ BAIL fast, do not build.
