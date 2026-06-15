# task051 — 25d487eb

**Rule:** "Laser beam from an arrowhead." A solid downward-narrowing TRIANGLE
(arrowhead) of colour c0 is drawn — widest at its base (width 2*depth-1),
narrowing to a single apex cell (depth in {3,4}) — with a single TIP pixel of
colour c1 at the centre of the base. A BEAM of colour c1 fires from the apex out
to the grid edge along the triangle's axis of symmetry. apply_gravity then
rotates/flips the whole figure into one of 4 cardinal orientations (arrowhead may
point up/down/left/right). INPUT = triangle + tip; OUTPUT additionally paints the
beam. Recovery rule (0 errors / 3000 fresh): tip colour = channel with pixel
count == 1 (triangle count > 1, exclude bg ch0); beam AXIS = the shorter triangle
span (base is the wide edge, so cspan>rspan => vertical/up-down, else
horizontal); beam DIRECTION = toward the apex = toward the side of the tip where
the triangle centroid lies (vertical: up iff centroid-row < tip-row; horizontal:
left iff centroid-col < tip-col). Beam fills the axis-line through the tip, in the
apex half-plane, on in-grid background cells.
**Current:** 15.00 pts (public gen:vyank6322) -> custom:task051 **15.03** pts, mem 21323, params 120
**Target tier:** detection (label-map B with non-local arrowhead detection). Output
colour is a non-local function (find arrowhead, infer axis+direction, project a
ray) => not S/A/separable. Floor = a per-cell colour label map + one fp32 colour
conv. 15.03 sits just past the 15.00 public baseline.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | label-map L=V+tipcol*beam; tip/tri masks as [1,10,30,30] selects; fp32 planes | det | 156655 | 104 | 13.04 | (stored ok) | too big |
| 2 | derive tip/tri masks from V==colour scalars (kill [1,10,30,30]) | det | 108099 | 104 | 13.41 | ok | better |
| 3 | all per-cell planes fp16; fuse beam into 1 outer product (linerow x linecol) | det | 55149 | 106 | 13.78 | ok | |
| 4 | channel-space ROW/COL PROFILES -> tip/tri stats are 1-D, no 2-D mask plane | det | 39923 | 106 | 14.40 | ok | big cut |
| 5 | in-grid-bg via ch0 conv; fuse sentinel chain | det | 37383 | 116 | 14.47 | ok | |
| 6 | 20x20 working canvas (grids <=20x20, top-left); Pad L back to 30x30 | det | 29163 | 129 | 14.71 | ok | big cut |
| 7 | drop ch0 conv: in-grid = rowany(outer)colany from profiles | det | 24163 | 119 | 14.90 | ok | |
| 8 | fold tipcol into 1-D linecol; Where(ingridB,Lf0,15) sentinel; Where bgin | det | **21323** | **120** | **15.03** | **200/200** | **BEST** |

## Best achieved
**15.03** @ mem 21323 params 120 — adopted? **N** (build-only per task scope).
Beats prior 15.00? **Y** (+0.03). Fresh 200/200 (isolated, in-memory vs fresh gen).

## Irreducible-floor analysis
Dominant intermediate = **V32 [1,1,30,30] fp32 = 3600 B**, the colour-index Conv
output (sum_c c*input[c]). It MUST be 30x30 fp32 (input is fp32; Conv preserves
dtype; Pad cannot retype) before it is cropped to the 20x20 working canvas (Vc
1600, V 800). Same "fp32 conv crop is the cheapest colour gateway" floor that
task020 hit. The rest is already minimal: ~10 fp16 [1,1,20,20] planes (800 B),
two [1,10,30,1]/[1,10,1,30] profile reductions (1200 B, also fp32-input-bound),
the 900 B uint8 L (free-output feeder). Everything non-2D is 1-D (<=20 elems) or
channel-space (10 elems).

## OPEN ANGLES (re-attack backlog)
- Eliminate V32's full-30 cost: get per-cell colours at 20x20 without a 30x30
  fp32 conv. A spatial input crop to [1,10,20,20] is 16000 B (worse); Gather-crop
  same. No cheap retype path found — this is the binding constraint on further
  gains. If a 20x20 colour plane could be produced directly (e.g. a custom op or
  a Conv that only emits the top-left 20x20), mem would drop ~3000 -> ~18000 (pts
  ~15.2).
- The two profile reductions (2400 B) are fp32 because input is fp32; if the
  colour conv and profiles shared one cropped fp32 source the convs could fuse,
  but the shared source itself is the expensive 30x30 plane.

## INSIGHT (transferable)
⭐ **Collapse 2-D detection to 1-D PROFILES in channel space before selecting a
colour.** Instead of building a [1,1,H,W] tip/triangle mask (and the 4-5 derived
planes per mask), reduce the input to row/col profiles `ReduceSum(input,axis3)` /
`(...,axis2)` = [1,10,H,1] / [1,10,1,W] (tiny), then select the tip/triangle
channel in profile space. tip-row/col, triangle centroid, row-span/col-span, AND
the in-grid rectangle (`rowany (outer) colany`, valid because the grid is a solid
H×W block anchored top-left) all fall out of 1-D vectors — zero 2-D mask planes.
⭐ **A separable axis-aligned ray = combine the two 1-D vectors BEFORE the outer
product.** vertical beam = vhalf(rows) ⊗ oncol(cols); horizontal = onrow ⊗ hhalf.
Pick the right pair with scalars (`linerow = vert*vhalf + (1-vert)*onrow`,
`linecol = vert*oncol + (1-vert)*hhalf`) so a SINGLE [1,1,H,W] Mul yields the beam
— and fold the output colour into the 1-D `linecol` so the same product is the
colour contribution. ⭐ **`Where(ingridB, value, sentinel)` beats additive
sentinels** (one op, reuses the boolean in-grid mask, no offgrid/sentadd planes),
and `Where(ingridB, notpres, 0)` builds in-grid-background with no extra Mul.
