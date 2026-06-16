# task086 — 3befdf3e

**Rule:** Each instance has 1-2 non-overlapping squares; each square has half-size L∈{1,2}.
In the INPUT a square is an (L+2)×(L+2) block — a 1-px border of color1 (c1) around an L×L
inner of color0 (c0); `colors=[c0,c1]` is GLOBAL per instance (c1 outnumbers c0 → c1 = most
frequent non-bg colour). OUTPUT per square is a fixed L-parametric concentric stamp at the
same location: `SQ`=the (L+2)² block; `er`=erode3(SQ)=L×L inner; perimeter ring `out0=SQ−er`
→ c0 (colours INVERT vs input); cross arms of length L extending L cells outward along the
block's row/col span (no corners) plus the inner → out1 → c1; everything else unchanged.
**Current:** 14.89 pts (public, BAIL?-flagged "overlaps").
**Target tier:** A — separable morphology + colour-index plane routed into the FREE output.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | chained Where(c1,Where(c0,input)) | A | 51180 | 73 | 14.16 | — | full [1,10,30,30] tmp plane (36KB) kills it |
| 2 | colour-index L → Equal-to-output (BOOL) | A | 29060 | 93 | 14.72 | — | killed the 10-ch tmp; occ/2× full planes remain |
| 3 | occ-derived SQ, build L at 12×12, pad once | A | 17312 | 99 | 15.24 | — | one 30×30 fp32 occ + 3 fp16 30×30 planes |
| 4 | drop occ (1-D row/col profiles for in-grid) | A | 15224 | 100 | 15.36 | — | removed 3600B occ |
| 5 | sentinel-mask+pad in 12×12, slice profiles to W | A | 11168 | 98 | **15.67** | 200/200 | adopted candidate |

## Best achieved
15.67 @ mem 11168 params 98 — adopted? candidate (do not self-adopt). Beats prior 14.89? **Y (+0.78)**.
Fresh ISOLATED 200/200 (and 500/500). Stored pass 266/266.

## Irreducible-floor analysis
Dominant intermediate = `L` [1,1,30,30] fp16 = 1800B: the colour-index plane (0=bg, c0/c1 on
the two regions, −1 off-grid sentinel) feeding `output = Equal(L, arange[1,10,1,1])` (BOOL,
free). Output colours COPY arbitrary input colours, so a per-cell colour-index plane is
mandatory; fp16 halves the documented 3600B fp32 floor to 1800B and cannot go lower without
removing the plane (not possible here). Everything else is a 12×12 fp16 working plane (~288B)
or a 1-D profile (~tens of B). The only fp32 left is the 576B ch0 slice (Slice preserves the
fp32 input dtype) and the [1,10] channel-count vector for colour recovery.

Variable L (no scalar): split SQ into L=1 vs L=2 cells via a 4×4-solid detector
(`Conv 4×4 ==16` top-left, dilated back over its 4×4 footprint with controlled ONNX pads),
then dilate each part with its own kernel (3-tap for L=1, 5-tap for L=2). out1 = (Σ of the
four directional Convs >0) AND NOT out0.

## OPEN ANGLES (re-attack backlog)
- The `L` plane is the floor; only escape would be a row⊗col-separable colour route, but the
  cross-arm + inverted-ring stamp is not separable (couples r&c), so 1800B looks terminal.
- Minor: the 576B fp32 ch0 slice could fold into the occupancy reduction, but it's tiny.

## INSIGHT (transferable)
⭐ A "size-parametric concentric stamp" with a SMALL discrete size set (here L∈{1,2}) is NOT a
flood-fill/correspondence wall: split the occupancy into per-size cell-masks via a fixed-size
solid-block detector (`Conv k×k == k²` top-left, dilated back over its footprint), then apply
each size's own separable dilation kernel and OR. The variable dilation amount is handled with
ZERO data-dependent ops or scalars. Colour inversion + outward cross = perimeter-ring(out0) +
(directional-dilation − block)(out1) over the input occupancy; recover the two global colours
as argmax / other-nonzero of channel counts and carry them as SCALAR indices into one fp16
colour-index plane → Equal-to-arange free BOOL output. This BAIL?-flagged "overlaps" task was
fully closed-form tier-A (+0.78).
