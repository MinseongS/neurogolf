"""task211 (ARC-AGI 8d5021e8) — mirror+tile a small grid into a 6-fold layout.

Rule (from the ARC-GEN generator, verified fresh):
  Input is a `height` x `width` grid (fixed 3 rows x 2 cols) holding pixels of a
  single colour on a black canvas.  The OUTPUT is a (3*height) x (2*width) grid
  (9 rows x 4 cols).  Each input pixel (r,c) of colour k is stamped at SIX output
  positions:
      rows: {height-1-r, height+r, 3*height-1-r}   (top-flip, mid-direct, bot-flip)
      cols: {width-1-c, width+c}                     (left-mirror, right-direct)
  i.e. the output is the input grid reflected about a horizontal mid-axis (cols
  0..1 = mirror, cols 2..3 = direct) and tiled vertically 3x as [flip|direct|flip].

  Every output cell (R,C) therefore COPIES exactly one input cell (r,c):
      r = row_src[R],  c = col_src[C]      (R in 0..8, C in 0..3)
      row_src = [2,1,0, 0,1,2, 2,1,0]      # 2-R | R-3 | 8-R
      col_src = [1,0, 0,1]                  # 1-C (C<2) | C-2 (C>=2)

Closed-form reconstruction = a pure spatial COPY (Tier-S), no detection.  The
cheapest route is GridSample on the tiny *active* region:
  1. Slice the active 3x2 block  (fp32, [1,10,3,2] = 240B).
  2. Cast it to fp16            ([1,10,3,2] = 120B) so the sampled tensor is fp16.
  3. GridSample(nearest, zeros) with a [1,9,4,2] grid that maps each output cell
     (R,C) to the normalized (col_src[C], row_src[R]) coordinate of the 3x2 input
     -> [1,10,9,4] fp16 = 720B  (the dominant intermediate, half the fp32 cost the
     public GridSample net pays for the same logical sample).
  4. Pad back to 30x30; the Pad result IS the (free) output.

  GridSample beats a two-stage Gather here because one op produces the [1,10,9,4]
  result, avoiding the intermediate [1,10,3,4] tensor a col-then-row gather needs.
  Channel-0 (background) is carried through exactly, so background output cells are
  one-hot; cells outside the 9x4 grid stay 0 via padding_mode=zeros + the zero-Pad.

Memory: 240 + 120 + 720 = 1080B intermediates; ~87 params (grid 72 + slice 6 +
pad 9).  Total ~1167  ->  ~17.94  (public net 17.67).
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

S = 30
H, W = 3, 2          # input grid: rows, cols
OH, OW = 9, 4        # output grid: rows, cols

ROW_SRC = [2, 1, 0, 0, 1, 2, 2, 1, 0]   # output row R -> input row
COL_SRC = [1, 0, 0, 1]                   # output col C -> input col


def build(task):
    inits, nodes = [], []

    def init(name, arr, dt):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    F16 = TensorProto.FLOAT16

    # ---- slice the active 3x2 region (fp32) --------------------------------
    init("sl_starts", np.array([0, 0], np.int64), np.int64)
    init("sl_ends", np.array([H, W], np.int64), np.int64)
    init("sl_axes", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["input", "sl_starts", "sl_ends", "sl_axes"], "act")  # [1,10,3,2] fp32

    # ---- to fp16 so the sampled plane is half-size -------------------------
    n("Cast", ["act"], "act16", to=F16)                              # [1,10,3,2] fp16

    # ---- sampling grid [1,OH,OW,2] = (x=col, y=row) normalized for a 3x2 in -
    # align_corners=0 normalization: coord(i,N) = (2*i + 1)/N - 1
    def xn(c):  # column -> normalized x (width W)
        return (2.0 * c + 1.0) / W - 1.0

    def yn(r):  # row -> normalized y (height H)
        return (2.0 * r + 1.0) / H - 1.0

    grid = np.zeros((1, OH, OW, 2), np.float32)
    for R in range(OH):
        for C in range(OW):
            grid[0, R, C, 0] = xn(COL_SRC[C])   # x
            grid[0, R, C, 1] = yn(ROW_SRC[R])   # y
    init("grid", grid, np.float32)
    n("GridSample", ["act16", "grid"], "gs",
      mode="nearest", padding_mode="zeros", align_corners=0)         # [1,10,9,4] fp16

    # ---- pad back to 30x30 (the output is FREE) ----------------------------
    pads = np.array([0, 0, 0, 0, 0, 0, S - OH, S - OW], np.int64)
    init("pad_amt", pads, np.int64)
    n("Pad", ["gs", "pad_amt"], "output", mode="constant")  # [1,10,30,30] fp16, val=0

    x = helper.make_tensor_value_info("input", F, [1, 10, S, S])
    y = helper.make_tensor_value_info("output", F16, [1, 10, S, S])
    graph = helper.make_graph(nodes, "task211", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 16)])
