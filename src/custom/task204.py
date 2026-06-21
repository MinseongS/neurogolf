"""Task 204 (ARC 868de0fa): hollow blue squares; fill each square's interior
with orange 7 (odd side length L) or red 2 (even L).  Blue walls stay blue (1),
in-grid background stays 0, off-grid stays all-zero.

Rule (exact, verified 0/2000 fresh in numpy + fresh-3000 against the generator):
  Several non-overlapping (gap >= 1) HOLLOW blue square boxes of side L in [3,10]
  on a size x size grid (size = 10..20).  Each box's (L-2)x(L-2) interior is filled
  orange if L is odd, red if L is even; the blue 1px outline is preserved.

Encoding (uint8 perimeter-anchor detection -> linear conv output assembly):
  * blue = input channel 1, sliced + cropped to the 20x20 active region (size<=20)
    and cast to uint8 (the QLinearConv input).
  * PER-SIZE PERIMETER ANCHOR: for each k in 3..10, a k x k uint8 perimeter kernel
    (1 on the ring, 0 inside) convolved (QLinearConv) over blue fires exactly 1 at a
    box's top-left corner iff a perfect k-box outline sits there.  The detection is
    SHARP because the QLinearConv weight-scale = 1/(2*peri-1): conv==peri rounds to 1,
    conv==peri-1 rounds to 0 (one missing ring pixel kills the anchor).
  * DILATE each anchor by a (k-2)x(k-2) MaxPool (pad k-3 each side) -> the (k-2)^2
    interior block, all aligned to 18x18; Max over the ODD sizes -> orange_core,
    over the EVEN sizes -> red_core.  Pad the cores 18x18 -> 20x20 with the (1,1)
    top-left offset so the interior lands inside its box.
  * IN-GRID mask via two 1-D occupancy profiles (ReduceMax over channel+one spatial
    axis -> a [1,1,30,1] / [1,1,1,30] vector, thresholded and AND'd) -> a 20x20 uint8
    grid plane.  The grid is a solid size x size square so this is exact, and it is
    ~2.5x cheaper than slicing the fp32 background channel.
  * OUTPUT via ONE 1x1 QLinearConv on Concat[grid, blue, red, orange] (the FREE
    output op, pads the 20x20 result out to 30x30 -> off-grid all-zero):
        ch0 (bg)     = grid - blue - red - orange   (interiors removed from bg)
        ch1 (blue)   = blue
        ch2 (red)    = red
        ch7 (orange) = orange
    int8 out_w; uint8 output clamps the bg subtraction at 0.  No per-cell Where/
    Equal label plane is ever materialised.

Beats the deployed kojimar net (15.62 pts, 11364B, 454 par): the bg fp32 slice
(1600B) + Concat(bg,blue) is replaced by a 1-D-occupancy grid plane and a single
[grid,blue,red,orange] output conv -> mem 11364 -> 10604, pts 15.62 -> 15.69.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
I8 = TensorProto.INT8
BOOL = TensorProto.BOOL
W = 20  # working canvas (grid size is always <= 20)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    # ---- blue channel sliced to the 20x20 active region -> uint8 ------------
    init("bst", np.array([1, 0, 0], np.int64), np.int64)
    init("ben", np.array([2, W, W], np.int64), np.int64)
    init("bax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "bst", "ben", "bax"], "blue_f")     # [1,1,20,20] f32 (entry)
    n("Cast", ["blue_f"], "blue_u8", to=U8)

    init("q_scale", np.array([1.0], np.float32), np.float32)
    init("q_zero_u8", np.array([0], np.uint8), np.uint8)

    def perim(k):
        w = np.ones((1, 1, k, k), np.uint8)
        w[0, 0, 1:k - 1, 1:k - 1] = 0
        return w

    # ---- per-size perimeter anchor (QLinearConv) + dilate to interior ------
    odd_fills, even_fills = [], []
    for k in range(3, 11):
        peri = int(perim(k).sum())
        init(f"dw_{k}", perim(k), np.uint8)
        # weight-scale 1/(2*peri-1): conv==peri -> 1, conv==peri-1 -> 0 (sharp)
        init(f"dws_{k}", np.array([1.0 / (2 * peri - 1)], np.float32), np.float32)
        n("QLinearConv",
          ["blue_u8", "q_scale", "q_zero_u8", f"dw_{k}", f"dws_{k}", "q_zero_u8",
           "q_scale", "q_zero_u8"],
          f"anchor_{k}", kernel_shape=[k, k])
        ks = k - 2
        pad = ks - 1
        n("MaxPool", [f"anchor_{k}"], f"fill_{k}",
          kernel_shape=[ks, ks], pads=[pad, pad, pad, pad], strides=[1, 1])
        (odd_fills if k % 2 else even_fills).append(f"fill_{k}")

    n("Max", odd_fills, "orange_core")                       # [1,1,18,18] uint8
    n("Max", even_fills, "red_core")
    init("fill_pad", np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64), np.int64)
    init("z_u8", np.array(0, np.uint8), np.uint8)
    n("Pad", ["orange_core", "fill_pad", "z_u8"], "orange20", mode="constant")
    n("Pad", ["red_core", "fill_pad", "z_u8"], "red20", mode="constant")

    # ---- in-grid mask via 1-D occupancy profiles -> uint8 grid plane -------
    n("ReduceMax", ["input"], "rowp30", axes=[1, 3], keepdims=1)  # [1,1,30,1] f32
    n("ReduceMax", ["input"], "colp30", axes=[1, 2], keepdims=1)  # [1,1,1,30] f32
    init("rst", np.array([0], np.int64), np.int64)
    init("ren", np.array([W], np.int64), np.int64)
    init("r2", np.array([2], np.int64), np.int64)
    init("c3", np.array([3], np.int64), np.int64)
    init("HalfF", np.array(0.5, np.float32), np.float32)
    n("Slice", ["rowp30", "rst", "ren", "r2"], "rowp")
    n("Slice", ["colp30", "rst", "ren", "c3"], "colp")
    n("Greater", ["rowp", "HalfF"], "rowb")
    n("Greater", ["colp", "HalfF"], "colb")
    n("Cast", ["rowb"], "rowb_u8", to=U8)                    # [1,1,20,1] uint8
    n("Cast", ["colb"], "colb_u8", to=U8)                    # [1,1,1,20] uint8
    # grid = rowany AND colany, built directly in uint8 (Min broadcasts both axes,
    # skipping the bool And + uint8 Cast of the full 20x20 plane)
    n("Min", ["rowb_u8", "colb_u8"], "grid_u8")              # [1,1,20,20] uint8

    # ---- output assembly: 1x1 QLinearConv on [grid, blue, red, orange] -----
    n("Concat", ["grid_u8", "blue_u8", "red20", "orange20"], "out_state", axis=1)
    ow = np.zeros((10, 4, 1, 1), np.int8)
    ow[0] = np.array([1, -1, -1, -1]).reshape(4, 1, 1)       # bg = grid-blue-red-orange
    ow[1] = np.array([0, 1, 0, 0]).reshape(4, 1, 1)          # blue
    ow[2] = np.array([0, 0, 1, 0]).reshape(4, 1, 1)          # red
    ow[7] = np.array([0, 0, 0, 1]).reshape(4, 1, 1)          # orange
    init("out_w", ow, np.int8)
    init("out_w_zero", np.array([0], np.int8), np.int8)
    n("QLinearConv",
      ["out_state", "q_scale", "q_zero_u8", "out_w", "q_scale", "out_w_zero",
       "q_scale", "q_zero_u8"],
      "output", kernel_shape=[1, 1], pads=[0, 0, 10, 10])    # -> [1,10,30,30] uint8

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", U8, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 13)])
