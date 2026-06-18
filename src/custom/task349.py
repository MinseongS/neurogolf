"""task349 (ARC-AGI db93a21d) -- "death stars": redraw beams + halos + centers.

Rule (from the generator, verified 0/500+ fresh in numpy and ORT):
  INPUT  = only the MAROON(=9) square centers, one per death-star.  Each center
           is a solid 2r x 2r block (r = radius).  The block WIDTH is always
           exactly 2r (never clipped: col in [0,size-2r]); the HEIGHT can be
           clipped at the top/bottom grid edge.
  OUTPUT = for every center:
    * MAROON center  : the 2r x 2r block copied from the input.
    * GREEN(=3) halo : the center block dilated by r on all sides -> a 4r x 4r
                       square centered on the block (Chebyshev dilation by r).
    * BLUE(=1) beam  : the center's columns filled from just BELOW the block
                       down to the bottom edge.
  Compositing priority per cell: MAROON > GREEN > BLUE > background.

Verified reductions (each 0/400-600 fresh):
  * GREEN = OR over d in 1..5 of  dilate_d( run-length-of-maroon >= 2d ).
    "a fully-maroon horizontal 2d-window exists within Chebyshev d" -- which is
    a SINGLE MaxPool over the left-anchored window-sum field cv_2d:
        cv_2d   = Conv(maroon, ones[1,1,1,2d], pad right 2d-1)   (window sums)
        green_d = MaxPool(cv_2d, kernel[2d+1, 4d],
                          pads=[d, 3d-1, d, d]) >= 2d
    (a width-2d window can sum to at most 2d, so ">=2d" == "a full window
    lies within reach": vertical reach d, horizontal reach d from either end
    of the 2d window -> x' window [x0-3d+1, x0+d]).
  * BLUE = a maroon cell exists strictly ABOVE in the same column
         = (downward prefix-OR of maroon) shifted down by one row.

Encoding: ONE fp32 1x1 Conv collapses the (bg,maroon) one-hot to colf in
{0:off-grid, 1:in-grid bg, 2:maroon} (the single fp32 plane).  Everything else
is fp16/bool.  L (uint8 colour index in {0,1,3,9}) -> Equal(L, arange) routes
the 10-channel one-hot into the FREE bool output; off-grid -> sentinel 255.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I64 = TensorProto.INT64

H = W = 30


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- colf: one 1x1 Conv collapsing (bg ch0, maroon ch9) ----------------
    # weight: ch0 -> 1, ch9 -> 2  =>  off-grid 0, in-grid bg 1, maroon 2.
    cw = np.zeros((1, 10, 1, 1), np.float32)
    cw[0, 0, 0, 0] = 1.0
    cw[0, 9, 0, 0] = 2.0
    init("colw", cw, np.float32)
    n("Conv", ["input", "colw"], "colf")                       # [1,1,30,30] f32
    init("h0", np.array(0.5, np.float32), np.float32)
    init("h1", np.array(1.5, np.float32), np.float32)
    n("Greater", ["colf", "h0"], "ingrid_b")                   # bool in-grid
    n("Greater", ["colf", "h1"], "meq")                        # bool maroon
    n("Cast", ["meq"], "m", to=F16)                            # f16 {0,1}

    # ---- GREEN = OR_d dilate_d( run>=2d ) ----------------------------------
    green_terms = []
    for d in range(1, 6):
        w = 2 * d
        init(f"ker{w}", np.ones((1, 1, 1, w), np.float16), np.float16)
        # left-anchored window-sum, pad right (w-1) to keep full width.
        n("Conv", ["m", f"ker{w}"], f"cv{w}", pads=[0, 0, 0, w - 1])
        # single MaxPool: vertical reach d, horizontal window [x-3d+1, x+d].
        n("MaxPool", [f"cv{w}"], f"gp{w}",
          kernel_shape=[2 * d + 1, 4 * d],
          pads=[d, 3 * d - 1, d, d], strides=[1, 1])
        init(f"wt{w}", np.array(float(w) - 0.5, np.float16), np.float16)
        n("Greater", [f"gp{w}", f"wt{w}"], f"gd{w}")           # bool green_d
        green_terms.append(f"gd{w}")
    cur = green_terms[0]
    for i, t in enumerate(green_terms[1:]):
        nxt = "green_b" if i == len(green_terms) - 2 else f"gor{i}"
        n("Or", [cur, t], nxt)
        cur = nxt

    # ---- BLUE: maroon at-or-above in same column (inclusive prefix-OR) ------
    # The block's own cells get blue=1 too, but the MAROON Where below has higher
    # priority and overwrites them; cells in the halo ABOVE a block have no maroon
    # above so stay 0.  -> inclusive downward OR is exact (verified 0/500).
    n("MaxPool", ["m"], "colcum",
      kernel_shape=[H, 1], pads=[H - 1, 0, 0, 0], strides=[1, 1])  # f16 incl-self
    init("half", np.array(0.5, np.float16), np.float16)
    n("Greater", ["colcum", "half"], "blue_b")

    # ---- compose L in {0,1,3,9} (priority maroon > green > blue) -----------
    # blue is already 0/1; use it as the base layer (Cast bool->uint8).
    n("Cast", ["blue_b"], "blueu", to=U8)                      # 0/1 uint8
    init("c3", np.array(3, np.uint8), np.uint8)
    init("c9", np.array(9, np.uint8), np.uint8)
    n("Where", ["green_b", "c3", "blueu"], "L2")
    n("Where", ["meq", "c9", "L2"], "L3")
    # off-grid -> sentinel 255 (matches no colour channel).
    init("sent30", np.full((1, 1, H, W), 255, np.uint8), np.uint8)
    n("Where", ["ingrid_b", "L3", "sent30"], "L")              # [1,1,30,30] u8

    # ---- output = Equal(L, arange[0..9]) -> BOOL [1,10,30,30] (FREE) -------
    init("arange", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "arange"], "output")

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task349", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
