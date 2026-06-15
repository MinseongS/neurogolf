"""task329 (ARC-AGI d23f8c26) — keep only the middle column, clear the rest.

Rule (from the generator): the grid is a square `size x size` of random colours
(size always ODD = 2*randint(1,4)+1, anchored top-left).  The output keeps the
cell at column `size//2` (the middle column) unchanged for every row, and clears
every other cell to background (0).

Memory floor-break (single uint8 label map + final Equal):
  v[r,c] = 16*ingrid + colour_index   (one 1x1 Conv; weights are params)
     off-grid      -> 0
     in-grid black -> 16
     in-grid col k -> 16 + k
  ingrid[r,c] = (v > 15).  The in-grid region is a square anchored top-left, so
  every in-grid column is full; the in-grid width W = sum_c colHas[c] where
  colHas[c] = max over rows of ingrid.  size is ODD => the middle column index is
  mid = (W-1)/2 (exact integer).  A column-index plane is compared to mid.

  L[r,c] = vu8 (16..25)   if in-grid AND col == mid   (passthrough colour)
           16             elif in-grid                 (cleared -> background 0)
           0              else (off-grid)              (matches nothing)
  output = Equal(L, arange(16..25))  (BOOL, opset 11)

  Only one ~900B uint8 plane (L) is materialised; everything else is 1-D / small.
  All values are small integers, exact in float32 / uint8.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    U8 = TensorProto.UINT8
    B = TensorProto.BOOL

    init("fifteen", np.array(15.5, np.float32), np.float32)

    # v[r,c] = 16*ingrid + colour_index via one 1x1 Conv.
    cw = (np.arange(10, dtype=np.float32) + 16.0).reshape(1, 10, 1, 1)
    cw[0, 0, 0, 0] = 16.0  # ch0 (black/background): only the +16 in-grid term
    init("cw", cw, np.float32)
    n("Conv", ["input", "cw"], "vf")                   # [1,1,30,30] fp32

    n("Greater", ["vf", "fifteen"], "ingrid_b")        # v > 15  (in-grid)
    n("Cast", ["vf"], "vu8", to=U8)                    # [1,1,30,30] uint8

    # colHas[c] = 1 iff column c is in-grid.  Derive from a 1-D reduction over vf
    # (avoids materialising a 30x30 ingrid float plane): colmax[c] = max over rows
    # of v; column is in-grid iff colmax > 15.
    n("ReduceMax", ["vf"], "colmax", axes=[2], keepdims=1)        # [1,1,1,30]
    n("Greater", ["colmax", "fifteen"], "colHas_b")              # [1,1,1,30] bool
    n("Cast", ["colHas_b"], "colHas", to=F)                      # [1,1,1,30] {0,1}
    # in-grid width W = sum_c colHas[c]
    n("ReduceSum", ["colHas"], "W", axes=[3], keepdims=1)         # [1,1,1,1]
    # size is odd => mid column index = (W-1)/2
    init("one", np.array(1.0, np.float32), np.float32)
    init("two", np.array(2.0, np.float32), np.float32)
    n("Sub", ["W", "one"], "Wm1")
    n("Div", ["Wm1", "two"], "mid")                    # [1,1,1,1] integer-valued

    # column-index plane [1,1,1,30] = 0,1,2,...,29
    colidx = np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30)
    init("colidx", colidx, np.float32)
    n("Equal", ["colidx", "mid"], "colmid_b")          # [1,1,1,30] bool
    # keep cell iff in-grid AND col == mid
    n("And", ["colmid_b", "ingrid_b"], "keep_b")       # [1,1,30,30] broadcast

    # cleared in-grid cells -> 16 (colour 0 in the 16-shifted space => background)
    init("v16", np.array(16, np.uint8), np.uint8)
    n("Where", ["ingrid_b", "v16", "vu8"], "Lclear")   # in-grid->16 else off (0)
    # kept cells restore their colour (vu8); everything else stays Lclear
    n("Where", ["keep_b", "vu8", "Lclear"], "L")       # [1,1,30,30] uint8

    chan = (np.arange(10, dtype=np.uint8) + 16).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["L", "chan"], "output")                # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task329", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
