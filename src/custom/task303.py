"""task303 (ARC-AGI c1d99e64) — paint full-black row/col red, else passthrough.

Rule (from the generator): the grid is filled with one non-red colour plus black
(0) cells.  A set of "straightaway" rows and columns are forced entirely black.
The generator guarantees every NON-straightaway in-grid row and column contains
at least one coloured (non-black) cell.  Output: every cell whose row OR column
is a straightaway becomes red (colour 2); all other cells pass the input through
unchanged.  Equivalently, a cell is red iff its whole (in-grid) row is black OR
its whole (in-grid) column is black.

Memory floor-break (separable conditions + single uint8 label map + final Equal):
  colored[r,c]   = max over channels 1..9 of input  (1 if cell is non-black)
  rowHasColor[r] = max over c of colored            (0 => row is all-black)
  colHasColor[c] = max over r of colored            (0 => col is all-black)
  ingrid[r,c]    = max over all channels (any channel set => cell is on-grid)
  colorIdx[r,c]  = sum_c c*input[c]                  (input colour 0..9, exact)

  L[r,c] = 10                          if off-grid               (matches nothing)
           2 (red)                     elif rowHasColor[r]==0 OR colHasColor[c]==0
           colorIdx[r,c]               else (passthrough)
  output = Equal(L, arange[1,10,1,1])  (BOOL, opset 11)

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
    H = TensorProto.FLOAT16
    U8 = TensorProto.UINT8
    B = TensorProto.BOOL

    init("fifteen", np.array(15.5, np.float32), np.float32)

    # ONE 1x1 Conv encodes everything we need per cell (weights are params, not
    # memory).  v[r,c] = 16*ingrid + colour_index, where ingrid = sum_c input[c]
    # and colour_index = sum_c c*input[c]:
    #   off-grid       -> v = 0
    #   in-grid black  -> v = 16  (colour index 0)
    #   in-grid col k  -> v = 16 + k
    # so ingrid = (v > 15), colour index = v - 16 (when in-grid), and a cell is
    # "coloured" iff v > 16.  All values are small integers, exact in float32.
    cw = (np.arange(10, dtype=np.float32) + 16.0).reshape(1, 10, 1, 1)
    cw[0, 0, 0, 0] = 16.0  # ch0 (black): only the +16 in-grid term, index 0
    init("cw", cw, np.float32)
    n("Conv", ["input", "cw"], "vf")                   # [1,1,30,30] fp32

    n("Greater", ["vf", "fifteen"], "ingrid_b")        # v > 15  (in-grid)

    # quantise v to uint8 ONCE; we keep the label map in the SHIFTED space
    # (in-grid colour k -> 16+k), so no subtraction is needed.  The final Equal
    # compares against arange(16..25); off-grid (v=0) matches nothing.
    n("Cast", ["vf"], "vu8", to=U8)                    # [1,1,30,30] uint8

    # Straightaway detection via 1-D reductions over vf directly (cheap, 120B
    # each).  A row's max v is: 16 if every in-grid cell is black (=straightaway),
    # >=17 if it has any colour, 0 if the row is entirely off-grid.  So a row is
    # a straightaway iff its max v == 16, i.e. 15 < max < 17.
    n("ReduceMax", ["vf"], "rowmax", axes=[3], keepdims=1)  # [1,1,30,1] fp32
    n("ReduceMax", ["vf"], "colmax", axes=[2], keepdims=1)  # [1,1,1,30] fp32
    init("sixteen", np.array(16.5, np.float32), np.float32)
    # straightaway iff 15 < max < 17  (== 16 exactly)
    n("Greater", ["rowmax", "fifteen"], "rgt15")
    n("Less", ["rowmax", "sixteen"], "rlt16")
    n("And", ["rgt15", "rlt16"], "rowred_b")           # [1,1,30,1] bool
    n("Greater", ["colmax", "fifteen"], "cgt15")
    n("Less", ["colmax", "sixteen"], "clt16")
    n("And", ["cgt15", "clt16"], "colred_b")           # [1,1,1,30] bool
    n("Or", ["rowred_b", "colred_b"], "red_b")         # [1,1,30,30] broadcast
    # only paint red on IN-GRID cells; off-grid cells keep vu8 (=0 sentinel).
    n("And", ["red_b", "ingrid_b"], "redin_b")         # [1,1,30,30]

    # --- assemble label map L[1,1,30,30] (in 16-shifted space) -------------
    # red (colour 2) -> 16+2 = 18.  Off-grid v=0 -> 0 (matches nothing in 16..25),
    # so it needs no explicit branch; in-grid passthrough = vu8 (16..25).
    init("v18", np.array(18, np.uint8), np.uint8)
    n("Where", ["redin_b", "v18", "vu8"], "L")         # red else passthrough

    chan = (np.arange(10, dtype=np.uint8) + 16).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["L", "chan"], "output")                # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task303", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
