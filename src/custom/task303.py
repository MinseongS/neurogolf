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
    U8 = TensorProto.UINT8
    B = TensorProto.BOOL

    init("zerof", np.array(0.0, np.float32), np.float32)

    # --- colour index per cell:  sum_c c*input[c]  (ch0 weight 0) -----------
    cw = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("cw", cw, np.float32)
    n("Mul", ["input", "cw"], "weighted")              # [1,10,30,30]
    n("ReduceSum", ["weighted"], "coloridxf", axes=[1], keepdims=1)  # [1,1,30,30]

    # --- coloured mask:  max over channels 1..9 -----------------------------
    # weight channels 1..9 = 1, channel 0 = 0, then ReduceMax over channel axis.
    mw = np.array([0] + [1] * 9, dtype=np.float32).reshape(1, 10, 1, 1)
    init("mw", mw, np.float32)
    n("Mul", ["input", "mw"], "colmasked")             # zero out ch0
    n("ReduceMax", ["colmasked"], "coloredf", axes=[1], keepdims=1)  # [1,1,30,30]

    # rowHasColor / colHasColor
    n("ReduceMax", ["coloredf"], "rowhasf", axes=[3], keepdims=1)  # [1,1,30,1]
    n("ReduceMax", ["coloredf"], "colhasf", axes=[2], keepdims=1)  # [1,1,1,30]
    # straightaway = no colour in that row/col
    n("Equal", ["rowhasf", "zerof"], "rowred_b")       # [1,1,30,1] bool
    n("Equal", ["colhasf", "zerof"], "colred_b")       # [1,1,1,30] bool
    n("Or", ["rowred_b", "colred_b"], "red_b")         # [1,1,30,30] broadcast

    # in-grid: any channel set
    n("ReduceMax", ["input"], "ingridf", axes=[1], keepdims=1)  # [1,1,30,30]
    n("Greater", ["ingridf", "zerof"], "ingrid_b")

    # --- assemble label map L[1,1,30,30] -----------------------------------
    n("Cast", ["coloridxf"], "coloridxU8", to=U8)      # passthrough colour
    init("v2", np.array(2, np.uint8), np.uint8)
    init("v10", np.array(10, np.uint8), np.uint8)
    # red where straightaway else passthrough colour
    n("Where", ["red_b", "v2", "coloridxU8"], "Lon")   # [1,1,30,30]
    # off-grid -> sentinel 10
    n("Where", ["ingrid_b", "Lon", "v10"], "L")        # [1,1,30,30]

    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["L", "chan"], "output")                # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task303", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
