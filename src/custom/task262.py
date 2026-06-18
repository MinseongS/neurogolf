"""task262 (ARC-AGI a85d4709) — "fill each row with the colour selected by its gray pixel".

Rule (from the generator):
  A 3x3 grid.  Background is 0.  Each row r has exactly one gray(5) pixel at column
  cols[r] in {0,1,2}.  In the OUTPUT every cell of row r (cols 0..2) is filled with
  colors[cols[r]] where colors=(2,4,3):  col0->2, col1->4, col2->3.
  The 3x3 grid sits at the canvas top-left; everything else stays background(0).

IO note: the harness one-hot leaves OFF-grid cells all-zero (no channel set), and
in-grid colours are always {2,3,4} so channel 0 is never set anywhere.

Encoding (Tier S-ish — NO full-canvas plane materialised):
  gray3 = input[:,5:6,:3,:3]                                    [1,1,3,3] f32
  rowval3 = MatMul(gray3, wcol)  wcol=[2,4,3]^T                 [1,1,3,1] -> per-row colour idx
  rowval_full = Pad(rowval3, 0) to [1,1,30,1]                   (0 for r>=3)
  rowEq = Equal(rowval_full, cvals[1,10,1,1])                   [1,10,30,1] bool
          cvals = [-1,1,2,...,9]  (channel 0 sentinel -1 => ch0 never fires)
  output = And(rowEq[1,10,30,1], colmask[1,1,1,30])             [1,10,30,30] BOOL
           colmask = (c<3).  In-grid -> colour channel; r>=3 or c>=3 -> all false.
  All broadcasts land in the FREE bool output; no [1,1,30,30] intermediate exists.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- gray slice on the 3x3 active region (channel 5) -------------------
    init("g_s", np.array([5, 0, 0], np.int64), np.int64)
    init("g_e", np.array([6, 3, 3], np.int64), np.int64)
    init("g_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "g_s", "g_e", "g_ax"], "gray3")   # [1,1,3,3] f32

    # ---- per-row colour index: rowval3[r] = sum_c gray3[r,c]*wcol[c] -------
    # wcol = [2,4,3]^T  (col0->2, col1->4, col2->3). exactly one gray per row.
    init("wcol", np.array([[[[2.0], [4.0], [3.0]]]], np.float32), np.float32)  # [1,1,3,1]
    n("MatMul", ["gray3", "wcol"], "rowval3")              # [1,1,3,1] f32

    # ---- pad rows 3..29 with 0 (no gray there -> background) ---------------
    init("pads", np.array([0, 0, 0, 0, 0, 0, 27, 0], np.int64), np.int64)
    init("ZEROF", np.array(0.0, np.float32), np.float32)
    n("Pad", ["rowval3", "pads", "ZEROF"], "rowval_full", mode="constant")  # [1,1,30,1]

    # ---- per-row, per-channel equality (colour one-hot over rows) ----------
    # channel 0 uses sentinel -1 so it NEVER matches (off-grid + bg stay all-false).
    cvals = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    cvals[0, 0, 0, 0] = -1.0
    init("cvals", cvals, np.float32)
    n("Equal", ["rowval_full", "cvals"], "rowEq")          # [1,10,30,1] bool

    # ---- route into the FREE output via column gate ------------------------
    # colmask = (c < 3).  in-grid coloured cell -> its channel; else all-false.
    colmask = np.zeros((1, 1, 1, 30), dtype=bool)
    colmask[0, 0, 0, :3] = True
    init("colmask", colmask, np.bool_)                     # [1,1,1,30]
    n("And", ["rowEq", "colmask"], "output")               # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task262", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
