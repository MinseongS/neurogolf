"""task292 (ARC-AGI ...) — recolor yellow cells by column.

Rule (generator, height=3, width 10..20): input has yellow(4) at row1 of every
column plus yellow at row0-or-2. Output keeps the SAME cells but recolors: pink(6)
where col%3==0, else yellow(4). Background/off-grid unchanged.

  => pure recolor of the yellow occupancy: L = ymask * (4 + 2*[c%3==0]). The in-grid
  column mask is FREE from row1 (all-yellow in-grid): ingrid_col = input[yellow, row1].
  Off-grid cols -> sentinel 99; pad rows 3..29 -> 99. Equal(L, arange) -> bool output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8

YEL = 4
PINK = 6
HACT = 3  # active rows


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    init("axHW", np.array([2, 3], np.int64), np.int64)
    # ymask = input[:, YEL:YEL+1, 0:3, :] -> [1,1,3,30]
    init("ym_s", np.array([YEL, 0, 0], np.int64), np.int64)
    init("ym_e", np.array([YEL + 1, HACT, 30], np.int64), np.int64)
    init("axCHW", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "ym_s", "ym_e", "axCHW"], "ymask")        # [1,1,3,30] f32

    # ingrid_col = input[:, YEL, 1:2, :] (row1 all-yellow in-grid) -> [1,1,1,30]
    init("ig_s", np.array([YEL, 1, 0], np.int64), np.int64)
    init("ig_e", np.array([YEL + 1, 2, 30], np.int64), np.int64)
    n("Slice", ["input", "ig_s", "ig_e", "axCHW"], "ingrid_col")   # [1,1,1,30] f32

    # color-per-col template: 4 + 2*[c%3==0]  -> [1,1,1,30]
    colval = np.full((1, 1, 1, 30), float(YEL), np.float32)
    colval[..., ::3] = float(PINK)
    init("colval", colval, np.float32)

    n("Mul", ["ymask", "colval"], "L")                             # [1,1,3,30] f32

    # off-grid cols -> +99: L_s = L + 99*(1 - ingrid_col)
    init("C99", np.array(99.0, np.float32), np.float32)
    init("ONE", np.array(1.0, np.float32), np.float32)
    n("Sub", ["ONE", "ingrid_col"], "offcol")                      # [1,1,1,30]
    n("Mul", ["offcol", "C99"], "offcol99")
    n("Add", ["L", "offcol99"], "Ls")                              # [1,1,3,30] f32

    n("Cast", ["Ls"], "Lu8", to=U8)
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - HACT, 0], np.int64), np.int64)
    init("SENT", np.array(99, np.uint8), np.uint8)
    n("Pad", ["Lu8", "pads", "SENT"], "L30", mode="constant")      # [1,1,30,30] u8

    init("ar", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L30", "ar"], "output")                            # [1,10,30,30] bool

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task292", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
