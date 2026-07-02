"""Task 055 — line-delimited 3x3 grid fill via category-augmented LUT.

Rows and columns are classified into five categories: band 0..2, separator
line, and off-grid.  A single 5x5 uint8 LUT then emits the block colour, cyan
line colour, or off-grid sentinel directly, avoiding the old full-canvas line
and in-grid overlay planes.
"""

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ..harness import IR_VERSION


LUT = np.zeros((5, 5), dtype=np.uint8)
LUT[:3, :3] = np.array(
    [
        [0, 2, 0],
        [4, 6, 3],
        [0, 1, 0],
    ],
    dtype=np.uint8,
)
LUT[3, :] = 8
LUT[:, 3] = 8
LUT[4, :] = 10
LUT[:, 4] = 10
LUT[4, 3] = 10
LUT[3, 4] = 10


def build(task):
    inits = []
    nodes = []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dtype), name))

    def node(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))

    init("ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    init("hs", np.array([0, 8, 0, 0], np.int64), np.int64)
    init("he", np.array([1, 9, 30, 1], np.int64), np.int64)
    init("ve", np.array([1, 9, 1, 30], np.int64), np.int64)
    node("Slice", ["input", "hs", "he", "ax"], "hline")
    node("Slice", ["input", "hs", "ve", "ax"], "vline")

    init("ax2", np.array(2, np.int64), np.int64)
    init("ax3", np.array(3, np.int64), np.int64)
    node("CumSum", ["hline", "ax2"], "rowband", exclusive=1)
    node("CumSum", ["vline", "ax3"], "colband", exclusive=1)

    init("kr", np.arange(3, dtype=np.float32).reshape(1, 1, 1, 3), np.float32)
    init("kc", np.arange(3, dtype=np.float32).reshape(1, 1, 3, 1), np.float32)
    node("Equal", ["rowband", "kr"], "Reqb")
    node("Equal", ["colband", "kc"], "Ceqb")

    init("half", np.array(0.5, np.float32), np.float32)
    node("Greater", ["hline", "half"], "hb")
    node("Greater", ["vline", "half"], "vb")
    node("ReduceSum", ["input"], "rowsum", axes=[1, 3], keepdims=1)
    node("ReduceSum", ["input"], "colsum", axes=[1, 2], keepdims=1)
    node("Greater", ["rowsum", "half"], "rin")
    node("Greater", ["colsum", "half"], "cin")
    node("Not", ["rin"], "roff")
    node("Not", ["cin"], "coff")
    node("Not", ["hb"], "not_hb")
    node("Not", ["vb"], "not_vb")
    node("And", ["rin", "not_hb"], "rband_ok")
    node("And", ["cin", "not_vb"], "cband_ok")
    node("And", ["Reqb", "rband_ok"], "Rbands_b")
    node("And", ["Ceqb", "cband_ok"], "Cbands_b")
    node("Concat", ["Rbands_b", "hb", "roff"], "Rcat_b", axis=3)
    node("Concat", ["Cbands_b", "vb", "coff"], "Ccat_b", axis=2)
    node("Cast", ["Rcat_b"], "Rcat", to=TensorProto.UINT8)
    node("Cast", ["Ccat_b"], "Ccat", to=TensorProto.UINT8)

    init("lut5", LUT.reshape(1, 1, 5, 5), np.uint8)
    init("qscale", np.array(1.0, np.float32), np.float32)
    init("qzero", np.array(0, np.uint8), np.uint8)
    q = ["qscale", "qzero"]
    node("QLinearMatMul", ["Rcat", *q, "lut5", *q, *q], "RL")
    node("QLinearMatMul", ["RL", *q, "Ccat", *q, *q], "L")
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    node("Equal", ["L", "chan"], "output")

    graph = helper.make_graph(
        nodes,
        "task055_category_lut",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        inits,
    )
    return helper.make_model(
        graph,
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)],
    )
