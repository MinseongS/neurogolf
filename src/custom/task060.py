"""task060 (ARC-AGI 29c11459) — mirror-fill marked rows.

Rule (from generator, grid ALWAYS 11x5): each "marker" row r carries a single
colored cell at col 0 (left_color) and col 10 (right_color). The output fills that
row as: cols 0..4 = left_color, col 5 = gray(5), cols 6..10 = right_color. Non-marker
rows and off-grid stay background.

  => fully separable per-row broadcast (Tier A-ish): read left/right color index per
  row from cols 0 and 10 (channel-contract), build a small [1,1,5,11] color-index plane
  L = lv*leftmask + rv*rightmask + gray*marker*midmask, pad to 30x30 with sentinel 99
  (off-grid -> all-False), Equal(L, arange) -> bool output [1,10,30,30].
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I64 = TensorProto.INT64

W = 11          # grid width (fixed)
H = 5           # grid height (fixed)
MID = W // 2    # 5
GRAY = 5


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    init("kvec", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    init("axHW", np.array([2, 3], np.int64), np.int64)

    # left color per row: input[:, :, 0:H, 0:1] -> contract channel -> lv [1,1,H,1]
    init("c0_s", np.array([0, 0], np.int64), np.int64)
    init("c0_e", np.array([H, 1], np.int64), np.int64)
    init("c10_s", np.array([0, W - 1], np.int64), np.int64)
    init("c10_e", np.array([H, W], np.int64), np.int64)

    n("Slice", ["input", "c0_s", "c0_e", "axHW"], "col0")       # [1,10,H,1]
    n("Mul", ["col0", "kvec"], "col0k")
    n("ReduceSum", ["col0k"], "lv", axes=[1], keepdims=1)        # [1,1,H,1]
    n("Slice", ["input", "c10_s", "c10_e", "axHW"], "col10")    # [1,10,H,1]
    n("Mul", ["col10", "kvec"], "col10k")
    n("ReduceSum", ["col10k"], "rv", axes=[1], keepdims=1)       # [1,1,H,1]

    # marker = (lv > 0) as float
    init("ZERO", np.array(0.0, np.float32), np.float32)
    n("Greater", ["lv", "ZERO"], "mk_b")                        # [1,1,H,1] bool
    n("Cast", ["mk_b"], "mk", to=F32)

    # column masks [1,1,1,W]
    lm = np.zeros((1, 1, 1, W), np.float32); lm[..., 0:MID] = 1.0
    rm = np.zeros((1, 1, 1, W), np.float32); rm[..., MID + 1:W] = 1.0
    mm = np.zeros((1, 1, 1, W), np.float32); mm[..., MID] = 1.0
    init("lm", lm, np.float32)
    init("rm", rm, np.float32)
    init("mm", mm, np.float32)
    init("GRAYf", np.array(float(GRAY), np.float32), np.float32)

    n("Mul", ["lv", "lm"], "tL")        # [1,1,H,W]
    n("Mul", ["rv", "rm"], "tR")        # [1,1,H,W]
    n("Mul", ["mk", "mm"], "tMm")       # [1,1,H,W]
    n("Mul", ["tMm", "GRAYf"], "tM")
    n("Add", ["tL", "tR"], "tLR")
    n("Add", ["tLR", "tM"], "Lf")       # [1,1,H,W] f32, exact small ints

    # cast to uint8, pad to 30x30 with sentinel 99 (off-grid -> all-False)
    n("Cast", ["Lf"], "Lu8", to=U8)
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - H, 30 - W], np.int64), np.int64)
    init("SENT", np.array(99, np.uint8), np.uint8)
    n("Pad", ["Lu8", "pads", "SENT"], "L30", mode="constant")   # [1,1,30,30] u8

    init("ar", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L30", "ar"], "output")                         # [1,10,30,30] bool

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task060", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
