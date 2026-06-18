"""Task 103 (ARC-AGI 44f52bb0) — horizontal-symmetry classifier of a 3x3 grid.

Rule (from the generator): the input is a `size x size` (size==3) grid holding
red (2) pixels.  The OUTPUT is a 1x1 grid whose single cell is

    blue   (1)  if the red pattern is left<->right symmetric, else
    orange (7).

Left/right symmetry of a size==3 grid only depends on column 0 vs column 2 of
the red channel (column 1 maps to itself).  So:

    asymm = ReduceSum( |red[:, :, :, 0] - red[:, :, :, 2]| ) > 0

is a scalar boolean.  The output colour channel is then

    idx = 1 + 6 * asymm          # 1 (blue) when symmetric, 7 (orange) else

and the whole 10-channel one-hot is routed into the FREE output by
Equal(idx, arange8) -> Cast uint8 -> Pad to [1,10,30,30].  No 30x30 plane is
ever materialised; every intermediate is a tiny scalar / 3x1 vector / 8-elem
one-hot.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
I64 = TensorProto.INT64
U8 = TensorProto.UINT8


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- red column 0 and column 2 (channel 2), rows 0..2 ----
    init("axes", np.array([1, 2, 3], np.int64), np.int64)
    init("c0_s", np.array([2, 0, 0], np.int64), np.int64)
    init("c0_e", np.array([3, 3, 1], np.int64), np.int64)
    init("c2_s", np.array([2, 0, 2], np.int64), np.int64)
    init("c2_e", np.array([3, 3, 3], np.int64), np.int64)
    n("Slice", ["input", "c0_s", "c0_e", "axes"], "col0")   # [1,1,3,1]
    n("Slice", ["input", "c2_s", "c2_e", "axes"], "col2")   # [1,1,3,1]

    # ---- asymmetry scalar ----
    n("Sub", ["col0", "col2"], "diff")                      # [1,1,3,1]
    n("Abs", ["diff"], "ad")
    n("ReduceSum", ["ad"], "asum", axes=[1, 2, 3], keepdims=1)  # [1,1,1,1]
    init("zero", np.array([[[[0.0]]]], np.float32), np.float32)
    n("Greater", ["asum", "zero"], "asym")                  # [1,1,1,1] bool
    n("Cast", ["asym"], "asymf", to=F32)                    # 0.0 / 1.0

    # ---- channel index = 1 + 6*asymm ----
    init("six", np.array([[[[6.0]]]], np.float32), np.float32)
    init("one", np.array([[[[1.0]]]], np.float32), np.float32)
    n("Mul", ["asymf", "six"], "m6")
    n("Add", ["m6", "one"], "idx")                          # [1,1,1,1] = 1 or 7

    # ---- one-hot over 8 channels, then Pad INTO the free output ----
    init("arange8", np.arange(8, dtype=np.float32).reshape(1, 8, 1, 1), np.float32)
    n("Equal", ["idx", "arange8"], "oneh")                  # [1,8,1,1] bool
    n("Cast", ["oneh"], "oneh8", to=U8)                     # [1,8,1,1] uint8
    pads = np.array([0, 0, 0, 0, 0, 2, 29, 29], np.int64)
    init("pads", pads, np.int64)
    init("zerov", np.array(0, np.uint8), np.uint8)
    n("Pad", ["oneh8", "pads", "zerov"], "output", mode="constant")  # [1,10,30,30]

    out_vi = helper.make_tensor_value_info("output", U8, [1, 10, 30, 30])
    in_vi = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task103", [in_vi], [out_vi], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
