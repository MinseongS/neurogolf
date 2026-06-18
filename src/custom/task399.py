"""Task 399 (ARC-AGI ff28f65a) — count red 2x2 boxes -> fixed 3x3 blue pattern.

Rule (from the generator): the input is a size x size grid (size 3..7) holding
`num_boxes` non-overlapping red (2) 2x2 squares.  The OUTPUT is ALWAYS a 3x3
grid whose blue (1) pixels are turned on by a fixed schedule keyed only on the
COUNT n = num_boxes (1..5):
    n>=1 -> (0,0)   n>=2 -> (0,2)   n>=3 -> (1,1)   n>=4 -> (2,0)   n>=5 -> (2,2)
every other 3x3 cell is black (0); everything outside the 3x3 is unset.

Because the boxes never overlap, the red pixel count is exactly 4*n, so
    n = ReduceSum(red channel) / 4   (a scalar, integer-exact in fp32).
A cell turns blue iff its fixed schedule-rank `sched` (1..5, or 99 for the four
never-blue cells) satisfies sched <= n, i.e. NOT(sched > n).

Output: build the 3x3 colour-index label lab3 = Where(blue, 1, 0), Pad to
30x30 with the off-grid sentinel 10, then Equal(lab3, arange[1,10,1,1]) routes
the whole 10-channel one-hot into the FREE bool output.  All intermediates are
tiny scalars / 3x3 / a 30x30 int32 label plane.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
I32 = TensorProto.INT32
U8 = TensorProto.UINT8
B = TensorProto.BOOL


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- red count -> n scalar (NO full 30x30 plane) ----
    # Per-channel pixel counts [1,10,1,1] (40B); slice channel 2 = red count.
    n("ReduceSum", ["input"], "cnts", axes=[2, 3], keepdims=1)   # [1,10,1,1] fp32
    init("c2a", np.array([2], np.int64), np.int64)
    init("c2b", np.array([3], np.int64), np.int64)
    init("ax1", np.array([1], np.int64), np.int64)
    n("Slice", ["cnts", "c2a", "c2b", "ax1"], "cnt")             # [1,1,1,1]
    init("four", np.array([[[[4.0]]]], np.float32), np.float32)
    n("Div", ["cnt", "four"], "nf")                              # [1,1,1,1] = n

    # ---- blue = NOT( sched > n ) on the 3x3 schedule ----
    sched = np.array([[1, 99, 2],
                      [99, 3, 99],
                      [4, 99, 5]], np.float32).reshape(1, 1, 3, 3)
    init("sched", sched, np.float32)
    n("Greater", ["sched", "nf"], "gt")                          # [1,1,3,3] bool
    n("Not", ["gt"], "blue")                                     # [1,1,3,3] bool

    # ---- build the tiny [1,10,3,3] uint8 one-hot, then Pad it INTO the free
    #      output (no 30x30 carrier plane is ever materialised) ----
    init("one3", np.ones((1, 1, 3, 3), np.uint8), np.uint8)
    init("zero3", np.zeros((1, 1, 3, 3), np.uint8), np.uint8)
    n("Where", ["blue", "one3", "zero3"], "ch1")                 # [1,1,3,3] blue
    n("Where", ["blue", "zero3", "one3"], "ch0")                 # [1,1,3,3] black
    n("Concat", ["ch0", "ch1"], "oneh3", axis=1)                 # [1,2,3,3] u8

    # ---- Pad [1,2,3,3] -> [1,10,30,30] with 0 (8 trailing channels + spatial);
    #      this Pad IS the output, so no carrier plane is materialised ----
    pads = np.array([0, 0, 0, 0, 0, 8, 27, 27], np.int64)
    init("pads", pads, np.int64)
    init("zerov", np.array(0, np.uint8), np.uint8)
    n("Pad", ["oneh3", "pads", "zerov"], "output", mode="constant")  # [1,10,30,30]

    out_vi = helper.make_tensor_value_info("output", U8, [1, 10, 30, 30])
    in_vi = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task399", [in_vi], [out_vi], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
