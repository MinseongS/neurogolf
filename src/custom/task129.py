"""Task 129 (ARC-AGI 5582e5ca) — solid-fill with the most-frequent colour.

Rule (from the generator): the input is a 3x3 grid whose 9 cells are filled
with 6 sampled colours via a fixed multiplicity schedule:
    colors[0] -> 3 cells (rows[0..2]), colors[1] -> 2 cells, colors[2..5] -> 1 cell each
    (colors[5] may equal colors[4], so colours[4]/[5] reach at most 2 cells).
colors[0] is sampled DISTINCT from colours[1..4], so it is the UNIQUE colour
appearing 3 times; every other colour appears <= 2 times.  The OUTPUT is a
solid 3x3 grid filled entirely with colors[0] (the mode colour); off-grid
(the rest of the 30x30 canvas) is all-zero in every channel.

Because the 3x3 grid is placed at the top-left and `convert_to_numpy` sets a
one-hot ONLY for the 9 real cells (off-grid stays all-zero, NOT channel-0=1),
per-channel pixel counts come straight from ReduceSum over spatial axes:
    counts = ReduceSum(input, axes=[2,3]) -> [1,10,1,1] fp32 (40B)
The mode channel is the unique one with count == 3, isolated by ONE threshold
    modehot = Greater(counts, 2.5) -> [1,10,1,1] bool
(no ReduceMax/ArgMax needed — exactly-3 is guaranteed by the generator).

Output: Where(modehot, 1, 0) with the constant operands shaped [1,1,3,3]
broadcasts the [1,10,1,1] mode selector across the 3x3 active block in ONE op
-> [1,10,3,3] uint8, then a single Pad zero-fills to the FREE [1,10,30,30]
output.  No [1,10,30,30] intermediate is ever materialised.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
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

    # per-channel pixel counts over the 3x3 grid (off-grid is all-zero)
    n("ReduceSum", ["input"], "counts", axes=[2, 3], keepdims=1)  # [1,10,1,1] fp32

    # mode channel = the unique colour with count == 3 (others <= 2)
    init("thr", np.array([[[[2.5]]]], np.float32), np.float32)
    n("Greater", ["counts", "thr"], "modehot")                    # [1,10,1,1] bool

    # broadcast the [1,10,1,1] selector across the 3x3 active block (uint8)
    init("one3", np.ones((1, 1, 3, 3), np.uint8), np.uint8)
    init("zero3", np.zeros((1, 1, 3, 3), np.uint8), np.uint8)
    n("Where", ["modehot", "one3", "zero3"], "block")             # [1,10,3,3] u8

    # Pad the [1,10,3,3] one-hot block INTO the free [1,10,30,30] output
    init("pads", np.array([0, 0, 0, 0, 0, 0, 27, 27], np.int64), np.int64)
    init("zerov", np.array(0, np.uint8), np.uint8)
    n("Pad", ["block", "pads", "zerov"], "output", mode="constant")  # [1,10,30,30]

    out_vi = helper.make_tensor_value_info("output", U8, [1, 10, 30, 30])
    in_vi = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task129", [in_vi], [out_vi], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
