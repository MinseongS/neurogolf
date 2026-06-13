"""Task 312 (c9f8e694): recolor every gray (channel 5) cell to the column-0
pattern color of its own row; non-gray cells pass through unchanged.

Rule (exact, from ARC-GEN generator): column 0 holds a per-row pattern color
(never gray). Gray boxes sit at columns >= 2. Output(r,c) = pattern(r) if
input(r,c) is gray else input(r,c). In one-hot terms this is a single Where:

    output = Where(gray_mask, col0_onehot_broadcast, input)

Channel 5 is automatically zeroed: where gray, col0[5]=0 (col 0 never gray);
where not gray, input[5]=0 (cell isn't gray). col0 [1,10,30,1] broadcasts over
columns; gray [1,1,30,30] broadcasts over channels.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ..builders import _model


def build(task):
    inits = []
    nodes = []

    def init(name, arr, dtype=np.int64):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    # gray mask = input channel 5, [1,1,30,30]
    init("g_st", np.array([5], np.int64))
    init("g_en", np.array([6], np.int64))
    init("c_ax", np.array([1], np.int64))
    n("Slice", ["input", "g_st", "g_en", "c_ax"], "gray_f")
    # condition must be bool (1 byte intermediate)
    n("Cast", ["gray_f"], "gray_b", to=int(onnx.TensorProto.BOOL))

    # col0 = input column 0 (the per-row pattern one-hot), [1,10,30,1]
    init("w_st", np.array([0], np.int64))
    init("w_en", np.array([1], np.int64))
    init("w_ax", np.array([3], np.int64))
    n("Slice", ["input", "w_st", "w_en", "w_ax"], "col0")

    # output = gray ? col0(broadcast over cols) : input
    n("Where", ["gray_b", "col0", "input"], "output")

    return _model(nodes, inits)
