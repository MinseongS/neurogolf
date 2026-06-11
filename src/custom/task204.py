"""Task 204: hollow squares of color 1; fill interior with 7 (odd side) or 2 (even).

Method: detect horizontal-wall interior cells Hm (1 with left&right 1) via
Conv+Relu; vertical ray-casting parity (strict prefix count of Hm above, odd
=> enclosed) via triangular MatMul + Floor; interior size parity from nearest
H-wall above/below row indices via index-weighted MaxPool cummax; final 1x1
Conv combines ch0/ch1 pass-through with parity masks (sign-correct output).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ..builders import _model


def build(task):
    inits = []
    nodes = []

    def init(name, arr, dtype=np.float32):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    # 1. c1 = Conv(input): picks channel 1, 1x3 ones kernel, bias -2
    W1 = np.zeros((1, 10, 1, 3), np.float32)
    W1[0, 1, 0, :] = 1.0
    init("W1", W1)
    init("b1", np.array([-2.0], np.float32))
    n("Conv", ["input", "W1", "b1"], "c1", pads=[0, 1, 0, 1])
    # 2. Hm = Relu(c1): 1 at horizontal-wall interior cells
    n("Relu", ["c1"], "Hm")

    # 3. cu = 0.5 * strict prefix sum down columns (triangular MatMul)
    Tl = np.tril(np.full((30, 30), 0.5, np.float32), -1)
    init("Tl", Tl)
    n("MatMul", ["Tl", "Hm"], "cu")
    # 4-5. pu = frac(cu): 0.5 iff odd count of H-walls strictly above
    n("Floor", ["cu"], "fl")
    n("Sub", ["cu", "fl"], "pu")

    # 6-9. nearest H-wall above/below via index-weighted cummax (MaxPool)
    j = np.arange(30, dtype=np.float32)
    init("idx1", ((j + 1) / 2).reshape(30, 1))
    init("idx2", ((30 - j) / 2).reshape(30, 1))
    n("Mul", ["Hm", "idx1"], "v1")
    n("MaxPool", ["v1"], "Pup", kernel_shape=[30, 1], pads=[29, 0, 0, 0])
    n("Mul", ["Hm", "idx2"], "v2")
    n("MaxPool", ["v2"], "Pdn", kernel_shape=[30, 1], pads=[0, 0, 29, 0])
    # 10-12. ps = frac(Pup + Pdn): 0.5 iff interior height odd
    n("Add", ["Pup", "Pdn"], "sf")
    n("Floor", ["sf"], "fl2")
    n("Sub", ["sf", "fl2"], "ps")

    # 13. ch01 = input channels 0..1
    init("sl_st", np.array([0], np.int64), np.int64)
    init("sl_en", np.array([2], np.int64), np.int64)
    init("sl_ax", np.array([1], np.int64), np.int64)
    n("Slice", ["input", "sl_st", "sl_en", "sl_ax"], "ch01")

    # 14. cat = [ch0, ch1, pu, ps]
    n("Concat", ["ch01", "pu", "ps"], "cat", axis=1)

    # 15. final 1x1 Conv -> output (sign-correct one-hot)
    Wf = np.zeros((10, 4, 1, 1), np.float32)
    bf = np.zeros(10, np.float32)
    Wf[0] = np.array([1, 0, -2, 0]).reshape(4, 1, 1)   # ch0 - 2*pu
    Wf[1] = np.array([0, 1, 0, 0]).reshape(4, 1, 1)    # ch1
    Wf[2] = np.array([8, 0, 2, -2]).reshape(4, 1, 1)   # even interior -> 2
    bf[2] = -8.5
    Wf[7] = np.array([8, 0, 2, 2]).reshape(4, 1, 1)    # odd interior -> 7
    bf[7] = -9.5
    init("Wf", Wf)
    init("bf", bf)
    n("Conv", ["cat", "Wf", "bf"], "output")

    return _model(nodes, inits)
