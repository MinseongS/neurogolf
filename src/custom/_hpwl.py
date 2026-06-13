"""Shared builder for the common.hpwl tasks (246: a2fd1cf0, 335: d4a91cb9).

Dot of color c0 at (r0,c0), dot of color c1 at (r1,c1); draw an L-path of
color cp between them. Same outer-product construction as task246 (see that
module's docstring), parameterized by the three colors.
"""

import numpy as np
from onnx import helper, numpy_helper

from ..builders import _model


def build_hpwl(col0, col1, colp):
    inits, nodes = [], []

    def init(name, arr, dtype=np.float32):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    # row stream
    Wr = np.zeros((1, 10, 1, 18), np.float32)
    Wr[0, col0, 0, :] = 1.0
    Wr[0, col1, 0, :] = 2.0
    init("Wr", Wr)
    n("Conv", ["input", "Wr"], "vr", strides=[1, 18])
    n("Clip", ["vr"], "wr", min=0.0, max=1.0)
    n("Sub", ["vr", "wr"], "gr")
    n("MaxPool", ["wr"], "pr", kernel_shape=[30, 1], pads=[29, 0, 0, 0])
    n("MaxPool", ["wr"], "qr", kernel_shape=[30, 1], pads=[0, 0, 29, 0])
    n("Mul", ["pr", "qr"], "cr")
    n("ReduceMax", ["input"], "rowin", axes=[1, 3], keepdims=1)
    n("Concat", ["rowin", "vr", "gr", "cr"], "VrT", axis=3)  # [1,1,30,4]

    # col stream
    Wc = np.zeros((1, 10, 18, 1), np.float32)
    Wc[0, col0, :, 0] = 1.0
    Wc[0, col1, :, 0] = 2.0
    init("Wc", Wc)
    n("Conv", ["input", "Wc"], "vc", strides=[18, 1])
    n("Clip", ["vc"], "wc", min=0.0, max=1.0)
    n("Sub", ["vc", "wc"], "gc")
    n("MaxPool", ["wc"], "pc", kernel_shape=[1, 30], pads=[0, 29, 0, 0])
    n("MaxPool", ["wc"], "qc", kernel_shape=[1, 30], pads=[0, 0, 0, 29])
    n("Mul", ["pc", "qc"], "cc")
    n("ReduceMax", ["input"], "colin", axes=[1, 2], keepdims=1)
    n("Concat", ["colin", "vc", "gc", "cc"], "VcT", axis=2)  # [1,1,4,30]

    # per-channel outer-product coefficients
    My = np.zeros((1, 10, 4, 4), np.float32)
    My[0, 0, 0, 0] = 1
    My[0, 0, 1, 3] = -1
    My[0, 0, 2, 3] = 2
    My[0, 0, 3, 2] = -1
    My[0, col0, 1, 1] += 1
    My[0, col0, 1, 2] += -2
    My[0, col0, 2, 1] += -2
    My[0, col0, 2, 2] += 4
    My[0, col1, 2, 2] += 1
    My[0, colp, 1, 3] += 1
    My[0, colp, 1, 1] += -1
    My[0, colp, 1, 2] += 1
    My[0, colp, 2, 3] += -2
    My[0, colp, 2, 1] += 2
    My[0, colp, 2, 2] += -3
    My[0, colp, 3, 2] += 1
    init("My", My)

    n("MatMul", ["My", "VcT"], "R10")
    n("MatMul", ["VrT", "R10"], "output")
    return _model(nodes, inits)
