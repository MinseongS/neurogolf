"""Task 246 (ARC a2fd1cf0): red dot (2) at (r0,c0), green dot (3) at (r1,c1);
draw cyan (8) L-path: along row r0 from c0 (excl) to c1 (excl), then along
col c1 from r0 (incl) to r1 (excl).

Method: every output channel is a sum of <=4 outer products of row-basis and
col-basis vectors. Row basis [rowin, v=e_r0+2*e_r1, e_r1, closed_r] (and the
column analogue) is built from 120-byte [1,1,30,1] vectors: Conv 1x30 reduction
extracts v, Clip+Sub split the two dots, MaxPool-cummax from both ends gives
the closed interval [min,max]. A constant [1,10,4,4] coefficient tensor turns
the col basis into per-channel factors (MatMul), and a final broadcast MatMul
[1,1,30,4]@[1,10,4,30] assembles the whole canvas directly into `output`
(free). All values are small integers -> exact in float32.
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

    # --- row stream: [1,1,30,1] vectors -------------------------------
    # vr[r] = 1 at red row r0, 2 at green row r1 (one dot of each color).
    # Generator places dots at coords in range(1, dim-2), dim <= 20, so all
    # dot coords are <= 17: an 18-wide kernel (stride 18 -> single window
    # covering cols 0..17) sees every dot. 180 params instead of 300.
    Wr = np.zeros((1, 10, 1, 18), np.float32)
    Wr[0, 2, 0, :] = 1.0
    Wr[0, 3, 0, :] = 2.0
    init("Wr", Wr)
    n("Conv", ["input", "Wr"], "vr", strides=[1, 18])
    n("Clip", ["vr"], "wr", min=0.0, max=1.0)        # e_r0 + e_r1
    n("Sub", ["vr", "wr"], "gr")                     # e_r1
    n("MaxPool", ["wr"], "pr", kernel_shape=[30, 1], pads=[29, 0, 0, 0])
    n("MaxPool", ["wr"], "qr", kernel_shape=[30, 1], pads=[0, 0, 29, 0])
    n("Mul", ["pr", "qr"], "cr")                     # [min(r0,r1) <= r <= max]
    n("ReduceMax", ["input"], "rowin", axes=[1, 3], keepdims=1)
    n("Concat", ["rowin", "vr", "gr", "cr"], "VrT", axis=3)  # [1,1,30,4]

    # --- col stream: [1,1,1,30] vectors -------------------------------
    Wc = np.zeros((1, 10, 18, 1), np.float32)
    Wc[0, 2, :, 0] = 1.0
    Wc[0, 3, :, 0] = 2.0
    init("Wc", Wc)
    n("Conv", ["input", "Wc"], "vc", strides=[18, 1])
    n("Clip", ["vc"], "wc", min=0.0, max=1.0)        # f_c0 + f_c1
    n("Sub", ["vc", "wc"], "gc")                     # f_c1
    n("MaxPool", ["wc"], "pc", kernel_shape=[1, 30], pads=[0, 29, 0, 0])
    n("MaxPool", ["wc"], "qc", kernel_shape=[1, 30], pads=[0, 0, 0, 29])
    n("Mul", ["pc", "qc"], "cc")                     # [min(c0,c1) <= c <= max]
    n("ReduceMax", ["input"], "colin", axes=[1, 2], keepdims=1)
    n("Concat", ["colin", "vc", "gc", "cc"], "VcT", axis=2)  # [1,1,4,30]

    # --- per-channel outer-product coefficients ------------------------
    # row basis i: 0=rowin, 1=v, 2=g(e_r1), 3=c(closed_r);  e_r0 = v - 2g
    # col basis j: 0=colin, 1=u, 2=h(f_c1), 3=d(closed_c);  f_c0 = u - 2h
    My = np.zeros((1, 10, 4, 4), np.float32)
    # ch0 = rowin*colin - e_r0*d - c*h
    My[0, 0, 0, 0] = 1
    My[0, 0, 1, 3] = -1
    My[0, 0, 2, 3] = 2
    My[0, 0, 3, 2] = -1
    # ch2 = e_r0 * f_c0 = (v-2g)*(u-2h)
    My[0, 2, 1, 1] = 1
    My[0, 2, 1, 2] = -2
    My[0, 2, 2, 1] = -2
    My[0, 2, 2, 2] = 4
    # ch3 = g * h
    My[0, 3, 2, 2] = 1
    # ch8 = (v-2g)*(d-u+h) + (c-g)*h
    My[0, 8, 1, 3] = 1
    My[0, 8, 1, 1] = -1
    My[0, 8, 1, 2] = 1
    My[0, 8, 2, 3] = -2
    My[0, 8, 2, 1] = 2
    My[0, 8, 2, 2] = -3
    My[0, 8, 3, 2] = 1
    init("My", My)

    n("MatMul", ["My", "VcT"], "R10")                # [1,10,4,30]
    n("MatMul", ["VrT", "R10"], "output")            # [1,10,30,30] free

    return _model(nodes, inits)
