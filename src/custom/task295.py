"""Task 295 (bbc9ae5d): growing staircase from a 1-row seed.

Rule: input is a 1 x w row (w = 2*randint(3,9) in {6..18}), first L cells of
color k (1-9), rest black.  Output is (w/2) x w; row r has its first L+r
cells in color k, the remainder of the row black (color 0).

Graph: per-channel counts give L (colored count), w (total), h = w/2, and the
color one-hot e_k = (masked counts > 0).  On a cropped 9x18 canvas (the max
output extent) build two bool masks: M1[r,c] = (c - r < L) & (r < h) (colored
region) and Mf[r,c] = (c < w) & (r < h) (full grid).  Final op is a 1x1 Conv
with runtime weights W[:,0] = e_k - e_0, W[:,1] = e_0 over the 2-channel mask
stack, using pads to emit the [1,10,30,30] output directly (free tensor):
out[ch] = (e_k - e_0)[ch]*M1 + e_0[ch]*Mf  =  e_k*M1 + e_0*(Mf - M1).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ..builders import _model

H, W = 9, 18  # max output extent (w <= 18, h = w/2 <= 9)


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

    # --- scalars from per-channel counts ---
    n("ReduceSum", ["input"], "cnt", axes=[2, 3], keepdims=1)      # [1,10,1,1]
    c01 = np.ones((1, 10, 1, 1), np.float32)
    c01[0, 0] = 0.0
    init("c01", c01)
    n("Mul", ["cnt", "c01"], "cntm")                               # colored only
    n("ReduceSum", ["cntm"], "L", axes=[1], keepdims=1)            # [1,1,1,1]
    n("ReduceSum", ["cnt"], "w", axes=[1], keepdims=1)             # [1,1,1,1]
    init("half", np.full((1, 1, 1, 1), 0.5, np.float32))
    n("Mul", ["w", "half"], "h")                                   # h = w/2

    # --- runtime conv weights [10,2,1,1] ---
    init("zero", np.zeros((1, 1, 1, 1), np.float32))
    n("Greater", ["cntm", "zero"], "ekb")                          # [1,10,1,1] b
    n("Cast", ["ekb"], "ekf", to=onnx.TensorProto.FLOAT)
    e0 = np.zeros((1, 10, 1, 1), np.float32)
    e0[0, 0] = 1.0
    init("e0", e0)
    n("Sub", ["ekf", "e0"], "dk")                                  # e_k - e_0
    init("wsh", np.array([10, 1, 1, 1], np.int64), np.int64)
    n("Reshape", ["dk", "wsh"], "dkr")                             # [10,1,1,1]
    init("e0w", e0.reshape(10, 1, 1, 1))
    n("Concat", ["dkr", "e0w"], "Wt", axis=1)                      # [10,2,1,1]

    # --- masks on cropped 9x18 canvas ---
    D = (np.arange(W, dtype=np.float32)[None, :]
         - np.arange(H, dtype=np.float32)[:, None]).reshape(1, 1, H, W)
    init("D", D)                                                   # c - r
    init("I", np.arange(H, dtype=np.float32).reshape(1, 1, H, 1))
    init("J", np.arange(W, dtype=np.float32).reshape(1, 1, 1, W))
    n("Less", ["D", "L"], "A")                                     # c - r < L
    n("Less", ["I", "h"], "B")                                     # r < h
    n("And", ["A", "B"], "M1")                                     # colored
    n("Less", ["J", "w"], "Cj")                                    # c < w
    n("And", ["Cj", "B"], "Mf")                                    # full grid
    n("Concat", ["M1", "Mf"], "Xb", axis=1)                        # [1,2,9,18] b
    n("Cast", ["Xb"], "Xf", to=onnx.TensorProto.FLOAT)

    # --- fused write: 1x1 Conv padded out to the 30x30 canvas ---
    n("Conv", ["Xf", "Wt"], "output",
      pads=[0, 0, 30 - H, 30 - W])                                 # [1,10,30,30]

    return _model(nodes, inits)
