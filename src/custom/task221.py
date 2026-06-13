"""Task 221 (91413438): tile the 3x3 grid n times into a (3v)x(3v) canvas.

Rule (from ARC-GEN): input is a 3x3 grid with n colored pixels (n=2..6),
v = 9 - n voids. Output is (3v)x(3v); the input grid is copied into the
first n blocks of the v-wide block grid in row-major order; every other
cell inside the canvas is color 0.

Graph: v = ReduceSum(input ch0); block masks on a [1,1,10,10] block grid:
  B1[br,bc] = bc < min(n - br*v, v * [br<v])   (pattern blocks, k < n)
  Rg[br,bc] = bc < v * [br<v]                  (in-region blocks)
  B2 = Xor(B1, Rg)                             (background-only blocks)
Stack [B1,B2] as a 2-channel [1,2,10,10] image and ConvTranspose with
stride 3 and a runtime kernel [2,10,3,3] = [input 3x3 slice; ones->ch0],
writing [1,10,30,30] directly into output (free).
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

    # --- scalars: v (void count), n = 9 - v ---
    n("ReduceSum", ["input"], "chsum", axes=[2, 3], keepdims=0)  # [1,10] f
    init("g0", np.array([0], np.int64), np.int64)
    n("Gather", ["chsum", "g0"], "v", axis=1)                    # [1,1] f
    init("c9", np.array(9.0, np.float32))
    n("Sub", ["c9", "v"], "nn")                                  # [1,1] f

    # --- block masks on the 10x10 block grid ---
    init("br", np.arange(10, dtype=np.float32).reshape(1, 1, 10, 1))
    init("bc", np.arange(10, dtype=np.float32).reshape(1, 1, 1, 10))
    n("Mul", ["br", "v"], "brv")                                 # [1,1,10,1]
    n("Sub", ["nn", "brv"], "t")                                 # n - br*v
    n("Less", ["br", "v"], "lbr")                                # [1,1,10,1] b
    n("Cast", ["lbr"], "lbrf", to=onnx.TensorProto.FLOAT)
    n("Mul", ["v", "lbrf"], "u")                                 # v*[br<v]
    n("Min", ["t", "u"], "m")                                    # [1,1,10,1]
    n("Less", ["bc", "m"], "B1")                                 # [1,1,10,10] b
    n("Less", ["bc", "u"], "Rg")                                 # [1,1,10,10] b
    n("Xor", ["B1", "Rg"], "B2")                                 # [1,1,10,10] b
    n("Concat", ["B1", "B2"], "Bb", axis=1)                      # [1,2,10,10] b
    n("Cast", ["Bb"], "Bf", to=onnx.TensorProto.FLOAT)           # [1,2,10,10]

    # --- runtime ConvTranspose kernel [2,10,3,3] ---
    init("sl_st", np.array([0, 0], np.int64), np.int64)
    init("sl_en", np.array([3, 3], np.int64), np.int64)
    init("sl_ax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["input", "sl_st", "sl_en", "sl_ax"], "P")        # [1,10,3,3]
    K = np.zeros((1, 10, 3, 3), np.float32)
    K[0, 0] = 1.0
    init("K", K)
    n("Concat", ["P", "K"], "W", axis=0)                         # [2,10,3,3]

    # --- expand blocks to the canvas, straight into output ---
    n("ConvTranspose", ["Bf", "W"], "output", strides=[3, 3])

    return _model(nodes, inits)
