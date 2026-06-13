"""Task 269 (ac0a08a4): pixel-count zoom of a 3x3 grid.

Rule: input is always 3x3 with k colored (non-black) pixels, k in 1..9.
Output is 3k x 3k; each input cell (r,c) becomes a k x k block at
(r*k, c*k) of the same color (black cells become black blocks).

So out[ch,R,C] = in[ch, R//k, C//k] for R,C < 3k, zero elsewhere -- a
separable expansion out = M @ X @ M^T with M[R,r] = (r*k <= R < (r+1)*k).

Graph: k = sum of channels 1..9 over the canvas (ReduceSum + MatMul with a
0/1 selector); D = R - r*k on a [30,3] index grid; M = (D >= 0) & (D < k)
(pure integer float math, exact); MT = Transpose(M); the two batched
MatMuls expand columns then rows, the second writing `output` (free).
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

    # --- k = number of non-black pixels (sum of channels 1..9) ---
    n("ReduceSum", ["input"], "chsum", axes=[2, 3], keepdims=0)    # [1,10] f
    init("w01", np.array([[0]] + [[1]] * 9, np.float32))           # [10,1]
    n("MatMul", ["chsum", "w01"], "k")                             # [1,1] f

    # --- expansion matrix M[R,r] = (r*k <= R < (r+1)*k), [30,3] ---
    init("ridx", np.arange(3, dtype=np.float32).reshape(1, 3))     # [1,3]
    n("Mul", ["ridx", "k"], "rk")                                  # [1,3] f
    init("Ridx", np.arange(30, dtype=np.float32).reshape(30, 1))   # [30,1]
    n("Sub", ["Ridx", "rk"], "D")                                  # [30,3] f
    n("Less", ["D", "k"], "m1")                                    # D < k
    init("negh", np.array(-0.5, np.float32))
    n("Greater", ["D", "negh"], "m2")                              # D >= 0
    n("And", ["m1", "m2"], "mb")                                   # [30,3] bool
    n("Cast", ["mb"], "M", to=onnx.TensorProto.FLOAT)              # [30,3] f
    n("Transpose", ["M"], "MT", perm=[1, 0])                       # [3,30] f

    # --- out = M @ in3 @ MT ---
    init("sl_st", np.array([0, 0], np.int64), np.int64)
    init("sl_en", np.array([3, 3], np.int64), np.int64)
    init("sl_ax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["input", "sl_st", "sl_en", "sl_ax"], "in3")        # [1,10,3,3]
    n("MatMul", ["in3", "MT"], "T1")                               # [1,10,3,30]
    n("MatMul", ["M", "T1"], "output")                             # [1,10,30,30]

    return _model(nodes, inits)
