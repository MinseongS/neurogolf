"""Task 203 (85c4e7cd): concentric square rings, ring colors reversed.

Rule: square grid of size s = 2L (L = #colors, 3..9); cell (r,c) belongs to
ring k = min(r, c, s-1-r, s-1-c) and shows colors[k]; the output shows
colors[L-1-k].  Since the colors are distinct (sampled without replacement),
the output is simply a per-cell COLOR PERMUTATION of the input:
colors[k] -> colors[L-1-k].

Graph: s = max column occupancy; L-1 = s/2 - 1.  The palette in order is read
from column L-1: cell (d, L-1) = colors[d] for d <= L-1 (one MatMul with a
runtime one-hot column selector, then Slice to the first 9 rows).  The
reversal matrix P[d,k] = [d+k == L-1] (Equal on a constant 9x9 index-sum
matrix) gives Drev = D @ P with Drev[:,k] = onehot(colors[L-1-k]) (zero for
k >= L).  The 10x10 channel-permutation matrix M = Drev @ D^T is reshaped to
a runtime 1x1 Conv weight applied straight into `output` (free tensor).
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

    # --- grid size s and L-1 = s/2 - 1 ---
    n("ReduceSum", ["input"], "colsum", axes=[1, 2], keepdims=0)   # [1,30] f
    n("ReduceMax", ["colsum"], "s_f", axes=[1], keepdims=0)        # [1] f
    n("Cast", ["s_f"], "s_i", to=onnx.TensorProto.INT32)           # [1] i32
    init("c2", np.array([2], np.int32), np.int32)
    init("c1", np.array([1], np.int32), np.int32)
    n("Div", ["s_i", "c2"], "L_i")                                 # [1] L
    n("Sub", ["L_i", "c1"], "Lm1")                                 # [1] L-1

    # --- column selector e[c] = [c == L-1], extract column L-1 ---
    init("ar30", np.arange(30, dtype=np.int32).reshape(30, 1), np.int32)
    n("Equal", ["ar30", "Lm1"], "e_b")                             # [30,1] bool
    n("Cast", ["e_b"], "e_f", to=onnx.TensorProto.FLOAT)           # [30,1] f
    n("MatMul", ["input", "e_f"], "col")                           # [1,10,30,1]

    # --- palette D[v,d] = onehot(colors[d])[v] for d <= L-1 (rows 0..8) ---
    init("sl_st", np.array([0], np.int64), np.int64)
    init("sl_en", np.array([9], np.int64), np.int64)
    init("sl_ax", np.array([2], np.int64), np.int64)
    n("Slice", ["col", "sl_st", "sl_en", "sl_ax"], "D4")           # [1,10,9,1]
    init("dshape", np.array([10, 9], np.int64), np.int64)
    n("Reshape", ["D4", "dshape"], "D2")                           # [10,9] f

    # --- reversal P[d,k] = [d+k == L-1]; Drev[:,k] = onehot(colors[L-1-k]) ---
    S9 = (np.arange(9)[:, None] + np.arange(9)[None, :]).astype(np.int32)
    init("S9", S9, np.int32)
    n("Equal", ["S9", "Lm1"], "P_b")                               # [9,9] bool
    n("Cast", ["P_b"], "P_f", to=onnx.TensorProto.FLOAT)           # [9,9] f
    n("MatMul", ["D2", "P_f"], "Drev")                             # [10,9] f

    # --- permutation M = Drev @ D^T, apply as runtime 1x1 Conv weight ---
    n("Transpose", ["D2"], "D2T", perm=[1, 0])                     # [9,10] f
    n("MatMul", ["Drev", "D2T"], "M")                              # [10,10] f
    init("wshape", np.array([10, 10, 1, 1], np.int64), np.int64)
    n("Reshape", ["M", "wshape"], "W")                             # [10,10,1,1]
    n("Conv", ["input", "W"], "output")                            # free

    return _model(nodes, inits)
