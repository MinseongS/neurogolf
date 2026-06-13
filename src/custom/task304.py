"""Task 304 (c3e719e8): 3x3 grid -> 9x9 Kronecker self-stamp on mode cells.

Rule: input is always 3x3; the mode color (unique max count, guaranteed by
the generator) marks cells; the 9x9 output stamps a copy of the whole 3x3
grid at block (r,c) iff grid[r][c] == mode, background (channel 0) elsewhere
inside the 9x9.

Graph: X = input[.., :3, :3] via negative-pads Pad (attribute, 0 params);
counts = ReduceSum(X); modemask mm = Greater(counts, max(counts) - 0.5)
(exact: integer counts in float32); mode-cell map M = 1x1 Conv of X
with the *computed* mm as weight; complement Mc = 1 - M (valid on the grid
region); Pad [M;Mc] to [1,2,10,10]; final ConvTranspose with the *computed*
weight [X; C0] (C0 = channel-0 ones block) and stride 3 writes output (free):
  output[k, 3r+dr, 3c+dc] = M[r,c] * X[k,dr,dc] + (1-M)[r,c] * [k == 0].
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ..builders import _model


def build(task):
    inits = []
    nodes = []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    # X = input[:, :, :3, :3]  -> [1,10,3,3] (negative pads = free crop)
    n("Pad", ["input"], "X", pads=[0, 0, 0, 0, 0, 0, -27, -27])

    # per-color counts and mode mask (unique max guaranteed):
    # mm_k = [cnt_k > max(cnt) - 0.5], exact since counts are small ints
    n("ReduceSum", ["X"], "cnt", axes=[2, 3], keepdims=1)       # [1,10,1,1] f
    n("ReduceMax", ["cnt"], "mx", axes=[1], keepdims=1)         # [1,1,1,1] f
    init("half", np.array(0.5, np.float32), np.float32)
    n("Sub", ["mx", "half"], "mxh")                             # [1,1,1,1] f
    n("Greater", ["cnt", "mxh"], "eq")                          # [1,10,1,1] b
    n("Cast", ["eq"], "mm", to=onnx.TensorProto.FLOAT)          # [1,10,1,1] f

    # M[r,c] = 1 iff grid[r][c] == mode: 1x1 Conv with computed weight mm
    n("Conv", ["X", "mm"], "M")                                 # [1,1,3,3] f
    init("one", np.array(1.0, np.float32), np.float32)
    n("Sub", ["one", "M"], "Mc")                                # [1,1,3,3] f
    n("Concat", ["M", "Mc"], "MM", axis=1)                      # [1,2,3,3] f

    # pad to [1,2,10,10] so stride-3 ConvTranspose lands on [1,10,30,30]
    n("Pad", ["MM"], "MM10", pads=[0, 0, 0, 0, 0, 0, 7, 7])     # [1,2,10,10]

    # weight [2,10,3,3]: channel 0 stamps the grid, channel 1 stamps bg ones
    c0 = np.zeros((1, 10, 3, 3), np.float32)
    c0[0, 0] = 1.0
    init("C0", c0, np.float32)
    n("Concat", ["X", "C0"], "W", axis=0)                       # [2,10,3,3] f

    # output[k,3r+dr,3c+dc] = M[r,c]*X[k,dr,dc] + (1-M)[r,c]*[k==0]
    n("ConvTranspose", ["MM10", "W"], "output", strides=[3, 3])

    return _model(nodes, inits)
