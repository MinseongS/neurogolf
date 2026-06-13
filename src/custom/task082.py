"""Task 082 (3ac3eb23): split-and-stripe a row-0 pixel pattern.

Rule (from the ARC-GEN generator): the input has colored pixels only in row 0,
output height is always 6:
  - r even (0,2,4): output[r] == input row 0 (color stays at its column c)
  - r odd  (1,3,5): the pixel at c moves to neighbours c-1, c+1 (c -> black)

Everything is computed in a 2-row template [1,10,2,30] then broadcast to the
canvas by a single MatMul P (maps source row 0/1 -> output rows 0..5 by parity;
rows 6..29 stay 0) routed straight into `output` (free, no canvas intermediate).

Even row = x (the row-0 slice, already a valid one-hot incl. black channel 0).
Odd row is produced from x by ONE 1x3 convolution G (zero padded):
  colored ch in 1..9:  odd[ch,w] = x[ch,w-1] + x[ch,w+1]
  black   ch 0:        odd[0,w]  = sum_k x[k,w] - sum_{k>=1}(x[k,w-1]+x[k,w+1])
The first term sum_k x[k,w] equals the in-width grid mask (row 0 is a valid
one-hot, so its channels sum to 1 inside the width and 0 beyond), so channel 0
is set wherever the cell is inside the grid and not a colour neighbour, and is
0 outside the grid.  Source columns are >=3 apart so neighbours never collide.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ..builders import _model


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    # single 1x3 conv x -> odd-row vector (full channel mix), kernel cols [w-1,w,w+1]
    G = np.zeros((10, 10, 1, 3), np.float32)
    for ch in range(1, 10):                 # colored channels: neighbour sum
        G[ch, ch, 0, 0] = 1.0
        G[ch, ch, 0, 2] = 1.0
    for k in range(10):                     # ch0 += sum_k x[k,w]  (grid mask)
        G[0, k, 0, 1] = 1.0
    for k in range(1, 10):                  # ch0 -= colour neighbours
        G[0, k, 0, 0] = -1.0
        G[0, k, 0, 2] = -1.0
    init("G", G, np.float32)

    # parity broadcast P [30,2]: rows 0..5 only (6..29 stay zero)
    P = np.zeros((30, 2), np.float32)
    for r in range(6):
        P[r, r % 2] = 1.0
    init("P", P, np.float32)

    init("s0", np.array([0], np.int64), np.int64)
    init("s1", np.array([1], np.int64), np.int64)
    init("ax2", np.array([2], np.int64), np.int64)

    # even row = row-0 slice x [1,10,1,30]
    n("Slice", ["input", "s0", "s1", "ax2"], "x")
    # odd row from one conv, zero-padded to width 30 -> [1,10,1,30]
    n("Conv", ["x", "G"], "odd", kernel_shape=[1, 3], pads=[0, 1, 0, 1])
    # 2-row template -> [1,10,2,30]
    n("Concat", ["x", "odd"], "tpl", axis=2)
    # broadcast to canvas straight into output (free)
    n("MatMul", ["P", "tpl"], "output")
    return _model(nodes, inits)
