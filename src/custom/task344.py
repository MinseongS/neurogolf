"""Task 344 (ARC d90796e8): green next to red -> green becomes cyan, red erased.

Exact rule (colors: black=0, red=2, green=3, gray=5, cyan=8):
- a green(3) cell with >=1 red(2) 4-neighbor becomes cyan(8)
- a red(2) cell with >=1 green(3) 4-neighbor becomes black(0)
- everything else (incl. gray) is unchanged

The scorer accepts a channel iff `result > 0`. For each cell the output color is
a *linearly separable* function of the plus-shaped neighborhood one-hot, so the
whole rule is ONE 3x3 Conv (plus-shaped weights) + bias written straight into the
free `output` tensor. Zero intermediates.

Let c0,c2,c3,c5 = center one-hots, n2/n3 = #red/#green among the 4 neighbors.
Per-channel score (>0 selects that channel), scaled x4 to keep all weights int:
  ch0 = 8*c0 + 4*c2 + 1*n3 - 4   (black, or red erased by a green neighbor)
  ch2 = 4*c2           - 4*n3 - 2 (red kept iff no green neighbor)
  ch3 = 4*c3           - 4*n2 - 2 (green kept iff no red neighbor)
  ch5 = 4*c5                      (gray, unchanged)
  ch8 = 4*c3 + 1*n2           - 4 (green -> cyan iff a red neighbor)
All other channels score 0 (never selected). Verified mutually exclusive over
every center color x neighbor count, so exactly one channel is >0 per cell.
"""

import numpy as np
from onnx import helper, numpy_helper

from ..builders import _model

# plus-shaped neighbor positions in a 3x3 kernel (no center, no diagonals)
_NBR = [(0, 1), (2, 1), (1, 0), (1, 2)]
_CEN = (1, 1)


def build(task):
    W = np.zeros((10, 10, 3, 3), np.float32)
    b = np.zeros(10, np.float32)

    # ch0 (black): 8*c0 + 4*c2 + 1*n3 - 4
    W[0, 0, _CEN[0], _CEN[1]] = 8.0
    W[0, 2, _CEN[0], _CEN[1]] = 4.0
    for r, c in _NBR:
        W[0, 3, r, c] = 1.0
    b[0] = -4.0

    # ch2 (red): 4*c2 - 4*n3 - 2
    W[2, 2, _CEN[0], _CEN[1]] = 4.0
    for r, c in _NBR:
        W[2, 3, r, c] = -4.0
    b[2] = -2.0

    # ch3 (green): 4*c3 - 4*n2 - 2
    W[3, 3, _CEN[0], _CEN[1]] = 4.0
    for r, c in _NBR:
        W[3, 2, r, c] = -4.0
    b[3] = -2.0

    # ch5 (gray): 4*c5
    W[5, 5, _CEN[0], _CEN[1]] = 4.0

    # ch8 (cyan): 4*c3 + 1*n2 - 4
    W[8, 3, _CEN[0], _CEN[1]] = 4.0
    for r, c in _NBR:
        W[8, 2, r, c] = 1.0
    b[8] = -4.0

    inits = [
        numpy_helper.from_array(W, "W"),
        numpy_helper.from_array(b, "b"),
    ]
    nodes = [
        helper.make_node("Conv", ["input", "W", "b"], ["output"],
                         pads=[1, 1, 1, 1]),
    ]
    return _model(nodes, inits)
