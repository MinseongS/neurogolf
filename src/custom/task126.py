"""Task 126 (ARC-AGI): mark the bottom of each shooter's centre column yellow.

Rule (from generator): U-shaped "shooter" brackets (5 cells: top bar at columns
c,c+1,c+2 plus legs at c and c+2 one row down) sit in the grid, each a non-yellow
colour. They appear unchanged in the output, PLUS a yellow(4) pixel is placed at
grid[height-1][c+1] — the bottom grid row, in the bracket's centre column.

The centre column c+1 is the only column of a bracket holding exactly ONE coloured
cell (the leg columns c, c+2 hold two). Brackets never share columns (spaced >=3),
so: a column is a centre iff its total coloured-cell count == 1. height-1 is the
last in-grid row (rows below it are all-zero).

Floor-break: build the boolean yellow-cell plane as the outer product of
rowmask (last in-grid row) and colmask (centre columns), then one Where into the
free `output` overlays yellow there and leaves every other cell as input.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..builders import _model


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype=np.float32):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    # shift-up matrix: (shiftM @ rowhas)[i] = rowhas[i+1]
    S = np.zeros((1, 1, 30, 30), dtype=np.float32)
    for i in range(29):
        S[0, 0, i, i + 1] = 1.0
    init("shiftM", S)
    # tall 30x1 conv summing coloured channels (1..9) down each column
    Wcc = np.zeros((1, 10, 30, 1), dtype=np.float32)
    Wcc[0, 1:, :, :] = 1.0
    init("Wcc", Wcc)
    init("half", np.array([0.5], dtype=np.float32))
    init("oneandhalf", np.array([1.5], dtype=np.float32))
    yel = np.zeros((1, 10, 1, 1), dtype=np.float32)
    yel[0, 4, 0, 0] = 1.0
    init("yellow", yel)

    def n(op, ins, out, **kw):
        nodes.append(helper.make_node(op, ins, [out], **kw))

    # centre columns: coloured count down each column == 1 (excludes ch0 bg)
    n("Conv", ["input", "Wcc"], "colcount")                         # [1,1,1,30]
    n("Greater", ["colcount", "half"], "c_pos")
    n("Less", ["colcount", "oneandhalf"], "c_lo")
    n("And", ["c_pos", "c_lo"], "center")                            # [1,1,1,30] bool

    # last in-grid row: rowhas[r] & not rowhas[r+1]
    n("ReduceMax", ["input"], "rowhas", axes=[1, 3], keepdims=1)     # [1,1,30,1]
    n("MatMul", ["shiftM", "rowhas"], "rownext")                     # [1,1,30,1] = rowhas[r+1]
    n("Sub", ["rowhas", "rownext"], "rowdiff")
    n("Greater", ["rowdiff", "half"], "rowmask")                     # [1,1,30,1] bool

    n("And", ["rowmask", "center"], "ycell")                         # [1,1,30,30] bool outer
    n("Where", ["ycell", "yellow", "input"], "output")

    return _model(nodes, inits)
