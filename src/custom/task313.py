"""Task 313 (caa06a1f): periodic stripe pattern, phase-shifted by one.

Rule: square grid size s, palette of L = 2 + s//12 colors; output cell (r,c)
is colors[(r%2 + c + 1) % L] everywhere in the s x s grid. The palette is
read from input row 0 (cells (0,0..2) are always inside the visible corner
region since border <= s//2 <= s-3).

Graph: s = max column occupancy; L = 2 + s//12 (int ops); index pattern
(p + c + 1) % L on a tiny [2,30] parity-row canvas via Mod; one-hot planes
[1,3,2,30] via Equal + column mask; palette one-hots from Slice(input row0,
cols 0..2) reshaped into a 1x1 Conv weight -> B[1,10,2,30]; final MatMul
with a masked [30,2] parity-row selector writes output (free) directly.
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

    # --- grid size s and column mask from occupancy ---
    # colsum[c] = s for c < s, else 0  (every in-grid cell is one-hot)
    n("ReduceSum", ["input"], "colsum", axes=[1, 2], keepdims=0)   # [1,30] f
    n("ReduceMax", ["colsum"], "s_f", axes=[1], keepdims=0)        # [1] f
    n("Cast", ["s_f"], "s_i", to=onnx.TensorProto.INT32)           # [1] i32
    init("c12", np.array([12], np.int32), np.int32)
    init("c2", np.array([2], np.int32), np.int32)
    n("Div", ["s_i", "c12"], "d")                                  # s // 12
    n("Add", ["d", "c2"], "L")                                     # L = 2+s//12
    init("half", np.array([[0.5]], np.float32))
    n("Greater", ["colsum", "half"], "colmask")                    # [1,30] bool

    # --- index pattern (p + c + 1) % L on parity rows p in {0,1} ---
    Q = np.arange(30, dtype=np.int32)[None, :] + \
        np.array([[1], [2]], np.int32)                             # p + c + 1
    init("Q", Q, np.int32)
    n("Mod", ["Q", "L"], "R")                                      # [2,30] i32
    init("J", np.arange(3, dtype=np.int32).reshape(1, 3, 1, 1), np.int32)
    n("Equal", ["R", "J"], "E")                                    # [1,3,2,30] bool
    n("And", ["E", "colmask"], "Em")                               # column-masked
    n("Cast", ["Em"], "Ef", to=onnx.TensorProto.FLOAT)             # [1,3,2,30] f

    # --- palette one-hots from input cells (0,0),(0,1),(0,2) ---
    init("sl_st", np.array([0, 0], np.int64), np.int64)
    init("sl_en", np.array([1, 3], np.int64), np.int64)
    init("sl_ax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["input", "sl_st", "sl_en", "sl_ax"], "tl")         # [1,10,1,3]
    init("wshape", np.array([10, 3, 1, 1], np.int64), np.int64)
    n("Reshape", ["tl", "wshape"], "cvw")                          # [10,3,1,1]

    # --- per-parity-row colored pattern: B[0,k,p,c] ---
    n("Conv", ["Ef", "cvw"], "B")                                  # [1,10,2,30] f

    # --- expand parity rows to full grid, row-masked, into output ---
    init("rshape", np.array([30, 1], np.int64), np.int64)
    n("Reshape", ["colmask", "rshape"], "rmask")                   # [30,1] bool
    n("Cast", ["rmask"], "rmf", to=onnx.TensorProto.FLOAT)         # [30,1] f
    Par = np.zeros((30, 2), np.float32)
    Par[np.arange(30), np.arange(30) % 2] = 1.0
    init("Par", Par)
    n("Mul", ["Par", "rmf"], "RowSel")                             # [30,2] f
    n("MatMul", ["RowSel", "B"], "output")                         # [1,10,30,30]

    return _model(nodes, inits)
