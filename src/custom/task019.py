"""Task 019 (10fcaaa3): 2x2 tile with cyan diagonal halo.

Rule (from ARC-GEN generator): input is an H x W grid (H,W in 2..6) holding a
few cells of a single non-cyan color.  Output is 2H x 2W: the input grid is
tiled 2x2 (color stamps at (r,c),(r+H,c),(r,c+W),(r+H,c+W)); cyan(8) is painted
on the 4 diagonal neighbours of every stamped colour cell (in full output
coords, clipped to the 2H x 2W region), but the colour overwrites cyan.

Graph: recover H,W from per-row/col occupancy of the all-channel "cell exists"
plane; Gather precomputed index tables to 2x2-tile the colored & exists planes;
cyan = diagonal 3x3 Conv of the colour mask, clamped to {0,1} and ANDed with
in-grid & non-colour; build a single int8 label plane
L = k*Color + 8*Cyan + G - 1  (-1 outside, 0 bg, 8 cyan, k colour) where
k = sum_c c*[channel c used]; output[c] = (L == c).

All values are tiny integers (exact in float32).  Canvas mask planes are int8;
the only [1,10,30,30] tensor is the final bool Equal feeding the free `output`.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..builders import _model


def _idx_tables():
    """[5,30] int32: tile maps for H (or W) = 2..6.

    idx[H-2][R] = R % H for R < 2H, else 29 (a line always outside the
    2H x 2W <= 12x12 grid, hence all-zero in the tiled plane)."""
    rows = []
    for H in range(2, 7):
        rows.append(np.array(
            [(R % H if R < 2 * H else 29) for R in range(30)], np.int32))
    return np.array(rows, np.int32)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- constants ----
    init("IDX", _idx_tables(), np.int32)                          # [5,30]
    init("Wcol", np.array([0] + [1] * 9, np.float32).reshape(1, 10, 1, 1),
         np.float32)                                              # colored mask
    init("Wk", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1),
         np.float32)                                              # k accumulator
    init("diag", np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]],
                          np.float32).reshape(1, 1, 3, 3), np.float32)
    init("two", np.array(2.0, np.float32), np.float32)
    init("half", np.array(0.5, np.float32), np.float32)
    init("v8", np.array(8, np.int32), np.int32)
    init("v0", np.array(0, np.int32), np.int32)
    init("vm1", np.array(-1, np.int32), np.int32)
    init("chan", np.arange(10, dtype=np.int32).reshape(1, 10, 1, 1), np.int32)

    # ---- compact input planes ----
    n("Conv", ["input", "Wcol"], "colored")        # [1,1,30,30] 1 at colored cells
    n("ReduceMax", ["input"], "exists", axes=[1], keepdims=1)    # 1 inside HxW

    # ---- recover H, W ----
    n("ReduceMax", ["exists"], "rowocc", axes=[3], keepdims=1)   # [1,1,30,1]
    n("ReduceMax", ["exists"], "colocc", axes=[2], keepdims=1)   # [1,1,1,30]
    n("ReduceSum", ["rowocc"], "Hf", keepdims=0)                 # scalar H
    n("ReduceSum", ["colocc"], "Wf", keepdims=0)                 # scalar W
    n("Sub", ["Hf", "two"], "Hm")
    n("Sub", ["Wf", "two"], "Wm")
    n("Cast", ["Hm"], "Hi0", to=TensorProto.INT32)              # H-2 in 0..4
    n("Cast", ["Wm"], "Wi0", to=TensorProto.INT32)
    n("Squeeze", ["Hi0"], "Hi", axes=[0])                       # -> scalar index
    n("Squeeze", ["Wi0"], "Wi", axes=[0])
    n("Gather", ["IDX", "Hi"], "idxr", axis=0)                  # [30] int32
    n("Gather", ["IDX", "Wi"], "idxc", axis=0)

    # ---- 2x2 tile the colored plane as bool, then float only for the Conv ----
    n("Greater", ["colored", "half"], "colb")                  # bool 1 at colored cells
    n("Gather", ["colb", "idxr"], "cr", axis=2)                # bool
    n("Gather", ["cr", "idxc"], "Cb", axis=3)                  # bool colour cell
    n("Cast", ["Cb"], "Cf", to=TensorProto.FLOAT)             # float for Conv
    n("Conv", ["Cf", "diag"], "raw", pads=[1, 1, 1, 1])        # 0..4 f32
    n("Greater", ["raw", "half"], "rawpos")                    # bool: diag-neighbour

    # ---- 2x2 tile the exists plane as bool (in-grid mask) ----
    n("Greater", ["exists", "half"], "exb")                    # bool, 1 inside HxW
    n("Gather", ["exb", "idxr"], "er", axis=2)                 # bool
    n("Gather", ["er", "idxc"], "Gb", axis=3)                  # bool in-grid

    # ---- cyan (bool) = diag-neighbour & in-grid & not-colour ----
    n("Not", ["Cb"], "notc")
    n("And", ["rawpos", "Gb"], "cy0")
    n("And", ["cy0", "notc"], "Cyan")                           # bool

    # ---- colour index k (int32 scalar) ----
    n("ReduceMax", ["input"], "pres", axes=[2, 3], keepdims=1)  # [1,10,1,1] f32
    n("Mul", ["pres", "Wk"], "kparts")
    n("ReduceSum", ["kparts"], "kf", keepdims=1)                # [1,1,1,1] f32 = k
    n("Cast", ["kf"], "ki", to=TensorProto.INT32)             # int32 k (broadcast)

    # ---- L = colour? k : cyan? 8 : in-grid? 0 : -1   (int32 label plane) ----
    n("Where", ["Gb", "v0", "vm1"], "Lg")                      # 0 in-grid else -1
    n("Where", ["Cyan", "v8", "Lg"], "Lc")                     # cyan = 8
    n("Where", ["Cb", "ki", "Lc"], "L")                        # colour overrides

    # ---- output[c] = (L == c) ----
    n("Equal", ["L", "chan"], "eq")                             # [1,10,30,30] bool
    n("Cast", ["eq"], "output", to=TensorProto.FLOAT)
    return _model(nodes, inits)
