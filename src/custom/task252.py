"""Task 252 (ARC a5f85a15): diagonal stripes of a single color C; every
colored cell in an odd column turns yellow (4), even columns keep C, black
unchanged.

Method: all grids are square and <= 15x15 (generator: size in 3..15), so work
on the top-left 15x15 patch with bool masks. Build g (grid mask) from row/col
occupancy, s = g XOR black-channel (colored mask), split s by column parity
into s_e / s_o. Decoder: output = e0*g + (e_C - e0)*s_e + (e4 - e0)*s_o, a
single 1x1 Conv whose 3-column weight has one runtime column (e_C from the
per-channel presence vector u = ReduceMax(input); black is always present so
u = e0 + e_C, giving e_C - e0 = u - 2*e0). Conv pads the 15x15 patch back to
the 30x30 canvas directly into `output` (free). All values in {-1,0,1,2} ->
exact in float32.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ..builders import _model

S = 15  # max grid size seen across train/test/arc-gen (generator: 3..15)


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

    i64 = np.int64
    init("st0", [0], i64)
    init("enS", [S], i64)
    init("axH", [2], i64)
    init("axW", [3], i64)
    init("st000", [0, 0, 0], i64)
    init("en1SS", [1, S, S], i64)
    init("axCHW", [1, 2, 3], i64)

    # --- grid mask g on the SxS patch (bool) ---------------------------
    n("ReduceMax", ["input"], "rocc", axes=[1, 3], keepdims=1)  # [1,1,30,1]
    n("ReduceMax", ["input"], "cocc", axes=[1, 2], keepdims=1)  # [1,1,1,30]
    n("Cast", ["rocc"], "rb", to=onnx.TensorProto.BOOL)
    n("Cast", ["cocc"], "cb", to=onnx.TensorProto.BOOL)
    n("Slice", ["rb", "st0", "enS", "axH"], "rbS")              # [1,1,S,1]
    n("Slice", ["cb", "st0", "enS", "axW"], "cbS")              # [1,1,1,S]
    n("And", ["rbS", "cbS"], "g")                               # [1,1,S,S]

    # --- colored mask s, split by column parity ------------------------
    n("Slice", ["input", "st000", "en1SS", "axCHW"], "in0f")    # [1,1,S,S] f32
    n("Cast", ["in0f"], "in0b", to=onnx.TensorProto.BOOL)
    n("Xor", ["g", "in0b"], "s")                # colored cells (in0b subset g)
    even = (np.arange(S) % 2 == 0).reshape(1, 1, 1, S)
    init("evenC", even, np.bool_)
    n("And", ["s", "evenC"], "se")
    n("Xor", ["s", "se"], "so")

    # --- 3-channel decoder input ---------------------------------------
    n("Concat", ["g", "se", "so"], "Zb", axis=1)                # [1,3,S,S] bool
    n("Cast", ["Zb"], "Z", to=onnx.TensorProto.FLOAT)

    # --- runtime 1x1 Conv weight [10,3,1,1] -----------------------------
    n("ReduceMax", ["input"], "u", axes=[2, 3], keepdims=1)     # [1,10,1,1]
    init("shp", [10, 1, 1, 1], i64)
    n("Reshape", ["u", "shp"], "uR")                            # [10,1,1,1]
    two_e0 = np.zeros((10, 1, 1, 1)); two_e0[0] = 2.0
    init("twoE0", two_e0, np.float32)
    n("Sub", ["uR", "twoE0"], "w1")             # e_C - e0
    e0 = np.zeros((10, 1, 1, 1)); e0[0] = 1.0
    init("e0c", e0, np.float32)
    e4me0 = np.zeros((10, 1, 1, 1)); e4me0[4] = 1.0; e4me0[0] = -1.0
    init("e4me0", e4me0, np.float32)
    n("Concat", ["e0c", "w1", "e4me0"], "W", axis=1)            # [10,3,1,1]

    # --- decode straight into the 30x30 canvas --------------------------
    n("Conv", ["Z", "W"], "output", pads=[0, 0, 30 - S, 30 - S])

    return _model(nodes, inits)
