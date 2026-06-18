"""task078 (ARC-AGI ...) — drop the red bar up to sit under the blue bar.

Rule (generator, grid ALWAYS 10x10): each active column has a blue(1) bar of height
`top` from row0, and a red(2) bar of height `bottom` at the BOTTOM of the grid. Output:
blue bar (rows 0..top-1) unchanged, red bar MOVED to rows top..top+bottom-1 (directly
below the blue). Other columns / off-grid = background.

  => per-column counts: top = #blue in col, bottom = #red in col (anywhere). Then
  blue_mask[r,c] = r < top[c]; red_mask[r,c] = top[c] <= r < top[c]+bottom[c].
  L = blue_mask*1 + red_mask*2, pad 10->30 with sentinel 99, Equal(L, arange).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8

BLUE = 1
RED = 2
W = 10  # grid always 10x10


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    init("axCHW", np.array([1, 2, 3], np.int64), np.int64)
    # blue channel, rows 0:W, cols 0:W -> [1,1,W,W]; reduce rows -> count [1,1,1,W]
    init("b_s", np.array([BLUE, 0, 0], np.int64), np.int64)
    init("b_e", np.array([BLUE + 1, W, W], np.int64), np.int64)
    n("Slice", ["input", "b_s", "b_e", "axCHW"], "blue")        # [1,1,W,W]
    n("ReduceSum", ["blue"], "topc", axes=[2], keepdims=1)       # [1,1,1,W]
    init("r_s", np.array([RED, 0, 0], np.int64), np.int64)
    init("r_e", np.array([RED + 1, W, W], np.int64), np.int64)
    n("Slice", ["input", "r_s", "r_e", "axCHW"], "red")          # [1,1,W,W]
    n("ReduceSum", ["red"], "botc", axes=[2], keepdims=1)        # [1,1,1,W]

    # rowramp [1,1,W,1]
    init("rowramp", np.arange(W, dtype=np.float32).reshape(1, 1, W, 1), np.float32)

    n("Less", ["rowramp", "topc"], "bmask")                      # [1,1,W,W] bool r<top
    n("Add", ["topc", "botc"], "tpb")                            # [1,1,1,W]
    n("Less", ["rowramp", "tpb"], "below_tpb")                   # r < top+bot
    n("Not", ["bmask"], "ge_top")                                # r >= top
    n("And", ["below_tpb", "ge_top"], "rmask")                   # top<=r<top+bot

    # L = bmask*BLUE + rmask*RED
    n("Cast", ["bmask"], "bf", to=F32)
    n("Cast", ["rmask"], "rf", to=F32)
    init("BLUEf", np.array(float(BLUE), np.float32), np.float32)
    init("REDf", np.array(float(RED), np.float32), np.float32)
    n("Mul", ["bf", "BLUEf"], "bl")
    n("Mul", ["rf", "REDf"], "rl")
    n("Add", ["bl", "rl"], "L")                                  # [1,1,W,W] f32

    n("Cast", ["L"], "Lu8", to=U8)
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), np.int64)
    init("SENT", np.array(99, np.uint8), np.uint8)
    n("Pad", ["Lu8", "pads", "SENT"], "L30", mode="constant")    # [1,1,30,30] u8

    init("ar", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L30", "ar"], "output")                          # [1,10,30,30] bool

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task078", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
