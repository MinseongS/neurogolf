"""Task 271 (ae4f1146): emit the 3x3 box that has the MOST blue pixels.

Rule (from ARC-GEN generator): a 9x9 black grid holds four NON-overlapping 3x3
cyan(8) boxes; each box carries some blue(1) pixels at local positions inside it.
The four boxes carry a strictly-increasing number of blue pixels (lengths are
`common.sample(range(9),4)` then sorted ascending, assigned to boxes 0..3).  The
output is the 3x3 content of the box with the MOST blue pixels: output[r][c] is
blue(1) where that box has a blue pixel, else cyan(8).  The winning box is unique
(verified 0 ties over 2000 fresh samples).

Approach (no flood-fill, no [1,10,*,*] intermediate):
  * occupancy = ch1+ch8 (box cells).  A 3x3 all-ones Conv over a 9x9 slice gives,
    at each of the 7x7 top-left positions, the box-occupancy count (==9 => real
    box top-left) and the blue count (Conv of ch1).
  * score = blue-count where occupancy==9 else -1.  M = ReduceMax(score).
    winner = (score==M) is a unique cell; its (row,col) = ReduceMax of
    win*rowramp / win*colramp gives the scalar top-left (minrow,mincol).
  * Gather the 3x3 blue window at (minrow,mincol), build a uint8 label map
    (blue->1, else 8), pad to 30x30 with sentinel 10, final Equal -> BOOL output.

Memory floor: the largest intermediates are the 9x9 (81-elem) and 7x7 (49-elem)
fp16 planes; everything else is scalar / 1-D / 3x3.  No 10-channel plane is ever
materialised (the 10-way expansion lands in the FREE bool `output`).
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

G = 9   # grid side
T = G - 2  # 7 top-left positions per axis


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
    init("sl_b_st", np.array([0, 1, 0, 0], np.int64), np.int64)
    init("sl_b_en", np.array([1, 2, G, G], np.int64), np.int64)
    init("sl_ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    init("sl_c_st", np.array([0, 8, 0, 0], np.int64), np.int64)
    init("sl_c_en", np.array([1, 9, G, G], np.int64), np.int64)

    init("ones3", np.ones((1, 1, 3, 3), np.float16), np.float16)
    init("nine", np.array(9.0, np.float16), np.float16)
    init("neg1", np.array(-1.0, np.float16), np.float16)
    init("rowramp", np.arange(T, dtype=np.float16).reshape(1, 1, T, 1), np.float16)
    init("colramp", np.arange(T, dtype=np.float16).reshape(1, 1, 1, T), np.float16)
    init("off3", np.array([0, 1, 2], np.int64), np.int64)

    init("v1", np.array(1, np.uint8), np.uint8)
    init("v8", np.array(8, np.uint8), np.uint8)
    init("half16", np.array(0.5, np.float16), np.float16)
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - 3, 30 - 3], np.int64), np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)

    # ---- slice the two colour channels over the 9x9 grid ----
    n("Slice", ["input", "sl_b_st", "sl_b_en", "sl_ax"], "blue9")   # [1,1,9,9] f32
    n("Slice", ["input", "sl_c_st", "sl_c_en", "sl_ax"], "cyan9")   # [1,1,9,9] f32
    n("Cast", ["blue9"], "blue9h", to=TensorProto.FLOAT16)
    n("Cast", ["cyan9"], "cyan9h", to=TensorProto.FLOAT16)
    n("Add", ["blue9h", "cyan9h"], "occ9")                          # box cells

    # ---- 3x3 sum-convs over the 7x7 top-left grid ----
    n("Conv", ["blue9h", "ones3"], "bcount")                        # [1,1,7,7]
    n("Conv", ["occ9", "ones3"], "ocount")                          # [1,1,7,7]
    n("Equal", ["ocount", "nine"], "isbox")                         # bool box top-left
    n("Where", ["isbox", "bcount", "neg1"], "score")               # [1,1,7,7] f16

    # ---- argmax position (unique winner) ----
    n("ReduceMax", ["score"], "M", keepdims=1)                      # [1,1,1,1]
    n("Equal", ["score", "M"], "iswin")                            # bool [1,1,7,7]
    n("Cast", ["iswin"], "winf", to=TensorProto.FLOAT16)
    n("Mul", ["winf", "rowramp"], "wrgrid")
    n("ReduceMax", ["wrgrid"], "wr", keepdims=0)                    # scalar f16 = minrow
    n("Mul", ["winf", "colramp"], "wcgrid")
    n("ReduceMax", ["wcgrid"], "wc", keepdims=0)                    # scalar f16 = mincol

    # ---- build the 3x3 row / col gather indices ----
    n("Cast", ["wr"], "wri", to=TensorProto.INT64)                 # scalar
    n("Cast", ["wc"], "wci", to=TensorProto.INT64)
    n("Add", ["off3", "wri"], "rowidx")                            # [3] int64
    n("Add", ["off3", "wci"], "colidx")                            # [3] int64

    # ---- gather the 3x3 blue window of the winning box ----
    n("Gather", ["blue9h", "rowidx"], "winrows", axis=2)          # [1,1,3,9]
    n("Gather", ["winrows", "colidx"], "winblue", axis=3)         # [1,1,3,3] f16
    n("Greater", ["winblue", "half16"], "winb")                   # bool blue mask

    # ---- label map: blue->1 else cyan 8, pad to 30x30 (sentinel 10) ----
    n("Where", ["winb", "v1", "v8"], "L3")                        # [1,1,3,3] uint8
    n("Pad", ["L3", "padpads", "padval"], "L", mode="constant")   # [1,1,30,30] uint8
    n("Equal", ["L", "chan"], "output")                           # -> free BOOL output

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
