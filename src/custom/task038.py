"""Task 038 (ARC-AGI 1fad071e) — count 2x2 BLUE boxes, emit that many blue cells.

Rule (from generator):
  * 9x9 input grid holds red(2)/blue(1) pixels.  Some pixels form 2x2 SOLID boxes
    (length 2), others are isolated singletons.  Boxes never overlap (gap >= 1) and
    singletons never touch a same-colour pixel, so any 2x2 all-blue block is exactly
    one genuine blue box.
  * big_blue = number of 2x2 BLUE boxes.  Output is a 1-row x 5-col grid whose first
    big_blue cells are blue(1) and the rest are background(0).  big_blue in 1..5.

Encoding (no full 30x30 plane):
  * Slice the BLUE channel cropped to the 9x9 active grid: B = input[:,1:2,0:9,0:9]
    -> [1,1,9,9] (324B).  A no-pad 2x2 all-ones Conv gives the 2x2 blue-sum
    resp[1,1,8,8]; resp==4 marks each box's top-left exactly once.
  * cnt = ReduceSum(resp==4)  (scalar big_blue).
  * Value row VL[1,1,1,30]:  blue=Less(colramp,cnt), ingrid=Less(colramp,5),
    VL = blue + ingrid - 1  -> {1 at c<cnt, 0 at cnt<=c<5, -1 (sentinel) at c>=5}.
  * lbl[1,10,1,30] = Equal(VL, arange[1,10,1,1])  (sentinel -1 hits no channel).
  * output[1,10,30,30] = And(rowmask[1,1,30,1] (row 0 only), lbl)  -> BOOL, FREE.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- slice blue channel cropped to 9x9 active grid ----
    init("s_starts", np.array([1, 0, 0], np.int64), np.int64)
    init("s_ends", np.array([2, 9, 9], np.int64), np.int64)
    init("s_axes", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "s_starts", "s_ends", "s_axes"], "blue")  # [1,1,9,9]

    # ---- 2x2 all-ones conv -> 2x2 blue sum ----
    init("k22", np.ones((1, 1, 2, 2), np.float32), np.float32)
    n("Conv", ["blue", "k22"], "resp")  # [1,1,8,8], no pad

    # ---- resp == 4  -> box top-lefts ----
    init("four", np.array(4.0, np.float32), np.float32)
    n("Equal", ["resp", "four"], "isbox")          # bool [1,1,8,8]
    n("Cast", ["isbox"], "isboxf", to=F32)
    n("ReduceSum", ["isboxf"], "cnt", keepdims=1)  # [1,1,1,1] scalar big_blue

    # ---- value row VL[1,1,1,30] ----
    colramp = np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30)
    init("colramp", colramp, np.float32)
    init("five", np.array(5.0, np.float32), np.float32)

    n("Less", ["colramp", "cnt"], "blue_b")        # c < cnt
    n("Cast", ["blue_b"], "blue_f", to=F32)
    n("Less", ["colramp", "five"], "ingrid_b")     # c < 5
    n("Cast", ["ingrid_b"], "ingrid_f", to=F32)
    n("Add", ["blue_f", "ingrid_f"], "vsum")
    init("one", np.array(1.0, np.float32), np.float32)
    n("Sub", ["vsum", "one"], "VL")                # 1 / 0 / -1

    # ---- expand to channels via Equal, gate to row 0 ----
    arange = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("arange", arange, np.float32)
    n("Equal", ["VL", "arange"], "lbl")            # [1,10,1,30] bool

    rowmask = np.zeros((1, 1, 30, 1), np.float32)
    rowmask[0, 0, 0, 0] = 1.0
    init("rowmask", (rowmask > 0), np.bool_)
    n("And", ["rowmask", "lbl"], "output")         # [1,10,30,30] bool FREE

    graph = helper.make_graph(
        nodes, "task038", [
            helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
