"""Task 152 (67e8384a): 4-fold mirror (D2 symmetry) of a 3x3 grid -> 6x6.

Rule (from ARC-GEN generator):
  Input is a size x size grid (size = 3).  Output is 2*size x 2*size where each
  output row r<size = grid[r] ++ grid[r][::-1] (horizontal mirror) and rows
  size..2*size-1 = the vertical mirror of those rows.  So:
    out[R][C] = grid[mr(R)][mc(C)]  with  mr(R)=R if R<size else (2*size-1-R),
                                          mc(C)=C if C<size else (2*size-1-C).
  Everything is a pure COPY of input one-hot cells -> Tier S.

Encoding (pure copy, uint8 working planes, uint8 free output):
  Slice input to the active 3x3 corner (the only fp32 plane, 360B), Cast->uint8
  (one-hot is exactly {0,1} so uint8 is loss-free and 4x smaller than fp32),
  Gather rows (gr=[0,1,2,2,1,0]) then cols (gc=[0,1,2,2,1,0]) building the
  6x6x10 one-hot, then Pad to 30x30 with zeros (the FREE uint8 output).  Outside
  the 6x6 block the output is all-zero (no channel set) = background, matching
  the generator (input cells outside the 3x3 grid are all-zero too).

  Output is declared UINT8: the harness scores (out > 0) booleans, so a uint8
  {0,1} one-hot passes identically while quartering every working plane.  ORT
  supports uint8 for both Gather and Pad under ORT_DISABLE_ALL.

Result: mem 990, params 27 -> 18.075 pts (beats public import 17.67 by +0.40),
isolated fresh 200/200.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

SIZE = 3
OUT = 2 * SIZE  # 6


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # crop input to the active 3x3 corner (all channels kept)
    init("crop_st", np.array([0, 0], np.int64), np.int64)
    init("crop_en", np.array([SIZE, SIZE], np.int64), np.int64)
    init("crop_ax", np.array([2, 3], np.int64), np.int64)

    # mirror gather index vectors: [0,1,2,2,1,0]
    mirror = list(range(SIZE)) + list(range(SIZE - 1, -1, -1))
    init("gr", np.array(mirror, np.int64), np.int64)   # rows (axis 2)
    init("gc", np.array(mirror, np.int64), np.int64)   # cols (axis 3)

    # pad the 6x6 block out to 30x30 with zeros -> free uint8 output
    init("pad_pads", np.array([0, 0, 0, 0, 0, 0, 30 - OUT, 30 - OUT], np.int64),
         np.int64)
    init("pad_val", np.array(0, np.uint8), np.uint8)

    n("Slice", ["input", "crop_st", "crop_en", "crop_ax"], "g33")  # [1,10,3,3] f32
    n("Cast", ["g33"], "g33u", to=TensorProto.UINT8)               # [1,10,3,3] u8
    n("Gather", ["g33u", "gr"], "gr6", axis=2)                     # [1,10,6,3] u8
    n("Gather", ["gr6", "gc"], "gc6", axis=3)                      # [1,10,6,6] u8
    n("Pad", ["gc6", "pad_pads", "pad_val"], "output", mode="constant")

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
