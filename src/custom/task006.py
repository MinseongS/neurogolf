"""Task 006 (0520fde7): logical-AND of the two halves of a 3x7 grid.

Rule (from ARC-GEN common.grid_intersect, width=3 height=3):
  Input is a 3-row x 7-col grid.  Column 3 is a gray separator.  The left half
  is cols 0..2, the right half is cols 4..6.  Blue (colour 1) pixels are placed
  in both halves (background = black, colour 0).  The 3x3 OUTPUT is:
      output[r][c] = red(2)   iff   left[r][c] == blue  AND  right[r][c] == blue
                   = black(0) otherwise
  i.e. the cellwise logical AND of the two 3x3 halves (on channel-1 / blue),
  recoloured red.  Everything outside the 3x3 output grid is all-zero (bg).

Encoding (Tier B, uint8 free output):
  Slice channel-1 (blue) of the input into the left 3x3 block (cols 0..2) and the
  right 3x3 block (cols 4..6) -- the only fp32 planes, 36B each.  AND them
  (Cast->bool, And) to get the 3x3 red mask.  A single Where places the 10-channel
  one-hot directly: red one-hot at the masked cells, background one-hot elsewhere,
  yielding a tiny [1,10,3,3] uint8 block (90B).  Pad that block out to 30x30 with
  zeros -- the FREE uint8 output.  No fp32 30x30 plane and no Conv ever materialise.

  Output is declared UINT8: the harness scores (out > 0) booleans, so a uint8
  {0,1} one-hot passes identically while quartering every working plane.  Both
  Where and Pad run for uint8 under ORT_DISABLE_ALL.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

SIZE = 3
N = 10
BLUE = 1
RED = 2


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # Slice channel-1 (blue), rows 0..2.  Left cols 0..2, right cols 4..6.
    init("ax", np.array([1, 2, 3], np.int64), np.int64)        # ch, row, col
    init("l_st", np.array([BLUE, 0, 0], np.int64), np.int64)
    init("l_en", np.array([BLUE + 1, SIZE, SIZE], np.int64), np.int64)
    init("r_st", np.array([BLUE, 0, SIZE + 1], np.int64), np.int64)
    init("r_en", np.array([BLUE + 1, SIZE, 2 * SIZE + 1], np.int64), np.int64)

    # 10-channel one-hots for the Where branches.
    redhot = np.zeros((1, N, 1, 1), np.uint8); redhot[0, RED, 0, 0] = 1
    bghot = np.zeros((1, N, 1, 1), np.uint8); bghot[0, 0, 0, 0] = 1
    init("redhot", redhot, np.uint8)
    init("bghot", bghot, np.uint8)

    # pad the 3x3 block out to 30x30 with zeros -> free uint8 output
    init("pad_pads",
         np.array([0, 0, 0, 0, 0, 0, 30 - SIZE, 30 - SIZE], np.int64), np.int64)
    init("pad_val", np.array(0, np.uint8), np.uint8)

    n("Slice", ["input", "l_st", "l_en", "ax"], "left")    # [1,1,3,3] f32
    n("Slice", ["input", "r_st", "r_en", "ax"], "right")   # [1,1,3,3] f32
    n("Cast", ["left"], "lb", to=TensorProto.BOOL)
    n("Cast", ["right"], "rb", to=TensorProto.BOOL)
    n("And", ["lb", "rb"], "fg")                           # [1,1,3,3] bool
    n("Where", ["fg", "redhot", "bghot"], "blk")           # [1,10,3,3] u8
    n("Pad", ["blk", "pad_pads", "pad_val"], "output", mode="constant")

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
