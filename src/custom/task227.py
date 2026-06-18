"""task227 (ARC-AGI 94f9d214) — per-pixel NOR of the two stacked half-grids.

Rule (from the generator):
  The input is a `size`-tall x `2*size`-wide grid (size=4 always, so 8x4).
  The TOP half (rows 0..size-1) holds green pixels, the BOTTOM half
  (rows size..2*size-1) holds blue pixels, at random positions.  The output is
  a size x size grid:
      output[r][c] = red  iff  grid[r][c] == bg  AND  grid[size+r][c] == bg
  i.e. red exactly where BOTH stacked halves are EMPTY at column c, otherwise
  the cell stays background.

Encoding facts:
  Each grid cell sets exactly one of the 10 one-hot channels; channel 0 is the
  background, so "cell empty" == "channel-0 == 1".  Outside the 8x4 grid the
  whole 30x30 canvas is all-zero.  The 4x4 output region: red -> channel 2,
  every other in-grid cell -> channel 0 (background).

Pipeline (opset 11), all working tensors are tiny 4x4 blocks:
  1. top = input[:, 0:1, 0:4, 0:4]    (bg channel, rows 0..3)   [1,1,4,4]
     bot = input[:, 0:1, 4:8, 0:4]    (bg channel, rows 4..7)   [1,1,4,4]
  2. red = top * bot                  (1 where both empty)       [1,1,4,4]
     bg  = 1 - red                    (background elsewhere)     [1,1,4,4]
  3. concat into channels:  ch0=bg, ch2=red, the rest 0   -> [1,10,4,4] uint8
  4. Pad spatially to [1,10,30,30] (the graph output, zero-filled border).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
I64 = TensorProto.INT64


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # --- 1. slice the bg channel (ch0) of the top and bottom halves --------
    init("top_s", np.array([0, 0, 0], np.int64), np.int64)
    init("top_e", np.array([1, 4, 4], np.int64), np.int64)
    init("top_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "top_s", "top_e", "top_ax"], "top")      # [1,1,4,4] f32

    init("bot_s", np.array([0, 4, 0], np.int64), np.int64)
    init("bot_e", np.array([1, 8, 4], np.int64), np.int64)
    n("Slice", ["input", "bot_s", "bot_e", "top_ax"], "bot")      # [1,1,4,4] f32

    # --- 2. red = top AND bot (both empty);  bg = 1 - red ------------------
    n("Mul", ["top", "bot"], "red")                               # [1,1,4,4] f32
    init("one", np.array([[[[1.0]]]], np.float32), np.float32)
    n("Sub", ["one", "red"], "bg")                                # [1,1,4,4] f32

    # cast both to uint8 (output scored as out>0)
    n("Cast", ["red"], "red_u", to=U8)                            # [1,1,4,4] u8
    n("Cast", ["bg"], "bg_u", to=U8)                              # [1,1,4,4] u8

    # --- 3. build a 3-channel block: ch0=bg, ch1=0, ch2=red ---------------
    init("z1", np.zeros((1, 1, 4, 4), np.uint8), np.uint8)        # ch1 (blank)
    n("Concat", ["bg_u", "z1", "red_u"], "block", axis=1)         # [1,3,4,4] u8

    # --- 4. Pad channels (+7) AND spatially (+26) to [1,10,30,30] ---------
    init("pads", np.array([0, 0, 0, 0, 0, 7, 26, 26], np.int64), np.int64)
    init("padval", np.array(0, np.uint8), np.uint8)
    n("Pad", ["block", "pads", "padval"], "output", mode="constant")

    out_vi = helper.make_tensor_value_info("output", U8, [1, 10, 30, 30])
    in_vi = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])

    graph = helper.make_graph(nodes, "task227", [in_vi], [out_vi], inits)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 11)],
        ir_version=IR_VERSION)
    onnx.checker.check_model(model, full_check=True)
    return model
