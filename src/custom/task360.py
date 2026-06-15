"""Task 360 (ARC-GEN e3497940): fold two mirrored half-grids onto the left half.

Rule (verified 200/200 fresh):
  Input is a 10-row x 9-col grid with a gray(5) separator column at col 4.
  The LEFT block is cols 0..3, the RIGHT block is cols 5..8.  Each output cell
  (10 rows x 4 cols) is the union of the left cell and the horizontally-MIRRORED
  right cell (the two share the same colour whenever both are present):
      output[r][c] = left[r][c]  if left[r][c] != 0  else  reverse(right)[r][c]
  where output col c == width-c-1 maps left col (width-c-1) and right col
  (width+c+1), i.e. the right block reversed along the width axis.

Encoding (one-hot domain, ZERO params, no colour recovery):
  The benchmark feeds a one-hot [1,10,30,30] tensor and scores pred = out>0.
  Left and mirrored-right are one-hot, so for the COLOUR channels (1..9) the
  per-cell union is just an element-wise Max of the two slices (the rule
  guarantees both sides carry the same colour whenever both are present).  The
  BLACK channel (0) cannot be Max'd -- a cell that is black on one side and
  coloured on the other must end up coloured, not multi-hot -- so channel 0 is
  rebuilt as "no colour present" = 1 - ReduceMax over the folded colour channels.
  Slice left/right blocks (reverse right via negative-step Slice), Max the colour
  channels, derive channel 0, Concat, then Pad straight into the free
  [1,10,30,30] output.  Zero params; intermediates are a handful of [1,9,10,4]
  / [1,10,10,4] f32 tensors.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

H = 10  # active grid height (rows 0..9)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # left COLOUR channels (1..9), cols 0..3, rows 0..9  -> [1,9,10,4]
    init("L_st", np.array([1, 0, 0], np.int64), np.int64)
    init("L_en", np.array([10, H, 4], np.int64), np.int64)
    init("crc_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "L_st", "L_en", "crc_ax"], "left_f")  # [1,9,10,4] f32
    n("Cast", ["left_f"], "left", to=TensorProto.FLOAT16)      # [1,9,10,4] f16

    # right COLOUR channels (1..9), cols 5..8 REVERSED along width -> [1,9,10,4]
    # Slice start col=8, end=4 (exclusive), step=-1 yields cols 8,7,6,5.
    init("R_st", np.array([1, 0, 8], np.int64), np.int64)
    init("R_en", np.array([10, H, 4], np.int64), np.int64)
    init("R_step", np.array([1, 1, -1], np.int64), np.int64)
    n("Slice", ["input", "R_st", "R_en", "crc_ax", "R_step"], "right_f")  # [1,9,10,4] f32
    n("Cast", ["right_f"], "right", to=TensorProto.FLOAT16)               # [1,9,10,4] f16

    # colour union (channels 1..9 only): element-wise max of the two one-hot slices
    n("Max", ["left", "right"], "fold")  # [1,9,10,4] f16

    # channel 0 (black) = "no colour present" = 1 - max over folded colour channels
    n("ReduceMax", ["fold"], "anycol", axes=[1], keepdims=1)  # [1,1,10,4] f16
    init("one", np.array(1.0, np.float16), np.float16)
    n("Sub", ["one", "anycol"], "ch0")  # [1,1,10,4] f16 (1 where black)

    # reassemble the full one-hot [1,10,10,4] = [ch0, fold]
    n("Concat", ["ch0", "fold"], "full", axis=1)  # [1,10,10,4] f16

    # pad straight into the free [1,10,30,30] output (zeros outside the 10x4 region)
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - H, 30 - 4], np.int64), np.int64)
    init("padval", np.array(0.0, np.float16), np.float16)
    n("Pad", ["full", "padpads", "padval"], "output", mode="constant")  # [1,10,30,30] f16 (free)

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
