"""Task 236 (99b1bc43): XOR of a top blue 4x4 pattern and a bottom red 4x4
pattern, painted green.

Rule (from ARC-GEN generator, size=4):
  Input is a 9x4 colour grid (height 2*size+1 = 9).  Rows 0..3 hold blue (1)
  pixels (the "top" pattern), row 4 is a solid yellow (4) divider, rows 5..8
  hold red (2) pixels (the "bottom" pattern, originally at the same 0..3 rows
  before being shifted down by size+1=5).  The 4x4 output is green (3) at every
  cell where exactly ONE of {top has a pixel, bottom has a pixel} is true, i.e.
  the per-cell XOR of the two patterns; elsewhere background.

Encoding (closed-form XOR, uint8 working planes, uint8 free output):
  L = blue channel (1) over rows 0..3, cols 0..3  -> top occupancy [1,1,4,4]
  R = red  channel (2) over rows 5..8, cols 0..3  -> bottom occupancy [1,1,4,4]
  Both are one-hot {0,1}.  xor = |L - R| (Sub+Abs in fp32, the only op ORT runs
  for arithmetic), then Cast to uint8.  The green output channel (index 3) = xor;
  the background channel (index 0) = 1 - xor (so background cells carry ch0=1 as
  the harness expects); all other channels are 0.

  Channels 0 and 3 are non-adjacent, so we Concat [ch0, 0, 0, xor] (a tiny
  [1,4,4,4] uint8 stack, 64 B) and Pad the trailing 6 colour channels and the
  26x26 spatial border with zeros.  Pad's output IS the graph output, declared
  UINT8 (harness scores out>0, so a {0,1} one-hot passes identically while every
  working plane is 1 byte/elem instead of 4).

  No [1,10,30,30] or [1,10,4,4] fp32 plane ever materialises; the dominant
  intermediate is the 64 B uint8 channel stack.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

SIZE = 4


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # top blue occupancy: channel 1, rows 0..3, cols 0..3
    init("L_st", np.array([1, 0, 0], np.int64), np.int64)
    init("L_en", np.array([2, SIZE, SIZE], np.int64), np.int64)
    init("axes", np.array([1, 2, 3], np.int64), np.int64)
    # bottom red occupancy: channel 2, rows 5..8, cols 0..3
    init("R_st", np.array([2, SIZE + 1, 0], np.int64), np.int64)
    init("R_en", np.array([3, 2 * SIZE + 1, SIZE], np.int64), np.int64)

    init("pad_pads",
         np.array([0, 0, 0, 0, 0, 6, 30 - SIZE, 30 - SIZE], np.int64),
         np.int64)
    init("pad_val", np.array(0, np.uint8), np.uint8)
    init("z", np.zeros((1, 1, SIZE, SIZE), np.uint8), np.uint8)

    n("Slice", ["input", "L_st", "L_en", "axes"], "L")     # [1,1,4,4] f32
    n("Slice", ["input", "R_st", "R_en", "axes"], "R")     # [1,1,4,4] f32
    # XOR == (L != R); match == (L == R) is the NOT-xor background mask.
    n("Equal", ["L", "R"], "match")                         # bool, NOT xor
    n("Not", ["match"], "xorb")                             # bool, xor
    n("Cast", ["match"], "bg", to=TensorProto.UINT8)        # ch0 (bg) u8
    n("Cast", ["xorb"], "xor", to=TensorProto.UINT8)        # ch3 (green) u8
    # channels: 0=bg, 1=0, 2=0, 3=xor  (z is a const zero plane)
    n("Concat", ["bg", "z", "z", "xor"], "stack", axis=1)   # [1,4,4,4] u8
    n("Pad", ["stack", "pad_pads", "pad_val"], "output", mode="constant")

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
