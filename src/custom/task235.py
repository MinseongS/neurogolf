"""task235 (ARC-AGI 995c5fa3) — decode 3 gray-block glyphs to a 3x3 colour grid.

Rule (from the ARC-GEN generator, verified fresh):
  Input is a 4x14 grid: three 4x4 GRAY (colour 5) blocks at columns
  idx*5 .. idx*5+3 (idx=0,1,2), separated by single background columns.  Each
  block has BLACK (colour 0 == background) cells punched out of the gray in a
  pattern that encodes one of four colours {2,3,4,8}:
      colour 8: centre 2x2 punched   -> local cells (1,1),(1,2),(2,1),(2,2)
      colour 3: cols 0 & 3 punched   -> local cells (1,0),(1,3),(2,0),(2,3)
      colour 4: bottom-centre punched-> local cells (2,1),(2,2),(3,1),(3,2)
      colour 2: nothing punched (all gray)
  Output is 3x3 where row r is filled entirely with colours[r] (the colour of
  block r).

Decoder (closed form, no argmax / template-matmul):
  Three local cells fully discriminate the four glyphs — read the GRAY channel
  (5) at, per block idx (absolute coords):
      (row 1, col idx*5+1) punched  <=>  colour 8   (only 8 punches the centre)
      (row 1, col idx*5)   punched  <=>  colour 3   (only 3 punches col 0)
      (row 3, col idx*5+1) punched  <=>  colour 4   (only 4 punches row 3)
      none of the three punched     <=>  colour 2
  "punched" = gray-channel value is 0 at that cell, so the indicator is
  v = 1 - gray.  The blocks are regularly spaced (stride 5), so the three
  per-block indicator vectors come from one strided Slice each (cols start:+:5).

  Build the colour one-hot directly: scatter v8/v3/v4 into channels 8/3/4 and
  v2 = 1-(v8+v3+v4) into channel 2 of a tiny [1,10,3,1] tensor (block axis ->
  output-row axis), Expand the column axis to 3, Pad to 30x30.  No 3600B colour
  plane is ever materialised; every working tensor is fp16 and <100 elems.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

S = 30


def build(task):
    inits, nodes = [], []
    seen = set()

    def init(name, arr, dt):
        if name in seen:
            return name
        seen.add(name)
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    H = TensorProto.FLOAT16

    # ---- slice gray channel (5), grid rows 1..3, cols 0..11 -----------------
    init("g_starts", np.array([5, 1, 0], np.int64), np.int64)
    init("g_ends", np.array([6, 4, 12], np.int64), np.int64)
    init("g_axes", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "g_starts", "g_ends", "g_axes"], "gblk")  # [1,1,3,12] fp32
    n("Cast", ["gblk"], "gblkh", to=H)                             # [1,1,3,12] fp16

    # local row index within gblk: grid row1 -> 0, grid row3 -> 2
    # strided col slices (start:end:5) pick block columns idx*5(+1)
    init("rs0", np.array([0, 1], np.int64), np.int64)   # row 0, col 1
    init("re0", np.array([1, 12], np.int64), np.int64)
    init("ax23", np.array([2, 3], np.int64), np.int64)
    init("st5", np.array([1, 5], np.int64), np.int64)
    n("Slice", ["gblkh", "rs0", "re0", "ax23", "st5"], "s8")  # gray row1 cols 1,6,11

    init("rs1", np.array([0, 0], np.int64), np.int64)   # row 0, col 0
    init("re1", np.array([1, 11], np.int64), np.int64)
    n("Slice", ["gblkh", "rs1", "re1", "ax23", "st5"], "s3")  # gray row1 cols 0,5,10

    init("rs2", np.array([2, 1], np.int64), np.int64)   # row 2 (grid row3), col 1
    init("re2", np.array([3, 12], np.int64), np.int64)
    n("Slice", ["gblkh", "rs2", "re2", "ax23", "st5"], "s4")  # gray row3 cols 1,6,11

    # indicators v = 1 - gray  (1 == punched == this colour)
    init("one", np.array(1.0, np.float16), np.float16)
    n("Sub", ["one", "s8"], "v8")   # [1,1,1,3]
    n("Sub", ["one", "s3"], "v3")
    n("Sub", ["one", "s4"], "v4")
    # v2 = 1 - v8 - v3 - v4  (mutually exclusive; default colour)
    n("Add", ["v8", "v3"], "v83")
    n("Add", ["v83", "v4"], "v834")
    n("Sub", ["one", "v834"], "v2")

    # reshape each [1,1,1,3] -> [1,1,3,1] (block axis -> output-row axis)
    init("rowsh", np.array([1, 1, 3, 1], np.int64), np.int64)
    n("Reshape", ["v2", "rowsh"], "c2")
    n("Reshape", ["v3", "rowsh"], "c3")
    n("Reshape", ["v4", "rowsh"], "c4")
    n("Reshape", ["v8", "rowsh"], "c8")

    # scatter into channels: [0,0,c2,c3,c4,0,0,0,c8,0] -> [1,10,3,1]
    init("z2", np.zeros((1, 2, 3, 1), np.float16), np.float16)  # ch 0,1
    init("z3", np.zeros((1, 3, 3, 1), np.float16), np.float16)  # ch 5,6,7
    init("z1", np.zeros((1, 1, 3, 1), np.float16), np.float16)  # ch 9
    n("Concat", ["z2", "c2", "c3", "c4", "z3", "c8", "z1"], "chan", axis=1)  # [1,10,3,1]

    # expand cols to 3, pad to 30x30
    init("expsh", np.array([1, 10, 3, 3], np.int64), np.int64)
    n("Expand", ["chan", "expsh"], "blk33")            # [1,10,3,3] fp16
    init("pads", np.array([0, 0, 0, 0, 0, 0, S - 3, S - 3], np.int64), np.int64)
    init("padv", np.array(0.0, np.float16), np.float16)
    n("Pad", ["blk33", "pads", "padv"], "output")      # [1,10,30,30] fp16 (FREE)

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", H, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task235", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
