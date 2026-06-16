"""Task 180 (ARC-GEN 75b8110e): 4-quadrant overlay of an 8x8 grid into 4x4.

Rule (from ARC-GEN generator, verified fresh 200/200):
  The input is a 2*size x 2*size grid with size=4 (so 8x8).  It is split into
  four size x size (4x4) quadrants by (row_offset, col_offset):
      idx 0  color 4  offset (0,0) -> top-left     (rows 0..3, cols 0..3)
      idx 1  color 5  offset (0,1) -> top-right    (rows 0..3, cols 4..7)
      idx 2  color 6  offset (1,0) -> bottom-left  (rows 4..7, cols 0..3)
      idx 3  color 9  offset (1,1) -> bottom-right (rows 4..7, cols 4..7)
  Each quadrant only ever contains pixels of its own colour (or background 0).
  The output (4x4) overlays the four quadrants; the generator paints them in
  order [0, 3, 2, 1] and a later painter overwrites an earlier one, so the
  effective per-cell priority (highest first) is:
      idx 1 (5)  >  idx 2 (6)  >  idx 3 (9)  >  idx 0 (4)  >  background 0
  i.e. output[r][c] = the colour of the highest-priority quadrant that has a
  pixel at (r,c), else 0.

Encoding (label-map + final Equal, Tier B, all-tiny intermediates):
  The benchmark feeds a one-hot [1,10,30,30] tensor; the 8x8 grid sits at the
  top-left.  Each quadrant only ever contains its own fixed colour, so I never
  need to recover a colour index at all -- I only need PRESENCE per quadrant.
  Presence at quadrant cell (r,c) is exactly the input one-hot channel of that
  quadrant's colour, sliced over that quadrant's region (a Slice of the FREE
  input):
      TL_p = input[:, 4:5, 0:4, 0:4]   (colour 4)
      TR_p = input[:, 5:6, 0:4, 4:8]   (colour 5)
      BL_p = input[:, 6:7, 4:8, 0:4]   (colour 6)
      BR_p = input[:, 9:10,4:8, 4:8]   (colour 9)
  Each is [1,1,4,4].  Build the label map by priority (later painter wins):
      L = 5 if TR else (6 if BL else (9 if BR else (4 if TL else 0)))
  via nested Where with constant [1,1,4,4] colour fills.  Cast presence (>0) to
  bool, Pad L to 30x30 with sentinel 10 (matches no colour channel -> all-zero
  off-grid, exactly the benchmark's off-grid encoding), and the final op
  Equal(Lp, arange[0..9]) writes straight into the FREE BOOL output, so the
  10-channel expansion never costs memory.  No Conv -> no [1,1,30,30] colour
  plane; the four presence slices stay [1,1,4,4].

  Dominant intermediate: the padded label map Lp (uint8 [1,1,30,30] = 900 B),
  irreducible because it must be 30x30 to broadcast against the 10 colour
  channels in the final Equal.  Everything else is <=64 B (the 4x4 tensors).
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

S = 4  # quadrant / output size


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # Slice the four quadrant presence masks straight from the FREE input,
    # selecting each quadrant's own colour channel over its 4x4 region.
    init("tl_s", np.array([0, 4, 0, 0], np.int64), np.int64)          # ch 4, TL
    init("tl_e", np.array([1, 5, S, S], np.int64), np.int64)
    init("tr_s", np.array([0, 5, 0, S], np.int64), np.int64)          # ch 5, TR
    init("tr_e", np.array([1, 6, S, 2 * S], np.int64), np.int64)
    init("bl_s", np.array([0, 6, S, 0], np.int64), np.int64)          # ch 6, BL
    init("bl_e", np.array([1, 7, 2 * S, S], np.int64), np.int64)
    init("br_s", np.array([0, 9, S, S], np.int64), np.int64)          # ch 9, BR
    init("br_e", np.array([1, 10, 2 * S, 2 * S], np.int64), np.int64)
    init("ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "tl_s", "tl_e", "ax"], "tl")  # [1,1,4,4] f32 presence
    n("Slice", ["input", "tr_s", "tr_e", "ax"], "tr")
    n("Slice", ["input", "bl_s", "bl_e", "ax"], "bl")
    n("Slice", ["input", "br_s", "br_e", "ax"], "br")

    # presence booleans (value > 0)
    init("half", np.array(0.5, np.float32), np.float32)
    n("Greater", ["tr", "half"], "tr_p")
    n("Greater", ["bl", "half"], "bl_p")
    n("Greater", ["br", "half"], "br_p")
    n("Greater", ["tl", "half"], "tl_p")

    # constant colour fills [1,1,4,4] uint8
    init("c4", np.full((1, 1, S, S), 4, np.uint8), np.uint8)
    init("c5", np.full((1, 1, S, S), 5, np.uint8), np.uint8)
    init("c6", np.full((1, 1, S, S), 6, np.uint8), np.uint8)
    init("c9", np.full((1, 1, S, S), 9, np.uint8), np.uint8)
    init("zero_u", np.zeros((1, 1, S, S), np.uint8), np.uint8)

    # priority overlay: TR(5) > BL(6) > BR(9) > TL(4) > 0
    n("Where", ["tl_p", "c4", "zero_u"], "L0")   # TL=4 else 0
    n("Where", ["br_p", "c9", "L0"], "L1")       # BR=9 else L0
    n("Where", ["bl_p", "c6", "L1"], "L2")       # BL=6 else L1
    n("Where", ["tr_p", "c5", "L2"], "L")        # TR=5 else L2  -> [1,1,4,4] uint8

    # pad to 30x30 with sentinel 10 (matches no colour channel -> off-grid zero)
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - S, 30 - S], np.int64), np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    n("Pad", ["L", "pads", "padval"], "Lp", mode="constant")  # [1,1,30,30] uint8

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["Lp", "chan"], "output")  # -> FREE BOOL [1,10,30,30]

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
