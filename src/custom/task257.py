"""Task 257 (a68b268e): overlay 4 colored quadrants into one 4x4 grid.

Rule (from ARC-GEN generator):
  Input is a (2*size+1)x(2*size+1) = 9x9 grid (size=4) split by a blue cross
  (row 4, col 4) into four 4x4 quadrants, each holding ONE fixed colour:
    TL (rows 0-3, cols 0-3) = colour 7
    TR (rows 0-3, cols 5-8) = colour 4
    BL (rows 5-8, cols 0-3) = colour 8
    BR (rows 5-8, cols 5-8) = colour 6
  Output is 4x4 where each cell takes the colour of the FIRST non-background
  quadrant in priority order TL(7) > TR(4) > BL(8) > BR(6) (the generator writes
  idx 3,2,1,0 in that loop order, so idx 0 = colour 7 overwrites last = wins).

Encoding (pure {0,1} masks -> bool whole-pipeline, bool FREE output):
  The harness scores (out > 0), so the output is a pure one-hot mask and dtype
  is irrelevant.  Slice the 4 quadrant colour-channels from the FREE fp32 input,
  Cast each to BOOL (one-hot is exactly {0,1}, loss-free, 4x smaller than fp32),
  then build the priority overlay with And/Or/Not (all valid on bool under
  ORT_DISABLE_ALL):
    out7 = a                       (TL wins outright)
    out4 = b & ~a
    out8 = c & ~(a|b)
    out6 = d & ~(a|b|c)
  Concat the 10 output channels (the 6 unused colours are a zero [1,1,4,4] bool
  const) into [1,10,4,4] bool, Pad to 30x30 -> the FREE bool output.  Everything
  after the 4 fp32 slices runs in bool (itemsize 1), halving the fp16 net.
"""

import numpy as np
import onnx
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

    # axis const for channel/row/col slices
    init("ax", np.array([1, 2, 3], np.int64), np.int64)

    # quadrant slices: (channel, row0, col0)
    quads = {
        "a": (7, 0, 0),  # TL colour 7
        "b": (4, 0, 5),  # TR colour 4
        "c": (8, 5, 0),  # BL colour 8
        "d": (6, 5, 5),  # BR colour 6
    }
    for name, (ch, r0, c0) in quads.items():
        init(f"st_{name}", np.array([ch, r0, c0], np.int64), np.int64)
        init(f"en_{name}", np.array([ch + 1, r0 + S, c0 + S], np.int64), np.int64)
        n("Slice", ["input", f"st_{name}", f"en_{name}", "ax"], f"f_{name}")  # [1,1,4,4] f32
        n("Cast", [f"f_{name}"], name, to=TensorProto.BOOL)                   # [1,1,4,4] bool

    # priority overlay (bool ops)
    n("Or", ["a", "b"], "ab")          # a|b
    n("Or", ["ab", "c"], "abc")        # a|b|c
    n("Or", ["abc", "d"], "abcd")      # a|b|c|d  (any quadrant present)
    n("Not", ["a"], "na")
    n("Not", ["ab"], "nab")
    n("Not", ["abc"], "nabc")
    n("Not", ["abcd"], "o0")           # ch0 background = no quadrant present
    out7 = "a"
    n("And", ["b", "na"], "o4")
    n("And", ["c", "nab"], "o8")
    n("And", ["d", "nabc"], "o6")

    # zero uint8 plane for unused channels (ORT Pad rejects bool, accepts uint8)
    init("z", np.zeros((1, 1, S, S), np.uint8), np.uint8)

    # Cast the active bool planes to uint8 for Concat/Pad
    n("Cast", ["o0"], "u0", to=TensorProto.UINT8)
    n("Cast", [out7], "u7", to=TensorProto.UINT8)
    n("Cast", ["o4"], "u4", to=TensorProto.UINT8)
    n("Cast", ["o8"], "u8", to=TensorProto.UINT8)
    n("Cast", ["o6"], "u6", to=TensorProto.UINT8)

    # channels 0..9 ; ch0 background, 4->u4, 6->u6, 7->u7, 8->u8, rest zero
    chans = ["u0", "z", "z", "z", "u4", "z", "u6", "u7", "u8", "z"]
    n("Concat", chans, "cat10", axis=1)  # [1,10,4,4] uint8

    # pad the 4x4 block out to 30x30 -> FREE uint8 output
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - S, 30 - S], np.int64), np.int64)
    init("pv", np.array(0, np.uint8), np.uint8)
    n("Pad", ["cat10", "pads", "pv"], "output", mode="constant")

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
