"""Task 169 (6e82a1ae): recolour each gray sprite by 5 - (its pixel count).

Rule (from ARC-GEN generator):
  A 10x10 grid (inside the 30x30 canvas) holds 4-6 small gray (colour 5) sprites.
  Sprite bounding boxes are placed with spacing>=1, so the minimum chebyshev
  distance between gray cells of DIFFERENT sprites is 2 (measured) -- i.e. sprites
  are 8-connected-isolated.  Each sprite has 2, 3 or 4 gray pixels; every pixel of
  a sprite is recoloured  5 - count:
      count 2 -> colour 3
      count 3 -> colour 2
      count 4 -> colour 1

Encoding (closed-form, no flood-fill, tiny 10x10 working planes):
  Because sprites are isolated by a >=2 chebyshev gap, every 3x3-window operation
  on a gray cell only ever touches that cell's OWN sprite -- leak-free.  So we can
  build leak-free, multi-resolution local "mass" features by repeated gray-gated
  3x3 sum convolution:
      F1 = (3x3-sum of gray) * gray          # degree+1 per cell
      F2 = (3x3-sum of F1)   * gray          # radius-2 mass (still leak-free)
  Their per-sprite MAXima (broadcast across the sprite by 2 gray-gated MaxPool3x3
  iterations -- the max sits at a sprite-central cell so all cells are within 2 hops)
  form a fingerprint that separates the three counts exactly (verified 0/3000 fresh):
      M1 := max F1 over sprite ;  M2 := max F2 over sprite
      count 2  <=>  M1 == 2
      count 4  <=>  M1 >= 4   OR   M2 == 8     (M2==8 is the I-tetromino case)
      count 3  <=>  everything else (gray)
  The 10-channel one-hot is routed straight into the FREE uint8 output via a Where
  priority chain, then Pad'd from 10x10 to 30x30.  No [1,10,30,30] plane and no
  per-cell colour-index plane is ever materialised; all working planes are 10x10.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

N = 10          # channels
S = 10          # active grid (generator size)
GRAY = 5


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F16 = TensorProto.FLOAT16

    # --- slice gray channel (5), rows 0..S, cols 0..S -> [1,1,S,S] fp32 ---
    init("sl_ax", np.array([1, 2, 3], np.int64), np.int64)
    init("sl_st", np.array([GRAY, 0, 0], np.int64), np.int64)
    init("sl_en", np.array([GRAY + 1, S, S], np.int64), np.int64)
    n("Slice", ["input", "sl_st", "sl_en", "sl_ax"], "gray_f32")  # [1,1,S,S] f32
    n("Cast", ["gray_f32"], "gray", to=F16)                       # fp16 working dtype

    # 3x3 all-ones sum kernel (SAME pad), fp16
    init("k33", np.ones((1, 1, 3, 3), np.float16), np.float16)

    # F1 = (3x3 sum of gray) * gray
    n("Conv", ["gray", "k33"], "s1", pads=[1, 1, 1, 1])
    n("Mul", ["s1", "gray"], "F1")
    # F2 = (3x3 sum of F1) * gray
    n("Conv", ["F1", "k33"], "s2", pads=[1, 1, 1, 1])
    n("Mul", ["s2", "gray"], "F2")

    # broadcast per-sprite max of F1, F2 via 2 gray-gated MaxPool3x3 iterations
    # (the max value sits at a sprite-central cell, so every cell is within 2 hops;
    #  verified exact 0/3000)
    cur1, cur2 = "F1", "F2"
    for i in range(2):
        p1 = n("MaxPool", [cur1], "mp1_%d" % i, kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        cur1 = n("Mul", [p1, "gray"], "M1_%d" % i)
        p2 = n("MaxPool", [cur2], "mp2_%d" % i, kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        cur2 = n("Mul", [p2, "gray"], "M2_%d" % i)
    M1, M2 = cur1, cur2

    # classification masks (fp16 Equal/Greater are integer-exact here)
    init("c2", np.array(2.0, np.float16), np.float16)
    init("c35", np.array(3.5, np.float16), np.float16)
    init("c8", np.array(8.0, np.float16), np.float16)

    n("Equal", [M1, "c2"], "is2")                       # M1 == 2  -> count 2
    n("Greater", [M1, "c35"], "m1ge4")                  # M1 >= 4
    n("Equal", [M2, "c8"], "m2eq8")                     # M2 == 8 (I-tetromino)
    n("Or", ["m1ge4", "m2eq8"], "is4")                  # count 4
    n("Cast", ["gray"], "graybool", to=TensorProto.BOOL)

    # build a SINGLE colour-index plane [1,1,S,S] (uint8) via a Where priority chain
    #   bg -> 0 ; gray default -> 2 (count3) ; is4 -> 1 (count4) ; is2 -> 3 (count2)
    init("v0", np.array(0, np.uint8), np.uint8)
    init("v1", np.array(1, np.uint8), np.uint8)
    init("v2", np.array(2, np.uint8), np.uint8)
    init("v3", np.array(3, np.uint8), np.uint8)
    n("Where", ["graybool", "v2", "v0"], "idxA")        # [1,1,S,S] u8
    n("Where", ["is4", "v1", "idxA"], "idxB")
    n("Where", ["is2", "v3", "idxB"], "idx")            # [1,1,S,S] u8

    # pad index plane 10x10 -> 30x30 with 0, then Equal vs colour arange -> bool output
    init("pad_pads",
         np.array([0, 0, 0, 0, 0, 0, 30 - S, 30 - S], np.int64), np.int64)
    # pad OFF-GRID with sentinel 99 so it matches NO colour channel (all-zero off-grid,
    # exactly like the benchmark); in-grid bg cells keep idx 0 -> channel-0 one-hot.
    init("pad_val", np.array(99, np.uint8), np.uint8)
    n("Pad", ["idx", "pad_pads", "pad_val"], "idx30", mode="constant")  # [1,1,30,30] u8
    init("arange", np.arange(N, dtype=np.uint8).reshape(1, N, 1, 1), np.uint8)
    n("Equal", ["idx30", "arange"], "output")           # [1,10,30,30] bool

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
