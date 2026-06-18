"""task143 (ARC-AGI 63613498) — recolour the placed sprite that matches the
corner reference shape to gray.

Rule (from the generator, size=10 grid):
  A gray (5) L-marker is drawn at grid[i][3]=grid[3][i] for i in 0..3.
  3..5 monochrome "creature" sprites (3-4 contiguous pixels, each fitting in a
  3x3 box anchored at (0,0)) are placed at non-overlapping positions on the
  10x10 grid with DISTINCT colours (excluding gray and one special `bcolor`).
  Constraint: no sprite sits with both brow<5 and bcol<5 (top-left kept clear).
  Sprite 0 is special: its raw creature shape is ALSO drawn in `bcolor` at the
  raw coords (always inside the top-left 3x3 corner rows0-2/cols0-2), and its
  PLACED copy is recoloured gray in the OUTPUT.
  INPUT vs OUTPUT differ ONLY at sprite-0's placed cells: input has colour
  colors[0], output has gray(5).
  Each colour appears in at most one sprite, and creatures are unique, so
  sprite 0 is THE unique placed sprite whose shape matches the corner ref R.

Encoding (active region is only 10x10 — slice first; foreground = channels 1..9,
the bg channel 0 dropped; gray channel 5 kept as a candidate harmlessly since
its 7-cell marker fails the K-pixel count test):
  cand  = Slice(input, ch 1:10, rows/cols 0:10) fp32 [1,9,10,10] -> Cast fp16
  R     = ReduceSum over channels of cand[:, :, 0:3, 0:3]  -> [1,1,3,3] fp16
          (the top-left 3x3 corner holds only the bcolor reference; marker is at
           row3/col3, outside it);  K = ReduceSum(R) = ref pixel count.
  corr  = grouped Conv(cand, R replicated to [9,1,3,3], pad bottom/right=2)
          -> [1,9,10,10] = per-channel overlap of R-at-(i,j) with that sprite.
  corr2 = corr - BIG * cornermask   (kill the reference's own (0,0) self-match)
  has_match = ReduceMax(corr2, spatial) == K    -> [1,9,1,1] bool
  count_ok  = ReduceSum(cand, spatial)  == K     -> [1,9,1,1] bool
              (rejects a 4-px sprite that merely CONTAINS the 3-px R, and the
               7-cell gray marker channel)
  target_ch = has_match AND count_ok            -> [1,9,1,1] bool
  Each colour is one sprite, so the matched channel's full 10x10 plane IS the
  target mask:  mask = Conv(cand, target_ch as a [1,9,1,1] selector weight).
  Pad mask to 30x30 (uint8) and route the recolour into the FREE 30x30 output:
  output = Where(mask30, gray_onehot[1,10,1,1], input[1,10,30,30]).

Dominant intermediate: the fp32 9-channel 10x10 entry slice (3600B, irreducible
— Slice keeps fp32) and the per-channel corr / corr2 planes (1800B fp16 each,
intrinsic to the shape correlation).  All else <=1800B.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I64 = TensorProto.INT64

NCH = 9          # foreground channels 1..9
BIG = np.float16(1000.0)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- crop to fg channels 1..9 of the 10x10 active region, work in fp16 --
    init("s1", np.array([1, 0, 0], np.int64), np.int64)
    init("e10", np.array([10, 10, 10], np.int64), np.int64)
    init("ax123", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "s1", "e10", "ax123"], "inner")         # f32 [1,9,10,10]
    n("Cast", ["inner"], "cand", to=F16)                         # f16 [1,9,10,10]

    # ---- reference shape R from the top-left 3x3 corner ---------------------
    init("c0", np.array([0, 0], np.int64), np.int64)
    init("c3", np.array([3, 3], np.int64), np.int64)
    init("ax23", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["cand", "c0", "c3", "ax23"], "corner")           # f16 [1,9,3,3]
    n("ReduceSum", ["corner"], "R", axes=[1], keepdims=1)        # f16 [1,1,3,3]
    n("ReduceSum", ["R"], "K", axes=[1, 2, 3], keepdims=1)       # f16 [1,1,1,1]

    init("reps", np.array([NCH, 1, 1, 1], np.int64), np.int64)
    n("Tile", ["R", "reps"], "R8")                               # f16 [9,1,3,3]

    # per-channel correlation; pad bottom/right so window-top-left aligns to (i,j)
    n("Conv", ["cand", "R8"], "corr", group=NCH, kernel_shape=[3, 3],
      pads=[0, 0, 2, 2])                                          # f16 [1,9,10,10]

    # kill the reference's own self-match at window (0,0)
    cmask = np.zeros((1, 1, 10, 10), np.float16)
    cmask[0, 0, 0, 0] = BIG
    init("cmask", cmask, np.float16)
    n("Sub", ["corr", "cmask"], "corr2")                         # f16 [1,9,10,10]
    n("ReduceMax", ["corr2"], "best", axes=[2, 3], keepdims=1)   # f16 [1,9,1,1]
    n("Equal", ["best", "K"], "has_match")                       # bool [1,9,1,1]

    # channel pixel count must equal K (rejects R-supersets + gray marker)
    n("ReduceSum", ["cand"], "chcount", axes=[2, 3], keepdims=1)  # f16 [1,9,1,1]
    n("Equal", ["chcount", "K"], "count_ok")                     # bool [1,9,1,1]

    n("And", ["has_match", "count_ok"], "target_ch_b")           # bool [1,9,1,1]
    n("Cast", ["target_ch_b"], "target_ch", to=F16)             # f16 [1,9,1,1]

    # select matched channel's full plane: 1x1 conv with runtime selector weight
    n("Conv", ["cand", "target_ch"], "mask_f", kernel_shape=[1, 1])  # f16 [1,1,10,10]
    init("half", np.array(0.5, np.float16), np.float16)
    n("Greater", ["mask_f", "half"], "mask_b")                   # bool [1,1,10,10]
    n("Cast", ["mask_b"], "mask_u8", to=U8)                      # u8  [1,1,10,10]

    # pad the 10x10 mask up to the full 30x30 canvas
    init("pad30", np.array([0, 0, 0, 0, 0, 0, 20, 20], np.int64), np.int64)
    init("u0", np.array(0, np.uint8), np.uint8)
    n("Pad", ["mask_u8", "pad30", "u0"], "mask30_u8", mode="constant")  # u8 [1,1,30,30]
    n("Cast", ["mask30_u8"], "mask30", to=BOOL)                  # bool [1,1,30,30]

    # recolour matched cells to gray(5); everything else copies the free input
    gray = np.zeros((1, 10, 1, 1), np.float32)
    gray[0, 5, 0, 0] = 1.0
    init("gray_oh", gray, np.float32)
    n("Where", ["mask30", "gray_oh", "input"], "output")        # f32 [1,10,30,30]

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task143", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
