"""Task 384 (f25fbde4): 2x nearest-neighbor upscale of the yellow bounding box.

Rule (from the ARC-GEN generator): the input (always 9x9, color 4 = yellow)
contains a set of yellow pixels. Crop to their bounding box (height h, width w)
and upscale 2x so each cell becomes a 2x2 block; the output grid is 2h x 2w.

Output cell (R, C) = input(minr + R//2, minc + C//2). Inside the 2h x 2w output
box every cell is yellow (channel 4) or background (channel 0); outside the box
all channels are 0.

Memory floor-break (label map + final Equal on a small canvas):
  The output never exceeds 8x10 (bbox <= 4x5, upscaled 2x). All per-cell work
  runs on a WORK x WORK = 10x10 uint8 canvas; the label map L is Padded to
  30x30 (sentinel 10) just before the final Equal.

  * yellow mask comes from a 1x1 Conv selecting channel 4 (the 9x9 grid is
    top-left, rows/cols 9..29 are all-zero).
  * yellow row/col extent (the bbox) comes from reduce-max of that mask;
    min/max -> 2x-upscale source-index vectors base = [0,0,1,1,...] + min,
    redirected to an empty edge row/col (29) past max.
  * the mask is cast to uint8 (1 B/cell) and gathered along rows then cols on
    the 10-wide canvas, giving the 0/1 yellow stamp y.
  * L = 4 where yellow else sentinel 10; Pad to 30x30; the final op
        output = Equal(L, arange[1,10,1,1])
    writes straight into the free BOOL output (channel 4 true where yellow,
    every other channel all-false -> background, exactly the >0 scorer wants).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

WORK = 10  # working canvas side (max output dim = 8x10 -> 10)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    U8 = TensorProto.UINT8
    B = TensorProto.BOOL

    SRC = 14  # yellow-source corner (9x9 grid + empty redirect rows)
    # 2x source-index base on the WORK canvas: [0,0,1,1,2,2,...].
    base = np.repeat(np.arange((WORK + 1) // 2), 2)[:WORK].astype(np.int32)
    init("BASE", base, np.int32)                              # [WORK]
    init("RIDX", np.arange(SRC, dtype=np.int32), np.int32)    # [SRC]
    init("BIG", np.full(SRC, 999, dtype=np.int32), np.int32)
    init("NEG", np.full(SRC, -1, dtype=np.int32), np.int32)
    init("CEMP", np.full(WORK, SRC - 1, dtype=np.int32), np.int32)  # empty edge
    init("zero_f", np.array(0.0, dtype=np.float32), np.float32)

    # 2-D yellow stamp: slice input channel 4 to the SRC x SRC top-left corner
    # (the 9x9 grid is top-left, so the full yellow region fits inside SRC).
    init("y_st", np.array([4, 0, 0], np.int64), np.int64)
    init("y_en", np.array([5, SRC, SRC], np.int64), np.int64)
    init("y_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "y_st", "y_en", "y_ax"], "y_f")      # [1,1,SRC,SRC] f
    n("Cast", ["y_f"], "y_u8", to=U8)                         # uint8 mask (gather)

    # row/col yellow occupancy from the SRC corner (max>0 -> row/col has yellow).
    n("ReduceMax", ["y_f"], "rowmax", axes=[0, 1, 3], keepdims=0)  # [SRC] float
    n("ReduceMax", ["y_f"], "colmax", axes=[0, 1, 2], keepdims=0)  # [SRC] float

    def bbox(occ, lo_name, hi_name):
        n("Greater", [occ, "zero_f"], occ + "_m")             # bool [30]
        n("Where", [occ + "_m", "RIDX", "BIG"], occ + "_lo")
        n("ReduceMin", [occ + "_lo"], lo_name, axes=[0], keepdims=0)  # scalar
        n("Where", [occ + "_m", "RIDX", "NEG"], occ + "_hi")
        n("ReduceMax", [occ + "_hi"], hi_name, axes=[0], keepdims=0)  # scalar

    bbox("rowmax", "minr", "maxr")
    bbox("colmax", "minc", "maxc")

    def srcvec(mn, mx, name):
        n("Add", ["BASE", mn], name + "_raw")                 # [WORK]
        n("Greater", [name + "_raw", mx], name + "_over")     # bool [WORK]
        n("Where", [name + "_over", "CEMP", name + "_raw"], name)  # int32 [WORK]
        return name

    srcvec("minr", "maxr", "ridx")
    srcvec("minc", "maxc", "cidx")

    # gather yellow mask (uint8) -> y in {0,1} on the WORK x WORK canvas
    n("Gather", ["y_u8", "ridx"], "yr", axis=2)               # [1,1,WORK,30] u8
    n("Gather", ["yr", "cidx"], "yc", axis=3)                 # [1,1,WORK,WORK] u8

    # in-box rectangle: rows < 2h, cols < 2w (h = maxr-minr+1, top-left aligned).
    # 2h, 2w as scalar floats; compare against the WORK-length index ramp.
    init("one_f", np.array(1.0, np.float32), np.float32)
    init("ar_w", np.arange(WORK, dtype=np.float32).reshape(1, 1, WORK, 1),
         np.float32)
    init("ac_w", np.arange(WORK, dtype=np.float32).reshape(1, 1, 1, WORK),
         np.float32)
    n("Cast", ["minr"], "minrf", to=F)
    n("Cast", ["maxr"], "maxrf", to=F)
    n("Cast", ["minc"], "mincf", to=F)
    n("Cast", ["maxc"], "maxcf", to=F)
    n("Sub", ["maxrf", "minrf"], "hm1")                       # h-1
    n("Add", ["hm1", "one_f"], "h")
    n("Add", ["h", "h"], "twoh")                              # 2h
    n("Sub", ["maxcf", "mincf"], "wm1")
    n("Add", ["wm1", "one_f"], "w")
    n("Add", ["w", "w"], "twow")                              # 2w
    n("Less", ["ar_w", "twoh"], "rin")                        # [1,1,WORK,1] bool
    n("Less", ["ac_w", "twow"], "cin")                        # [1,1,1,WORK] bool
    n("And", ["rin", "cin"], "inbox")                         # [1,1,WORK,WORK] bool

    # L = 4 (yellow) else 0 (in-box bg) else sentinel 10 (outside)  uint8
    init("v0u", np.array(0, np.uint8), np.uint8)
    init("v4", np.array(4, np.uint8), np.uint8)
    init("v10", np.array(10, np.uint8), np.uint8)
    n("Where", ["inbox", "v0u", "v10"], "Lbg")               # 0 in box else 10
    n("Cast", ["yc"], "yb", to=B)                             # bool yellow (0/1)
    n("Where", ["yb", "v4", "Lbg"], "Lwk")                   # yellow overrides

    # pad to 30x30 (sentinel 10), then final Equal -> free BOOL output
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    n("Pad", ["Lwk", "padpads", "padval"], "L", mode="constant")  # [1,1,30,30]
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task384", [x], [y], inits)
    return helper.make_model(
        g, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
