"""Task 189 (ARC-AGI 7c008303): quadrant-legend recolor of a green stamp.

Generator (size is always 6 -> input 9x9, output 6x6):
  A 2x2 legend of four colors sits in a corner, separated from a 6x6 region by a
  cyan(8) cross at index 2 (a full cyan row + full cyan column). Each green(3)
  pixel in the 6x6 region is recolored by the QUADRANT it falls in: the cell at
  (r>=3, c>=3) of the legend, with colors[2*(r>=3)+(c>=3)]. Then a global
  flip_horiz / flip_vert is applied to BOTH grid and output identically. Legend
  colors are sampled from {1,2,4,5,6,7,9} (1..9 minus green/cyan), so channel 0
  (bg) carries no information.

Read in the flipped frame (flips act identically on grid and output):
  vflip <=> cyan(8) at (6,0); hflip <=> cyan(8) at (0,6).
  legend corner rows/cols = {0,1} (no flip) or {7,8} (flip);
  green block rows/cols    = {3..8} (no flip) or {0..5} (flip);
  out[R][C] = legend[R//3][C//3] if green at the matching block cell, else 0.

Floor-break (vs the 16.15 / 16.40 builds): never form the 10-channel 9x9 corner.
Three cheap single-purpose reads instead:
  * green9  = channel-3 slice [1,1,9,9] -> green presence (block via sr@.@sc).
  * cyan9   = channel-8 slice [1,1,9,9] -> the two flip bits (cells 6,0 / 0,6).
  * legtop/legbot = two 2-row x 9-col channel-1..9 strips (162B each) ->
    channel-collapsed legend planes; pick rows by vflip, cols by hflip.
Everything downstream is fp16 (color indices <=9 are exact); the 2x2 legend is
expanded to the 6x6 quadrant grid, gated by green, sentinel-padded to 30x30 and
emitted via Equal straight into the FREE bool output.
Verified exact on all 266 stored examples and fresh arc-gen instances.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F = TensorProto.FLOAT
H = TensorProto.FLOAT16
U8 = TensorProto.UINT8
B = TensorProto.BOOL

SRC = 9  # the legend + green region live entirely in the top-left 9x9 corner


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype=np.float32):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    # ---- single-channel reads (green = ch3, cyan = ch8) over the 9x9 corner ----
    init("g_st", np.array([3, 0, 0], np.int64), np.int64)
    init("g_en", np.array([4, SRC, SRC], np.int64), np.int64)
    init("g_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "g_st", "g_en", "g_ax"], "green9f")  # [1,1,9,9] fp32
    n("Cast", ["green9f"], "green9", to=H)                    # fp16
    # ---- flip bits: cyan(8) at (6,0) <=> vflip, at (0,6) <=> hflip.
    #      slice exactly those two single cells (4B each) ----
    init("cv_st", np.array([8, 6, 0], np.int64), np.int64)
    init("cv_en", np.array([9, 7, 1], np.int64), np.int64)
    n("Slice", ["input", "cv_st", "cv_en", "g_ax"], "cy_v")  # [1,1,1,1] cyan(6,0)
    init("ch_st", np.array([8, 0, 6], np.int64), np.int64)
    init("ch_en", np.array([9, 1, 7], np.int64), np.int64)
    n("Slice", ["input", "ch_st", "ch_en", "g_ax"], "cy_h")  # [1,1,1,1] cyan(0,6)
    init("half", np.array(0.5, np.float32))
    n("Greater", ["cy_v", "half"], "vbool")
    n("Greater", ["cy_h", "half"], "hbool")
    n("Cast", ["vbool"], "vfi", to=TensorProto.INT64)
    n("Cast", ["hbool"], "hfi", to=TensorProto.INT64)
    n("ReduceSum", ["vfi"], "vflip", axes=[0, 1, 2, 3], keepdims=0)  # scalar 0/1
    n("ReduceSum", ["hfi"], "hflip", axes=[0, 1, 2, 3], keepdims=0)  # scalar 0/1

    # ---- legend: slice the FOUR 2x2 corner blocks (channels 1..9), collapse
    #      each to a color-index 2x2, pick the legend's corner by (vflip,hflip) ----
    Wk = np.arange(1, 10, dtype=np.float32).reshape(1, 9, 1, 1)
    init("Wk", Wk)
    corners = [("tl", 0, 0), ("tr", 0, 7), ("bl", 7, 0), ("br", 7, 7)]
    kc_names = []
    for name, r0, c0 in corners:
        init(f"{name}_st", np.array([1, r0, c0], np.int64), np.int64)
        init(f"{name}_en", np.array([10, r0 + 2, c0 + 2], np.int64), np.int64)
        n("Slice", ["input", f"{name}_st", f"{name}_en", "g_ax"], f"{name}c")  # [1,9,2,2]
        n("Conv", [f"{name}c", "Wk"], f"{name}kf")               # [1,1,2,2] fp32
        n("Cast", [f"{name}kf"], f"{name}k", to=H)               # [1,1,2,2] fp16
        kc_names.append(f"{name}k")
    n("Concat", kc_names, "kstack", axis=0)                      # [4,1,2,2] fp16
    # corner index = vflip*2 + hflip  (tl=0, tr=1, bl=2, br=3)
    init("two", np.array(2, np.int64), np.int64)
    n("Mul", ["vflip", "two"], "v2")                             # [1]
    n("Add", ["v2", "hflip"], "cidx")                            # [1] in 0..3
    n("Gather", ["kstack", "cidx"], "L22idx", axis=0)            # [1,1,2,2] fp16

    # ---- expand legend 2x2 -> 6x6 quadrant index (fp16) ----
    Kr = np.zeros((6, 2), np.float16); Kc = np.zeros((2, 6), np.float16)
    for R in range(6):
        Kr[R, R // 3] = 1.0
        Kc[R // 3, R] = 1.0
    init("Kr", Kr, np.float16); init("Kc", Kc, np.float16)
    n("MatMul", ["Kr", "L22idx"], "lemid")                       # [1,1,6,2] fp16
    n("MatMul", ["lemid", "Kc"], "leidx")                        # [1,1,6,6] fp16

    # ---- 6x6 green block: gb = (green presence at the block cell) ----
    Sr = np.zeros((2, 6, SRC), np.float16); Sc = np.zeros((2, SRC, 6), np.float16)
    for R in range(6):
        Sr[0, R, R + 3] = 1.0   # no vflip: green rows 3..8
        Sr[1, R, R] = 1.0       # vflip:    green rows 0..5
        Sc[0, R + 3, R] = 1.0   # no hflip: green cols 3..8
        Sc[1, R, R] = 1.0       # hflip:    green cols 0..5
    init("SrB", Sr, np.float16); init("ScB", Sc, np.float16)
    n("Gather", ["SrB", "vflip"], "sr", axis=0)                  # [6,9] fp16
    n("Gather", ["ScB", "hflip"], "sc", axis=0)                  # [9,6] fp16
    n("MatMul", ["sr", "green9"], "grow")                        # [1,1,6,9] fp16
    n("MatMul", ["grow", "sc"], "gblk")                          # [1,1,6,6] fp16
    init("half16", np.array(0.5, np.float16), np.float16)
    n("Greater", ["gblk", "half16"], "gb")                       # [1,1,6,6] bool

    # ---- L = legend color where green, else 0; pad -> 30x30; final Equal ----
    n("Cast", ["leidx"], "leidx_u8", to=U8)
    init("v0", np.array(0, np.uint8), np.uint8)
    n("Where", ["gb", "leidx_u8", "v0"], "Lwk")                  # [1,1,6,6] uint8
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 24, 24], np.int64), np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    n("Pad", ["Lwk", "padpads", "padval"], "L", mode="constant")  # [1,1,30,30]
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task189", [x], [y], inits)
    return helper.make_model(
        g, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
