"""task196 (ARC-AGI 810b9b61) — recolour CLOSED box outlines blue->green.

Rule (from generator task_810b9b61.py):
  INPUT: a size x size grid (size = 3*randint(3,5) in {9,12,15}) with several
  axis-aligned rectangle OUTLINES drawn in blue (=1) on a black (=0) background.
  Boxes are pairwise 8-separated (overlaps(...,spacing=1) rejected).  Some boxes
  have a single perimeter cell knocked out to black (the "gap"/chunk).
  OUTPUT: each box outline is recoloured GREEN (=3) iff the box is a COMPLETE
  closed rectangle AND wide>=3 AND tall>=3; otherwise it stays BLUE (=1).  Flat
  boxes (wide==1 or tall==1), thin boxes (wide==2 or tall==2 -> no interior) and
  any box with a gap stay blue.  Black cells (incl. the gap) stay black.

  Closure is a PER-BOX (global) property, so it must be detected by enclosure:
    interior  = black cell that CANNOT reach the grid exterior through black
                (4/8-conn) -- a gap lets the interior leak out, a flat/thin box
                has no interior.
    green cell = blue cell 8-ADJACENT to an interior cell (corners reach the
                 centre interior diagonally, edges orthogonally).
  Verified exactly (0 mismatch / 400+ fresh) as:
    reach   = flood black inward from the canvas border (8-conn, gated by black)
    interior= black AND NOT reach
    green   = blue AND dilate8(interior)

Encoding (everything on a 15x15 WORK crop -- size<=15 always -- in fp16):
  * Slice ch0(black) & ch1(blue) to 15x15, cast fp16.
  * reach0 = black * border-mask;  10x:  reach = black * MaxPool3x3(reach).
    (8-conn flood = MaxPool 3x3, 0 params; gating by black every step blocks
     walls.  Walls are 1px blue so no diagonal corner leak.)
  * interior = black - reach;  green15 = blue * MaxPool3x3(interior).
  * Cast green15 -> uint8, Pad back to 30x30.
  * Label L (uint8, 30x30) = blue_full + 2*green30  in {0,1,3}; the free BOOL
    output = Equal(L, arange[1,10,1,1]).  (green subset of blue -> 1+2=3; black=0;
    blue-not-green=1.)  No per-cell colour Conv, no [1,10,*,*] intermediate.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
B = TensorProto.BOOL
I64 = TensorProto.INT64

WORK = 15
ITERS = 11


def build(task):
    inits, nodes = [], []

    npmap = {F32: np.float32, F16: np.float16, U8: np.uint8, I64: np.int64, B: bool}

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=npmap[dtype]), name))
        return name

    ctr = [0]

    def n(op, ins, **attrs):
        ctr[0] += 1
        out = f"{op}_{ctr[0]}"
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- slice ch1 (blue) to the 15x15 active crop ----
    init("ss1", np.array([0, 1, 0, 0], np.int64), I64)
    init("se1", np.array([1, 2, WORK, WORK], np.int64), I64)
    init("ax0", np.array([0, 1, 2, 3], np.int64), I64)
    nodes.append(helper.make_node("Slice", ["input", "ss1", "se1", "ax0"], ["blue_f32"]))
    blue = n("Cast", ["blue_f32"], to=F16)     # [1,1,15,15] fp16 {0,1}

    # notblue = 1 - blue : floodable region (in-grid bg + off-grid).  IMPORTANT:
    # off-grid one-hot cells are ALL-ZERO (ch0 is NOT set off-grid), so we must
    # flood through "not a blue wall", not through ch0=black.
    init("one_f16", np.array(1.0, np.float16), F16)
    notblue = n("Sub", ["one_f16", blue])      # [1,1,15,15] fp16 {0,1}

    # ---- seed reach = floodable cells on the 15x15 border ----
    bmask = np.zeros((1, 1, WORK, WORK), np.float16)
    bmask[0, 0, 0, :] = 1; bmask[0, 0, -1, :] = 1
    bmask[0, 0, :, 0] = 1; bmask[0, 0, :, -1] = 1
    init("bmask", bmask, F16)
    reach = n("Mul", [notblue, "bmask"])       # [1,1,15,15] fp16 seed

    # ---- flood: 8-conn dilation (MaxPool 3x3) gated by notblue, ITERS times ----
    for _ in range(ITERS):
        d = n("MaxPool", [reach], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])
        reach = n("Mul", [notblue, d])         # stay outside blue walls

    # ---- interior = notblue AND NOT reach (reach subset of notblue -> {0,1}) ----
    interior = n("Sub", [notblue, reach])      # [1,1,15,15] fp16 {0,1}

    # ---- green = blue AND dilate8(interior) ----
    di = n("MaxPool", [interior], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])
    green15 = n("Mul", [blue, di])             # [1,1,15,15] fp16 {0,1}
    green_b = n("Cast", [green15], to=B)       # [1,1,15,15] bool

    # ---- build label L on the 15x15 crop, then ONE Pad to 30x30 ----
    blue_u8 = n("Cast", [blue], to=U8)         # [1,1,15,15] uint8 {0,1}
    init("three_u8", np.array(3, np.uint8), U8)
    Lbg = n("Where", [green_b, "three_u8", blue_u8])  # {0,1,3} uint8

    # in-grid mask on the crop (off-grid cells must stay all-zero in the output):
    # in-grid <=> any input channel set <=> ch0(bg) OR ch1(blue).
    init("ssc", np.array([0, 0, 0, 0], np.int64), I64)
    init("sec", np.array([1, 1, WORK, WORK], np.int64), I64)
    nodes.append(helper.make_node("Slice", ["input", "ssc", "sec", "ax0"], ["black15_f32"]))
    black_b = n("Cast", ["black15_f32"], to=B)
    blue_b = n("Cast", [blue], to=B)
    ingrid = n("Or", [black_b, blue_b])        # [1,1,15,15] bool
    init("s99", np.array(99, np.uint8), U8)
    L15 = n("Where", [ingrid, Lbg, "s99"])     # off-grid -> 99 (matches no channel)

    # pad the crop up to 30x30 with sentinel 99 (off-canvas -> all-zero output)
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), I64)
    L = n("Pad", [L15, "pads", "s99"], mode="constant")  # [1,1,30,30] uint8

    # ---- free BOOL output = Equal(L, arange[0..9]) ----
    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, U8)
    nodes.append(helper.make_node("Equal", [L, "chan"], ["output"]))

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task196", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
