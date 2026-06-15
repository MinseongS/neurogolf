"""task342 (ARC-AGI d89b689b) — "pull the 4 quadrant colors onto the cyan box".

Rule (from the generator, grid is exactly 10x10):
  A 2x2 cyan(8) box sits at (brow,bcol)..(brow+1,bcol+1), brow,bcol in [2,7].
  Four single colored pixels (colors != cyan, all distinct) are scattered, one
  in each quadrant relative to the box:
      colors[0] = the pixel with row<brow  and col<bcol    (top-left)
      colors[1] = the pixel with row<brow  and col>bcol+1  (top-right)
      colors[2] = the pixel with row>brow+1 and col<bcol   (bottom-left)
      colors[3] = the pixel with row>brow+1 and col>bcol+1 (bottom-right)
  Output is all background(0) except the 2x2 box cells, filled by quadrant:
      output[brow  ][bcol  ] = colors[0]   output[brow  ][bcol+1] = colors[1]
      output[brow+1][bcol  ] = colors[2]   output[brow+1][bcol+1] = colors[3]

Encoding (data-dependent scatter on the 10x10 active canvas):
  * colorV[1,1,10,10] = sum_k w[k]*input[k] with w[k]=k except w[0]=w[8]=0, so
    cyan + background vanish and each colored pixel keeps its digit (1x1 Conv).
  * brow/bcol = min cyan row/col (ReduceMax presence -> ramp/Where/ReduceMin).
  * 1-D row/col bounds (arange compared to scalars): rowlt=row<brow,
    rowgt=row>brow+1, collt=col<bcol, colgt=col>bcol+1.  Each quadrant color =
    ReduceMax over the doubly-masked colorV (exactly one nonzero pixel there).
  * Corner equality masks rEQ0=row==brow, rEQ1=row==brow+1, cEQ0=col==bcol,
    cEQ1=col==bcol+1 (separable).  L = sum of (quadcolor * cornermask), a 10x10
    float label plane; cast uint8; Pad to 30x30 with sentinel 10; final
    Equal(L, arange[0..9]) -> free BOOL output (in-grid bg=0->ch0 on, off-grid
    sentinel 10 -> all channels off, matching the harness's unset off-grid cells).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64

W = 10  # grid is exactly 10x10 for this task


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- slice the active 10x10 corner (all 10 channels) -------------------
    init("in_s", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("in_e", np.array([1, 10, W, W], np.int64), np.int64)
    init("in_ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "in_s", "in_e", "in_ax"], "in10")   # [1,10,10,10] f32

    # ---- colorV: digit value per cell, cyan(8) & bg(0) zeroed ---------------
    w = np.arange(10, dtype=np.float32)
    w[8] = 0.0
    init("colW", w.reshape(1, 10, 1, 1), np.float32)         # [1,10,1,1]
    n("Conv", ["in10", "colW"], "colorV")                    # [1,1,10,10] f32

    # ---- cyan plane (channel 8) for box location ---------------------------
    init("cy_s", np.array([8, 0, 0], np.int64), np.int64)
    init("cy_e", np.array([9, W, W], np.int64), np.int64)
    init("cy_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["in10", "cy_s", "cy_e", "cy_ax"], "cyan")    # [1,1,10,10] f32

    init("ZEROF", np.array(0.0, np.float32), np.float32)
    init("BIG", np.array(1e6, np.float32), np.float32)
    ax2 = init("ax2", np.arange(W, dtype=np.float32).reshape(1, 1, W, 1), np.float32)
    ax3 = init("ax3", np.arange(W, dtype=np.float32).reshape(1, 1, 1, W), np.float32)

    # min cyan index along an axis -> box top-left scalar
    def min_index(axis_keep, tag):
        red_axes = [3] if axis_keep == 2 else [2]
        n("ReduceMax", ["cyan"], f"pres_{tag}", axes=red_axes, keepdims=1)
        n("Greater", [f"pres_{tag}", "ZEROF"], f"presb_{tag}")
        ramp = "ax2" if axis_keep == 2 else "ax3"
        n("Where", [f"presb_{tag}", ramp, "BIG"], f"idx_{tag}")
        n("ReduceMin", [f"idx_{tag}"], f"min_{tag}", axes=[2, 3], keepdims=1)
        return f"min_{tag}"  # [1,1,1,1] f32

    min_index(2, "br")  # brow scalar
    min_index(3, "bc")  # bcol scalar

    init("ONEF", np.array(1.0, np.float32), np.float32)
    n("Add", ["min_br", "ONEF"], "br1")   # brow+1
    n("Add", ["min_bc", "ONEF"], "bc1")   # bcol+1

    # ---- 1-D quadrant bounds (broadcast), as fp16 multipliers --------------
    def bound(op, ref, tag):
        n(op, ["ax2" if tag.startswith("row") else "ax3", ref], f"{tag}_b")
        n("Cast", [f"{tag}_b"], tag, to=F16)
        return tag

    bound("Less", "min_br", "rowlt")     # row<brow      [1,1,W,1] f16
    bound("Greater", "br1", "rowgt")     # row>brow+1    [1,1,W,1] f16
    bound("Less", "min_bc", "collt")     # col<bcol      [1,1,1,W] f16
    bound("Greater", "bc1", "colgt")     # col>bcol+1    [1,1,1,W] f16

    # ---- collapse colorV to top/bottom col-profiles (1-D), then split L/R --
    # exactly one colored pixel per quadrant; TL&TR sit on rows<brow but in
    # disjoint col bands, so the col-profile of the top band carries both.
    n("Cast", ["colorV"], "colorV16", to=F16)               # [1,1,W,W] f16
    n("Mul", ["colorV16", "rowlt"], "top")                  # [1,1,W,W] f16
    n("Mul", ["colorV16", "rowgt"], "bot")                  # [1,1,W,W] f16
    n("ReduceMax", ["top"], "topcol", axes=[2], keepdims=1)  # [1,1,1,W] f16
    n("ReduceMax", ["bot"], "botcol", axes=[2], keepdims=1)  # [1,1,1,W] f16

    def lr_color(prof, cmask, tag):
        n("Mul", [prof, cmask], f"p_{tag}")                  # [1,1,1,W] f16
        n("ReduceMax", [f"p_{tag}"], f"col_{tag}", axes=[3], keepdims=1)
        return f"col_{tag}"                                  # [1,1,1,1] f16

    lr_color("topcol", "collt", "TL")
    lr_color("topcol", "colgt", "TR")
    lr_color("botcol", "collt", "BL")
    lr_color("botcol", "colgt", "BR")

    # ---- corner equality masks (separable) ---------------------------------
    def eqmask(ax, ref, tag):
        n("Equal", [ax, ref], f"{tag}_b")
        n("Cast", [f"{tag}_b"], tag, to=F16)
        return tag

    eqmask("ax2", "min_br", "rEQ0")   # row==brow    [1,1,W,1] f16
    eqmask("ax2", "br1", "rEQ1")       # row==brow+1
    eqmask("ax3", "min_bc", "cEQ0")    # col==bcol    [1,1,1,W] f16
    eqmask("ax3", "bc1", "cEQ1")       # col==bcol+1

    # toprow[1,1,1,W] = cEQ0*TLcol + cEQ1*TRcol ; botrow likewise.
    def make_row(cEQ0, c0, cEQ1, c1, tag):
        n("Mul", [cEQ0, c0], f"a_{tag}")        # [1,1,1,W] f16
        n("Mul", [cEQ1, c1], f"b_{tag}")        # [1,1,1,W] f16
        n("Add", [f"a_{tag}", f"b_{tag}"], f"row_{tag}")
        return f"row_{tag}"

    make_row("cEQ0", "col_TL", "cEQ1", "col_TR", "top")
    make_row("cEQ0", "col_BL", "cEQ1", "col_BR", "bot")

    # L = rEQ0 (x) toprow + rEQ1 (x) botrow  (outer-product broadcast Mul)
    n("Mul", ["rEQ0", "row_top"], "Lt")     # [1,1,W,W] f16
    n("Mul", ["rEQ1", "row_bot"], "Lb")     # [1,1,W,W] f16
    n("Add", ["Lt", "Lb"], "Lf")            # [1,1,W,W] f16
    n("Cast", ["Lf"], "L10", to=U8)         # [1,1,W,W] uint8

    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), np.int64)
    init("V10", np.array(10, np.uint8), np.uint8)
    n("Pad", ["L10", "pads", "V10"], "L", mode="constant")  # [1,1,30,30] uint8

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                     # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task342", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
