"""task353 (ARC-AGI dc433765) — move the green pixel one step toward the yellow pixel.

Rule (from the generator):
  Input has exactly two coloured pixels: a green (3) at (gr,gc) and a yellow (4)
  at (yr,yc).  Chebyshev distance(green,yellow) >= 2 (the generator excludes
  "too close" placements).  Output: the yellow pixel stays put; the green pixel
  moves ONE cell toward the yellow along each axis independently:
      ng_r = gr + sign(yr - gr),  ng_c = gc + sign(yc - gc).
  Everything else is background (channel 0) inside the HxW grid; off-grid cells
  are all-channels-off.

Approach (Tier A, small working canvas, no full/10-ch planes materialized):
  The grid is anchored top-left, at most height 14 x width 12.  We never slice
  the 10-channel input into a plane (that would cost >=3600B).  Instead we use
  cheap 1-D per-channel profiles:
    - rowprof = ReduceSum(input,[3]) -> [1,10,30,1] (1200B);  the only sizeable
      intermediate.  colprof = ReduceSum(input,[2]) -> [1,10,1,30].
    - Slice channels 3 (green) & 4 (yellow) from these tiny profiles, and the
      coordinate of each single pixel = Sum(ramp * profile).
    - H,W from occupancy ReduceMax over [1,3] / [1,2].
    - dr=sign(yr-gr), dc=sign(yc-gc); ng_r=gr+dr, ng_c=gc+dc.
  Then build separable row/col one-hots on a WR(=14) x WC(=12) canvas, assemble
  the label map L (10 off-grid / 4 yellow / 3 green / 0 background), Pad to
  30x30 (sentinel 10), output = Equal(L, arange) -> free BOOL one-hot.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
I64 = TensorProto.INT64
B = TensorProto.BOOL

WR = 14   # working rows  (height <= 14)
WC = 12   # working cols  (width  <= 12)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- green/yellow row & col signatures as ONE no-pad Conv each ------------
    # rowsig[1,1,30,1] = 1*(green in row r) + 100*(yellow in row r).  Kernel
    # contracts the channel axis (ch3 weight 1, ch4 weight 100, else 0) and the
    # full column axis in one Conv -> avoids the [1,10,30,1] profile (1200B).
    Wr = np.zeros((1, 10, 1, 30), np.float32); Wr[0, 3] = 1.0; Wr[0, 4] = 100.0
    Wc = np.zeros((1, 10, 30, 1), np.float32); Wc[0, 3] = 1.0; Wc[0, 4] = 100.0
    init("Wr", Wr, np.float32)
    init("Wc", Wc, np.float32)
    n("Conv", ["input", "Wr"], "rowsigF")                       # [1,1,30,1] f32
    n("Conv", ["input", "Wc"], "colsigF")                       # [1,1,1,30] f32
    n("Cast", ["rowsigF"], "rowsig", to=F16)                    # fp16 (values<=101 exact)
    n("Cast", ["colsigF"], "colsig", to=F16)

    rrF = np.arange(30, dtype=np.float16).reshape(1, 1, 30, 1)
    rcF = np.arange(30, dtype=np.float16).reshape(1, 1, 1, 30)
    init("rrF", rrF, np.float16)
    init("rcF", rcF, np.float16)

    init("zero", np.array(0.0, np.float32), np.float32)
    init("fifty", np.array(50.0, np.float16), np.float16)
    init("c100", np.array(100.0, np.float16), np.float16)

    # decompose each signature (sig = greenprof + 100*yellowprof, profiles 0/1):
    #   yellow coord yc = Sum(yellowprof*ramp);  total = Sum(sig*ramp) = green + 100*yc
    #   => green coord = total - 100*yc.  Avoids materialising the green profile.
    def coord(sig, ramp, axis, tag):
        n("Greater", [sig, "fifty"], "yp_" + tag)               # yellow profile bool
        n("Cast", ["yp_" + tag], "ypf_" + tag, to=F16)          # 0/1 fp16
        n("Mul", ["ypf_" + tag, ramp], "ym_" + tag)
        n("ReduceSum", ["ym_" + tag], "yc_" + tag, axes=[axis], keepdims=1)
        n("Mul", [sig, ramp], "sm_" + tag)
        n("ReduceSum", ["sm_" + tag], "tot_" + tag, axes=[axis], keepdims=1)
        n("Mul", ["yc_" + tag, "c100"], "y100_" + tag)
        n("Sub", ["tot_" + tag, "y100_" + tag], "gc_" + tag)    # green coord
        return "yc_" + tag, "gc_" + tag
    yr, gr = coord("rowsig", "rrF", 2, "r")                     # yellow/green row
    yc, gc = coord("colsig", "rcF", 3, "c")                     # yellow/green col
    # coords stay fp16 (values<30, integer-exact); alias to fixed names
    n("Identity", [yr], "yr"); n("Identity", [gr], "gr")
    n("Identity", [yc], "yc"); n("Identity", [gc], "gc")

    # ---- in-grid mask from per-row/col occupancy (sliced to WR x WC) ----------
    # The grid is a solid origin-anchored rectangle, so occupied rows/cols are a
    # contiguous prefix; slicing the occupancy vectors to WRxWC gives the in-grid
    # mask directly (no H/W counts / Less needed).
    n("ReduceMax", ["input"], "rowocc", axes=[1, 3], keepdims=1)  # [1,1,30,1] f32
    n("ReduceMax", ["input"], "colocc", axes=[1, 2], keepdims=1)  # [1,1,1,30] f32
    n("Greater", ["rowocc", "zero"], "rowany30")                  # [1,1,30,1] bool
    n("Greater", ["colocc", "zero"], "colany30")                  # [1,1,1,30] bool
    init("z0", np.array([0], np.int64), np.int64)
    init("wrI", np.array([WR], np.int64), np.int64)
    init("wcI", np.array([WC], np.int64), np.int64)
    init("ax2", np.array([2], np.int64), np.int64)
    init("ax3", np.array([3], np.int64), np.int64)
    n("Slice", ["rowany30", "z0", "wrI", "ax2"], "rowin")         # [1,1,WR,1] bool
    n("Slice", ["colany30", "z0", "wcI", "ax3"], "colin")         # [1,1,1,WC] bool

    # ---- dr=sign(yr-gr), dc=sign(yc-gc); new green position (fp16) ------------
    init("zero16", np.array(0.0, np.float16), np.float16)
    init("one", np.array(1.0, np.float16), np.float16)
    init("none", np.array(-1.0, np.float16), np.float16)
    n("Sub", ["yr", "gr"], "dyr")
    n("Greater", ["dyr", "zero16"], "drp"); n("Less", ["dyr", "zero16"], "drn")
    n("Where", ["drp", "one", "zero16"], "dr1"); n("Where", ["drn", "none", "dr1"], "dr")
    n("Sub", ["yc", "gc"], "dyc")
    n("Greater", ["dyc", "zero16"], "dcp"); n("Less", ["dyc", "zero16"], "dcn")
    n("Where", ["dcp", "one", "zero16"], "dc1"); n("Where", ["dcn", "none", "dc1"], "dc")
    n("Add", ["gr", "dr"], "ngr")                               # [1,1,1,1] new green row
    n("Add", ["gc", "dc"], "ngc")

    # ---- working-canvas (WR x WC) ramps & one-hots (fp16) ---------------------
    rr = np.arange(WR, dtype=np.float16).reshape(1, 1, WR, 1)
    rc = np.arange(WC, dtype=np.float16).reshape(1, 1, 1, WC)
    init("rr", rr, np.float16)                                  # [1,1,WR,1]
    init("rc", rc, np.float16)                                  # [1,1,1,WC]

    # yellow / green cell one-hots
    n("Equal", ["rr", "yr"], "yrow1"); n("Equal", ["rc", "yc"], "ycol1")
    n("Equal", ["rr", "ngr"], "grow1"); n("Equal", ["rc", "ngc"], "gcol1")

    # ---- label map L on WR x WC (priority Where chain) ------------------------
    init("u0", np.array(0, np.uint8), np.uint8)
    init("u3", np.array(3, np.uint8), np.uint8)
    init("u4", np.array(4, np.uint8), np.uint8)
    init("u10", np.array(10, np.uint8), np.uint8)
    n("And", ["rowin", "colin"], "ingrid")                      # [1,1,WR,WC] bool
    n("Where", ["ingrid", "u0", "u10"], "Lbg")                  # [1,1,WR,WC] u8
    n("And", ["yrow1", "ycol1"], "ycell")
    n("Where", ["ycell", "u4", "Lbg"], "Ly")
    n("And", ["grow1", "gcol1"], "gcell")
    n("Where", ["gcell", "u3", "Ly"], "Lw")                     # [1,1,WR,WC] u8

    # ---- pad to 30x30 (sentinel 10) and final Equal ---------------------------
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 30 - WR, 30 - WC], np.int64), np.int64)
    n("Pad", ["Lw", "padpads", "u10"], "L", mode="constant")    # [1,1,30,30] u8
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                         # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task353", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
