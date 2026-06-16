"""task310 (ARC-AGI c909285e) — crop the box region to the top-left.

Rule (from the generator):
  A `size`x`size` grid (size 20..30) is filled with 3..4 "wires": for each
  (colour, spacing) every cell with (r+1)%spacing==0 OR (c+1)%spacing==0 gets
  that colour (later wires overwrite earlier).  Then a square box of side
  `boxlength` (5..8) at (boxrow, boxcol) has its PERIMETER drawn in `boxcolor`
  (a colour NOT used by any wire) on the grid.
  Output = the box subregion grid[boxrow:boxrow+L, boxcol:boxcol+L] (L=boxlength)
  cropped to the top-left of a fresh grid.  Inside the box the cells are wire
  colours plus the boxcolor perimeter, so the per-cell colour value must be
  carried (arbitrary per-instance colours).

  Identification: wire colours fill entire rows/cols, so their bbox span is
  ~size (>=15).  boxcolor only spans the box (span = L-1 <= 7).  So
  boxcolor = the non-background colour with the MINIMUM bbox span (task036
  min-span idiom), and (boxrow, boxcol, L) = (minrow, mincol, rowspan+1) of
  that colour.

Encoding (Tier B — min-span colour id + data-dependent crop window):
  1. Per-channel 1-D occupancy profiles -> per-channel (minr, minc, span).
     Mask ch0 + absent channels to +BIG span; ArgMin -> boxcolor.
  2. Gather boxcolor's (minr, minc, span) -> (boxrow, boxcol, L=span+1).
  3. colf[1,1,30,30] f32 = sum_k k*input_k (per-cell colour index 0..9).
     Gather a WORK=8 x WORK window starting at (boxrow, boxcol).
  4. Label L_u8 = colour where (r<L and c<L) else sentinel 10; Pad to 30x30.
  5. output = Equal(L, arange[0..9]) -> free BOOL one-hot output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
I64 = TensorProto.INT64
BOOL = TensorProto.BOOL

N = 30
WORK = 8  # max boxlength


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    BIG = 1000.0

    # ---- per-channel occupancy profiles --------------------------------------
    n("ReduceMax", ["input"], "rowocc", axes=[3], keepdims=1)  # [1,10,30,1] f32
    n("ReduceMax", ["input"], "colocc", axes=[2], keepdims=1)  # [1,10,1,30] f32
    n("ReduceMax", ["rowocc"], "anyc", axes=[2], keepdims=1)   # [1,10,1,1] f32

    init("half", np.array(0.5, np.float32), np.float32)
    n("Greater", ["rowocc", "half"], "rowb")                  # bool [1,10,30,1]
    n("Greater", ["colocc", "half"], "colb")                  # bool [1,10,1,30]

    # row / col index ramps (fp16: values <30 and +/-BIG(=1000) are fp16-exact)
    ramp_r = np.arange(30, dtype=np.float16).reshape(1, 1, 30, 1)
    ramp_c = np.arange(30, dtype=np.float16).reshape(1, 1, 1, 30)
    init("rr", ramp_r, np.float16)
    init("rc", ramp_c, np.float16)
    init("PBIG", np.array(BIG, np.float16), np.float16)
    init("NBIG", np.array(-BIG, np.float16), np.float16)

    # min/max occupied row & col per channel
    n("Where", ["rowb", "rr", "PBIG"], "rmin_src")
    n("ReduceMin", ["rmin_src"], "rmin", axes=[2], keepdims=1)   # [1,10,1,1]
    n("Where", ["rowb", "rr", "NBIG"], "rmax_src")
    n("ReduceMax", ["rmax_src"], "rmax", axes=[2], keepdims=1)   # [1,10,1,1]
    n("Where", ["colb", "rc", "PBIG"], "cmin_src")
    n("ReduceMin", ["cmin_src"], "cmin", axes=[3], keepdims=1)   # [1,10,1,1]
    n("Where", ["colb", "rc", "NBIG"], "cmax_src")
    n("ReduceMax", ["cmax_src"], "cmax", axes=[3], keepdims=1)   # [1,10,1,1]

    # spans
    n("Sub", ["rmax", "rmin"], "rspan")                       # [1,10,1,1] fp16
    n("Sub", ["cmax", "cmin"], "cspan")
    n("Max", ["rspan", "cspan"], "span")                      # [1,10,1,1] fp16

    # mask out absent channels and channel 0 -> span := +BIG
    n("Greater", ["anyc", "half"], "present")                 # [1,10,1,1] bool
    ch0kill = np.zeros((1, 10, 1, 1), np.bool_)
    ch0kill[0, 0, 0, 0] = True
    init("ch0kill", ch0kill, np.bool_)
    n("Not", ["ch0kill"], "notch0")
    n("And", ["present", "notch0"], "valid")
    n("Where", ["valid", "span", "PBIG"], "span2")            # [1,10,1,1] fp16

    # ---- boxcolor = argmin span over channel axis ----------------------------
    n("ArgMin", ["span2"], "bc_i", axis=1, keepdims=1)        # [1,1,1,1] int64

    init("shp10", np.array([1, 10], np.int64), np.int64)
    init("shp11", np.array([1, 1], np.int64), np.int64)
    init("shp1", np.array([1], np.int64), np.int64)

    n("Reshape", ["rmin", "shp10"], "rmin10")                 # [1,10] fp16
    n("Reshape", ["cmin", "shp10"], "cmin10")
    n("Reshape", ["bc_i", "shp11"], "bc11")                   # [1,1] int64
    n("GatherElements", ["rmin10", "bc11"], "minr16", axis=1)  # [1,1] fp16
    n("GatherElements", ["cmin10", "bc11"], "minc16", axis=1)
    n("Cast", ["minr16"], "minr", to=F32)                     # [1,1] f32
    n("Cast", ["minc16"], "minc", to=F32)

    # L = span+1 of boxcolor (square box, rowspan==colspan==L-1)
    n("Reshape", ["span", "shp10"], "span10")                 # [1,10] fp16
    n("GatherElements", ["span10", "bc11"], "Lm116", axis=1)  # [1,1] fp16 (L-1)
    n("Cast", ["Lm116"], "Lm1", to=F32)
    init("one", np.array(1.0, np.float32), np.float32)
    n("Add", ["Lm1", "one"], "L")                             # [1,1] f32 boxlength

    # ---- per-cell colour-index plane (1x1 Conv sum_k k*input_k) --------------
    w = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("convw", w, np.float32)
    n("Conv", ["input", "convw"], "colf")                     # [1,1,30,30] f32

    # ---- gather a WORK x WORK window starting at (boxrow, boxcol) ------------
    baseW = np.arange(WORK, dtype=np.float32)
    init("baseW", baseW, np.float32)                          # [WORK]
    init("c0", np.array(0.0, np.float32), np.float32)
    init("c29", np.array(29.0, np.float32), np.float32)

    n("Reshape", ["minr", "shp1"], "minr_s")                  # [1]
    n("Add", ["baseW", "minr_s"], "ridx_f")                   # [WORK]
    n("Clip", ["ridx_f", "c0", "c29"], "ridx_cl")
    n("Cast", ["ridx_cl"], "ridx", to=I64)                    # [WORK] int64

    n("Reshape", ["minc", "shp1"], "minc_s")                  # [1]
    n("Add", ["baseW", "minc_s"], "cidx_f")
    n("Clip", ["cidx_f", "c0", "c29"], "cidx_cl")
    n("Cast", ["cidx_cl"], "cidx", to=I64)                    # [WORK] int64

    n("Gather", ["colf", "ridx"], "Vr", axis=2)               # [1,1,WORK,30] f32
    n("Gather", ["Vr", "cidx"], "Vw", axis=3)                 # [1,1,WORK,WORK] f32
    n("Cast", ["Vw"], "Vu8", to=U8)                           # [1,1,WORK,WORK] u8

    # ---- box mask (r < L) and (c < L) on WORK x WORK canvas ------------------
    rampw_r = np.arange(WORK, dtype=np.float32).reshape(1, 1, WORK, 1)
    rampw_c = np.arange(WORK, dtype=np.float32).reshape(1, 1, 1, WORK)
    init("wr", rampw_r, np.float32)
    init("wc", rampw_c, np.float32)
    n("Less", ["wr", "L"], "rmask")                           # [1,1,WORK,1] bool
    n("Less", ["wc", "L"], "cmask")                           # [1,1,1,WORK] bool
    n("And", ["rmask", "cmask"], "boxmask")                   # [1,1,WORK,WORK] bool

    # ---- label map: colour inside box, sentinel 10 outside -------------------
    init("u10", np.array(10, np.uint8), np.uint8)
    n("Where", ["boxmask", "Vu8", "u10"], "Lw")               # [1,1,WORK,WORK] u8
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, N - WORK, N - WORK], np.int64), np.int64)
    n("Pad", ["Lw", "padpads", "u10"], "Lfull", mode="constant")  # [1,1,30,30]

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["Lfull", "chan"], "output")                   # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, N, N])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, N, N])
    g = helper.make_graph(nodes, "task310", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
