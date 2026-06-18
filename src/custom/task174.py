"""task174 (ARC-AGI 72ca375d) — crop the horizontally-symmetric box.

Rule (from the generator, verified):
  The 10x10 grid holds exactly 3 monochrome "boxes" (creatures) in 3 distinct
  colours.  Box-0 (colors[0]) is built to be BOTH horizontally(column)-mirror
  symmetric AND 180-rotationally symmetric; boxes 1,2 are built to be NEITHER.
  The output is box-0 cropped tight to its bounding box, placed at the top-left
  corner of a fresh grid (channel-0 fills the holes inside the HxW bbox, every
  cell outside the bbox is all-channels-off).

  Key invariant (verified 0/8000 fresh): box-0 is the UNIQUE colour whose
  bbox-cropped shape equals its own horizontal mirror.  Tested in closed form
  with a per-channel reflection MatMul: reflect each channel's columns about its
  own axis a = c0 + c1 (Cmat[k,c',c] = (c'+c == a_k)); the channel where the
  reflected plane equals the original (and c != background) is box-0.

Pipeline (ONNX, opset 11):
  1. Slice input to the 10x10 active region A [1,10,10,10].
  2. Per-channel col bbox (cmin,cmax) and row bbox (rmin,rmax) from 1-D occupancy
     profiles; a = cmin+cmax.
  3. Reflection matrix Cmat[1,10,10,10] = Equal(c'+c, a); Mf = MatMul(A, Cmat);
     sym = ReduceSum(|A-Mf|,[2,3]) == 0.  box0color = the present (c!=0) channel
     whose shape is symmetric.
  4. Gather that channel's plane, shift a WORK=5 window to the top-left by
     Gather(axis=2, arange+rmin) then Gather(axis=3, arange+cmin).
  5. Label map L uint8: box0color on the box cells, 0 inside the HxW bbox, 10
     outside; Pad to 30x30 (sentinel 10); Equal(L, arange[0..9]) -> BOOL output.
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

ACT = 10   # active region size (grid is always 10x10)
WORK = 5   # box bbox is always <= 5x5


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

    # ---- slice to the 10x10 active region ------------------------------------
    init("s_start", np.array([0, 0], np.int64), np.int64)
    init("s_end", np.array([ACT, ACT], np.int64), np.int64)
    init("s_axes", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["input", "s_start", "s_end", "s_axes"], "A32")  # [1,10,10,10] f32
    n("Cast", ["A32"], "A", to=F16)                         # [1,10,10,10] fp16

    # ---- per-channel occupancy profiles (fp16) -------------------------------
    n("ReduceMax", ["A"], "rowocc", axes=[3], keepdims=1)   # [1,10,10,1] fp16
    n("ReduceMax", ["A"], "colocc", axes=[2], keepdims=1)   # [1,10,1,10] fp16
    n("ReduceMax", ["rowocc"], "anyc", axes=[2], keepdims=1)  # [1,10,1,1] fp16

    init("half", np.array(0.5, np.float16), np.float16)
    n("Greater", ["rowocc", "half"], "rowb")                # bool [1,10,10,1]
    n("Greater", ["colocc", "half"], "colb")                # bool [1,10,1,10]

    ramp_r = np.arange(ACT, dtype=np.float16).reshape(1, 1, ACT, 1)
    ramp_c = np.arange(ACT, dtype=np.float16).reshape(1, 1, 1, ACT)
    init("rr", ramp_r, np.float16)                          # [1,1,10,1]
    init("rc", ramp_c, np.float16)                          # [1,1,1,10]
    init("PBIG", np.array(BIG, np.float16), np.float16)
    init("NBIG", np.array(-BIG, np.float16), np.float16)

    # min/max occupied row & col per channel
    n("Where", ["rowb", "rr", "PBIG"], "rmin_src")          # [1,10,10,1]
    n("ReduceMin", ["rmin_src"], "rmin", axes=[2], keepdims=1)  # [1,10,1,1]
    n("Where", ["rowb", "rr", "NBIG"], "rmax_src")
    n("ReduceMax", ["rmax_src"], "rmax", axes=[2], keepdims=1)  # [1,10,1,1]
    n("Where", ["colb", "rc", "PBIG"], "cmin_src")          # [1,10,1,10]
    n("ReduceMin", ["cmin_src"], "cmin", axes=[3], keepdims=1)  # [1,10,1,1]
    n("Where", ["colb", "rc", "NBIG"], "cmax_src")
    n("ReduceMax", ["cmax_src"], "cmax", axes=[3], keepdims=1)  # [1,10,1,1]

    # ---- reflection axis a = cmin + cmax (per channel) -----------------------
    n("Add", ["cmin", "cmax"], "axis_a")                    # [1,10,1,1] f32

    # ---- reflection matrix Cmat[1,10,10,10] = Equal(c'+c, a) -----------------
    cp = np.arange(ACT, dtype=np.float16).reshape(1, 1, ACT, 1)
    cc = np.arange(ACT, dtype=np.float16).reshape(1, 1, 1, ACT)
    init("cp", cp, np.float16)                              # [1,1,10,1]
    init("cc", cc, np.float16)                              # [1,1,1,10]
    n("Add", ["cp", "cc"], "cpc")                           # [1,1,10,10] fp16
    n("Equal", ["cpc", "axis_a"], "Cmat_b")                 # [1,10,10,10] bool
    n("Cast", ["Cmat_b"], "Cmat", to=F16)                   # [1,10,10,10] fp16

    # ---- Mf = A @ Cmat  (column reflection per channel) ----------------------
    # MatMul batches over [1,10]; contracts last axis of A with 2nd-last of Cmat.
    # Cmat is symmetric in (c',c) so Cmat == Cmat^T.  All values are tiny
    # integers (sums of <=10 ones) so fp16 is exact.
    n("MatMul", ["A", "Cmat"], "Mf")                        # [1,10,10,10] fp16

    # ---- symmetry per channel: overlap(A,Mf) == count(A) ---------------------
    # A and Mf are binary with the same pixel count (reflection preserves count),
    # so the masks are equal iff their overlap equals the count.  Uses ONE extra
    # full plane (A*Mf) instead of two (diff + diff^2).
    n("Mul", ["A", "Mf"], "AMf")                            # [1,10,10,10] fp16
    n("ReduceSum", ["AMf"], "overlap", axes=[2, 3], keepdims=1)   # [1,10,1,1]
    n("ReduceSum", ["A"], "cntA", axes=[2, 3], keepdims=1)        # [1,10,1,1]
    n("Equal", ["overlap", "cntA"], "is_sym")               # [1,10,1,1] bool

    # present (c has pixels) and c != background channel 0
    n("Greater", ["anyc", "half"], "present")               # [1,10,1,1] bool
    ch0kill = np.zeros((1, 10, 1, 1), np.bool_)
    ch0kill[0, 0, 0, 0] = True
    init("ch0kill", ch0kill, np.bool_)
    n("Not", ["ch0kill"], "notch0")
    n("And", ["present", "notch0"], "valid")                # [1,10,1,1] bool
    n("And", ["is_sym", "valid"], "is_box0")                # unique True channel

    # box0color = the channel index where is_box0 is True (argmax over channel)
    n("Cast", ["is_box0"], "is_box0_f", to=F16)             # [1,10,1,1] fp16
    n("ArgMax", ["is_box0_f"], "bc_i", axis=1, keepdims=1)  # [1,1,1,1] int64
    n("Cast", ["bc_i"], "bcf", to=F32)                      # box0color as float

    # ---- gather min_row / min_col / spans of the box0 channel ----------------
    init("shp10", np.array([1, 10], np.int64), np.int64)
    init("shp11", np.array([1, 1], np.int64), np.int64)
    n("Reshape", ["rmin", "shp10"], "rmin10")               # [1,10]
    n("Reshape", ["cmin", "shp10"], "cmin10")
    n("Reshape", ["bc_i", "shp11"], "bc11")                 # [1,1]
    n("GatherElements", ["rmin10", "bc11"], "minr", axis=1)  # [1,1] f32
    n("GatherElements", ["cmin10", "bc11"], "minc", axis=1)

    # spans (H-1, W-1)
    n("Sub", ["rmax", "rmin"], "rspan")                     # [1,10,1,1]
    n("Sub", ["cmax", "cmin"], "cspan")
    n("Reshape", ["rspan", "shp10"], "rspan10")
    n("Reshape", ["cspan", "shp10"], "cspan10")
    n("GatherElements", ["rspan10", "bc11"], "Hm1", axis=1)  # [1,1] fp16
    n("GatherElements", ["cspan10", "bc11"], "Wm1", axis=1)
    init("one", np.array(1.0, np.float16), np.float16)
    n("Add", ["Hm1", "one"], "H")                           # [1,1] fp16
    n("Add", ["Wm1", "one"], "W")

    # ---- box0 plane = the box0 colour channel, shift to top-left -------------
    init("shp1d", np.array([1], np.int64), np.int64)
    n("Reshape", ["bc_i", "shp1d"], "bc1")                  # [1] int64
    n("Gather", ["A", "bc1"], "bplane", axis=1)             # [1,1,10,10] f32 mask

    baseW = np.arange(WORK, dtype=np.float16)
    init("baseW", baseW, np.float16)                        # [WORK]
    init("shp1", np.array([1], np.int64), np.int64)
    init("c0", np.array(0.0, np.float16), np.float16)
    init("clo", np.array(float(ACT - 1), np.float16), np.float16)
    n("Reshape", ["minr", "shp1"], "minr_s")                # [1]
    n("Add", ["baseW", "minr_s"], "ridx_f")                 # [WORK]
    n("Clip", ["ridx_f", "c0", "clo"], "ridx_cl")
    n("Cast", ["ridx_cl"], "ridx", to=I64)                  # [WORK] int64
    n("Reshape", ["minc", "shp1"], "minc_s")                # [1]
    n("Add", ["baseW", "minc_s"], "cidx_f")
    n("Clip", ["cidx_f", "c0", "clo"], "cidx_cl")
    n("Cast", ["cidx_cl"], "cidx", to=I64)                  # [WORK] int64

    n("Gather", ["bplane", "ridx"], "Vr", axis=2)           # [1,1,WORK,10] f32
    n("Gather", ["Vr", "cidx"], "Vs", axis=3)               # [1,1,WORK,WORK] f32

    # ---- bbox mask (r < H) and (c < W) on the WORK x WORK canvas -------------
    rampw_r = np.arange(WORK, dtype=np.float16).reshape(1, 1, WORK, 1)
    rampw_c = np.arange(WORK, dtype=np.float16).reshape(1, 1, 1, WORK)
    init("wr", rampw_r, np.float16)                         # [1,1,WORK,1]
    init("wc", rampw_c, np.float16)                         # [1,1,1,WORK]
    n("Less", ["wr", "H"], "rmask")                         # [1,1,WORK,1] bool
    n("Less", ["wc", "W"], "cmask")                         # [1,1,1,WORK] bool
    n("And", ["rmask", "cmask"], "boxmask")                 # [1,1,WORK,WORK] bool

    # ---- box mask M = (Vs > 0.5) AND boxmask ---------------------------------
    n("Cast", ["bcf"], "bc_u8", to=U8)                      # box0color uint8 [1,1,1,1]
    n("Greater", ["Vs", "half"], "iseq")                    # [1,1,WORK,WORK] bool
    n("And", ["iseq", "boxmask"], "M")                      # box0 cells

    # ---- label map L (WORK x WORK): box0color on M, 0 inside box, 10 outside -
    init("u0", np.array(0, np.uint8), np.uint8)
    init("u10", np.array(10, np.uint8), np.uint8)
    n("Where", ["M", "bc_u8", "u0"], "Lin")                 # [1,1,WORK,WORK] u8
    n("Where", ["boxmask", "Lin", "u10"], "Lw")            # outside box -> 10

    # ---- pad to 30x30 (sentinel 10) and final Equal --------------------------
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    n("Pad", ["Lw", "padpads", "u10"], "L", mode="constant")  # [1,1,30,30] u8
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                     # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task174", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
