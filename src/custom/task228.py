"""Task 228 (ARC-AGI 952a094c) — box with interior corner pixels ejected.

Rule (from the generator):
  A hollow rectangle of `bcolor` (outline ring; interior all black) sits in the
  10x10 grid.  Inside, at the 4 interior corners, are 4 distinct colours:
    colors[0] at (r0+1, c0+1)   (interior top-left)
    colors[1] at (r0+1, c1-1)   (interior top-right)
    colors[2] at (r1-1, c0+1)   (interior bottom-left)
    colors[3] at (r1-1, c1-1)   (interior bottom-right)
  where the box outer bounds are rows [r0,r1], cols [c0,c1].

  Output: the SAME hollow ring (interior all black, the interior corner pixels
  cleared) plus the 4 colours EJECTED to the 4 *outer* diagonal corners, each
  moving to the diagonally OPPOSITE outer corner (point-reflection through the
  box centre):
    (r0-1, c0-1) = colors[3]   (interior BR  -> outer TL)
    (r0-1, c1+1) = colors[2]   (interior BL  -> outer TR)
    (r1+1, c0-1) = colors[1]   (interior TR  -> outer BL)
    (r1+1, c1+1) = colors[0]   (interior TL  -> outer BR)

Lean reconstruction (NO 30x30 colour plane):
  The whole output is reconstructible from a handful of SCALARS:
    * bcolor       = ArgMax over channel pixel-counts (ch0 masked).
    * r0,r1,c0,c1  = min/max index of the bcolor channel's row/col occupancy.
    * 4 corner colours = the unique non-bg, non-bcolor colour present in each
      (top|bottom row-band) x (left|right col-band) combination.
  Box bounds + corner colours both come from the two per-channel occupancy
  reductions ReduceMax(input,[3]) [1,10,30,1] and ReduceMax(input,[2]) [1,10,1,30]
  (1200B each) — NO 3600B fp32 [1,1,30,30] colour Conv is ever materialised.
  The label map L is then assembled on a WORK=10 uint8 canvas (separable ring
  frame from 1-D bounds + 4 single-cell outer-corner Wheres), Pad'ed to 30x30
  with sentinel 10, and emitted as output = Equal(L, arange) (opset 11, BOOL).
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

WORK = 10  # grid is always 10x10 at the top-left of the 30x30 canvas


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- per-channel occupancy over the FREE input (fp32 reductions) --------
    n("ReduceMax", ["input"], "rowocc", axes=[3], keepdims=1)  # [1,10,30,1] f32
    n("ReduceMax", ["input"], "colocc", axes=[2], keepdims=1)  # [1,10,1,30] f32

    # ---- bcolor = argmax over channels 1..9 of cell counts ------------------
    n("ReduceSum", ["input"], "counts", axes=[2, 3], keepdims=1)  # [1,10,1,1]
    init("mask01", np.array([0] + [1] * 9, np.float32).reshape(1, 10, 1, 1), np.float32)
    n("Mul", ["counts", "mask01"], "counts1")
    n("ArgMax", ["counts1"], "bidx", axis=1, keepdims=1)        # [1,1,1,1] int64
    n("Squeeze", ["bidx"], "bidx_s", axes=[0, 1, 2, 3])        # scalar int64
    n("Cast", ["bidx"], "bcolor_u", to=U8)                     # uint8 scalar [1,1,1,1]

    # ---- bcolor-channel row/col occupancy (Gather one channel), sliced to 10 -
    # rowocc[1,10,30,1] -> pick channel bidx along axis=1 -> [1,30,1]
    n("Gather", ["rowocc", "bidx_s"], "browocc_g", axis=1)     # [1,30,1] f32
    n("Unsqueeze", ["browocc_g"], "browocc", axes=[0])         # [1,1,30,1]
    n("Gather", ["colocc", "bidx_s"], "bcolocc_g", axis=1)     # [1,1,30] f32
    n("Unsqueeze", ["bcolocc_g"], "bcolocc", axes=[0])         # [1,1,1,30]

    init("half32", np.array(0.5, np.float32), np.float32)
    n("Greater", ["browocc", "half32"], "browb30")            # [1,1,30,1] bool
    n("Greater", ["bcolocc", "half32"], "bcolb30")            # [1,1,1,30] bool
    init("s0", np.array([0], np.int64), np.int64)
    init("sW", np.array([WORK], np.int64), np.int64)
    init("ax2", np.array([2], np.int64), np.int64)
    init("ax3", np.array([3], np.int64), np.int64)
    n("Slice", ["browb30", "s0", "sW", "ax2"], "browb")       # [1,1,W,1] bool
    n("Slice", ["bcolb30", "s0", "sW", "ax3"], "bcolb")       # [1,1,1,W] bool

    # ---- box bounds r0,r1,c0,c1 (scalars, fp16) over the WORK=10 ramps -------
    init("wr", np.arange(WORK, dtype=np.float16).reshape(1, 1, WORK, 1), np.float16)
    init("wc", np.arange(WORK, dtype=np.float16).reshape(1, 1, 1, WORK), np.float16)
    init("PBIG", np.array(100.0, np.float16), np.float16)
    init("NBIG", np.array(-100.0, np.float16), np.float16)

    n("Where", ["browb", "wr", "PBIG"], "r0src")
    n("ReduceMin", ["r0src"], "r0", axes=[2], keepdims=1)     # [1,1,1,1] f16
    n("Where", ["browb", "wr", "NBIG"], "r1src")
    n("ReduceMax", ["r1src"], "r1", axes=[2], keepdims=1)
    n("Where", ["bcolb", "wc", "PBIG"], "c0src")
    n("ReduceMin", ["c0src"], "c0", axes=[3], keepdims=1)
    n("Where", ["bcolb", "wc", "NBIG"], "c1src")
    n("ReduceMax", ["c1src"], "c1", axes=[3], keepdims=1)

    # ---- interior / exterior corner row & col indices (scalars) -------------
    init("One16", np.array(1.0, np.float16), np.float16)
    n("Add", ["r0", "One16"], "ri_t")   # interior top row    = r0+1
    n("Sub", ["r1", "One16"], "ri_b")   # interior bottom row = r1-1
    n("Add", ["c0", "One16"], "ci_l")   # interior left  col  = c0+1
    n("Sub", ["c1", "One16"], "ci_r")   # interior right col  = c1-1
    n("Sub", ["r0", "One16"], "ro_t")   # outer top row    = r0-1
    n("Add", ["r1", "One16"], "ro_b")   # outer bottom row = r1+1
    n("Sub", ["c0", "One16"], "co_l")   # outer left  col  = c0-1
    n("Add", ["c1", "One16"], "co_r")   # outer right col  = c1+1

    # ---- 4 corner colours (channel presence at interior corner row/col) -----
    # present_top[k] = colour k present in row (r0+1); from rowocc Gather'd at
    # that row.  Likewise bottom/left/right.  bcolor & ch0 masked out so only the
    # 4 distinct corner colours survive.  A colour is in the TL interior corner
    # iff it is in the top colour-row AND the left colour-col, etc.
    n("Cast", ["ri_t"], "ri_t_i", to=I64); n("Squeeze", ["ri_t_i"], "ri_t_s", axes=[0, 1, 2, 3])
    n("Cast", ["ri_b"], "ri_b_i", to=I64); n("Squeeze", ["ri_b_i"], "ri_b_s", axes=[0, 1, 2, 3])
    n("Cast", ["ci_l"], "ci_l_i", to=I64); n("Squeeze", ["ci_l_i"], "ci_l_s", axes=[0, 1, 2, 3])
    n("Cast", ["ci_r"], "ci_r_i", to=I64); n("Squeeze", ["ci_r_i"], "ci_r_s", axes=[0, 1, 2, 3])

    # presence vectors : Gather (scalar idx) drops the gathered spatial axis ->
    # [1,10,1]; Unsqueeze back to [1,10,1,1] so broadcasts stay aligned.
    n("Gather", ["rowocc", "ri_t_s"], "ptop_r", axis=2); n("Unsqueeze", ["ptop_r"], "ptop_g", axes=[2])
    n("Gather", ["rowocc", "ri_b_s"], "pbot_r", axis=2); n("Unsqueeze", ["pbot_r"], "pbot_g", axes=[2])
    n("Gather", ["colocc", "ci_l_s"], "plft_r", axis=3); n("Unsqueeze", ["plft_r"], "plft_g", axes=[3])
    n("Gather", ["colocc", "ci_r_s"], "prgt_r", axis=3); n("Unsqueeze", ["prgt_r"], "prgt_g", axes=[3])

    # mask: exclude ch0 (background) and the bcolor channel (bool, [1,10,1,1])
    init("chan_u8", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    init("zero_u8", np.array(0, np.uint8), np.uint8)
    n("Equal", ["chan_u8", "zero_u8"], "is_ch0")              # [1,10,1,1] bool
    n("Equal", ["chan_u8", "bcolor_u"], "is_bc")
    n("Or", ["is_ch0", "is_bc"], "drop_ch")
    n("Not", ["drop_ch"], "keep_ch")                          # [1,10,1,1] bool

    # presence -> bool, then masked once: kpt = keep AND present.
    n("Greater", ["ptop_g", "half32"], "ptop_b")
    n("Greater", ["pbot_g", "half32"], "pbot_b")
    n("Greater", ["plft_g", "half32"], "plft_b")
    n("Greater", ["prgt_g", "half32"], "prgt_b")
    n("And", ["ptop_b", "keep_ch"], "ptop_k")                 # [1,10,1,1] bool
    n("And", ["pbot_b", "keep_ch"], "pbot_k")
    # corner colour: channel where (rowband AND colband) -> sum of k over that
    # single surviving channel, computed in fp16 (chan ramp <16, exact).
    init("chan16", np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1), np.float16)

    init("zero16c", np.array(0.0, np.float16), np.float16)

    def corner(rowk, colb, name):
        n("And", [rowk, colb], name + "_ind")                 # [1,10,1,1] bool
        n("Where", [name + "_ind", "chan16", "zero16c"], name + "_w")  # [1,10,1,1] f16
        n("ReduceSum", [name + "_w"], name + "_v", axes=[1], keepdims=1)  # [1,1,1,1] f16
        return name + "_v"

    cTL = corner("ptop_k", "plft_b", "ctl")   # colors[0]
    cTR = corner("ptop_k", "prgt_b", "ctr")   # colors[1]
    cBL = corner("pbot_k", "plft_b", "cbl")   # colors[2]
    cBR = corner("pbot_k", "prgt_b", "cbr")   # colors[3]

    # ---- build uint8 label map L on WORK=10 canvas (reuse wr/wc ramps) ------
    def eq_row(val, name):  # row index == val  -> [1,1,W,1] bool (fp16 Equal exact)
        n("Equal", ["wr", val], name)
        return name

    def eq_col(val, name):  # col index == val  -> [1,1,1,W] bool
        n("Equal", ["wc", val], name)
        return name

    def le_row(lo, hi, name):  # lo <= row <= hi -> [1,1,W,1] bool
        n("Less", ["wr", lo], name + "_lt"); n("Less", [hi, "wr"], name + "_gt")
        n("Or", [name + "_lt", name + "_gt"], name + "_out"); n("Not", [name + "_out"], name)
        return name

    def le_col(lo, hi, name):  # lo <= col <= hi -> [1,1,1,W] bool
        n("Less", ["wc", lo], name + "_lt"); n("Less", [hi, "wc"], name + "_gt")
        n("Or", [name + "_lt", name + "_gt"], name + "_out"); n("Not", [name + "_out"], name)
        return name

    # ring frame = (row in {r0,r1} AND c0<=col<=c1) OR (col in {c0,c1} AND r0<=row<=r1)
    r_is_r0 = eq_row("r0", "r_r0")
    r_is_r1 = eq_row("r1", "r_r1")
    c_is_c0 = eq_col("c0", "c_c0")
    c_is_c1 = eq_col("c1", "c_c1")
    r_in = le_row("r0", "r1", "r_in")
    c_in = le_col("c0", "c1", "c_in")
    n("Or", [r_is_r0, r_is_r1], "r_edge")                     # [1,1,W,1]
    n("Or", [c_is_c0, c_is_c1], "c_edge")                     # [1,1,1,W]
    n("And", ["r_edge", c_in], "hbar")                        # horizontal edges
    n("And", [r_in, "c_edge"], "vbar")                        # vertical edges
    n("Or", ["hbar", "vbar"], "frame_b")                      # [1,1,W,W] bool ring

    # outer-corner label as a sum of two rank-1 outer products (fp16, exact):
    #   Lcorner[r,c] = (r==ro_t)*topcols[c] + (r==ro_b)*botcols[c]
    #   topcols[c]   = cBR*(c==co_l) + cBL*(c==co_r)   (outer TL gets cBR, TR gets cBL)
    #   botcols[c]   = cTR*(c==co_l) + cTL*(c==co_r)   (outer BL gets cTR, BR gets cTL)
    or_t = eq_row("ro_t", "or_t")    # [1,1,W,1] bool
    or_b = eq_row("ro_b", "or_b")
    oc_l = eq_col("co_l", "oc_l")    # [1,1,1,W] bool
    oc_r = eq_col("co_r", "oc_r")
    n("Cast", [or_t], "ort_f", to=F16); n("Cast", [or_b], "orb_f", to=F16)
    n("Cast", [oc_l], "ocl_f", to=F16); n("Cast", [oc_r], "ocr_f", to=F16)
    # topcols / botcols  [1,1,1,W] f16
    n("Mul", [cBR, "ocl_f"], "tc_l"); n("Mul", [cBL, "ocr_f"], "tc_r")
    n("Add", ["tc_l", "tc_r"], "topcols")
    n("Mul", [cTR, "ocl_f"], "bc_l"); n("Mul", [cTL, "ocr_f"], "bc_r")
    n("Add", ["bc_l", "bc_r"], "botcols")
    # one MatMul instead of two outer products + Add:
    #   rows[1,1,W,2] = [ort_f | orb_f] , cols[1,1,2,W] = [topcols; botcols]
    #   Lcorner = rows @ cols  -> [1,1,W,W]
    n("Concat", ["ort_f", "orb_f"], "rowsel", axis=3)         # [1,1,W,2] f16
    n("Concat", ["topcols", "botcols"], "colsel", axis=2)     # [1,1,2,W] f16
    n("MatMul", ["rowsel", "colsel"], "Lcorner_f")            # [1,1,W,W] f16
    n("Cast", ["Lcorner_f"], "Lcorner_u8", to=U8)

    # assemble label map (uint8): ring=bcolor over the frame, else outer-corner.
    n("Where", ["frame_b", "bcolor_u", "Lcorner_u8"], "L5")   # [1,1,W,W] uint8

    # ---- pad to 30x30 with sentinel 10, final Equal -------------------------
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    n("Pad", ["L5", "padpads", "padval"], "L", mode="constant")  # [1,1,30,30] u8
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task228", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
