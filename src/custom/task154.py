"""task154 (ARC-AGI 6855a6e4) — fold the outside gray pixels into the red box.

Rule (from the generator):
  A red(2) "gripper" box of size wide x tall (wide in 5..7, tall in 8..9, so
  tall > wide ALWAYS) is drawn: full red top edge + full red bottom edge, with
  red also at the four corners-just-inside (so the two side columns each carry
  red at top, top+1, bot-1, bot).  Gray(5) pixels are scattered strictly OUTSIDE
  the box: some ABOVE the top edge, some BELOW the bottom edge (columns always
  fall within the box interior).  The whole figure may be transposed (xpose).
  Transform: each outside gray is REFLECTED across the nearest red EDGE line into
  the box interior:
     above-top gray at row R  ->  row 2*top - R     (reflect across top edge)
     below-bot gray at row R  ->  row 2*bot - R     (reflect across bottom edge)
  Columns are preserved.  The red box is copied unchanged.  In the xposed case the
  same fold happens across the two vertical red edge COLUMNS instead of rows.
  (Verified: reflection lands exactly inside; midpoint of in/out row == edge row.)

Orientation discriminator (exact, verified 1000/1000):
  box is taller than wide  <=> non-xpose (reflect rows);  wider than tall <=> xpose.

Encoding (Tier B — two boolean reflection MatMuls + orientation select):
  red occupancy 1-D profiles -> rt,rb (min/max red row), ct,cb (min/max red col)
  as [1,1,1,1] scalars (no full red plane for geometry).
  gray = input[:,5] sliced to WORK x WORK fp16.
  Rmat[O,i] = (O==2*rt-i)|(O==2*rb-i) ; rowrefl = Rmat @ gray.
  Cmat[O,i] = (O==2*ct-i)|(O==2*cb-i) ; colrefl = gray @ Cmat^T.
  xpose = (cb-ct) > (rb-rt) ; refl = Where(xpose, colrefl, rowrefl).
  keep = (refl>0) AND (rt<row<rb) AND (ct<col<cb).
  Label L (uint8 30x30): bg 0, red->2 (copied from input red mask), gray-keep->5.
  output = Equal(L, arange[0..9]) -> BOOL.  Off-grid stays bg (correct).
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

N = 30   # full canvas (output must be 30x30)
W = 15   # active working canvas (generator size is fixed 15x15)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    init("ZEROF", np.array(0.0, np.float32), np.float32)
    init("BIG", np.array(1e6, np.float32), np.float32)
    init("TWOF", np.array(2.0, np.float32), np.float32)

    # ---- red channel slice (single channel, WORK x WORK) -------------------
    init("red_s", np.array([2, 0, 0], np.int64), np.int64)
    init("red_e", np.array([3, W, W], np.int64), np.int64)
    init("chan_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "red_s", "red_e", "chan_ax"], "red")  # [1,1,W,W] f32

    # gray channel slice
    init("gray_s", np.array([5, 0, 0], np.int64), np.int64)
    init("gray_e", np.array([6, W, W], np.int64), np.int64)
    n("Slice", ["input", "gray_s", "gray_e", "chan_ax"], "grayf")  # [1,1,W,W] f32

    # ---- red row/col occupancy profiles (1-D) ------------------------------
    n("ReduceMax", ["red"], "redrow", axes=[3], keepdims=1)  # [1,1,W,1]
    n("ReduceMax", ["red"], "redcol", axes=[2], keepdims=1)  # [1,1,1,W]

    init("ax2", np.arange(W, dtype=np.float32).reshape(1, 1, W, 1), np.float32)  # row idx
    init("ax3", np.arange(W, dtype=np.float32).reshape(1, 1, 1, W), np.float32)  # col idx

    def minmax_index(prof, axis_keep, tag):
        # returns (min,max) occupied index along the kept spatial axis as scalars
        ramp = "ax2" if axis_keep == 2 else "ax3"
        n("Greater", [prof, "ZEROF"], f"pres_{tag}")           # bool
        # min: ramp where present else BIG -> ReduceMin
        n("Where", [f"pres_{tag}", ramp, "BIG"], f"idxmin_{tag}")
        n("ReduceMin", [f"idxmin_{tag}"], f"min_{tag}", axes=[2, 3], keepdims=1)
        # max: ramp where present else 0 -> ReduceMax
        n("Where", [f"pres_{tag}", ramp, "ZEROF"], f"idxmax_{tag}")
        n("ReduceMax", [f"idxmax_{tag}"], f"max_{tag}", axes=[2, 3], keepdims=1)
        return f"min_{tag}", f"max_{tag}"   # [1,1,1,1] f32 scalars each

    rt, rb = minmax_index("redrow", 2, "rr")  # min/max red row
    ct, cb = minmax_index("redcol", 3, "cc")  # min/max red col

    # ---- reflection matrices ------------------------------------------------
    # Rmat[O(axis2), i(axis3)] = (O == 2*rt - i) OR (O == 2*rb - i)
    def refl_mat(amin, amax, tag):
        # source coord on axis3, output coord on axis2 -> [1,1,W,W]
        n("Mul", [amin, "TWOF"], f"a2min_{tag}")
        n("Sub", [f"a2min_{tag}", "ax3"], f"reflmin_{tag}")   # 2*amin - i  [1,1,1,W]
        n("Equal", [f"reflmin_{tag}", "ax2"], f"emin_{tag}")  # [1,1,W,W] bool
        n("Mul", [amax, "TWOF"], f"a2max_{tag}")
        n("Sub", [f"a2max_{tag}", "ax3"], f"reflmax_{tag}")
        n("Equal", [f"reflmax_{tag}", "ax2"], f"emax_{tag}")
        n("Or", [f"emin_{tag}", f"emax_{tag}"], f"matb_{tag}")
        n("Cast", [f"matb_{tag}"], f"mat_{tag}", to=F16)
        return f"mat_{tag}"   # [1,1,W,W] fp16

    Rmat = refl_mat(rt, rb, "R")   # reflect rows
    Cmat = refl_mat(ct, cb, "C")   # reflect cols (built same way; used transposed)

    n("Cast", ["grayf"], "gray16", to=F16)              # [1,1,W,W] fp16

    # rowrefl = Rmat @ gray  (reflect along row axis)
    n("MatMul", [Rmat, "gray16"], "rowrefl")            # [1,1,W,W] fp16
    # colrefl = gray @ Cmat^T  -> reflect along col axis.
    # Cmat[O,i]=(O==2*c-i); we need out_col O from src_col i: gray[r,i]*Cmat[O,i]
    # => colrefl[r,O] = sum_i gray[r,i]*Cmat[O,i] = gray @ Cmat^T.
    n("Transpose", [Cmat], "CmatT", perm=[0, 1, 3, 2])  # [1,1,W,W]
    n("MatMul", ["gray16", "CmatT"], "colrefl")         # [1,1,W,W] fp16

    # ---- orientation select -------------------------------------------------
    n("Sub", [rb, rt], "rspan")     # box height
    n("Sub", [cb, ct], "cspan")     # box width
    n("Greater", ["cspan", "rspan"], "xpose")   # bool [1,1,1,1] (wider => xpose)
    n("Where", ["xpose", "colrefl", "rowrefl"], "refl")  # [1,1,W,W] fp16

    init("ZEROH", np.array(0.0, np.float16), np.float16)
    n("Greater", ["refl", "ZEROH"], "refl_b")           # bool [1,1,W,W]

    # ---- inside-box mask (strictly between edges in BOTH axes) -------------
    n("Greater", ["ax2", rt], "row_gt_t")               # [1,1,W,1] bool
    n("Less", ["ax2", rb], "row_lt_b")
    n("And", ["row_gt_t", "row_lt_b"], "insideR")        # [1,1,W,1]
    n("Greater", ["ax3", ct], "col_gt_t")               # [1,1,1,W]
    n("Less", ["ax3", cb], "col_lt_b")
    n("And", ["col_gt_t", "col_lt_b"], "insideC")        # [1,1,1,W]
    n("And", ["insideR", "insideC"], "inside")           # [1,1,W,W]
    n("And", ["refl_b", "inside"], "graykeep")           # [1,1,W,W] bool

    # ---- red mask (copied as-is) -------------------------------------------
    n("Greater", ["red", "ZEROF"], "red_b")              # [1,1,W,W] bool

    # ---- label map (uint8, WORK x WORK) ------------------------------------
    init("V0", np.array(0, np.uint8), np.uint8)
    init("V2", np.array(2, np.uint8), np.uint8)
    init("V5", np.array(5, np.uint8), np.uint8)
    n("Where", ["red_b", "V2", "V0"], "L1")              # red -> 2
    n("Where", ["graykeep", "V5", "L1"], "Lw")           # gray-keep -> 5  [1,1,W,W]

    # pad label back to full 30x30 with bg 0
    init("PADS", np.array([0, 0, 0, 0, 0, 0, N - W, N - W], np.int64), np.int64)
    init("PADV", np.array(99, np.uint8), np.uint8)  # sentinel >9 -> off-grid all-False
    n("Pad", ["Lw", "PADS", "PADV"], "L", mode="constant")  # [1,1,N,N] uint8

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                  # [1,10,N,N] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, N, N])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, N, N])
    gph = helper.make_graph(nodes, "task154", [x], [y], inits)
    return helper.make_model(gph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
