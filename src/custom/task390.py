"""task390 (ARC-AGI f8a8fe49) — "eject the gray contents out of the red box".

Rule (from the generator):
  A red(2) box (full horizontal red edges at rows brow and B=brow+tall-1, plus 4
  red pixels at the second/second-last rows on the left/right walls) sits on a 0
  background.  gray(5) "contents" are packed INSIDE the box in the INPUT.  In the
  OUTPUT the red box is copied UNCHANGED and every interior gray pixel is reflected
  OUTWARD across the NEARER horizontal red edge of the box:
      Rout = 2*brow - Rin       if  2*Rin < brow + B   (closer to top edge)
      Rout = 2*B   - Rin        otherwise              (closer to bottom edge)
  The COLUMN of each gray pixel is preserved.  brow = min full-red row, B = max
  full-red row.  Half the instances are TRANSPOSED (xpose=1): then the box has
  vertical full-red edges (columns bcol..Bc) and gray reflects across the nearer
  vertical edge, ROW preserved.  Grid is always exactly 15x15; colours {0,2,5}.

Encoding (data-dependent reflection realized as the boolean double-MatMul idiom):
  outgray = Rmat @ gray @ Cmat^T  (sum then threshold >0) where, for the row-box
  orientation, Rmat[o,i] = (refl_row(i) == o) reflects rows and Cmat = I; for the
  col-box orientation Rmat = I and Cmat^T[i,o] = (refl_col(i) == o).  Orientation is
  detected by "does a full red row (>=5 red cells) exist".  refl_*(i) is built from
  the scalar edge positions via a ramp + Where + Equal.  Whole 15x15 canvas is
  in-grid (fixed grid size) so no in-grid mask is needed; the 30x30 Pad sentinel
  produces the off-canvas all-zero cells.  Build uint8 label L = 2*box + 5*outgray
  (red wins), Pad to 30x30 with sentinel 10, output = Equal(L, arange[0..9]) -> BOOL.
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

W = 15  # active canvas (grid is exactly 15x15 for this task)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- channel slices on the 15x15 canvas --------------------------------
    def chan_slice(ch, name):
        init(f"{name}_s", np.array([ch, 0, 0], np.int64), np.int64)
        init(f"{name}_e", np.array([ch + 1, W, W], np.int64), np.int64)
        init(f"{name}_ax", np.array([1, 2, 3], np.int64), np.int64)
        n("Slice", ["input", f"{name}_s", f"{name}_e", f"{name}_ax"], name)
        return name  # [1,1,W,W] f32

    chan_slice(2, "red")   # [1,1,W,W] f32
    chan_slice(5, "gray")  # [1,1,W,W] f32

    init("ZEROF", np.array(0.0, np.float32), np.float32)
    init("BIG", np.array(1e6, np.float32), np.float32)
    init("NBIG", np.array(-1e6, np.float32), np.float32)
    init("FIVEF", np.array(5.0, np.float32), np.float32)

    ax2 = init("ax2", np.arange(W, dtype=np.float32).reshape(1, 1, W, 1), np.float32)
    ax3 = init("ax3", np.arange(W, dtype=np.float32).reshape(1, 1, 1, W), np.float32)

    # ---- full red row / col counts ----------------------------------------
    # red row count = ReduceSum over cols (axis 3) -> [1,1,W,1]
    n("ReduceSum", ["red"], "rowcnt", axes=[3], keepdims=1)   # [1,1,W,1]
    n("ReduceSum", ["red"], "colcnt", axes=[2], keepdims=1)   # [1,1,1,W]
    # "full" edge = count >= 5  (full red edge spans wide>=5 cells)
    n("Less", ["rowcnt", "FIVEF"], "lt_r")
    n("Not", ["lt_r"], "full_r")   # bool [1,1,W,1]  (count >= 5)
    n("Less", ["colcnt", "FIVEF"], "lt_c")
    n("Not", ["lt_c"], "full_c")   # bool [1,1,1,W]

    # orientation: row-box iff any full red ROW exists
    n("Cast", ["full_r"], "full_r_f", to=F32)
    n("ReduceMax", ["full_r_f"], "any_full_r", axes=[2, 3], keepdims=1)  # [1,1,1,1]
    n("Greater", ["any_full_r", "ZEROF"], "is_row_box")  # bool [1,1,1,1]

    # ---- edge positions (min/max index of full edges) ---------------------
    def edge_minmax(full_b, ramp, tag):
        # full_b bool same shape as ramp; ramp f32 index ramp on that axis
        n("Where", [full_b, ramp, "BIG"], f"idxlo_{tag}")
        n("ReduceMin", [f"idxlo_{tag}"], f"lo_{tag}", axes=[2, 3], keepdims=1)
        n("Where", [full_b, ramp, "NBIG"], f"idxhi_{tag}")
        n("ReduceMax", [f"idxhi_{tag}"], f"hi_{tag}", axes=[2, 3], keepdims=1)
        return f"lo_{tag}", f"hi_{tag}"  # [1,1,1,1] f32 scalars

    brow, B = edge_minmax("full_r", "ax2", "r")    # row edges (top, bottom)
    bcol, Bc = edge_minmax("full_c", "ax3", "c")   # col edges (left, right)

    # ---- reflection matrices ----------------------------------------------
    # refl(i) = 2*lo - i  if 2*i < lo+hi  else 2*hi - i
    # Mat[o,i] = (refl(i) == o) : place i on src axis, o on the other axis.
    init("TWOF", np.array(2.0, np.float32), np.float32)

    def refl_mat(lo, hi, src_axis, active, out_name, tag):
        # Build a [1,1,W,W] permutation matrix Mat[o,i] where i is on src_axis and
        # o on the other axis.  The per-index remap vector remaps i -> refl(i) when
        # `active` (a [1,1,1,1] bool), else i -> i (identity) -- gating happens on
        # the tiny [W] vector so no second full matrix / identity init is needed.
        srcv = "ax3" if src_axis == 3 else "ax2"
        outv = "ax2" if src_axis == 3 else "ax3"
        n("Add", [lo, hi], f"sum_{tag}")                 # lo+hi
        n("Mul", [srcv, "TWOF"], f"twoi_{tag}")          # 2*i  (broadcast on src axis)
        n("Less", [f"twoi_{tag}", f"sum_{tag}"], f"top_{tag}")  # bool, 2i < lo+hi
        n("Mul", [lo, "TWOF"], f"twolo_{tag}")
        n("Sub", [f"twolo_{tag}", srcv], f"rtop_{tag}")  # 2*lo - i
        n("Mul", [hi, "TWOF"], f"twohi_{tag}")
        n("Sub", [f"twohi_{tag}", srcv], f"rbot_{tag}")  # 2*hi - i
        n("Where", [f"top_{tag}", f"rtop_{tag}", f"rbot_{tag}"], f"refl_{tag}")  # refl(i)
        # gate: active ? refl(i) : i   (identity on the inactive axis)
        n("Where", [active, f"refl_{tag}", srcv], f"rvec_{tag}")
        n("Equal", [f"rvec_{tag}", outv], f"mat_b_{tag}")  # bool [1,1,W,W]
        n("Cast", [f"mat_b_{tag}"], out_name, to=F16)
        return out_name

    # is_row_box is the row-axis active flag; its negation is the col-axis flag.
    n("Not", ["is_row_box"], "is_col_box")
    # Rmat[o(axis2), i(axis3)] : src=i on axis3, out=o on axis2; active iff row-box
    refl_mat("lo_r", "hi_r", 3, "is_row_box", "Rmat", "R")
    # CmatT[i(axis2), o(axis3)] : src=i on axis2, out=o on axis3 (already T); active iff col-box
    refl_mat("lo_c", "hi_c", 2, "is_col_box", "CmatT", "C")

    # ---- outgray = Rmat @ gray @ CmatT (sum then >0) ----------------------
    n("Cast", ["gray"], "gray16", to=F16)
    n("MatMul", ["Rmat", "gray16"], "rowmapped")        # fp16 [1,1,W,W]
    n("MatMul", ["rowmapped", "CmatT"], "colmapped")    # fp16 [1,1,W,W]
    init("ZEROH", np.array(0.0, np.float16), np.float16)
    n("Greater", ["colmapped", "ZEROH"], "outgray_b")   # bool [1,1,W,W]

    # ---- box mask (red copied unchanged) ----------------------------------
    n("Greater", ["red", "ZEROF"], "redb")              # bool [1,1,W,W]

    # ---- label map (whole 15x15 in-grid; 0=bg) ----------------------------
    init("V0", np.array(0, np.uint8), np.uint8)
    init("V2", np.array(2, np.uint8), np.uint8)
    init("V5", np.array(5, np.uint8), np.uint8)
    init("V10", np.array(10, np.uint8), np.uint8)
    n("Where", ["outgray_b", "V5", "V0"], "L1")         # gray on bg
    n("Where", ["redb", "V2", "L1"], "Lw")              # red wins

    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), np.int64)
    n("Pad", ["Lw", "pads", "V10"], "L", mode="constant")  # [1,1,30,30] uint8

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                 # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task390", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
