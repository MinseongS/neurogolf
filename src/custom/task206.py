"""task206 (ARC-AGI 88a10436) — "copy the colored sprite onto the gray marker".

Rule (from the generator, verified fresh):
  A connected 3x3 conway sprite (its center cell (1,1) is always occupied) is
  drawn in colors from {1,2,3,6}.  The sprite spans EXACTLY rows {0,1,2} and cols
  {0,1,2} relative to its top-left (conway_sprite never lets a row/col vanish), so
  its center cell = (min_colored_row+1, min_colored_col+1).
  The INPUT shows this colored sprite at location 0, plus a SINGLE gray(5) pixel
  somewhere else (the "marker").
  The OUTPUT keeps the colored sprite where it is AND stamps an identical copy
  centered on the marker (gray removed).  The marker center is the new sprite
  center, so the copy is a pure TRANSLATION of the input sprite by
      delta = (gray_row - center_row, gray_col - center_col).
  The two copies never overlap (generator forces |dr|>=4 OR |dc|>=4), so the
  output is just the per-cell MAX of the original colored plane and its shift.

Encoding (Tier B — data-dependent translation via two boolean MatMuls):
  Lc = per-cell sprite color index (one 1x1 Conv with gray channel zeroed) on a
       small WORK x WORK canvas, fp16.
  Recover scalars gray_row/gray_col (the single gray pixel) and center_row/
  center_col (= min colored row/col + 1).  Build shift matrices
      Srow[R,r]   = (r + dr == R)        (dr = gray_row - center_row)
      ScolT[c,C]  = (c + dc == C)        (dc = gray_col - center_col)
  Lshift = Srow @ Lc @ ScolT  (one-hot rows -> exact translation, fp16 exact for
  color values <= 9).  L = Max(Lc, Lshift) -> uint8, Pad to 30x30 with sentinel
  10, final Equal(L, arange[0..9]) writes straight into the FREE BOOL output.
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

WORK = 12  # active canvas (width/height <= 12; sprite/marker fit inside)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- color label on WORK window ------------------------------------------
    # One 1x1 Conv over the whole one-hot input (weight k -> value k, gray ch5 ->
    # 0) gives the colour label; the 30x30 fp32 Conv plane is the single dominant
    # intermediate.  Crop to WORK and drop to fp16 at once.
    # ONE channel-cropped Conv encodes BOTH the colour label and the gray marker.
    # Slice input[:,1:7,:W,:W] -> [1,6,W,W] (covers channels 1..6 = all colours +
    # gray), then a 1x1 Conv with weight [1,2,3,0,50,6] maps colours to their
    # value and gray(ch5) to 50.  In Vf, cells > 9 are the gray marker; the rest
    # carry the colour value.  Avoids the full 30x30 Conv plane.
    init("col_ax", np.array([1, 2, 3], np.int64), np.int64)
    init("col6_s", np.array([1, 0, 0], np.int64), np.int64)
    init("col6_e", np.array([7, WORK, WORK], np.int64), np.int64)
    n("Slice", ["input", "col6_s", "col6_e", "col_ax"], "in16")  # [1,6,W,W] f32
    cw = np.array([1, 2, 3, 0, 50, 6], np.float32).reshape(1, 6, 1, 1)
    init("cw", cw, np.float32)
    n("Conv", ["in16", "cw"], "Vf")                    # [1,1,W,W] f32
    init("NINE", np.array(9.0, np.float32), np.float32)
    init("ZEROF2", np.array(0.0, np.float32), np.float32)
    n("Greater", ["Vf", "NINE"], "gray_b")             # [1,1,W,W] bool gray marker
    n("Where", ["gray_b", "ZEROF2", "Vf"], "Vc")       # colour label (gray -> 0)
    n("Cast", ["Vc"], "Lc", to=F16)                    # [1,1,W,W] fp16 colour plane

    # ---- in-grid mask as a rectangle (grid anchored at origin) ---------------
    # The grid is a full width x height rectangle from (0,0); every in-grid cell
    # has exactly one one-hot channel set.  Collapse the FREE input to 1-D row and
    # col occupancy profiles (sum over channels AND the other spatial axis) -> the
    # in-grid rectangle is rowprofile(>0) AND colprofile(>0).  Tiny 30-vectors.
    n("ReduceSum", ["input"], "rowprof30", axes=[1, 3], keepdims=1)  # [1,1,30,1]
    n("ReduceSum", ["input"], "colprof30", axes=[1, 2], keepdims=1)  # [1,1,1,30]
    init("rp_st", np.array([0], np.int64), np.int64)
    init("rp_en", np.array([WORK], np.int64), np.int64)
    init("rp_ax2", np.array([2], np.int64), np.int64)
    init("rp_ax3", np.array([3], np.int64), np.int64)
    n("Slice", ["rowprof30", "rp_st", "rp_en", "rp_ax2"], "rowprof")  # [1,1,W,1]
    n("Slice", ["colprof30", "rp_st", "rp_en", "rp_ax3"], "colprof")  # [1,1,1,W]
    init("HALF2", np.array(0.5, np.float32), np.float32)
    n("Greater", ["rowprof", "HALF2"], "row_in")   # [1,1,W,1] bool
    n("Greater", ["colprof", "HALF2"], "col_in")   # [1,1,1,W] bool
    n("And", ["row_in", "col_in"], "in_grid")      # [1,1,W,W] bool rectangle

    # ---- scalars: gray_row/gray_col, min colored row/col --------------------
    init("BIG", np.array(99.0, np.float32), np.float32)
    init("NEG", np.array(-1.0, np.float32), np.float32)
    init("ONEF", np.array(1.0, np.float32), np.float32)
    ar_row = init("ar_row", np.arange(WORK, dtype=np.float32).reshape(1, 1, WORK, 1),
                  np.float32)
    ar_col = init("ar_col", np.arange(WORK, dtype=np.float32).reshape(1, 1, 1, WORK),
                  np.float32)

    # gray single pixel -> its row & col.  In Vf the gray cell holds 50 (>9), so
    # max over the other axis exceeds 9 exactly on the gray row / gray col.  No
    # separate gray plane needed.
    n("ReduceMax", ["Vf"], "gy_rowp", axes=[3], keepdims=1)  # [1,1,W,1]
    n("ReduceMax", ["Vf"], "gy_colp", axes=[2], keepdims=1)  # [1,1,1,W]
    n("Greater", ["gy_rowp", "NINE"], "gy_rb")
    n("Greater", ["gy_colp", "NINE"], "gy_cb")
    n("Where", ["gy_rb", "ar_row", "NEG"], "gy_ridx")
    n("ReduceMax", ["gy_ridx"], "gray_row", axes=[2, 3], keepdims=1)  # [1,1,1,1]
    n("Where", ["gy_cb", "ar_col", "NEG"], "gy_cidx")
    n("ReduceMax", ["gy_cidx"], "gray_col", axes=[2, 3], keepdims=1)  # [1,1,1,1]

    # colored presence (Lc>0) -> min colored row/col ; center = min + 1
    init("HALFH", np.array(0.5, np.float16), np.float16)
    n("ReduceMax", ["Lc"], "c_rowp", axes=[3], keepdims=1)    # [1,1,W,1] fp16
    n("ReduceMax", ["Lc"], "c_colp", axes=[2], keepdims=1)    # [1,1,1,W] fp16
    n("Greater", ["c_rowp", "HALFH"], "c_rb")
    n("Greater", ["c_colp", "HALFH"], "c_cb")
    n("Where", ["c_rb", "ar_row", "BIG"], "c_ridx")
    n("ReduceMin", ["c_ridx"], "min_row", axes=[2, 3], keepdims=1)  # [1,1,1,1]
    n("Where", ["c_cb", "ar_col", "BIG"], "c_cidx")
    n("ReduceMin", ["c_cidx"], "min_col", axes=[2, 3], keepdims=1)  # [1,1,1,1]
    n("Add", ["min_row", "ONEF"], "center_row")
    n("Add", ["min_col", "ONEF"], "center_col")

    # ---- shift deltas (scalars) ---------------------------------------------
    n("Sub", ["gray_row", "center_row"], "dr")   # [1,1,1,1]
    n("Sub", ["gray_col", "center_col"], "dc")    # [1,1,1,1]

    # ---- shift matrices: Srow[R,r]=(r+dr==R), ScolT[c,C]=(c+dc==C) -----------
    # Srow on (R=axis2, r=axis3): value = (ar_col(=r,axis3) + dr == ar_row(=R,axis2))
    n("Add", ["ar_col", "dr"], "rshift")          # [1,1,1,W] = r + dr (broadcast)
    n("Equal", ["rshift", "ar_row"], "Srow_b")    # [1,1,W,W] bool (R,r)
    n("Cast", ["Srow_b"], "Srow", to=F16)
    # ScolT on (c=axis2, C=axis3): value = (ar_row(=c,axis2) + dc == ar_col(=C,axis3))
    n("Add", ["ar_row", "dc"], "cshift")          # [1,1,W,1] = c + dc
    n("Equal", ["cshift", "ar_col"], "ScolT_b")   # [1,1,W,W] bool (c,C)
    n("Cast", ["ScolT_b"], "ScolT", to=F16)

    # ---- Lshift = Srow @ Lc @ ScolT (exact translation, fp16) ----------------
    n("MatMul", ["Srow", "Lc"], "rowshift")        # [1,1,W,W] fp16
    n("MatMul", ["rowshift", "ScolT"], "Lshift")   # [1,1,W,W] fp16
    n("Max", ["Lc", "Lshift"], "Lboth")            # [1,1,W,W] fp16

    # ---- label map -> mask out-of-grid -> Pad -> output ---------------------
    n("Cast", ["Lboth"], "L8c", to=U8)             # [1,1,W,W] uint8
    init("V10", np.array(10, np.uint8), np.uint8)  # sentinel (no channel matches)
    n("Where", ["in_grid", "L8c", "V10"], "L8")    # out-of-grid -> 10
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64),
         np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    n("Pad", ["L8", "pads", "padval"], "L", mode="constant")  # [1,1,30,30] uint8
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")            # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task206", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
