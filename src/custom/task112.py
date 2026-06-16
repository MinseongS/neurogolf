"""task112 (ARC-AGI 4938f0c2) — reflect the red pattern around the green box (4-fold symmetrize).

Rule (from the generator):
  A 2x2 green(3) box sits at (brow,bcol)..(brow+1,bcol+1) on a 0 background.
  A small red(2) pattern is stamped into the FOUR quadrants around the green box,
  reflected across the box centre.  For each quadrant (dr,dc) in
  {(-1,-1),(-1,+1),(+1,-1),(+1,+1)}: a red arm-pixel at relative (r,c) lands at
      row = (brow-1) + dr*r    (rowoff = brow-1 if dr<0 else brow+2)
      col = (bcol-1) + dc*c    (coloff = bcol-1 if dc<0 else bcol+2)
  Algebraically row R reflects about (brow+0.5):  R -> (2*brow+1) - R, and column
  C reflects about (bcol+0.5): C -> (2*bcol+1) - C.  The INPUT may only show one
  quadrant (showall=0) or all four (showall=1); the OUTPUT always shows all four.
  => output = input symmetrized: OR of {red, flipRows red, flipCols red, flipBoth red},
  with the green box (already symmetric) copied unchanged.

Encoding (Tier B — data-dependent reflection realized as two boolean permutation MatMuls):
  green box: copied as-is (it is symmetric about both axes).
  red: redout = red OR Rmat@red OR red@CmatT OR Rmat@red@CmatT  (sum then >0),
  where Rmat[R,r]=[R == 2*brow+1-r],  CmatT[c,C]=[C == 2*bcol+1-c],  brow/bcol = min
  green row/col.  Build a uint8 label L (3=green, 2=red) on the 30x30 canvas; cells
  outside the input grid (all-channel-0) get sentinel 10 so the final Equal yields
  all-False there.  output = Equal(L, arange[0..9]) -> BOOL.
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

N = 30  # full canvas (grid size is variable up to 30x30)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- channel slices (full 30x30) ---------------------------------------
    def chan_slice(ch, name):
        init(f"{name}_s", np.array([ch, 0, 0], np.int64), np.int64)
        init(f"{name}_e", np.array([ch + 1, N, N], np.int64), np.int64)
        init(f"{name}_ax", np.array([1, 2, 3], np.int64), np.int64)
        n("Slice", ["input", f"{name}_s", f"{name}_e", f"{name}_ax"], name)
        return name  # [1,1,N,N] f32

    chan_slice(2, "red")    # [1,1,N,N] f32 (needed in full for the 2-D reflection)

    init("ZEROF", np.array(0.0, np.float32), np.float32)
    init("BIG", np.array(1e6, np.float32), np.float32)

    # ---- green row / col profiles (1-D, no full green plane) ---------------
    # ReduceMax over channels+other-axis gives per-channel profiles; slice ch3.
    n("ReduceMax", ["input"], "rowprof_all", axes=[3], keepdims=1)  # [1,10,N,1]
    n("ReduceMax", ["input"], "colprof_all", axes=[2], keepdims=1)  # [1,10,1,N]
    init("g_rs", np.array([3, 0, 0], np.int64), np.int64)
    init("g_re", np.array([4, N, 1], np.int64), np.int64)
    init("g_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["rowprof_all", "g_rs", "g_re", "g_ax"], "grow")  # [1,1,N,1] green-row occ
    init("g_cs", np.array([3, 0, 0], np.int64), np.int64)
    init("g_ce", np.array([4, 1, N], np.int64), np.int64)
    n("Slice", ["colprof_all", "g_cs", "g_ce", "g_ax"], "gcol")  # [1,1,1,N] green-col occ

    # ---- min green row / col (box top-left), from 1-D profiles -------------
    def min_index(prof, axis_keep, tag):
        ramp = (np.arange(N, dtype=np.float32).reshape(1, 1, N, 1) if axis_keep == 2
                else np.arange(N, dtype=np.float32).reshape(1, 1, 1, N))
        init(f"ramp_{tag}", ramp, np.float32)
        n("Greater", [prof, "ZEROF"], f"presb_{tag}")
        n("Where", [f"presb_{tag}", f"ramp_{tag}", "BIG"], f"idx_{tag}")
        n("ReduceMin", [f"idx_{tag}"], f"min_{tag}", axes=[2, 3], keepdims=1)
        return f"min_{tag}"  # [1,1,1,1] f32 scalar

    min_index("grow", 2, "br")   # brow
    min_index("gcol", 3, "bc")   # bcol

    # ---- reflection matrices ------------------------------------------------
    # Rmat[R, r] = (R == 2*brow+1 - r).  Build axis vectors and reflected vector.
    init("ONEF", np.array(1.0, np.float32), np.float32)
    init("TWOF", np.array(2.0, np.float32), np.float32)
    init("ax3", np.arange(N, dtype=np.float32).reshape(1, 1, 1, N), np.float32)  # col idx
    init("ax2", np.arange(N, dtype=np.float32).reshape(1, 1, N, 1), np.float32)  # row idx

    def reflect_mat(bscalar, src_axis, tag):
        # axis = 2*b+1 ; refl(src) = axis - src.  src on src_axis, out on other axis.
        srcv = "ax3" if src_axis == 3 else "ax2"
        outv = "ax2" if src_axis == 3 else "ax3"
        n("Mul", [bscalar, "TWOF"], f"b2_{tag}")
        n("Add", [f"b2_{tag}", "ONEF"], f"axisc_{tag}")   # 2*b+1
        n("Sub", [f"axisc_{tag}", srcv], f"refl_{tag}")   # reflected coord of src
        n("Equal", [f"refl_{tag}", outv], f"mat_b_{tag}")  # bool [1,1,N,N]
        n("Cast", [f"mat_b_{tag}"], f"mat_{tag}", to=F16)  # fp16 [1,1,N,N]
        return f"mat_{tag}"

    # Rmat[R(axis2), r(axis3)] : src=r on axis3, out=R on axis2
    Rmat = reflect_mat("min_br", 3, "R")
    # CmatT[c(axis2), C(axis3)] : src=c on axis2, out=C on axis3 (already transposed)
    CmatT = reflect_mat("min_bc", 2, "C")

    # ---- redout = red OR Rmat@red OR red@CmatT OR Rmat@red@CmatT -------------
    n("Cast", ["red"], "red16", to=F16)               # fp16 [1,1,N,N]
    n("MatMul", [Rmat, "red16"], "rB")                # flip rows
    n("MatMul", ["red16", CmatT], "rC")               # flip cols
    n("MatMul", ["rB", CmatT], "rD")                  # flip both
    nodes.append(helper.make_node(
        "Sum", ["red16", "rB", "rC", "rD"], ["rsum"]))  # 4-way OR (one plane)
    init("ZEROH", np.array(0.0, np.float16), np.float16)
    n("Greater", ["rsum", "ZEROH"], "redout_b")        # bool [1,1,N,N]

    # ---- green box (symmetric): separable 2x2 mask from (brow,bcol) ---------
    # gr in {brow,brow+1}: (ax2 - brow) in {0,1} <=> 0 <= ax2-brow <= 1.
    n("Sub", ["ax2", "min_br"], "drow")                # [1,1,N,1]
    n("Sub", ["ax3", "min_bc"], "dcol")                # [1,1,1,N]
    n("Not", [n("Less", ["drow", "ZEROF"], "drow_neg")], "drow_ge0")
    n("Not", [n("Greater", ["drow", "ONEF"], "drow_gt1")], "drow_le1")
    n("And", ["drow_ge0", "drow_le1"], "grow_m")       # bool [1,1,N,1]
    n("Not", [n("Less", ["dcol", "ZEROF"], "dcol_neg")], "dcol_ge0")
    n("Not", [n("Greater", ["dcol", "ONEF"], "dcol_gt1")], "dcol_le1")
    n("And", ["dcol_ge0", "dcol_le1"], "gcol_m")       # bool [1,1,1,N]
    n("And", ["grow_m", "gcol_m"], "greenb")           # bool [1,1,N,N]

    # ---- in-grid mask (separable rectangle [0,H)x[0,W)) --------------------
    # Background fills every in-grid cell -> in-grid is the full rectangle.
    # Reuse the per-channel row/col profiles: ReduceMax over channels gives the
    # 1-D row/col occupancy (any colour present).
    n("ReduceMax", ["rowprof_all"], "rowocc", axes=[1], keepdims=1)  # [1,1,N,1]
    n("ReduceMax", ["colprof_all"], "colocc", axes=[1], keepdims=1)  # [1,1,1,N]
    n("Greater", ["rowocc", "ZEROF"], "rowin")                   # bool [1,1,N,1]
    n("Greater", ["colocc", "ZEROF"], "colin")                   # bool [1,1,1,N]
    n("And", ["rowin", "colin"], "ingrid")                       # bool [1,1,N,N]

    # ---- label map ---------------------------------------------------------
    init("V0", np.array(0, np.uint8), np.uint8)
    init("V2", np.array(2, np.uint8), np.uint8)
    init("V3", np.array(3, np.uint8), np.uint8)
    init("V10", np.array(10, np.uint8), np.uint8)
    n("Where", ["redout_b", "V2", "V0"], "L1")         # red on bg=0
    n("Where", ["greenb", "V3", "L1"], "L2")           # green wins (disjoint anyway)
    n("Where", ["ingrid", "L2", "V10"], "L")           # off-grid -> sentinel 10

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                # [1,10,N,N] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, N, N])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, N, N])
    g = helper.make_graph(nodes, "task112", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
