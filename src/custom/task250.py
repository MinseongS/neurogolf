"""task250 (ARC-AGI a48eeaf7) — "pull each gray pixel onto the ring around the red box".

Rule (from the generator):
  A 2x2 red(2) box sits at (boxrow,boxcol)..(boxrow+1,boxcol+1).  Background is 0.
  Gray(5) pixels are scattered.  In the OUTPUT the red box is copied unchanged, and
  every input gray pixel at (r,c) is snapped toward the box:
      R = clamp(r, boxrow-1, boxrow+2)
      C = clamp(c, boxcol-1, boxcol+2)
  i.e. each gray lands on the 4x4 ring region [boxrow-1..boxrow+2] x [boxcol-1..boxcol+2].
  Multiple grays may collide onto the same output cell.  The original gray locations
  are NOT kept; output is box + clamped grays.  br=boxrow=min red row, bc=boxcol=min red col.

Encoding (Tier B — data-dependent clamp realized as two boolean MatMuls):
  Output gray plane factors as  outgray = Rmat @ gray @ Cmat^T  (OR-AND semiring,
  realized as sum then threshold >0) where
      Rmat[R,r] = [clamp(r,br-1,br+2) == R],  Cmat^T[c,C] = [clamp(c,bc-1,bc+2) == C].
  Both clamp matrices are built from the scalar br/bc via a clamped arange + Equal.
  Work on the 10x10 active canvas; build uint8 label L = 2*box + 5*outgray (red wins),
  sentinel 10 off the 10x10 region; Pad to 30x30; output = Equal(L, arange[0..9]) -> BOOL.
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

W = 10  # active canvas (grid is exactly 10x10 for this task)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- channel slices on the 10x10 canvas --------------------------------
    # red = channel 2, gray = channel 5, bg = channel 0.
    def chan_slice(ch, name):
        init(f"{name}_s", np.array([ch, 0, 0], np.int64), np.int64)
        init(f"{name}_e", np.array([ch + 1, W, W], np.int64), np.int64)
        init(f"{name}_ax", np.array([1, 2, 3], np.int64), np.int64)
        n("Slice", ["input", f"{name}_s", f"{name}_e", f"{name}_ax"], name)
        return name  # [1,1,W,W] f32

    # grid is always exactly 10x10, so every cell of the canvas is in-grid: no
    # bg slice / in-grid mask needed (the 30x30 Pad sentinel handles off-canvas).
    chan_slice(2, "red")   # [1,1,W,W] f32
    chan_slice(5, "gray")  # [1,1,W,W] f32

    init("ZEROF", np.array(0.0, np.float32), np.float32)
    init("BIG", np.array(1e6, np.float32), np.float32)

    # ---- min red row / min red col (box top-left) --------------------------
    def min_index(plane, axis_keep, tag):
        red_axes = [3] if axis_keep == 2 else [2]
        n("ReduceMax", [plane], f"pres_{tag}", axes=red_axes, keepdims=1)
        n("Greater", [f"pres_{tag}", "ZEROF"], f"presb_{tag}")
        ramp = (np.arange(W, dtype=np.float32).reshape(1, 1, W, 1) if axis_keep == 2
                else np.arange(W, dtype=np.float32).reshape(1, 1, 1, W))
        init(f"ramp_{tag}", ramp, np.float32)
        n("Where", [f"presb_{tag}", f"ramp_{tag}", "BIG"], f"idx_{tag}")
        n("ReduceMin", [f"idx_{tag}"], f"min_{tag}", axes=[2, 3], keepdims=1)
        return f"min_{tag}"  # [1,1,1,1] f32 scalar

    min_index("red", 2, "br")   # boxrow
    min_index("red", 3, "bc")   # boxcol

    # ---- clamp matrices ----------------------------------------------------
    # clampvec_row[r] = clamp(r, br-1, br+2) ; Rmat[R,r] = (clampvec_row[r] == R)
    init("ONEF", np.array(1.0, np.float32), np.float32)
    init("TWOF", np.array(2.0, np.float32), np.float32)
    ax3 = init("ax3", np.arange(W, dtype=np.float32).reshape(1, 1, 1, W), np.float32)
    ax2 = init("ax2", np.arange(W, dtype=np.float32).reshape(1, 1, W, 1), np.float32)

    def clamp_mat(bscalar, src_axis, tag):
        # clampvec[src] = clip(src, b-1, b+2) ; src placed on `src_axis`, output
        # index on the other axis -> Equal broadcasts to [1,1,W,W].
        srcv = "ax3" if src_axis == 3 else "ax2"
        outv = "ax2" if src_axis == 3 else "ax3"
        n("Sub", [bscalar, "ONEF"], f"lo_{tag}")
        n("Add", [bscalar, "TWOF"], f"hi_{tag}")
        n("Max", [srcv, f"lo_{tag}"], f"cl1_{tag}")
        n("Min", [f"cl1_{tag}", f"hi_{tag}"], f"clv_{tag}")
        n("Equal", [f"clv_{tag}", outv], f"mat_b_{tag}")   # bool [1,1,W,W]
        n("Cast", [f"mat_b_{tag}"], f"mat_{tag}", to=F16)  # fp16 [1,1,W,W]
        return f"mat_{tag}"

    # Rmat[R(axis2), r(axis3)] : src=r on axis3, out=R on axis2
    Rmat = clamp_mat("min_br", 3, "R")
    # CmatT[c(axis2), C(axis3)] : src=c on axis2, out=C on axis3 (already T)
    CmatT = clamp_mat("min_bc", 2, "C")

    # ---- outgray = Rmat @ gray @ CmatT (OR via sum then >0) ----------------
    # values are {0,1}, products/sums stay well under 2^11 so fp16 is exact.
    n("Cast", ["gray"], "gray16", to=F16)             # fp16 [1,1,W,W]
    n("MatMul", [Rmat, "gray16"], "rowmapped")        # fp16 [1,1,W,W] (R, c)
    n("MatMul", ["rowmapped", CmatT], "colmapped")    # fp16 [1,1,W,W] (R, C)
    init("ZEROH", np.array(0.0, np.float16), np.float16)
    n("Greater", ["colmapped", "ZEROH"], "outgray_b")  # bool [1,1,W,W]

    # ---- box mask (red, 2x2) -----------------------------------------------
    n("Greater", ["red", "ZEROF"], "redb")            # bool [1,1,W,W]

    # ---- label map (whole 10x10 is in-grid; 0=bg) --------------------------
    init("V0", np.array(0, np.uint8), np.uint8)
    init("V2", np.array(2, np.uint8), np.uint8)
    init("V5", np.array(5, np.uint8), np.uint8)
    init("V10", np.array(10, np.uint8), np.uint8)
    n("Where", ["outgray_b", "V5", "V0"], "L1")       # gray on bg=0 canvas
    n("Where", ["redb", "V2", "L1"], "Lw")            # red box wins

    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), np.int64)
    n("Pad", ["Lw", "pads", "V10"], "L", mode="constant")  # [1,1,30,30] uint8

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")               # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task250", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
