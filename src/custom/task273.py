"""task273 (ARC-AGI af902bf9) — fill the interior of each yellow-corner rectangle with red.

Rule (from the generator task_af902bf9.py):
  A 10x10 grid contains 1 or 2 axis-aligned rectangles.  Each rectangle is marked in the
  INPUT only by its four YELLOW(4) corners at (r0,c0),(r0,c1),(r1,c0),(r1,c1).  In the
  OUTPUT the four yellow corners stay, and every cell STRICTLY INSIDE the rectangle
  (r0<r<r1 AND c0<c<c1) is painted RED(2).  When there are 2 boxes they sit in opposite
  (top-left / bottom-right) quadrants, so they never share a row- or col-range.

  Closed form (verified 0/4000 fails on fresh instances):
    A cell becomes red iff there is a yellow corner in ALL FOUR strict quadrants around it:
      UL = (exists yellow with row<r and col<c)
      UR = (exists yellow with row<r and col>c)
      DL = (exists yellow with row>r and col<c)
      DR = (exists yellow with row>r and col>c)
    red(r,c) = UL & UR & DL & DR.
  This is the JOINT-quadrant test (not the separable per-direction up/down/left/right OR,
  which would cross-talk between the two diagonal boxes).  Each quadrant count is a double
  triangular MatMul  Arow @ Y @ Acol  over the yellow plane Y, then Greater>0.

Encoding (route the 10-ch expansion into the FREE bool output; never materialize [1,10,H,W]):
  Work on the active 10x10 canvas.  Y = input[:,4:5,0:10,0:10] -> fp16.
  Quadrant counts via the two strict triangulars SL/SU (left- and right-multiply);
  redcnt = Min(UL,UR,DL,DR) is exactly 1 on interior cells.  Since the whole task uses
  only colours {0,2,4}, build ONE colour-index plane L = 4*Y + 2*redcnt (terms disjoint),
  cast to uint8, Pad to 30x30 with a SENTINEL 99 (so off-grid stays all-False), and
  output = Equal(L, arange[1,10,1,1]) -> BOOL [1,10,30,30] (declared BOOL, FREE).
  Carrier is a SINGLE 30x30 uint8 plane (900B) instead of the 1800B Where uint8+bool pair.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I64 = TensorProto.INT64

W = 10  # active canvas (grid is always 10x10 for this task)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- yellow plane (channel 4) on the 10x10 canvas ----------------------
    init("y_s", np.array([4, 0, 0], np.int64), np.int64)
    init("y_e", np.array([5, W, W], np.int64), np.int64)
    init("y_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "y_s", "y_e", "y_ax"], "yellow_f32")  # [1,1,W,W] f32
    n("Cast", ["yellow_f32"], "Y", to=F16)                     # [1,1,W,W] f16 {0,1}

    # ---- strict triangular matrices ----------------------------------------
    # Only TWO distinct matrices exist:
    #   SL[i,j]=1 iff j<i  (strictly-lower)   SU[i,j]=1 iff j>i  (strictly-upper)
    # row side (left-multiply A @ Y):  above = SL ([r,k]=1 iff k<r), below = SU.
    # col side (right-multiply Y @ M): left needs M[k,c]=[k<c]=SU, right=[k>c]=SL.
    SL = np.tril(np.ones((W, W), np.float16), k=-1)
    SU = np.triu(np.ones((W, W), np.float16), k=1)
    init("SL", SL.reshape(1, 1, W, W), np.float16)
    init("SU", SU.reshape(1, 1, W, W), np.float16)

    # ---- row-accumulated yellow (above / below) ----------------------------
    n("MatMul", ["SL", "Y"], "Yabove")  # [1,1,W,W] : yellow count strictly above
    n("MatMul", ["SU", "Y"], "Ybelow")  # [1,1,W,W] : yellow count strictly below

    # ---- joint quadrant counts ---------------------------------------------
    n("MatMul", ["Yabove", "SU"], "UL")   # row<r & col<c
    n("MatMul", ["Yabove", "SL"], "UR")   # row<r & col>c
    n("MatMul", ["Ybelow", "SU"], "DL")   # row>r & col<c
    n("MatMul", ["Ybelow", "SL"], "DR")   # row>r & col>c

    # ---- "all four quadrants have a yellow" = Min of the four counts > 0 ----
    # fp16 Min works under ORT_DISABLE_ALL; counts are tiny (<=~4) so exact.
    n("Min", ["UL", "UR"], "top2")        # [1,1,W,W] f16
    n("Min", ["DL", "DR"], "bot2")        # [1,1,W,W] f16
    n("Min", ["top2", "bot2"], "redcnt")  # [1,1,W,W] f16 : >0 iff interior

    # ---- build a single colour-INDEX plane L in {0,2,4} --------------------
    # The whole task uses only bg(0), red(2), yellow(4), so a colour-index plane
    # fully describes the output and lets us route the 10-ch expansion into the
    # FREE bool output via Equal(L, arange) — carrier is just ONE 30x30 plane.
    #   L = 4*Y + 2*redcnt.  redcnt is exactly 1 on interior cells (verified:
    #   the Min quadrant count is always 1 there) so no clip is needed, and
    #   interior cells are never yellow corners so the two terms never overlap.
    init("FOURH", np.array(4.0, np.float16), np.float16)
    init("TWOH", np.array(2.0, np.float16), np.float16)
    n("Mul", ["Y", "FOURH"], "yterm")             # 4 where yellow corner
    n("Mul", ["redcnt", "TWOH"], "rterm")         # 2 where interior
    n("Add", ["yterm", "rterm"], "Lsmall")        # [1,1,W,W] f16 in {0,2,4}
    n("Cast", ["Lsmall"], "L_u8", to=U8)          # [1,1,W,W] uint8

    # Pad off-grid with a SENTINEL (99, not a valid colour) so the Equal one-hot
    # leaves every off-grid cell all-False (== target, which is all-zero there);
    # padding with 0 would wrongly light channel-0 off the 10x10 active grid.
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), np.int64)
    init("SENT", np.array(99, np.uint8), np.uint8)
    n("Pad", ["L_u8", "pads", "SENT"], "L30", mode="constant")  # [1,1,30,30] u8

    # ---- output = Equal(L30, arange) : FREE BOOL [1,10,30,30] ---------------
    init("arange", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L30", "arange"], "output")       # broadcast -> [1,10,30,30] bool

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task273", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
