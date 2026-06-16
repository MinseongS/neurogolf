"""task031 (ARC-AGI 1cf80156) — crop a single-colour connected blob to its bbox.

Rule (from the generator, verified over fresh instances):
  The input grid is fixed width=12, height in {10,11,12}. It holds ONE connected
  blob of 8..12 pixels drawn in a single foreground colour on a background of 0.
  The output is the tight bounding box of the blob, cropped to the top-left corner
  of a fresh grid (background 0 fills the holes inside the bbox; everything outside
  the HxW box is all-channels-off / channel-0).

  Invariants (measured 0/20000):
    - exactly ONE non-zero colour present; background = 0 (corner cell input[0][0]).
    - output bbox is at most 9x9.

Pipeline (ONNX, opset 11):
  1. colf[1,1,30,30] = sum_k k*input_k (1x1 Conv) — per-cell colour index, >0 ⇔ fg.
  2. row/col occupancy profiles from ReduceMax of colf; derive (min_row,min_col)
     and (H-1,W-1) = bbox spans as scalars.
  3. Gather a WORK=9 x WORK window of colf starting at (min_row,min_col).
  4. boxmask = (r<H) AND (c<W) on the WORK window.
  5. L[1,1,WORK,WORK] uint8: window colour where boxmask else sentinel 10; Pad to
     30x30 with sentinel 10; output = Equal(L, arange[0..9]) -> free BOOL one-hot.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
I64 = TensorProto.INT64
B = TensorProto.BOOL

WORK = 9


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

    # ---- colour-index plane: colf = sum_k k * input_k (1x1 Conv) -------------
    w = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("cw", w, np.float32)
    n("Conv", ["input", "cw"], "colf")                         # [1,1,30,30] f32

    # ---- occupancy profiles --------------------------------------------------
    n("ReduceMax", ["colf"], "rowocc", axes=[3], keepdims=1)   # [1,1,30,1] f32
    n("ReduceMax", ["colf"], "colocc", axes=[2], keepdims=1)   # [1,1,1,30] f32
    init("half", np.array(0.5, np.float32), np.float32)
    n("Greater", ["rowocc", "half"], "rowb")                   # bool [1,1,30,1]
    n("Greater", ["colocc", "half"], "colb")                   # bool [1,1,1,30]

    # row / col index ramps (fp32; values <30 exact)
    ramp_r = np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1)
    ramp_c = np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30)
    init("rr", ramp_r, np.float32)                             # [1,1,30,1]
    init("rc", ramp_c, np.float32)                             # [1,1,1,30]
    init("PBIG", np.array(BIG, np.float32), np.float32)
    init("NBIG", np.array(-BIG, np.float32), np.float32)

    # min/max occupied row & col
    n("Where", ["rowb", "rr", "PBIG"], "rmin_src")             # [1,1,30,1]
    n("ReduceMin", ["rmin_src"], "rmin", axes=[2], keepdims=1)  # [1,1,1,1]
    n("Where", ["rowb", "rr", "NBIG"], "rmax_src")
    n("ReduceMax", ["rmax_src"], "rmax", axes=[2], keepdims=1)  # [1,1,1,1]
    n("Where", ["colb", "rc", "PBIG"], "cmin_src")             # [1,1,1,30]
    n("ReduceMin", ["cmin_src"], "cmin", axes=[3], keepdims=1)  # [1,1,1,1]
    n("Where", ["colb", "rc", "NBIG"], "cmax_src")
    n("ReduceMax", ["cmax_src"], "cmax", axes=[3], keepdims=1)  # [1,1,1,1]

    # spans (H-1, W-1) and H, W
    n("Sub", ["rmax", "rmin"], "rspan")                        # [1,1,1,1] = H-1
    n("Sub", ["cmax", "cmin"], "cspan")                        # = W-1
    init("one", np.array(1.0, np.float32), np.float32)
    n("Add", ["rspan", "one"], "H")                            # [1,1,1,1]
    n("Add", ["cspan", "one"], "W")

    # ---- gather a WORK x WORK window of colf starting at (min_row,min_col) ----
    init("shp1", np.array([1], np.int64), np.int64)
    n("Reshape", ["rmin", "shp1"], "minr_s")                   # [1]
    n("Reshape", ["cmin", "shp1"], "minc_s")
    baseW = np.arange(WORK, dtype=np.float32)
    init("baseW", baseW, np.float32)                           # [WORK]
    init("c0", np.array(0.0, np.float32), np.float32)
    init("c29", np.array(29.0, np.float32), np.float32)
    n("Add", ["baseW", "minr_s"], "ridx_f")                    # [WORK]
    n("Clip", ["ridx_f", "c0", "c29"], "ridx_cl")
    n("Cast", ["ridx_cl"], "ridx", to=I64)
    n("Add", ["baseW", "minc_s"], "cidx_f")
    n("Clip", ["cidx_f", "c0", "c29"], "cidx_cl")
    n("Cast", ["cidx_cl"], "cidx", to=I64)
    n("Gather", ["colf", "ridx"], "Vr", axis=2)                # [1,1,WORK,30] f32
    n("Gather", ["Vr", "cidx"], "Vs", axis=3)                  # [1,1,WORK,WORK] f32

    # ---- box mask (r<H) and (c<W) on the WORK canvas -------------------------
    rampw_r = np.arange(WORK, dtype=np.float32).reshape(1, 1, WORK, 1)
    rampw_c = np.arange(WORK, dtype=np.float32).reshape(1, 1, 1, WORK)
    init("wr", rampw_r, np.float32)                            # [1,1,WORK,1]
    init("wc", rampw_c, np.float32)                            # [1,1,1,WORK]
    n("Less", ["wr", "H"], "rmask")                            # [1,1,WORK,1] bool
    n("Less", ["wc", "W"], "cmask")                            # [1,1,1,WORK] bool
    n("And", ["rmask", "cmask"], "boxmask")                    # [1,1,WORK,WORK]

    # ---- label map L (WORK x WORK): window colour on box, sentinel 10 outside -
    n("Cast", ["Vs"], "Vs_u8", to=U8)                          # [1,1,WORK,WORK] u8
    init("u10", np.array(10, np.uint8), np.uint8)
    n("Where", ["boxmask", "Vs_u8", "u10"], "Lw")             # outside box -> 10

    # ---- pad WORK x WORK label map to 30x30 (sentinel 10), final Equal -------
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    n("Pad", ["Lw", "padpads", "u10"], "L", mode="constant")   # [1,1,30,30] u8
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                        # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task031", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
