"""task300 (ARC-AGI be94b721) — crop the LARGEST sprite to its bounding box.

Rule (from the generator):
  The grid holds 3..4 small connected "creatures" (sprites), each drawn in a
  DISTINCT colour (`random_colors`, one per sprite).  Sizes are sorted DESCENDING,
  so sprite index 0 is the LARGEST (most pixels).  The output is sprite-0's shape
  cropped to its own bounding box (origin at top-left), in sprite-0's colour
  (channel-0 background fills the interior holes).

  Because each sprite has a UNIQUE colour, the colour channel with the MOST pixels
  (excluding background channel 0) is exactly sprite-0, and every pixel of that
  colour belongs to sprite-0.  So: pick the max-pixel-count colour channel,
  recover its bounding box, shift it to the origin, emit it monochrome.

Pipeline (ONNX, opset 11):
  1. counts = ReduceSum(input,[2,3]) [1,10,1,1] f32; kill ch0 (-BIG); mc = ArgMax.
  2. bplane = Gather(input, mc, axis=1) [1,1,30,30] f32 mask of sprite-0.
  3. From bplane's 1-D occupancy (single channel) recover (min_row,min_col,H,W).
  4. Gather a WORK x WORK window at (min_row,min_col) -> shifted to origin.
  5. M = (window>0) AND (r<H) AND (c<W); label L = mc on M else sentinel.
  6. Pad L to 30x30 (sentinel) and Equal(L, arange[0..9]) -> free BOOL one-hot.
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

WORK = 5  # sprite-0 bbox is at most 4 tall x 3 wide; 5 is a safe upper bound.


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- 1. max-pixel-count colour channel (exclude ch0) ---------------------
    n("ReduceSum", ["input"], "counts", axes=[2, 3], keepdims=1)  # [1,10,1,1] f32
    # subtract a big penalty on channel 0 so it never wins.
    bgpen = np.zeros((1, 10, 1, 1), np.float32)
    bgpen[0, 0, 0, 0] = 1e6
    init("bgpen", bgpen, np.float32)
    n("Sub", ["counts", "bgpen"], "counts_m")                     # [1,10,1,1]
    n("ArgMax", ["counts_m"], "mc_i", axis=1, keepdims=1)         # [1,1,1,1] i64

    # ---- 2. sprite-0 plane ----------------------------------------------------
    init("shp1", np.array([1], np.int64), np.int64)
    n("Reshape", ["mc_i", "shp1"], "mc1")                         # [1] i64
    n("Gather", ["input", "mc1"], "bplane", axis=1)               # [1,1,30,30] f32

    # ---- 3. bbox of the single plane -----------------------------------------
    n("ReduceMax", ["bplane"], "rowocc", axes=[3], keepdims=1)    # [1,1,30,1] f32
    n("ReduceMax", ["bplane"], "colocc", axes=[2], keepdims=1)    # [1,1,1,30] f32
    init("half", np.array(0.5, np.float32), np.float32)
    n("Greater", ["rowocc", "half"], "rowb")                      # bool [1,1,30,1]
    n("Greater", ["colocc", "half"], "colb")                      # bool [1,1,1,30]

    BIG = 1000.0
    ramp_r = np.arange(30, dtype=np.float16).reshape(1, 1, 30, 1)
    ramp_c = np.arange(30, dtype=np.float16).reshape(1, 1, 1, 30)
    init("rr", ramp_r, np.float16)
    init("rc", ramp_c, np.float16)
    init("PBIG", np.array(BIG, np.float16), np.float16)
    init("NBIG", np.array(-BIG, np.float16), np.float16)

    n("Where", ["rowb", "rr", "PBIG"], "rmin_src")                # [1,1,30,1] f16
    n("ReduceMin", ["rmin_src"], "rmin16", axes=[2], keepdims=1)  # [1,1,1,1] f16
    n("Where", ["rowb", "rr", "NBIG"], "rmax_src")
    n("ReduceMax", ["rmax_src"], "rmax16", axes=[2], keepdims=1)
    n("Where", ["colb", "rc", "PBIG"], "cmin_src")                # [1,1,1,30] f16
    n("ReduceMin", ["cmin_src"], "cmin16", axes=[3], keepdims=1)
    n("Where", ["colb", "rc", "NBIG"], "cmax_src")
    n("ReduceMax", ["cmax_src"], "cmax16", axes=[3], keepdims=1)

    n("Cast", ["rmin16"], "rmin", to=F32)                         # [1,1,1,1] f32
    n("Cast", ["cmin16"], "cmin", to=F32)
    n("Sub", ["rmax16", "rmin16"], "Hm116")                       # H-1 f16
    n("Sub", ["cmax16", "cmin16"], "Wm116")
    n("Cast", ["Hm116"], "Hm1", to=F32)
    n("Cast", ["Wm116"], "Wm1", to=F32)
    init("one", np.array(1.0, np.float32), np.float32)
    n("Add", ["Hm1", "one"], "H")                                 # [1,1,1,1] f32
    n("Add", ["Wm1", "one"], "W")

    # ---- 4. shift WORK x WORK window to origin --------------------------------
    baseW = np.arange(WORK, dtype=np.float32)
    init("baseW", baseW, np.float32)                              # [WORK]
    init("c0", np.array(0.0, np.float32), np.float32)
    init("c29", np.array(29.0, np.float32), np.float32)
    n("Reshape", ["rmin", "shp1"], "rmin_s")                      # [1]
    n("Add", ["baseW", "rmin_s"], "ridx_f")
    n("Clip", ["ridx_f", "c0", "c29"], "ridx_cl")
    n("Cast", ["ridx_cl"], "ridx", to=I64)                        # [WORK] i64
    n("Reshape", ["cmin", "shp1"], "cmin_s")
    n("Add", ["baseW", "cmin_s"], "cidx_f")
    n("Clip", ["cidx_f", "c0", "c29"], "cidx_cl")
    n("Cast", ["cidx_cl"], "cidx", to=I64)                        # [WORK] i64

    n("Gather", ["bplane", "ridx"], "Vr", axis=2)                 # [1,1,WORK,30]
    n("Gather", ["Vr", "cidx"], "Vs", axis=3)                     # [1,1,WORK,WORK]

    # ---- 5. box mask + blob mask ---------------------------------------------
    rampw_r = np.arange(WORK, dtype=np.float32).reshape(1, 1, WORK, 1)
    rampw_c = np.arange(WORK, dtype=np.float32).reshape(1, 1, 1, WORK)
    init("wr", rampw_r, np.float32)                               # [1,1,WORK,1]
    init("wc", rampw_c, np.float32)                               # [1,1,1,WORK]
    n("Less", ["wr", "H"], "rmask")                               # bool
    n("Less", ["wc", "W"], "cmask")
    n("And", ["rmask", "cmask"], "boxmask")                       # [1,1,WORK,WORK]
    n("Greater", ["Vs", "half"], "iseq")                          # [1,1,WORK,WORK]
    n("And", ["iseq", "boxmask"], "M")                            # blob cells bool

    # ---- 6. label map -> Pad -> one-hot --------------------------------------
    n("Cast", ["mc_i"], "mc_u8", to=U8)                           # [1,1,1,1] u8
    init("u0", np.array(0, np.uint8), np.uint8)
    init("u10", np.array(10, np.uint8), np.uint8)
    n("Where", ["M", "mc_u8", "u0"], "Lin")                       # [1,1,WORK,WORK]
    n("Where", ["boxmask", "Lin", "u10"], "Lw")                   # outside box ->10
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    n("Pad", ["Lw", "padpads", "u10"], "L", mode="constant")      # [1,1,30,30] u8
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                           # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task300", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
