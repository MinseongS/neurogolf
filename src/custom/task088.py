"""task088 (ARC-AGI 3de23699) — crop the box interior, recolour sprite -> corner colour.

Rule (verified 0/5000 fresh):
  The grid has a rectangular "box" marked ONLY by 4 corner pixels in colour
  `colors[1]` at the four positions just OUTSIDE the box interior:
    (brow-1, bcol-1), (brow-1, bcol+wide), (brow+tall, bcol-1), (brow+tall, bcol+wide).
  Inside the interior [brow..brow+tall-1] x [bcol..bcol+wide-1] there are
  `wide+tall+randint(-1,1)` sprite pixels in colour `colors[0]`.
  The output is the tall x wide interior cropped to the top-left of a fresh grid,
  with every sprite pixel painted in the CORNER colour `colors[1]` and every other
  interior cell background (0); outside the tall x wide box everything is off.

  Colour identification (exact): the corner colour appears EXACTLY 4 times; the
  sprite colour appears >=5 times (wide+tall>=6, +randint(-1,1) => >=5). So:
    cornercol = the non-bg channel whose pixel count == 4
    spritecol = the non-bg channel with the MAX pixel count
  Box geometry from the corner colour's bbox:
    brow = rmin_corner + 1, bcol = cmin_corner + 1
    tall = (rmax_corner - rmin_corner) - 1, wide = (cmax_corner - cmin_corner) - 1

Pipeline (ONNX, opset 11), mirrors task036's crop+shift idiom:
  1. per-channel pixel counts (ReduceSum) -> cornercol (count==4) & spritecol (argmax).
  2. corner bbox from cheap 1-D per-channel occupancy profiles (no full corner plane).
  3. sprite plane = Gather(input, spritecol, axis=1) [1,1,30,30] (the only full plane);
     shift to top-left & crop to a WORK=10 window via Gather(axis=2)+Gather(axis=3).
  4. label map L (WORK x WORK): cornercol where sprite present & in-box, 0 in-box-empty,
     sentinel 10 outside-box. Pad to 30x30 with sentinel, Equal(L, arange) -> BOOL output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
I64 = TensorProto.INT64
B = TensorProto.BOOL


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    WORK = 10  # wide, tall are each in 3..10

    # ---- per-channel pixel counts -------------------------------------------
    # counts[1,10,1,1] = number of pixels of each colour.
    n("ReduceSum", ["input"], "counts", axes=[2, 3], keepdims=1)  # [1,10,1,1] f32

    # corner colour = channel whose count == 4 (exactly). ch0 huge, sprite >=5.
    init("four", np.array(4.0, np.float32), np.float32)
    n("Equal", ["counts", "four"], "iscorner")               # [1,10,1,1] bool
    # cornercol scalar index = argmax of the boolean (cast to f32)
    n("Cast", ["iscorner"], "iscorner_f", to=F32)
    n("ArgMax", ["iscorner_f"], "cc_i", axis=1, keepdims=1)  # [1,1,1,1] int64

    # sprite colour = channel with MAX count, excluding ch0 (background huge).
    # zero out ch0 then argmax.
    ch0kill = np.zeros((1, 10, 1, 1), np.float32)
    ch0kill[0, 0, 0, 0] = 1.0
    init("ch0kill", ch0kill, np.float32)
    init("one_f", np.array(1.0, np.float32), np.float32)
    n("Sub", ["one_f", "ch0kill"], "notch0_f")               # [1,10,1,1] 0 at ch0 else 1
    n("Mul", ["counts", "notch0_f"], "counts_nb")            # [1,10,1,1] ch0 zeroed
    n("ArgMax", ["counts_nb"], "sc_i", axis=1, keepdims=1)   # [1,1,1,1] int64

    # ---- corner bbox from cheap 1-D per-channel occupancy profiles ----------
    n("ReduceMax", ["input"], "rowocc", axes=[3], keepdims=1)  # [1,10,30,1] f32
    n("ReduceMax", ["input"], "colocc", axes=[2], keepdims=1)  # [1,10,1,30] f32
    init("half", np.array(0.5, np.float32), np.float32)
    n("Greater", ["rowocc", "half"], "rowb")                 # [1,10,30,1] bool
    n("Greater", ["colocc", "half"], "colb")                 # [1,10,1,30] bool

    ramp_r = np.arange(30, dtype=np.float16).reshape(1, 1, 30, 1)
    ramp_c = np.arange(30, dtype=np.float16).reshape(1, 1, 1, 30)
    init("rr", ramp_r, np.float16)
    init("rc", ramp_c, np.float16)
    init("PBIG", np.array(1000.0, np.float16), np.float16)
    init("NBIG", np.array(-1000.0, np.float16), np.float16)

    # per-channel min/max occupied row/col (fp16, broadcast ramp over channels)
    n("Where", ["rowb", "rr", "PBIG"], "rmin_src")           # [1,10,30,1] fp16
    n("ReduceMin", ["rmin_src"], "rmin", axes=[2], keepdims=1)  # [1,10,1,1]
    n("Where", ["rowb", "rr", "NBIG"], "rmax_src")
    n("ReduceMax", ["rmax_src"], "rmax", axes=[2], keepdims=1)  # [1,10,1,1]
    n("Where", ["colb", "rc", "PBIG"], "cmin_src")           # [1,10,1,30]
    n("ReduceMin", ["cmin_src"], "cmin", axes=[3], keepdims=1)  # [1,10,1,1]
    n("Where", ["colb", "rc", "NBIG"], "cmax_src")
    n("ReduceMax", ["cmax_src"], "cmax", axes=[3], keepdims=1)  # [1,10,1,1]

    # spans (corner bbox) -> tall = rspan-1, wide = cspan-1
    n("Sub", ["rmax", "rmin"], "rspan")                      # [1,10,1,1] fp16
    n("Sub", ["cmax", "cmin"], "cspan")

    # gather corner colour's rmin/cmin and rspan/cspan via the cc index
    init("shp10", np.array([1, 10], np.int64), np.int64)
    init("shp11", np.array([1, 1], np.int64), np.int64)
    n("Reshape", ["cc_i", "shp11"], "cc11")                  # [1,1]
    n("Reshape", ["rmin", "shp10"], "rmin10")
    n("Reshape", ["cmin", "shp10"], "cmin10")
    n("Reshape", ["rspan", "shp10"], "rspan10")
    n("Reshape", ["cspan", "shp10"], "cspan10")
    n("GatherElements", ["rmin10", "cc11"], "minr16", axis=1)  # [1,1] fp16
    n("GatherElements", ["cmin10", "cc11"], "minc16", axis=1)
    n("GatherElements", ["rspan10", "cc11"], "rspan16", axis=1)
    n("GatherElements", ["cspan10", "cc11"], "cspan16", axis=1)
    n("Cast", ["minr16"], "minr", to=F32)                    # corner rmin [1,1]
    n("Cast", ["minc16"], "minc", to=F32)
    n("Cast", ["rspan16"], "rspan_f", to=F32)
    n("Cast", ["cspan16"], "cspan_f", to=F32)

    # box interior top-left = corner_min + 1 ; tall = rspan-1, wide = cspan-1
    init("one1", np.array(1.0, np.float32), np.float32)
    n("Add", ["minr", "one1"], "brow")                       # [1,1] f32
    n("Add", ["minc", "one1"], "bcol")
    n("Sub", ["rspan_f", "one1"], "tall")                    # [1,1] = tall
    n("Sub", ["cspan_f", "one1"], "wide")                    # [1,1] = wide

    # ---- sprite plane (the only full 30x30 plane) ---------------------------
    init("shp1d", np.array([1], np.int64), np.int64)
    n("Reshape", ["sc_i", "shp1d"], "sc1")                   # [1] int64
    n("Gather", ["input", "sc1"], "splane", axis=1)          # [1,1,30,30] f32 mask

    # gather a WORK-row x WORK-col window starting at (brow, bcol)
    baseW = np.arange(WORK, dtype=np.float32)
    init("baseW", baseW, np.float32)                         # [WORK]
    init("shp1", np.array([1], np.int64), np.int64)
    init("c0", np.array(0.0, np.float32), np.float32)
    init("c29", np.array(29.0, np.float32), np.float32)
    n("Reshape", ["brow", "shp1"], "brow_s")                 # [1]
    n("Add", ["baseW", "brow_s"], "ridx_f")                  # [WORK]
    n("Clip", ["ridx_f", "c0", "c29"], "ridx_cl")
    n("Cast", ["ridx_cl"], "ridx", to=I64)                   # [WORK] int64
    n("Reshape", ["bcol", "shp1"], "bcol_s")                 # [1]
    n("Add", ["baseW", "bcol_s"], "cidx_f")
    n("Clip", ["cidx_f", "c0", "c29"], "cidx_cl")
    n("Cast", ["cidx_cl"], "cidx", to=I64)                   # [WORK] int64

    n("Gather", ["splane", "ridx"], "Vr", axis=2)            # [1,1,WORK,30] f32
    n("Gather", ["Vr", "cidx"], "Vs", axis=3)                # [1,1,WORK,WORK] f32

    # ---- box mask (r < tall) and (c < wide) on the WORK x WORK canvas -------
    rampw_r = np.arange(WORK, dtype=np.float32).reshape(1, 1, WORK, 1)
    rampw_c = np.arange(WORK, dtype=np.float32).reshape(1, 1, 1, WORK)
    init("wr", rampw_r, np.float32)                          # [1,1,WORK,1]
    init("wc", rampw_c, np.float32)                          # [1,1,1,WORK]
    n("Less", ["wr", "tall"], "rmask")                       # [1,1,WORK,1] bool
    n("Less", ["wc", "wide"], "cmask")                       # [1,1,1,WORK] bool
    n("And", ["rmask", "cmask"], "boxmask")                  # [1,1,WORK,WORK] bool

    # ---- sprite-present mask (Vs > 0.5) AND boxmask -------------------------
    n("Greater", ["Vs", "half"], "issp")                     # [1,1,WORK,WORK] bool
    n("And", ["issp", "boxmask"], "M")                       # sprite cells in box

    # ---- label map L: cornercol on M, 0 on box&!M, sentinel 10 outside ------
    n("Cast", ["cc_i"], "cc_u8", to=U8)                      # cornercol uint8 [1,1,1,1]
    init("u0", np.array(0, np.uint8), np.uint8)
    init("u10", np.array(10, np.uint8), np.uint8)
    n("Where", ["M", "cc_u8", "u0"], "Lin")                  # [1,1,WORK,WORK] u8
    n("Where", ["boxmask", "Lin", "u10"], "Lw")             # outside box -> 10

    # pad WORK x WORK to 30x30 with sentinel, then Equal -> BOOL output
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    n("Pad", ["Lw", "padpads", "u10"], "L", mode="constant")  # [1,1,30,30] u8
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                      # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task088", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
