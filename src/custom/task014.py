"""task014 (ARC-AGI 0b148d64) — crop the rarest-colour quadrant.

Rule (from the generator):
  The grid is split into 4 quadrants by a thick all-background cross (a band of
  `rowthick` rows and `colthick` cols). Random pixels (density 0.9) in each
  quadrant are coloured: the quadrant indexed `quadrant` gets `color_list[0]`,
  every other quadrant gets `color_list[1]`. The generator only `break`s when
  the rarest FOREGROUND colour equals color_list[0] — i.e. the special quadrant
  uses the colour with the FEWEST pixels in the grid. The output is the grid
  cropped to the bounding box of that rarest colour (cells inside the bbox are
  either the rarest colour or background 0; nothing else lands in one quadrant).

Approach (mirrors src/custom/task036.py, the crop-the-cluster idiom):
  1. rarest colour = ArgMin over per-channel pixel COUNT (ch0 + absent channels
     forced to +BIG so background / unused channels can never win).
  2. bbox (min_row, min_col, H, W) of that channel from 1-D occupancy profiles.
  3. select the rarest channel plane (Gather axis=1), shift it to the top-left
     by Gather(axis=2, arange+min_row) then Gather(axis=3, arange+min_col),
     working in a WORK x WORK window.
  4. label map L[1,1,WORK,WORK] uint8 = rarest colour on the mask, 0 on the rest
     of the bbox, sentinel 10 outside; Pad to 30x30, final Equal -> free BOOL.
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

    BIG = 1000.0  # fp16-exact, > any row/col index (<30)
    WORK = 19  # measured max output extent over 20000 fresh = 18; +1 margin
    F16 = TensorProto.FLOAT16

    # ---- per-channel pixel COUNT -> rarest foreground colour -----------------
    n("ReduceSum", ["input"], "cnt", axes=[2, 3], keepdims=1)   # [1,10,1,1] f32
    # mask ch0 (background) and absent channels (count==0) to +BIG so the
    # genuine rarest foreground colour (smallest positive count, !=ch0) wins.
    init("half", np.array(0.5, np.float32), np.float32)
    n("Greater", ["cnt", "half"], "present")                    # bool [1,10,1,1]
    ch0kill = np.zeros((1, 10, 1, 1), np.bool_)
    ch0kill[0, 0, 0, 0] = True
    init("ch0kill", ch0kill, np.bool_)
    n("Not", ["ch0kill"], "notch0")
    n("And", ["present", "notch0"], "valid")                    # present AND c!=0
    init("PBIGf", np.array(BIG, np.float32), np.float32)
    n("Where", ["valid", "cnt", "PBIGf"], "cnt2")               # [1,10,1,1]
    n("ArgMin", ["cnt2"], "bc_i", axis=1, keepdims=1)           # [1,1,1,1] int64

    # ---- select rarest channel plane FIRST (so all bbox work is single-chan) -
    init("shp1d", np.array([1], np.int64), np.int64)
    n("Reshape", ["bc_i", "shp1d"], "bc1")                      # [1] int64
    n("Gather", ["input", "bc1"], "bplane32", axis=1)           # [1,1,30,30] f32

    # ---- bbox of the rarest colour from 1-D profiles on the SINGLE plane -----
    # occupancy is [1,1,30,1]/[1,1,1,30] = 120/60 B (vs 10-ch = 1200 B).
    n("ReduceMax", ["bplane32"], "rowocc", axes=[3], keepdims=1)  # [1,1,30,1] f32
    n("ReduceMax", ["bplane32"], "colocc", axes=[2], keepdims=1)  # [1,1,1,30] f32
    n("Greater", ["rowocc", "half"], "rowb")                    # bool [1,1,30,1]
    n("Greater", ["colocc", "half"], "colb")                    # bool [1,1,1,30]

    ramp_r = np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1)
    ramp_c = np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30)
    init("rr", ramp_r, np.float32)
    init("rc", ramp_c, np.float32)
    init("PBIG", np.array(BIG, np.float32), np.float32)
    init("NBIG", np.array(-BIG, np.float32), np.float32)

    n("Where", ["rowb", "rr", "PBIG"], "rmin_src")              # [1,1,30,1] f32
    n("ReduceMin", ["rmin_src"], "minr", axes=[2], keepdims=1)  # [1,1,1,1]
    n("Where", ["rowb", "rr", "NBIG"], "rmax_src")
    n("ReduceMax", ["rmax_src"], "rmaxr", axes=[2], keepdims=1)
    n("Where", ["colb", "rc", "PBIG"], "cmin_src")              # [1,1,1,30] f32
    n("ReduceMin", ["cmin_src"], "minc", axes=[3], keepdims=1)
    n("Where", ["colb", "rc", "NBIG"], "cmax_src")
    n("ReduceMax", ["cmax_src"], "cmaxc", axes=[3], keepdims=1)

    init("one", np.array(1.0, np.float32), np.float32)
    n("Sub", ["rmaxr", "minr"], "Hm1")                          # H-1 [1,1,1,1]
    n("Sub", ["cmaxc", "minc"], "Wm1")
    n("Add", ["Hm1", "one"], "H")
    n("Add", ["Wm1", "one"], "W")

    baseW = np.arange(WORK, dtype=np.float32)
    init("baseW", baseW, np.float32)                            # [WORK]
    init("shp1", np.array([1], np.int64), np.int64)
    init("c0", np.array(0.0, np.float32), np.float32)
    init("c29", np.array(29.0, np.float32), np.float32)
    n("Reshape", ["minr", "shp1"], "minr_s")                    # [1]
    n("Add", ["baseW", "minr_s"], "ridx_f")
    n("Clip", ["ridx_f", "c0", "c29"], "ridx_cl")
    n("Cast", ["ridx_cl"], "ridx", to=I64)                      # [WORK] int64
    n("Reshape", ["minc", "shp1"], "minc_s")
    n("Add", ["baseW", "minc_s"], "cidx_f")
    n("Clip", ["cidx_f", "c0", "c29"], "cidx_cl")
    n("Cast", ["cidx_cl"], "cidx", to=I64)                      # [WORK] int64

    n("Gather", ["bplane32", "ridx"], "Vr", axis=2)             # [1,1,WORK,30] f32
    n("Gather", ["Vr", "cidx"], "Vs", axis=3)                   # [1,1,WORK,WORK] f32

    # ---- bbox mask (r < H) and (c < W) on the WORK x WORK canvas -------------
    rampw_r = np.arange(WORK, dtype=np.float32).reshape(1, 1, WORK, 1)
    rampw_c = np.arange(WORK, dtype=np.float32).reshape(1, 1, 1, WORK)
    init("wr", rampw_r, np.float32)                             # [1,1,WORK,1]
    init("wc", rampw_c, np.float32)                             # [1,1,1,WORK]
    n("Less", ["wr", "H"], "rmask")                             # bool
    n("Less", ["wc", "W"], "cmask")
    n("And", ["rmask", "cmask"], "boxmask")                     # [1,1,WORK,WORK]

    # ---- rarest-colour mask M = (Vs > 0.5) AND boxmask -----------------------
    n("Cast", ["bc_i"], "bcf", to=F32)                          # rarest colour f32
    n("Cast", ["bcf"], "bc_u8", to=U8)                          # [1,1,1,1] uint8
    n("Greater", ["Vs", "half"], "iseq")                        # [1,1,WORK,WORK]
    n("And", ["iseq", "boxmask"], "M")

    # ---- label map L: rarest colour on M, 0 on box&!M, 10 outside ------------
    init("u0", np.array(0, np.uint8), np.uint8)
    init("u10", np.array(10, np.uint8), np.uint8)
    n("Where", ["M", "bc_u8", "u0"], "Lin")                     # [1,1,WORK,WORK]
    n("Where", ["boxmask", "Lin", "u10"], "Lw")

    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    n("Pad", ["Lw", "padpads", "u10"], "L", mode="constant")    # [1,1,30,30]
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                         # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task014", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
