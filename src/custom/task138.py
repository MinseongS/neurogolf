"""task138 (ARC-AGI 5daaa586) — boxed rays, cropped to the box.

Rule (verified fresh in numpy):
  H x W grid (H,W ~ 10..26) holding a rectangular box of four FULL lines:
  left & right vertical (cols `left`,`right`, every row) + top & bottom
  horizontal (rows `up`,`down`, every col), coloured colors[0..3]=
  (left,right,up,down) in a random draworder (corners take last-drawn colour).
  Scattered single pixels of one `drawcolor` (== exactly one of the four line
  colours) each emit a RAY in a single global direction toward the matching wall
  (left if drawcolor==colors[0], right==colors[1], up==colors[2], down==colors[3]),
  painting drawcolor from the pixel up to (not into) the wall.  Output = box region
  [up..down] x [left..right] moved to the top-left of a fresh canvas.

Encoding (opset 12 — uint8 crop-first label-map; opset12 enables uint8 MaxPool):
  1. colf = sum_k k*input_k -> ONE fp32 30x30 entry plane (3600B, forced: a 10->1
     channel reduction over the fp32 one-hot must output fp32). Cast -> uint8.
  2. box edges left/right/up/down recovered as int scalars from per-row/col
     occupancy COUNTS via two no-pad count-Convs (drop bg ch0): a FULL line column
     hits H cells / a full row hits W cells; scattered pixels never reach H/W.
     Only tiny [1,1,30,1]/[1,1,1,30] count planes -- no 30x30 occupancy plane.
  3. crop+shift the uint8 index plane to top-left; downstream planes CROP x CROP u8.
  4. line colours by single-cell Gather (non-corner line cells); drawcolor = max over
     the strict interior; direction = Equal(drawcolor, line-colour).
  5. ray = directional uint8 MAX-fill of seed*drawcolor: one active axis via
     transpose + one-sided full-length uint8 MaxPool (zero params, opset12).
  6. value = max(crop, ray) (rays carry drawcolor, never overwrite a wall since they
     stop before it); keep mask -> sentinel 10; Pad CROP->30; Equal -> free BOOL.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
I64 = TensorProto.INT64
B = TensorProto.BOOL

N = 30
CH = 24   # crop side (max output dim h<=24, w<=23 verified over 4000 instances)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- 1. entry colour-index plane (forced fp32), cast to uint8 -------------
    init("kW", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    n("Conv", ["input", "kW"], "colf")                      # [1,1,30,30] f32
    n("Cast", ["colf"], "colu", to=U8)                      # [1,1,30,30] u8

    # ---- 2. box edges from per-row/col occupancy COUNTS (count-Convs) --------
    # per-row count via a strided full-width count-Conv (drop bg ch0). Kernel is
    # 1x1 over channels summed across a 30-wide window by stride trick? Use the
    # explicit full-width / full-height kernels but keep them as cheap params.
    rowk = np.ones((1, 10, 1, N), np.float32); rowk[0, 0] = 0.0
    colk = np.ones((1, 10, N, 1), np.float32); colk[0, 0] = 0.0
    init("rowk", rowk, np.float32)
    init("colk", colk, np.float32)
    n("Conv", ["input", "rowk"], "rowcnt")                  # [1,1,30,1]  #cols/row
    n("Conv", ["input", "colk"], "colcnt")                  # [1,1,1,30]  #rows/col
    n("ReduceMax", ["rowcnt"], "Wf", axes=[2], keepdims=1)  # W
    n("ReduceMax", ["colcnt"], "Hf", axes=[3], keepdims=1)  # H

    n("Equal", ["colcnt", "Hf"], "iscol")                   # [1,1,1,30] bool
    n("Equal", ["rowcnt", "Wf"], "isrow")                   # [1,1,30,1] bool

    init("rampc", np.arange(N, dtype=np.float32).reshape(1, 1, 1, N), np.float32)
    init("rampr", np.arange(N, dtype=np.float32).reshape(1, 1, N, 1), np.float32)
    init("PBIG", np.array(1000.0, np.float32), np.float32)
    init("NBIG", np.array(-1000.0, np.float32), np.float32)

    n("Where", ["iscol", "rampc", "PBIG"], "lcand")
    n("ReduceMin", ["lcand"], "leftf", axes=[3], keepdims=1)   # [1,1,1,1]
    n("Where", ["iscol", "rampc", "NBIG"], "rcand")
    n("ReduceMax", ["rcand"], "rightf", axes=[3], keepdims=1)
    n("Where", ["isrow", "rampr", "PBIG"], "ucand")
    n("ReduceMin", ["ucand"], "upf", axes=[2], keepdims=1)
    n("Where", ["isrow", "rampr", "NBIG"], "dcand")
    n("ReduceMax", ["dcand"], "downf", axes=[2], keepdims=1)

    # ---- 3. crop+shift the uint8 index to top-left ---------------------------
    init("baseCH", np.arange(CH, dtype=np.float32), np.float32)
    init("shp1", np.array([1], np.int64), np.int64)
    init("f0", np.array(0.0, np.float32), np.float32)
    init("fNm1", np.array(float(N - 1), np.float32), np.float32)

    n("Reshape", ["upf", "shp1"], "up1")
    n("Add", ["baseCH", "up1"], "ridxf")
    n("Clip", ["ridxf", "f0", "fNm1"], "ridxc")
    n("Cast", ["ridxc"], "ridx", to=I64)                    # [CH]
    n("Reshape", ["leftf", "shp1"], "lf1")
    n("Add", ["baseCH", "lf1"], "cidxf")
    n("Clip", ["cidxf", "f0", "fNm1"], "cidxc")
    n("Cast", ["cidxc"], "cidx", to=I64)                    # [CH]

    n("Gather", ["colu", "ridx"], "cr", axis=2)             # [1,1,CH,30] u8
    n("Gather", ["cr", "cidx"], "crop", axis=3)             # [1,1,CH,CH] u8

    # ---- 4. interior mask, line colours, drawcolor (crop coords) -------------
    init("one16f", np.array(1.0, np.float32), np.float32)
    n("Sub", ["downf", "upf"], "ohm1")                      # oh-1
    n("Sub", ["rightf", "leftf"], "owm1")                   # ow-1

    init("crampc", np.arange(CH, dtype=np.float32).reshape(1, 1, 1, CH), np.float32)
    init("crampr", np.arange(CH, dtype=np.float32).reshape(1, 1, CH, 1), np.float32)
    n("Greater", ["crampr", "f0"], "r_gt0")
    n("Less", ["crampr", "ohm1"], "r_ltl")
    n("And", ["r_gt0", "r_ltl"], "inrow")                   # [1,1,CH,1]
    n("Greater", ["crampc", "f0"], "c_gt0")
    n("Less", ["crampc", "owm1"], "c_ltl")
    n("And", ["c_gt0", "c_ltl"], "incol")                   # [1,1,1,CH]
    n("And", ["inrow", "incol"], "interior")                # [1,1,CH,CH] bool

    init("i64_0", np.array([0], np.int64), np.int64)
    init("i64_1", np.array([1], np.int64), np.int64)
    n("Reshape", ["owm1", "shp1"], "owm1r")
    n("Cast", ["owm1r"], "ow_i", to=I64)
    n("Reshape", ["ohm1", "shp1"], "ohm1r")
    n("Cast", ["ohm1r"], "oh_i", to=I64)

    n("Gather", ["crop", "i64_1"], "row1", axis=2)          # [1,1,1,CH]
    n("Gather", ["row1", "i64_0"], "lc", axis=3)            # left colour
    n("Gather", ["row1", "ow_i"], "rc", axis=3)             # right colour
    n("Gather", ["crop", "i64_0"], "row0", axis=2)
    n("Gather", ["row0", "i64_1"], "uc", axis=3)            # up colour
    n("Gather", ["crop", "oh_i"], "rowd", axis=2)
    n("Gather", ["rowd", "i64_1"], "dc", axis=3)            # down colour

    # drawcolor = max index over the strict interior, via uint8 global MaxPool
    # (opset12 uint8 MaxPool gives the scalar max with NO fp32 plane).
    init("u0", np.array(0, np.uint8), np.uint8)
    n("Where", ["interior", "crop", "u0"], "intv")          # [1,1,CH,CH] u8
    n("MaxPool", ["intv"], "draw", kernel_shape=[CH, CH])   # [1,1,1,1] u8 scalar

    n("Equal", ["draw", "lc"], "isLeft")
    n("Equal", ["draw", "rc"], "isRight")
    n("Equal", ["draw", "uc"], "isUp")
    n("Equal", ["draw", "dc"], "isDown")
    n("Or", ["isLeft", "isRight"], "isHor")
    n("Or", ["isLeft", "isUp"], "isRevDir")

    # ---- 5. seed*drawcolor & directional uint8 ray ---------------------------
    n("Equal", ["crop", "draw"], "isdraw")                  # [1,1,CH,CH] bool
    n("And", ["isdraw", "interior"], "seedb")
    # seedv = draw where seed else 0  (uint8 value plane)
    n("Where", ["seedb", "draw", "u0"], "seedv")            # [1,1,CH,CH] u8

    n("Transpose", ["seedv"], "seedvT", perm=[0, 1, 3, 2])
    n("Where", ["isHor", "seedv", "seedvT"], "saxis")       # active axis -> axis3 u8
    n("MaxPool", ["saxis"], "fillF", kernel_shape=[1, CH], pads=[0, CH - 1, 0, 0])
    n("MaxPool", ["saxis"], "fillR", kernel_shape=[1, CH], pads=[0, 0, 0, CH - 1])
    n("Where", ["isRevDir", "fillR", "fillF"], "filla")     # u8
    n("Transpose", ["filla"], "fillaT", perm=[0, 1, 3, 2])
    n("Where", ["isHor", "filla", "fillaT"], "fillback")    # u8 ray value plane

    # ---- 6. compose value plane, crop+pad+Equal ------------------------------
    # ray cells hold drawcolor (>0) along the painted segment; clip to the strict
    # interior (rays never touch walls/corners), then overlay drawcolor.
    n("Greater", ["fillback", "u0"], "rayhit")              # bool
    n("And", ["rayhit", "interior"], "raymask")             # bool, interior-clipped
    n("Where", ["raymask", "draw", "crop"], "V")            # [1,1,CH,CH] u8

    n("Add", ["ohm1", "one16f"], "oh")
    n("Add", ["owm1", "one16f"], "ow")
    n("Less", ["crampr", "oh"], "rkeep")                    # [1,1,CH,1]
    n("Less", ["crampc", "ow"], "ckeep")                    # [1,1,1,CH]
    n("And", ["rkeep", "ckeep"], "keep")
    init("u10", np.array(10, np.uint8), np.uint8)
    n("Where", ["keep", "V", "u10"], "Lw")
    init("padpads", np.array([0, 0, 0, 0, 0, 0, N - CH, N - CH], np.int64), np.int64)
    n("Pad", ["Lw", "padpads", "u10"], "L", mode="constant")

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                     # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task138", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 12)])
