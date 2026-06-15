"""Task 159 (6b9890af): magnify a 3x3 sprite into a red-bordered box.

Rule (from ARC-GEN): the input contains a 3x3 sprite (single non-red color)
and a hollow red square outline of side outsize = 3*m+2 (m = magnifier 1..4).
The output is a fresh outsize x outsize grid: a red border ring, and inside it
the sprite magnified by m (each sprite cell -> an m x m block), placed at
offset +1 from the border. Interior non-sprite cells are background (color 0).
Everything outside the outsize x outsize rectangle is all-zero.

Recovery from input (1-D reductions, cheap):
  - m = (#red cells - 4) / 12       (red ring perimeter is 12m+4)
  - footprint = sum of channels except 0 (bg) and 2 (red)
  - (rmin,cmin) = top-left of the sprite footprint bbox
  - sprite colour = the single channel (not 0/2) that is non-empty

Memory floor-break (small 14x14 canvas + uint8 label map + final Equal):
  outsize = 3m+2 <= 14, so the whole output fits the top-left 14x14 corner.
  The 3x3 footprint plane (the one canvas-sized float) is Gathered to a 14x14
  sprite aligned at (0,0): rows rmin..rmin+13, cols cmin..cmin+13. The magnify
  is a second pair of Gathers with index (Ro-1)//m. A uint8 label map L (red
  ring = 2, magnified sprite = colour, in-grid background = 0, sentinel 10
  outside the outsize box) is padded to 30x30 and finished with
  Equal(L, arange[0..9]) into the free BOOL `output`.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

WORK = 14  # outsize = 3*m+2 <= 14 for m <= 4


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    I64 = TensorProto.INT64

    # ---- per-channel presence (sum over H,W) ----
    n("ReduceSum", ["input"], "chsum", axes=[2, 3], keepdims=1)   # [1,10,1,1]
    init("g2", np.array([2], np.int64), np.int64)
    n("Gather", ["chsum", "g2"], "redc", axis=1)                  # [1,1,1,1]
    init("c4", np.array(4.0, np.float32), np.float32)
    init("c12", np.array(12.0, np.float32), np.float32)
    n("Sub", ["redc", "c4"], "rm4")
    n("Div", ["rm4", "c12"], "m")                                 # scalar m

    # ---- footprint plane (sum channels != 0,2) via 1x1 Conv ----
    keepW = np.ones((1, 10, 1, 1), np.float32)
    keepW[0, 0] = 0.0
    keepW[0, 2] = 0.0
    init("keepW", keepW, np.float32)
    init("keep", keepW.reshape(1, 10, 1, 1), np.float32)
    n("Conv", ["input", "keepW"], "fpf")                          # [1,1,30,30] f32

    # ---- sprite colour one-hot vector -> scalar colour id (uint8) ----
    init("zeroS", np.array(0.0, np.float32), np.float32)
    n("Greater", ["chsum", "zeroS"], "present")                   # [1,10,1,1] b
    n("Cast", ["present"], "presentf", to=F)
    n("Mul", ["presentf", "keep"], "colvec")                      # [1,10,1,1]
    arc = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("arc", arc, np.float32)
    n("Mul", ["colvec", "arc"], "colid_parts")
    n("ReduceSum", ["colid_parts"], "colid_f", axes=[1, 2, 3], keepdims=1)
    n("Cast", ["colid_f"], "colid_u8", to=TensorProto.UINT8)      # scalar colour

    # ---- rmin, cmin from footprint (1-D occupancy) ----
    n("ReduceSum", ["fpf"], "rowsum", axes=[3], keepdims=1)       # [1,1,30,1]
    n("ReduceSum", ["fpf"], "colsum", axes=[2], keepdims=1)       # [1,1,1,30]
    n("Greater", ["rowsum", "zeroS"], "rocc")
    n("Greater", ["colsum", "zeroS"], "cocc")
    n("Cast", ["rocc"], "roccf", to=F)
    n("Cast", ["cocc"], "coccf", to=F)
    init("Rrow", np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1), np.float32)
    init("Ccol", np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30), np.float32)
    init("c99", np.array(99.0, np.float32), np.float32)
    init("c1", np.array(1.0, np.float32), np.float32)
    n("Mul", ["Rrow", "roccf"], "Rocc")
    n("Sub", ["c1", "roccf"], "rinv")
    n("Mul", ["rinv", "c99"], "rinv99")
    n("Add", ["Rocc", "rinv99"], "Rbig")
    n("ReduceMin", ["Rbig"], "rmin", axes=[2], keepdims=1)        # [1,1,1,1]
    n("Mul", ["Ccol", "coccf"], "Cocc")
    n("Sub", ["c1", "coccf"], "cinv")
    n("Mul", ["cinv", "c99"], "cinv99")
    n("Add", ["Cocc", "cinv99"], "Cbig")
    n("ReduceMin", ["Cbig"], "cmin", axes=[3], keepdims=1)        # [1,1,1,1]

    # ---- align footprint to top-left of a 14x14 canvas via two Gathers ----
    # row source index rmin + Ri  (Ri 0..13); col source cmin + Ci
    init("arW", np.arange(WORK, dtype=np.float32).reshape(WORK), np.float32)
    n("Add", ["rmin", "arW"], "rsrc_f")          # [1,1,1,WORK] broadcast scalar+vec
    n("Add", ["cmin", "arW"], "csrc_f")
    # clamp to 0..29 so Gather is in-bounds (out-of-range rows are background 0)
    init("c0f", np.array(0.0, np.float32), np.float32)
    init("c29f", np.array(29.0, np.float32), np.float32)
    n("Clip", ["rsrc_f", "c0f", "c29f"], "rsrc_c")
    n("Clip", ["csrc_f", "c0f", "c29f"], "csrc_c")
    n("Cast", ["rsrc_c"], "rsrc", to=I64)        # [1,1,1,WORK]
    n("Cast", ["csrc_c"], "csrc", to=I64)
    n("Squeeze", ["rsrc"], "rsrc1", axes=[0, 1, 2])  # [WORK] int64
    n("Squeeze", ["csrc"], "csrc1", axes=[0, 1, 2])  # [WORK]
    n("Gather", ["fpf", "rsrc1"], "fp_r", axis=2)   # [1,1,WORK,30]
    n("Gather", ["fp_r", "csrc1"], "sp14", axis=3)  # [1,1,WORK,WORK] sprite@TL

    # ---- magnify: gather sp14 with index (Ro-1)//m for Ro 0..13 ----
    # tgt[Ro] = (Ro-1)//m  (valid for interior Ro 1..3m; clamp to 0..13)
    n("Sub", ["arW", "c1"], "arWm1")             # Ro-1
    n("Div", ["arWm1", "m"], "arWdiv")
    n("Floor", ["arWdiv"], "tgt_f")              # [WORK]
    init("c13f", np.array(13.0, np.float32), np.float32)
    n("Clip", ["tgt_f", "c0f", "c13f"], "tgt_c")
    n("Cast", ["tgt_c"], "tgt4", to=I64)         # [1,1,1,WORK]
    n("Squeeze", ["tgt4"], "tgt", axes=[0, 1, 2])   # [WORK]
    n("Gather", ["sp14", "tgt"], "mag_r", axis=2)   # [1,1,WORK,WORK]
    n("Gather", ["mag_r", "tgt"], "mag", axis=3)    # [1,1,WORK,WORK] magnified
    init("half", np.array(0.5, np.float32), np.float32)
    n("Greater", ["mag", "half"], "magB")        # bool [1,1,WORK,WORK]

    # ---- region geometry on 14x14 (index comparisons, 1-D) ----
    init("c3", np.array(3.0, np.float32), np.float32)
    init("c2", np.array(2.0, np.float32), np.float32)
    n("Mul", ["m", "c3"], "m3")                  # 3m
    n("Add", ["m3", "c2"], "G")                  # outsize = 3m+2
    arWr = np.arange(WORK, dtype=np.float32).reshape(1, 1, WORK, 1)
    arWc = np.arange(WORK, dtype=np.float32).reshape(1, 1, 1, WORK)
    init("arWr", arWr, np.float32)
    init("arWc", arWc, np.float32)
    # interior: 1 <= idx <= 3m
    n("Add", ["m3", "half"], "m3h")
    n("Greater", ["arWr", "half"], "Rge1"); n("Less", ["arWr", "m3h"], "Rle")
    n("And", ["Rge1", "Rle"], "RintB")           # [1,1,WORK,1]
    n("Greater", ["arWc", "half"], "Cge1"); n("Less", ["arWc", "m3h"], "Cle")
    n("And", ["Cge1", "Cle"], "CintB")           # [1,1,1,WORK]
    n("And", ["RintB", "CintB"], "interiorB")    # [1,1,WORK,WORK]
    # in-grid: idx <= G-1 (idx < G-0.5)
    n("Sub", ["G", "half"], "Gh")
    n("Less", ["arWr", "Gh"], "RinB"); n("Less", ["arWc", "Gh"], "CinB")
    n("And", ["RinB", "CinB"], "inGridB")        # [1,1,WORK,WORK]
    # border ring: (idx==0 or idx==G-1) within grid
    n("Less", ["arWr", "half"], "Rzero")
    n("Sub", ["G", "c1"], "Gm1"); n("Sub", ["Gm1", "half"], "Gm1h")
    n("Greater", ["arWr", "Gm1h"], "Rlast"); n("Or", ["Rzero", "Rlast"], "RbB")
    n("Less", ["arWc", "half"], "Czero")
    n("Greater", ["arWc", "Gm1h"], "Clast"); n("Or", ["Czero", "Clast"], "CbB")
    n("Or", ["RbB", "CbB"], "anyB")
    n("And", ["anyB", "inGridB"], "borderB")     # [1,1,WORK,WORK]

    # ---- magnified sprite mask, restricted to interior ----
    n("And", ["magB", "interiorB"], "spriteB")   # [1,1,WORK,WORK]

    # ---- uint8 label map on 14x14 ----
    init("v0", np.array(0, np.uint8), np.uint8)
    init("v2", np.array(2, np.uint8), np.uint8)
    init("v10", np.array(10, np.uint8), np.uint8)
    n("Where", ["inGridB", "v0", "v10"], "L0")       # 0 in-grid else 10
    n("Where", ["borderB", "v2", "L0"], "L1")        # red ring
    n("Where", ["spriteB", "colid_u8", "L1"], "L14")  # sprite colour (overrides bg)

    # ---- pad to 30x30 (sentinel 10), final Equal ----
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64),
         np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    n("Pad", ["L14", "padpads", "padval"], "L", mode="constant")  # [1,1,30,30]
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")              # free BOOL

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task159", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
