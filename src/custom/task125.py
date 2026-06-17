"""task125 (ARC-AGI 543a7ed5) — "frame each pink rectangle: outline green, fill holes yellow".

Rule (from the generator task_543a7ed5.py):
  A 15x15 grid, background cyan(8).  Several non-overlapping (separated by >=2) solid pink(6)
  rectangles are drawn; each has a smaller rectangular HOLE punched out of its interior (the
  hole shows as background cyan inside the pink rect).  In the OUTPUT, per rectangle:
    - pink cells stay pink(6),
    - the interior hole becomes yellow(4),
    - a 1-cell green(3) outline is drawn around the rectangle's bounding box.
  Background stays cyan(8).

  Closed-form, NO flood-fill (verified 0/800 fails):
    pink = input==6 ;  cyan = input==8
    aL/aR/aU/aD = directional prefix/suffix-OR of pink along the 4 axes
                  (input has only pink & cyan, so "first non-cyan = first pink")
    hole  = cyan AND aL AND aR AND aU AND aD           (enclosed cyan == hole)
    region= pink OR hole                               (the full rectangle = its bbox)
    green = dilate3x3(region) AND NOT region           (1-ring outline; boxes separated >=2)
    out   = 8 ; green->3 ; hole->4 ; pink->6

Encoding (route the 10-ch one-hot into the FREE output):
  Work on the active 15x15 canvas (grid is always 15x15).
    pink  = input[:, 6:7, 0:15, 0:15]                  [1,1,15,15] f32 -> f16
    cyan  = 1 - pink                                   (only two colors in the input)
  prefix/suffix-OR via strict-triangular MatMuls (rows: right-mul; cols: left-mul), >0.
    hole  = cyan AND aL AND aR AND aU AND aD
    region= MaxPool(... no) ; region = pink OR hole
    green = (MaxPool3x3(region) > 0) AND NOT region
  L = 8 - 5*green - 4*hole - 2*pink   (disjoint sets; fp16 exact for these small ints)
  Pad L to 30x30 with sentinel 99 (off-canvas -> matches no channel), then
    output = Equal(L30, arange[1,10,1,1])  -> BOOL [1,10,30,30]  (the free output).
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

W = 15  # active canvas (grid is always 15x15 for this task)
G = 30  # full ONNX canvas


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- pink plane (channel 6) on the 15x15 canvas ------------------------
    init("p_s", np.array([6, 0, 0], np.int64), np.int64)
    init("p_e", np.array([7, W, W], np.int64), np.int64)
    init("p_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "p_s", "p_e", "p_ax"], "pink_f32")  # [1,1,15,15] f32
    n("Cast", ["pink_f32"], "pink", to=F16)                  # f16 {0,1}

    # ---- strict-triangular prefix/suffix matrices --------------------------
    # rows (axis=3): pink @ M
    SU = np.triu(np.ones((W, W), np.float16), k=1)  # SU[c',c]=1 iff c'<c  (any to the LEFT)
    SL = np.tril(np.ones((W, W), np.float16), k=-1)  # SL[c',c]=1 iff c'>c  (any to the RIGHT)
    init("SU", SU.reshape(1, 1, W, W), np.float16)
    init("SL", SL.reshape(1, 1, W, W), np.float16)

    n("MatMul", ["pink", "SU"], "sumL")   # anyL accumulator [1,1,15,15]
    n("MatMul", ["pink", "SL"], "sumR")   # anyR
    n("MatMul", ["SL", "pink"], "sumU")   # anyU  (r' above)
    n("MatMul", ["SU", "pink"], "sumD")   # anyD  (r' below)

    # pink as a BOOL mask (cheap; arithmetic pink-f16 only needed for the matmuls)
    init("zero", np.array([0.0], np.float16), np.float16)
    init("zf32", np.array([0.0], np.float32), np.float32)
    n("Greater", ["pink_f32", "zf32"], "pinkb")   # bool [1,1,15,15]

    # enclosed = (sumL>0)&(sumR>0)&(sumU>0)&(sumD>0).  Each sum is a nonneg fp16 count
    # (<=14), so the product is >0 iff BOTH halves are >0 (exact-at-zero, no overflow).
    n("Mul", ["sumL", "sumR"], "mLR")
    n("Mul", ["sumU", "sumD"], "mUD")
    n("Greater", ["mLR", "zero"], "encLR")        # bool
    n("Greater", ["mUD", "zero"], "encUD")        # bool
    n("And", ["encLR", "encUD"], "encb")          # bool: enclosed = hole OR pink-interior

    # region = enclosed OR pink (the full bbox).  hole/green need no NOT masking because the
    # Where priority chain below applies them BEFORE pink (which always overrides to 6) and
    # every region cell is pinkb OR encb, so the lower-priority "green=dilation" is overwritten.
    n("Or", ["encb", "pinkb"], "regionb")         # bool: full bbox region

    # green = dilate3x3(region).  MaxPool needs float -> cast region to f16.
    n("Cast", ["regionb"], "region", to=F16)      # f16 {0,1}
    n("MaxPool", ["region"], "dil", kernel_shape=[3, 3], pads=[1, 1, 1, 1],
      strides=[1, 1])                              # 3x3 dilation, f16 {0,1}
    n("Greater", ["dil", "zero"], "dilb")          # bool: region + 1-ring (dilation)

    # ---- color-index plane via Where priority chain (bool masks, fp16 scalars) ----
    # Lowest priority first: dilation(green=3) < enclosed(hole=4) < pink(6); default bg=8.
    #   dilb=1, not enc, not pink  -> the outline ring -> 3
    #   encb=1, not pink           -> hole -> 4
    #   pinkb=1                    -> pink -> 6
    init("c8", np.array([8.0], np.float16), np.float16)
    init("c3", np.array([3.0], np.float16), np.float16)
    init("c4", np.array([4.0], np.float16), np.float16)
    init("c6", np.array([6.0], np.float16), np.float16)
    n("Where", ["dilb", "c3", "c8"], "Lg")        # dilation->3 else 8
    n("Where", ["encb", "c4", "Lg"], "Lgh")       # enclosed->4 (overrides ring)
    n("Where", ["pinkb", "c6", "Lgh"], "L15f")    # pink->6 (overrides hole/ring)
    n("Cast", ["L15f"], "L15", to=U8)             # uint8 index plane (halves the 30x30 pad)

    # ---- pad to 30x30 with sentinel 99 (off-canvas -> matches no channel) ----
    init("pads", np.array([0, 0, 0, 0, 0, 0, G - W, G - W], np.int64), np.int64)
    init("sent", np.array(99, np.uint8), np.uint8)
    n("Pad", ["L15", "pads", "sent"], "L30", mode="constant")  # [1,1,30,30] u8

    # ---- one-hot expansion into the FREE bool output -----------------------
    init("arange", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L30", "arange"], "output")     # BOOL [1,10,30,30]

    graph = helper.make_graph(
        nodes, "task125",
        [helper.make_tensor_value_info("input", F32, [1, 10, G, G])],
        [helper.make_tensor_value_info("output", BOOL, [1, 10, G, G])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
