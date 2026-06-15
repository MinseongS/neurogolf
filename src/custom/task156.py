"""Task 156 (ARC-AGI 694f12f3): colour the interiors of two yellow rectangles by
size — the SMALLER gets blue(1), the LARGER gets red(2).

Rule (verified exact on all 265 stored examples + fresh arc-gen). The input has
exactly two solid yellow(4) rectangles, vertically separated. The output keeps
the yellow cells and recolours each rectangle's interior (the inset-by-1
sub-rectangle) by its area: the rectangle with the smaller interior area becomes
blue(1), the larger red(2).

Memory floor-break (10x10 working canvas + label map + final Equal):

  Grid is always 10x10.  All per-cell work is done on a [1,1,10,10] canvas
  (100 elements) instead of [1,1,30,30] (900 elements).

  1. Slice ch4 (yellow channel) of input to [1,1,10,10]; cast to fp16 for Conv.
  2. 3x3 all-ones Conv on ch4_16 -> cnt [1,1,10,10] fp16; threshold > 8.5 gives
     Imask [1,1,10,10] bool (interior = all 9 neighbours yellow).
  3. Band separation on 10-element row vectors:
     - rowint = ReduceSum(Imask_f, axis=3) -> [1,1,10,1] f32 = per-row interior count
     - R = rowint > 0 (bool -> f32) -> [1,1,10,1]
     - strict cumulative = Linc_10 @ R - R  (Linc_10 [10,10] lower-triangular params)
     - gap row = seen AND NOT R; gapcum = Linc_10 @ isgap; below = gapcum > 0.5
     - botcnt, topcnt -> bigbottom scalar -> per-row redrow/bluerow [1,1,10,1]
  4. Build uint8 label map L [1,1,10,10]:
       L = 0 (background)
       L = 4 (yellow) where ch4 is on (yellow_u8 overrides)
       L = 1 (blue)   where Imask AND bluerow
       L = 2 (red)    where Imask AND redrow
     Three Where ops, correct priority (interior overrides yellow).
  5. Pad L to [1,1,30,30] with sentinel 10.
  6. output = Equal(L, arange[1,10,1,1]) -> free BOOL output (opset 11).

No [1,10,30,30] or [1,1,30,30] float intermediates ever materialise; the only
30x30 tensor is the final padded uint8 label L (900B), just before Equal.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

WORK = 10   # grid is always 10x10


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- constants ----
    init("half_f", np.array(0.5, np.float32), np.float32)
    init("half_f16", np.array(0.5, np.float16), np.float16)
    init("th85_f16", np.array(8.5, np.float16), np.float16)
    init("zero_f", np.array(0.0, np.float32), np.float32)
    init("one_f", np.array(1.0, np.float32), np.float32)

    # ---- slice yellow channel (ch4) to [1,1,WORK,WORK] ----
    init("sl_st", np.array([4, 0, 0], np.int64), np.int64)
    init("sl_en", np.array([5, WORK, WORK], np.int64), np.int64)
    init("sl_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "sl_st", "sl_en", "sl_ax"], "ch4_f")    # [1,1,10,10] f32
    n("Cast", ["ch4_f"], "ch4_16", to=TensorProto.FLOAT16)        # [1,1,10,10] fp16

    # ---- 3x3 all-ones Conv counts yellow neighbours (incl. self) -> [1,1,10,10] fp16
    Wint = np.ones((1, 1, 3, 3), np.float16)
    init("Wint16", Wint, np.float16)
    n("Conv", ["ch4_16", "Wint16"], "cnt16",
      kernel_shape=[3, 3], pads=[1, 1, 1, 1])                     # [1,1,10,10] fp16
    # interior = all 9 neighbours are yellow (cnt == 9, i.e. > 8.5)
    n("Greater", ["cnt16", "th85_f16"], "Imask")                   # [1,1,10,10] bool

    # ---- per-row interior count -> band detection ----
    n("Cast", ["Imask"], "Imask_f", to=TensorProto.FLOAT)          # [1,1,10,10] f32
    n("ReduceSum", ["Imask_f"], "rowint",
      axes=[3], keepdims=1)                                        # [1,1,10,1] f32
    n("Greater", ["rowint", "zero_f"], "Rb")                       # [1,1,10,1] bool
    n("Cast", ["Rb"], "R", to=TensorProto.FLOAT)                   # [1,1,10,1] f32

    # ---- band separation: inclusive lower-triangular MatMul on 10x10 ----
    Linc = np.tril(np.ones((10, 10), np.float32)).reshape(1, 1, 10, 10)
    init("Linc10", Linc, np.float32)

    n("MatMul", ["Linc10", "R"], "incR")                          # [1,1,10,1] f32
    n("Sub", ["incR", "R"], "seenf")                              # strict cum
    n("Greater", ["seenf", "half_f"], "seenb")                    # [1,1,10,1] bool
    n("Cast", ["seenb"], "seen", to=TensorProto.FLOAT)
    n("Sub", ["one_f", "R"], "notR")
    n("Mul", ["seen", "notR"], "isgap")                           # gap rows
    n("MatMul", ["Linc10", "isgap"], "gapcum")                    # [1,1,10,1] f32
    n("Greater", ["gapcum", "half_f"], "belowb")
    n("Cast", ["belowb"], "below", to=TensorProto.FLOAT)          # [1,1,10,1]
    n("Sub", ["one_f", "below"], "above")                         # [1,1,10,1]

    # ---- compare band areas -> bigbottom scalar ----
    n("Mul", ["rowint", "below"], "botrows")
    n("Mul", ["rowint", "above"], "toprows")
    n("ReduceSum", ["botrows"], "botcnt", axes=[2], keepdims=1)   # [1,1,1,1]
    n("ReduceSum", ["toprows"], "topcnt", axes=[2], keepdims=1)   # [1,1,1,1]
    n("Greater", ["botcnt", "topcnt"], "bbb")
    n("Cast", ["bbb"], "bb", to=TensorProto.FLOAT)                # [1,1,1,1] f32

    # ---- per-row redrow/bluerow [1,1,10,1] ----
    n("Sub", ["one_f", "bb"], "nbb")
    n("Mul", ["below", "bb"], "t1")
    n("Mul", ["above", "nbb"], "t2")
    n("Add", ["t1", "t2"], "redrow")                              # [1,1,10,1]
    n("Sub", ["one_f", "redrow"], "bluerow")

    # ---- uint8 label map L [1,1,10,10] ----
    # L = 0 (bg) everywhere first
    # Override with yellow (4) where ch4 is on
    # Override with blue (1) or red (2) where Imask AND bluerow/redrow
    #
    # Build Imask_redrow / Imask_bluerow: broadcast [1,1,10,1] over [1,1,10,10]
    n("Greater", ["redrow", "half_f"], "redrow_b")                # [1,1,10,1] bool
    n("And", ["Imask", "redrow_b"], "red_cell")                   # [1,1,10,10] bool
    n("Greater", ["bluerow", "half_f"], "bluerow_b")
    n("And", ["Imask", "bluerow_b"], "blue_cell")                 # [1,1,10,10] bool

    # yellow_b: ch4_f > 0.5 (bool)
    n("Greater", ["ch4_f", "half_f"], "yellow_b")                 # [1,1,10,10] bool

    # Sentinel for outside: we use Pad to 30x30 with value 10 after
    # Build label: background=0, yellow=4, blue=1, red=2
    init("v0", np.array(0, np.uint8), np.uint8)
    init("v1", np.array(1, np.uint8), np.uint8)
    init("v2", np.array(2, np.uint8), np.uint8)
    init("v4", np.array(4, np.uint8), np.uint8)

    n("Where", ["yellow_b", "v4", "v0"], "Ly")                   # 4 where yellow, 0 else
    n("Where", ["blue_cell", "v1", "Ly"], "Lb")                  # 1 where blue interior
    n("Where", ["red_cell", "v2", "Lb"], "L10")                  # 2 where red interior

    # ---- Pad L to [1,1,30,30] with sentinel 10, then Equal -> BOOL output ----
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    n("Pad", ["L10", "padpads", "padval"], "L30", mode="constant") # [1,1,30,30] u8

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L30", "chan"], "output")                          # BOOL [1,10,30,30]

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task156", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
