"""Task 281 (b548a754): a hollow box (outer border + inner fill) is stretched
along the axis of an isolated cyan dot until its edge reaches the dot; the dot
is erased.  xpose/flip are applied to BOTH input and output, so the rule is the
orientation-agnostic stretch.

Key facts (verified exact over 50k arc-gen instances):
  - the stretched frame is exactly the FILLED bounding box of all nonzero cells
    (box + dot): rows min..max occupied, cols min..max occupied.
  - bbox border ring -> outer color; eroded interior -> inner color.
  - inner color = the colour whose cells form a SOLID filled rectangle
    (cell-count == its own bbox area); outer = the other present non-cyan colour
    (a hollow ring, count == perimeter < area).
  - the box is a fully-filled rectangle once stretched, so the output interior
    is the eroded filled bbox.

Memory floor-break (no 2-D colour/neighbour plane):
  Per-channel counts & bboxes are tiny [1,10,1,1] tensors, so inner_id / outer_id
  are recovered with pure 1-D arithmetic -- no 30x30 colorid / nonzero / 3x3
  neighbour planes.  The filled bbox and its erosion are separable outer
  products of 1-D row/col spans (one 900B bool plane each).  A single uint8
  label map L (10 = outside grid; 0 = in-grid background; outer_id on the
  border; inner_id on the interior) is finished with Equal(L, arange[1,10,1,1])
  into the free BOOL `output` (opset 11).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

N = 30
F16 = TensorProto.FLOAT16


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    init("half", np.array([0.5], np.float32), np.float32)
    init("one16", np.array(1.0, np.float16), np.float16)
    ih = np.arange(N, dtype=np.float16).reshape(1, 1, N, 1)
    iw = np.arange(N, dtype=np.float16).reshape(1, 1, 1, N)
    init("ih", ih, np.float16)
    init("iw", iw, np.float16)
    init("big", np.array(1000.0, np.float16), np.float16)
    init("nbig", np.array(-1000.0, np.float16), np.float16)

    # --- per-channel row/col counts (f32) and presence ---
    n("ReduceSum", ["input"], "rc", axes=[3], keepdims=1)   # [1,10,30,1]
    n("ReduceSum", ["input"], "cc", axes=[2], keepdims=1)   # [1,10,1,30]
    n("Greater", ["rc", "half"], "rpb")                     # [1,10,30,1] bool
    n("Greater", ["cc", "half"], "cpb")                     # [1,10,1,30] bool

    # per-channel bbox r0/r1/c0/c1 (f16 indices)
    n("Where", ["rpb", "ih", "big"], "rlo")
    n("ReduceMin", ["rlo"], "r0", axes=[2], keepdims=1)     # [1,10,1,1]
    n("Where", ["rpb", "ih", "nbig"], "rhi")
    n("ReduceMax", ["rhi"], "r1", axes=[2], keepdims=1)
    n("Where", ["cpb", "iw", "big"], "clo")
    n("ReduceMin", ["clo"], "c0", axes=[3], keepdims=1)
    n("Where", ["cpb", "iw", "nbig"], "chi")
    n("ReduceMax", ["chi"], "c1", axes=[3], keepdims=1)

    # per-channel area = (r1-r0+1)*(c1-c0+1)  (f16, <= 900, exact)
    n("Sub", ["r1", "r0"], "dh")
    n("Sub", ["c1", "c0"], "dw")
    n("Add", ["dh", "one16"], "bh")
    n("Add", ["dw", "one16"], "bw")
    n("Mul", ["bh", "bw"], "area")                          # [1,10,1,1]

    # per-channel total cell count (f16)
    n("ReduceSum", ["rc"], "cnt_f", axes=[2], keepdims=1)   # [1,10,1,1] f32
    n("Cast", ["cnt_f"], "cnt", to=F16)
    n("ReduceMax", ["input"], "present", axes=[2, 3], keepdims=1)  # [1,10,1,1]
    n("Cast", ["present"], "presentH", to=F16)

    # inner channel = present, count == area (solid rectangle), not cyan/bg.
    notc = np.ones((1, 10, 1, 1), np.float16)
    notc[0, 0, 0, 0] = 0.0   # background
    notc[0, 8, 0, 0] = 0.0   # cyan
    init("notc", notc, np.float16)
    n("Equal", ["cnt", "area"], "solidB")                   # [1,10,1,1] bool
    n("Cast", ["solidB"], "solid", to=F16)
    n("Mul", ["solid", "presentH"], "in0")
    n("Mul", ["in0", "notc"], "innerChan")                  # [1,10,1,1] onehot f16

    ar10 = np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1)
    init("ar10", ar10, np.float16)
    n("Mul", ["innerChan", "ar10"], "in_idp")
    n("ReduceMax", ["in_idp"], "inner_f", axes=[1, 2, 3], keepdims=1)  # [1,1,1,1]
    n("Cast", ["inner_f"], "inner_id", to=TensorProto.UINT8)  # scalar uint8

    # outer channel = present non-cyan, not the inner channel.
    n("Sub", ["presentH", "innerChan"], "ptmp")            # present minus inner
    n("Mul", ["ptmp", "notc"], "outerChan")                # [1,10,1,1] onehot f16
    n("Mul", ["outerChan", "ar10"], "out_idp")
    n("ReduceMax", ["out_idp"], "outer_f", axes=[1, 2, 3], keepdims=1)
    n("Cast", ["outer_f"], "outer_id", to=TensorProto.UINT8)  # scalar uint8

    # --- filled bbox of ALL nonzero cells (separable 1-D spans) ---
    # nonzero(colour) row/col occupancy = sum over channels 1..9 of the per-
    # channel row/col counts rc/cc (a 1x1 Conv over the channel axis): a row/col
    # is "occupied" iff it holds any non-background cell.
    n("ReduceMax", ["input"], "anyrow", axes=[1, 3], keepdims=1)  # [1,1,30,1]
    n("ReduceMax", ["input"], "anycol", axes=[1, 2], keepdims=1)  # [1,1,1,30]
    w_nz = np.zeros((1, 10, 1, 1), np.float32)
    w_nz[0, 1:, 0, 0] = 1.0
    init("w_nz", w_nz, np.float32)
    n("Conv", ["rc", "w_nz"], "nzrow")                          # [1,1,30,1] f32
    n("Conv", ["cc", "w_nz"], "nzcol")                          # [1,1,1,30] f32
    n("Greater", ["nzrow", "half"], "nzrowB")                   # [1,1,30,1] bool
    n("Greater", ["nzcol", "half"], "nzcolB")                   # [1,1,1,30] bool
    n("Cast", ["nzrowB"], "nzrowF", to=F16)
    n("Cast", ["nzcolB"], "nzcolF", to=F16)

    # filled span = prefix-max AND suffix-max of the 1-D occupancy (MaxPool)
    n("MaxPool", ["nzrowF"], "rpre", kernel_shape=[30, 1], pads=[29, 0, 0, 0])
    n("MaxPool", ["nzrowF"], "rsuf", kernel_shape=[30, 1], pads=[0, 0, 29, 0])
    n("Mul", ["rpre", "rsuf"], "rowocc")                        # [1,1,30,1] f16
    n("MaxPool", ["nzcolF"], "cpre", kernel_shape=[1, 30], pads=[0, 29, 0, 0])
    n("MaxPool", ["nzcolF"], "csuf", kernel_shape=[1, 30], pads=[0, 0, 0, 29])
    n("Mul", ["cpre", "csuf"], "colocc")                        # [1,1,1,30] f16

    # 3-wide erosion -> interior span
    init("kr", np.ones((1, 1, 3, 1), np.float16), np.float16)
    init("be", np.array([-2.0], np.float16), np.float16)
    n("Conv", ["rowocc", "kr", "be"], "rint_r", pads=[1, 0, 1, 0])
    n("Relu", ["rint_r"], "rowint")                             # [1,1,30,1]
    init("kc", np.ones((1, 1, 1, 3), np.float16), np.float16)
    n("Conv", ["colocc", "kc", "be"], "cint_c", pads=[0, 1, 0, 1])
    n("Relu", ["cint_c"], "colint")                             # [1,1,1,30]

    init("halfH", np.array([0.5], np.float16), np.float16)
    n("Greater", ["rowint", "halfH"], "rintB")                  # [1,1,30,1]
    n("Greater", ["colint", "halfH"], "cintB")                  # [1,1,1,30]
    n("And", ["rintB", "cintB"], "interiorB")                   # [1,1,30,30]
    n("Greater", ["rowocc", "halfH"], "roccB")                  # [1,1,30,1]
    n("Greater", ["colocc", "halfH"], "coccB")                  # [1,1,1,30]
    n("And", ["roccB", "coccB"], "bboxB")                       # [1,1,30,30]

    # in-grid region (separable, off `input`)
    n("Greater", ["anyrow", "half"], "growB")                   # [1,1,30,1]
    n("Greater", ["anycol", "half"], "gcolB")                   # [1,1,1,30]
    n("And", ["growB", "gcolB"], "gridB")                       # [1,1,30,30]

    # --- uint8 label map L ---
    init("v0", np.array(0, np.uint8), np.uint8)
    init("v10", np.array(10, np.uint8), np.uint8)
    n("Where", ["gridB", "v0", "v10"], "L0")
    n("Where", ["bboxB", "outer_id", "L0"], "L1")
    n("Where", ["interiorB", "inner_id", "L1"], "L")

    init("chan10", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan10"], "output")

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, N, N])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, N, N])
    graph = helper.make_graph(nodes, "task281", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
