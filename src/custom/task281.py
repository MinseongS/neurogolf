"""task281 (ARC-AGI b548a754) — stretch the bordered box toward the cyan dot.

Rule (from the generator task_b548a754.py):
  Input: a rectangular box (1px OUTER-colour border, INNER-colour interior) placed
  near one edge, plus a single cyan(8) dot offset away from the box along the box's
  axis (within the box's perpendicular span).  Output: the box is stretched so its
  edge reaches the dot's line — i.e. the output rectangle = the BOUNDING BOX of all
  non-background cells (box ∪ dot), drawn as a 1px outer-colour frame with an
  inner-colour interior.  The cyan dot is removed.  xpose/flip are applied to both
  grids so the stretch direction is arbitrary; "union-bbox -> framed rect" is
  orientation-independent.  Verified exact (0/500 fresh fails) by numpy reference.

Memory floor-break — NO 2-D colour/neighbour plane at all:
  * colours via tiny [1,10,1,1] arithmetic: in the INPUT the inner colour is a SOLID
    filled rectangle (count == its own bbox area), the outer colour is a hollow ring
    (count == perimeter < area), so inner_id = present non-cyan colour with cnt==area,
    outer_id = the OTHER present non-cyan colour (sum of both indices − inner).
  * union bbox is a separable outer product of 1-D row/col occupancy spans, recovered
    on a 13x13 active canvas (grids are 11..13) via NON-strict triangular prefix/
    suffix-OR (task070); the interior is the same with STRICT triangulars (erode 1).
  * non-bg row/col occupancy = a [0,1,..,1] channel Conv on the per-channel row/col
    counts (ch0 excluded) — 1-D, never a 2-D occupancy plane.
  Single uint8 label map L13 (0 in-grid bg / outer on rect / inner on interior /
  sentinel 10 off-grid), padded to 30x30 with 10, Equal(L,arange) into the FREE BOOL
  output.  Dominant intermediate is that one 30x30 uint8 map (~900B).
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

W = 13  # active canvas (grid is always 11..13)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    init("Z", np.array(0.0, np.float16), np.float16)
    init("half", np.array(0.5, np.float32), np.float32)
    init("ONE16", np.array(1.0, np.float16), np.float16)
    init("cl_ax", np.array([0, 1, 2, 3], np.int64), np.int64)

    # ====================================================================
    # COLOURS  (tiny [1,10,1,1] arithmetic — no 2-D plane)
    # ====================================================================
    # per-channel total cell count (f16) and presence.  The two box colours are the
    # only present non-cyan colours; the INNER colour is a solid (t-2)(w-2) rectangle
    # and the OUTER a 2(t+w)-4 ring — for t,w∈[3,5] the ring ALWAYS has more cells, so
    #   inner_id = MIN-count present non-cyan colour ; outer_id = MAX-count one.
    n("ReduceSum", ["input"], "cnt_f", axes=[2, 3], keepdims=1)    # [1,10,1,1] f32
    n("Cast", ["cnt_f"], "cnt", to=F16)
    n("ReduceMax", ["input"], "present", axes=[2, 3], keepdims=1)  # [1,10,1,1] f32
    n("Cast", ["present"], "presentH", to=F16)
    notc = np.ones((1, 10, 1, 1), np.float16)
    notc[0, 0, 0, 0] = 0.0
    notc[0, 8, 0, 0] = 0.0
    init("notc", notc, np.float16)
    n("Mul", ["presentH", "notc"], "pnc")          # present non-cyan mask [1,10,1,1]

    init("BIG", np.array(1000.0, np.float16), np.float16)
    ar10 = np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1)
    init("ar10", ar10, np.float16)

    # inner = channel with MIN count among present non-cyan
    n("Sub", ["ONE16", "pnc"], "absent")           # 1 where absent/cyan/bg
    n("Mul", ["absent", "BIG"], "pad_big")         # BIG where absent
    n("Add", ["cnt", "pad_big"], "cnt_in")         # absent channels -> huge
    n("ReduceMin", ["cnt_in"], "mincnt", axes=[1], keepdims=1)  # [1,1,1,1]
    n("Equal", ["cnt_in", "mincnt"], "innerChanB")  # [1,10,1,1] bool (the min ch)
    n("Cast", ["innerChanB"], "innerChan", to=F16)
    n("Mul", ["innerChan", "ar10"], "in_idp")
    n("ReduceMax", ["in_idp"], "inner_s", axes=[1, 2, 3], keepdims=1)  # [1,1,1,1]

    # outer = the OTHER present non-cyan channel (MAX count)
    n("Mul", ["cnt", "pnc"], "cnt_pnc")            # count, 0 elsewhere
    n("ReduceMax", ["cnt_pnc"], "maxcnt", axes=[1], keepdims=1)
    n("Equal", ["cnt_pnc", "maxcnt"], "outerChanB0")
    n("Cast", ["outerChanB0"], "outerChan0", to=F16)
    n("Mul", ["outerChan0", "pnc"], "outerChan")   # keep only present non-cyan
    n("Mul", ["outerChan", "ar10"], "out_idp")
    n("ReduceMax", ["out_idp"], "outer_s", axes=[1, 2, 3], keepdims=1)  # [1,1,1,1]

    # ====================================================================
    # MASKS  (separable 1-D occupancy on a 13x13 canvas)
    # ====================================================================
    # non-bg row/col occupancy = [0,1,..,1] channel Conv on per-channel row/col
    # counts (ch0 excluded), all in fp16, sliced to the 13x13 canvas.
    n("ReduceSum", ["input"], "rc_f", axes=[3], keepdims=1)   # [1,10,30,1] f32
    n("ReduceSum", ["input"], "cc_f", axes=[2], keepdims=1)   # [1,10,1,30] f32
    w_nz = np.zeros((1, 10, 1, 1), np.float32)
    w_nz[0, 1:, 0, 0] = 1.0
    init("w_nz", w_nz, np.float32)
    n("Conv", ["rc_f", "w_nz"], "nzrow30")                  # [1,1,30,1] f32
    n("Conv", ["cc_f", "w_nz"], "nzcol30")                  # [1,1,1,30] f32
    init("r13_s", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("r13_e", np.array([1, 1, W, 1], np.int64), np.int64)
    init("c13_e", np.array([1, 1, 1, W], np.int64), np.int64)
    n("Slice", ["nzrow30", "r13_s", "r13_e", "cl_ax"], "nzrow_f")  # [1,1,13,1] f32
    n("Slice", ["nzcol30", "r13_s", "c13_e", "cl_ax"], "nzcol_f")  # [1,1,1,13] f32
    n("Cast", ["nzrow_f"], "rowhas", to=F16)                # [1,1,13,1] f16
    n("Cast", ["nzcol_f"], "colhas", to=F16)                # [1,1,1,13] f16

    # in-grid occupancy (any channel incl. ch0) for the off-grid sentinel
    n("ReduceMax", ["input"], "rowany30", axes=[1, 3], keepdims=1)  # [1,1,30,1]
    n("ReduceMax", ["input"], "colany30", axes=[1, 2], keepdims=1)  # [1,1,1,30]
    n("Slice", ["rowany30", "r13_s", "r13_e", "cl_ax"], "rowany")
    n("Slice", ["colany30", "r13_s", "c13_e", "cl_ax"], "colany")
    n("Greater", ["rowany", "half"], "rowany_b")
    n("Greater", ["colany", "half"], "colany_b")
    n("And", ["rowany_b", "colany_b"], "ingrid_b")          # [1,1,13,13]

    # triangular prefix/suffix matrices (non-strict for rect, strict for interior)
    NS_low = np.tril(np.ones((W, W), np.float16))           # k<=r
    NS_up = np.triu(np.ones((W, W), np.float16))            # k>=r
    S_low = np.tril(np.ones((W, W), np.float16), -1)        # k<r
    S_up = np.triu(np.ones((W, W), np.float16), 1)          # k>r
    init("NSlow", NS_low.reshape(1, 1, W, W), np.float16)
    init("NSup", NS_up.reshape(1, 1, W, W), np.float16)
    init("Slow", S_low.reshape(1, 1, W, W), np.float16)
    init("Sup", S_up.reshape(1, 1, W, W), np.float16)

    # rect span: (∃ occ ≤ r) AND (∃ occ ≥ r)
    n("MatMul", ["NSlow", "rowhas"], "pr_r")
    n("MatMul", ["NSup", "rowhas"], "sr_r")
    n("Greater", ["pr_r", "Z"], "pr_rb")
    n("Greater", ["sr_r", "Z"], "sr_rb")
    n("And", ["pr_rb", "sr_rb"], "row_in")        # [1,1,13,1]
    n("MatMul", ["colhas", "NSup"], "pr_c")       # prefix M[k,c]=[k<=c]=NSup
    n("MatMul", ["colhas", "NSlow"], "sr_c")      # suffix [k>=c]=NSlow
    n("Greater", ["pr_c", "Z"], "pr_cb")
    n("Greater", ["sr_c", "Z"], "sr_cb")
    n("And", ["pr_cb", "sr_cb"], "col_in")        # [1,1,1,13]

    # interior span: STRICT (∃ occ < r) AND (∃ occ > r) -> erode by 1 each end
    n("MatMul", ["Slow", "rowhas"], "ipr_r")
    n("MatMul", ["Sup", "rowhas"], "isr_r")
    n("Greater", ["ipr_r", "Z"], "ipr_rb")
    n("Greater", ["isr_r", "Z"], "isr_rb")
    n("And", ["ipr_rb", "isr_rb"], "row_int")     # [1,1,13,1]
    n("MatMul", ["colhas", "Sup"], "ipr_c")       # ∃ occ k<c -> [k<c]=Sup
    n("MatMul", ["colhas", "Slow"], "isr_c")      # ∃ occ k>c -> [k>c]=Slow
    n("Greater", ["ipr_c", "Z"], "ipr_cb")
    n("Greater", ["isr_c", "Z"], "isr_cb")
    n("And", ["ipr_cb", "isr_cb"], "col_int")     # [1,1,1,13]

    n("And", ["row_in", "col_in"], "rect_b")      # [1,1,13,13]
    n("And", ["row_int", "col_int"], "intr_b")    # [1,1,13,13]

    # ====================================================================
    # LABEL MAP L13 -> pad -> Equal -> FREE BOOL output
    # ====================================================================
    n("Cast", ["rect_b"], "rect_f", to=F16)
    n("Cast", ["intr_b"], "intr_f", to=F16)
    n("Mul", ["rect_f", "outer_s"], "L_outer")
    n("Sub", ["inner_s", "outer_s"], "diff_s")
    n("Mul", ["intr_f", "diff_s"], "L_diff")
    n("Add", ["L_outer", "L_diff"], "Lcol")        # [1,1,13,13] : 0/outer/inner
    n("Cast", ["Lcol"], "Lcol_u8", to=U8)
    init("v10", np.array(10, np.uint8), np.uint8)
    n("Where", ["ingrid_b", "Lcol_u8", "v10"], "L13")   # off-grid -> sentinel 10

    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), np.int64)
    n("Pad", ["L13", "pads", "v10"], "L30", mode="constant")  # [1,1,30,30] u8
    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["L30", "chan"], "output")          # [1,10,30,30] BOOL (free)

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task281", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
