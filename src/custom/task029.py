"""task029 (ARC 1c786137): crop the interior of the unique rectangular ring.

Rule: random static (3-4 colours) + one hollow rectangle drawn in a colour the
static never uses.  Output = the static content strictly inside that ring,
translated to the top-left of the (zero-padded 30x30) output.

Encoding (count-based detection, colorid double-Gather crop):
  * Per-channel row/col counts (ReduceSum on the FREE input) drive ring
    detection: the ring colour is the unique channel whose total pixel count
    equals 2*rowmax + 2*colmax - 4 (rectangle perimeter) AND whose max-count
    row/col each occur at >=2 distinct positions.
  * bbox edges (r0,r1,c0,c1) recovered per channel via ArgMax-first / ArgMax-
    last on the counts, then Gather'd by the winning colour index.
  * crop = double Gather of a sentinel-padded uint8 colour-id plane; out-of-
    region output rows/cols index the pad row/col (sentinel 10) so the gathered
    label map L is 0..9 inside the interior, 10 outside.
  * output = Equal(L_uint8, arange[1,10,1,1]) -> free BOOL (sentinel 10 hits
    nothing -> all channels off outside the interior).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from src.harness import IR_VERSION

N = 30
I32 = TensorProto.INT32
I64 = TensorProto.INT64
F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL


def build(task):
    nodes, inits = [], []

    def init(name, arr, dtype=None):
        a = np.ascontiguousarray(arr, dtype=dtype) if dtype is not None else np.ascontiguousarray(arr)
        inits.append(numpy_helper.from_array(a, name))
        return name

    def nd(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- per-channel row/col counts (FREE input) -------------------------
    nd("ReduceSum", ["input"], "row_counts", axes=[3], keepdims=1)  # [1,10,30,1] f32
    nd("ReduceSum", ["input"], "col_counts", axes=[2], keepdims=1)  # [1,10,1,30] f32

    # ---- ring detection: total == 2*rowmax + 2*colmax - 4 ----------------
    nd("ReduceSum", ["row_counts"], "total_count", axes=[2], keepdims=1)   # [1,10,1,1]
    nd("ReduceMax", ["row_counts"], "row_max", axes=[2], keepdims=1)       # box width
    nd("ReduceMax", ["col_counts"], "col_max", axes=[3], keepdims=1)       # box height
    init("minus_four_f", np.array(-4.0, np.float32))
    nd("Sum", ["row_max", "row_max", "col_max", "col_max", "minus_four_f"], "perim_count")
    nd("Equal", ["total_count", "perim_count"], "total_ok_b")

    # max-count row/col must occur at >=2 distinct positions (first<last argmax)
    nd("ArgMax", ["row_counts"], "row_first", axis=2, keepdims=1)
    nd("ArgMax", ["row_counts"], "row_last", axis=2, keepdims=1, select_last_index=1)
    nd("Less", ["row_first", "row_last"], "two_rows_b")
    nd("ArgMax", ["col_counts"], "col_first", axis=3, keepdims=1)
    nd("ArgMax", ["col_counts"], "col_last", axis=3, keepdims=1, select_last_index=1)
    nd("Less", ["col_first", "col_last"], "two_cols_b")

    nd("And", ["total_ok_b", "two_rows_b"], "valid0_b")
    nd("And", ["valid0_b", "two_cols_b"], "valid_b")          # [1,10,1,1] bool
    nd("Cast", ["valid_b"], "valid_u8", to=U8)
    nd("ArgMax", ["valid_u8"], "color_idx", axis=1, keepdims=0)   # [1,1,1] int64

    # winning bbox edges (scalars via Gather over channel axis)
    for src, dst in [("row_first", "r0"), ("row_last", "r1"),
                     ("col_first", "c0"), ("col_last", "c1")]:
        nd("Gather", [src, "color_idx"], dst + "_g", axis=1)        # [1,1,1,1,1,1] int64
        nd("Squeeze", [dst + "_g"], dst + "_i64", axes=[0, 1, 2, 3, 4, 5])
        nd("Cast", [dst + "_i64"], dst, to=I32)                     # scalar int32

    # interior origin (r0+1, c0+1) and interior span (r1-r0-1, c1-c0-1)
    init("one_i32", np.array(1, np.int32))
    nd("Add", ["r0", "one_i32"], "ir0")     # interior top
    nd("Add", ["c0", "one_i32"], "ic0")     # interior left
    nd("Sub", ["r1", "r0"], "rspan0")
    nd("Sub", ["rspan0", "one_i32"], "ih")  # interior height
    nd("Sub", ["c1", "c0"], "cspan0")
    nd("Sub", ["cspan0", "one_i32"], "iw")  # interior width

    # ---- colour-id plane (0..9) + sentinel pad ---------------------------
    w_id = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("w_id", w_id)
    nd("Conv", ["input", "w_id"], "colorid_f")          # [1,1,30,30] f32
    nd("Cast", ["colorid_f"], "colorid", to=U8)          # [1,1,30,30] u8
    init("pads", np.array([0, 0, 0, 0, 0, 0, 1, 1], np.int64))
    init("ten_u8", np.array(10, np.uint8))
    nd("Pad", ["colorid", "pads", "ten_u8"], "colpad")   # [1,1,31,31] u8

    # ---- gather indices: in-region -> origin+i ; else -> sentinel row N ---
    init("ar30", np.arange(N, dtype=np.int32))
    init("sent", np.array(N, np.int32))
    # rows
    nd("Add", ["ar30", "ir0"], "gr0")                    # ir0 + i
    nd("Less", ["ar30", "ih"], "rin_b")                  # i < interior height
    nd("Where", ["rin_b", "gr0", "sent"], "gr")          # [30] int32
    # cols
    nd("Add", ["ar30", "ic0"], "gc0")
    nd("Less", ["ar30", "iw"], "cin_b")
    nd("Where", ["cin_b", "gc0", "sent"], "gc")

    nd("Gather", ["colpad", "gr"], "crop_r", axis=2)     # [1,1,30,31] u8
    nd("Gather", ["crop_r", "gc"], "L", axis=3)          # [1,1,30,30] u8

    init("chan10", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1))
    nd("Equal", ["L", "chan10"], "output")               # -> free BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, N, N])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, N, N])
    graph = helper.make_graph(nodes, "task029", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 12)])
