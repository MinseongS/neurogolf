"""Task 316 (cdecee7f): read coloured pixels in ascending-column order into a
3x3 grid, with the middle row reversed.

Rule (from ARC-GEN generator cdecee7f): the input is a 10x10 grid holding
6..9 single coloured pixels (colours 1..9, may repeat), exactly one per
DISTINCT column, and the columns are sorted ascending.  The output is a 3x3
grid filled in reading order with the colours taken in ascending-column order:

    out[0] = [c0, c1, c2]
    out[1] = [c3, c4, c5]   then REVERSED -> [c5, c4, c3]
    out[2] = [c6, c7, c8]

Missing slots (when fewer than 9 colours) stay background (0).  Cells outside
the 3x3 grid are all background.

Closed form, NO 30x30 plane (active canvas is only 3x3):
  * colf = sum_k k*input_k  -> [1,1,10,10] (>0 exactly at coloured cells)
  * colcolor = ReduceSum over rows -> [1,1,1,10] : per-column colour value
    (exactly one nonzero per occupied column).
  * occupied = colcolor>0; idx[c] = EXCLUSIVE CumSum(occupied) = number of
    occupied columns to the left = the destination reading-order index.
  * dest[c] = permtable[idx[c]] folds in the middle-row reversal
    (permtable = [0,1,2,5,4,3,6,7,8]).
  * scatter via a runtime one-hot matrix P[c,j]=(dest[c]==j) & occupied[c]:
    colorvec[1,9] = colcolor[1,10] @ P[10,9]; reshape -> L[1,1,3,3].
  * Equal(L,arange[1,10,1,1]) -> [1,10,3,3] bool one-hot of the 3x3, Cast to
    uint8 and Pad to [1,10,30,30] (off-grid stays all-zero -> background).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION


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
    # ONE no-pad Conv folds "sum over rows" + "contract channels with weight k"
    # into a single op: kernel [out=1,in=10,kh=30,kw=1], w[0,k,r,0]=k.
    #   colcolor[0,0,0,c] = sum_r sum_k k * input[k,r,c]
    _cw = np.zeros((1, 10, 30, 1), np.float32)
    for k in range(10):
        _cw[0, k, :, 0] = k
    init("cw", _cw, np.float32)
    init("half", np.array(0.5, np.float32), np.float32)
    init("axcol", np.array(1, np.int64), np.int64)             # cumsum axis (col)
    init("perm", np.array([0, 1, 2, 5, 4, 3, 6, 7, 8], np.int64), np.int64)
    init("lo", np.array(0.0, np.float32), np.float32)
    init("hi", np.array(8.0, np.float32), np.float32)
    init("ar9", np.arange(9, dtype=np.int32).reshape(1, 9), np.int32)
    init("shp_col10", np.array([1, 10], np.int64), np.int64)   # colcolor -> [1,10]
    init("shp_dest", np.array([10, 1], np.int64), np.int64)    # dest -> [10,1]
    init("sl0", np.array([0], np.int64), np.int64)
    init("sl10", np.array([10], np.int64), np.int64)
    init("sl_ax3", np.array([3], np.int64), np.int64)
    init("shp_L", np.array([1, 1, 3, 3], np.int64), np.int64)  # colorvec -> [1,1,3,3]
    init("chan", np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1), np.float16)
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 27, 27], np.int64), np.int64)
    init("padval", np.array(0, np.uint8), np.uint8)

    # ---- per-column colour value (no full plane, ONE no-pad Conv) ----
    n("Conv", ["input", "cw"], "colcolor30")                    # [1,1,1,30]
    n("Slice", ["colcolor30", "sl0", "sl10", "sl_ax3"], "colcolor4")  # [1,1,1,10]
    n("Reshape", ["colcolor4", "shp_col10"], "colcolor")        # [1,10]
    n("Greater", ["colcolor", "half"], "occ_b")                 # [1,10] bool
    n("Cast", ["occ_b"], "occ_f", to=TensorProto.FLOAT)         # [1,10]

    # exclusive prefix count -> destination index (reading order)
    n("CumSum", ["occ_f", "axcol"], "idx_f", exclusive=1)       # [1,10] f32
    n("Clip", ["idx_f", "lo", "hi"], "idx_c")                   # clamp to [0,8]
    n("Cast", ["idx_c"], "idx_i", to=TensorProto.INT64)         # [1,10]
    n("Gather", ["perm", "idx_i"], "dest2")                     # [1,10] int64
    n("Reshape", ["dest2", "shp_dest"], "dest")                 # [10,1]
    n("Cast", ["dest"], "dest_i32", to=TensorProto.INT32)       # [10,1]

    # one-hot scatter matrix P[c,j] = (dest[c]==j) & occupied[c]
    n("Equal", ["dest_i32", "ar9"], "Peq")                      # [10,9] bool
    n("Reshape", ["occ_b", "shp_dest"], "occ_col")              # [10,1] bool
    n("And", ["Peq", "occ_col"], "P_b")                         # [10,9] bool
    n("Cast", ["P_b"], "P_f", to=TensorProto.FLOAT16)           # [10,9] fp16

    # colorvec[1,9] = colcolor[1,10] @ P[10,9]  (fp16: colours 0..9 exact)
    n("Cast", ["colcolor"], "colcolor_h", to=TensorProto.FLOAT16)
    n("MatMul", ["colcolor_h", "P_f"], "colorvec")              # [1,9] fp16
    n("Reshape", ["colorvec", "shp_L"], "L3")                   # [1,1,3,3] fp16

    # one-hot the 3x3, cast, pad to [1,10,30,30]
    n("Equal", ["L3", "chan"], "oh3")                           # [1,10,3,3] bool
    n("Cast", ["oh3"], "oh3_u", to=TensorProto.UINT8)           # [1,10,3,3]
    n("Pad", ["oh3_u", "padpads", "padval"], "output", mode="constant")

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
