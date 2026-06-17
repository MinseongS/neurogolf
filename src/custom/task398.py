"""Task 398 (feca6190): anti-diagonal rays from a 1x5 color row.

Rule: input is a 1x5 row with n nonzero colors (n=1..5). Output is an s x s
grid, s = 5n. Each input column c (color v, zeros included) draws the
anti-diagonal ray output[r][s-1+c-r] = v for r in c..s-1. Equivalently, the
output value at (r,j) depends only on the anti-diagonal t = r+j and on the
grid extent: column index c = t - (s-1); inside the grid the cell shows
input column c when 0 <= c <= 4 (else black/color-0); outside the s x s box
the canvas is all-zero.

ONE fp16 plane build (no int32 30x30 index plane, no [1,10,30,30] palette
gather):

  z   = number of zero colors = Sum(channel-0 over input row cols 0..4); it
        selects the grid size s = 25 - 5z and thus one of 5 structural maps.
  KEY [30,30] int32  (const) : per-cell *tuple id* k in 0..50, where the tuple
        is the 5-vector of structural data-column indices over z=0..4. KEY is
        a constant -> free as params, 0 memory.
  BIGVEC [51,5] int32 (const): BIGVEC[k,z] = structural data-column index in
        {0..4 colored, 5 inside-black, 6 outside-grid} for tuple k at size z.
  bvec  = Gather(BIGVEC, z, axis=1) -> [51] : structural index per tuple now.
  data6 = [colors[0..4], 0(black), 99(sentinel)]  (fp16, length 7).
  vtab  = Gather(data6, bvec, axis=0) -> [51] fp16 : the actual OUTPUT VALUE
        per tuple (real colour 0..9, 0 for inside-black, 99 sentinel outside).
  V     = Gather(vtab, KEY, axis=0) -> [30,30] fp16 : the value plane (the one
        and only canvas-sized intermediate, 1800B).
  output= Equal(V, arange[1,10,1,1]) -> BOOL [1,10,30,30] (the FREE output):
        99 -> all-false (all-zero canvas outside grid); 0 -> channel-0 (black);
        colour v -> channel v. fp16 Equal is integer-exact for these values.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

F16 = onnx.TensorProto.FLOAT16
F32 = onnx.TensorProto.FLOAT
I32 = onnx.TensorProto.INT32
I64 = onnx.TensorProto.INT64
BOOL = onnx.TensorProto.BOOL


def _tables():
    """KEY [30,30] int32 (tuple id per cell), BIGVEC [51,5] int32."""
    P = np.empty((5, 30, 30), np.int32)
    for z in range(5):
        s = 25 - 5 * z
        for r in range(30):
            for col in range(30):
                if r < s and col < s:
                    c = r + col - (s - 1)
                    P[z, r, col] = c if 0 <= c <= 4 else 5
                else:
                    P[z, r, col] = 6
    tup = P.transpose(1, 2, 0).reshape(900, 5)
    uniq, inv = np.unique(tup, axis=0, return_inverse=True)
    key = inv.astype(np.int32).reshape(30, 30)
    bigvec = uniq.astype(np.int32)               # [51, 5]
    return key, bigvec


def build(task):
    inits = []
    nodes = []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    key, bigvec = _tables()
    init("KEY", key, np.int32)                    # [30,30] const
    init("BIGVEC", bigvec, np.int32)              # [51,5] const

    # --- z = number of zero colours among input row cols 0..4 (scalar) ---
    init("z_st", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("z_en", np.array([1, 1, 1, 5], np.int64), np.int64)
    init("z_ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "z_st", "z_en", "z_ax"], "ch0")   # [1,1,1,5] (channel 0)
    n("ReduceSum", ["ch0"], "zf", keepdims=0)              # scalar fp32
    n("Cast", ["zf"], "zi", to=I32)                        # scalar int32

    # --- structural index per tuple at this size: bvec = BIGVEC[:, z] ---
    n("Gather", ["BIGVEC", "zi"], "bvec", axis=1)          # [51] int32

    # --- palette: actual colours of input row cols 0..4 + [black, sentinel] ---
    init("cw_st", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("cw_en", np.array([1, 10, 1, 5], np.int64), np.int64)
    init("cw_ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "cw_st", "cw_en", "cw_ax"], "rowslc")  # [1,10,1,5]
    n("Squeeze", ["rowslc"], "row3", axes=[0, 2])               # [10,5]
    init("CHAN", np.arange(10, dtype=np.float32).reshape(1, 10), np.float32)
    n("MatMul", ["CHAN", "row3"], "colvals")                    # [1,5] fp32 colour values
    init("BS", np.array([[0.0, 99.0]], np.float32), np.float32)  # black, sentinel
    n("Concat", ["colvals", "BS"], "data6f", axis=1)            # [1,7] fp32
    n("Squeeze", ["data6f"], "data6f1", axes=[0])               # [7] fp32
    n("Cast", ["data6f1"], "data6", to=F16)                     # [7] fp16

    # --- output value per tuple, then the single value plane V ---
    n("Gather", ["data6", "bvec"], "vtab", axis=0)             # [51] fp16
    n("Gather", ["vtab", "KEY"], "V", axis=0)                  # [30,30] fp16

    # --- route the 10-channel expansion into the FREE bool output ---
    init("ARANGE", np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1), np.float16)
    n("Equal", ["V", "ARANGE"], "output")                     # BOOL [1,10,30,30]

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=10,
        opset_imports=[helper.make_opsetid("", 11)])
