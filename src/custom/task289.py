"""Task 289 (ARC-AGI b91ae062) — nearest-neighbour upscale by a data-dependent
factor K = number of distinct non-background colours.

Rule (from the generator, common.grid_enhance):
  * Input is a 3x3 grid (size=3) with 3..5 scattered coloured pixels on black.
  * enhance = len(colors) = number of DISTINCT colours used = K  (1..5).
  * Output is a (3K)x(3K) grid: each input cell (r,c) of colour v becomes a
    K x K solid block of colour v at (r*K, c*K).  I.e. output[u,v] =
    input[u//K, v//K] for u,v < 3K, and OUTSIDE that footprint every channel is 0.

Construction (label-map + final Equal):
  * src[r,c] = colour-index plane of the 3x3 input (1x1 Conv Sum_k k*input_k on a
    [1,10,3,3] slice -> [1,1,3,3] fp32, 36 B).  Cast to fp16 (18 B) so the gathered
    planes are fp16.
  * K = number of distinct colours = sum over channels 1..9 of per-channel presence
    (ReduceMax over space -> [1,10,1,1], slice ch1..9, ReduceSum).  Scalar.
  * row/col gather index ri[u] = clip(floor(u / K), 0, 2)  (a [30] int vector).
    Gather src along axis 2 (rows) -> [1,1,30,3], then axis 3 (cols) -> the
    [1,1,30,30] colour plane Lf (fp16, 1800 B = the sole big intermediate / label floor).
  * validity mask: cell is in-footprint iff u<3K and v<3K  <=>  floor(u/K)<3 along
    each axis.  rowok[u] = (floor(u/K) < 3); colok analogously.  Outside -> sentinel.
  * L = where(rowok & colok, round(Lf), 99) cast to uint8; output = Equal(L, arange)
    (BOOL, the FREE output tensor; opset 11).

Memory floor: the lone 30x30 plane is the gathered colour plane Lf [1,1,30,30] fp16
(1800 B) plus the uint8 label L (900 B); everything else is <=120 B (1-D vectors,
3x3 source, [30] index maps).  The gather is intrinsically a full-canvas 30x30 read.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
I64 = TensorProto.INT64
B = TensorProto.BOOL


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype=None):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- 3x3 source colour-index plane ----
    # Slice the top-left 3x3 of the 10-channel input -> [1,10,3,3] (90 elem fp32).
    init("ss", np.array([0, 0, 0, 0], np.int64), I64)
    init("se", np.array([1, 10, 3, 3], np.int64), I64)
    init("sax", np.array([0, 1, 2, 3], np.int64), I64)
    n("Slice", ["input", "ss", "se", "sax"], "in33")          # [1,10,3,3]

    # 1x1 Conv weight = [0,1,...,9] over channels -> colour index plane [1,1,3,3].
    w = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("cw", w, F32)
    n("Conv", ["in33", "cw"], "src_f")                        # [1,1,3,3] fp32
    n("Cast", ["src_f"], "src3", to=U8)                       # [1,1,3,3] uint8
    # Pad row/col index 3 with sentinel 99 -> [1,1,4,4]; out-of-footprint cells
    # (row or col index clipped to 3) read the sentinel, so no validity mask /
    # Where plane is ever materialised.
    init("padshape", np.array([0, 0, 0, 0, 0, 0, 1, 1], np.int64), I64)
    init("pad99", np.array(99, np.uint8), U8)
    n("Pad", ["src3", "padshape", "pad99"], "src", mode="constant")  # [1,1,4,4] u8

    # ---- K = number of distinct non-bg colours ----
    # per-channel presence over the WHOLE input, channels 1..9.
    n("ReduceMax", ["input"], "pres", axes=[2, 3], keepdims=1)  # [1,10,1,1] fp32
    init("ps", np.array([0, 1, 0, 0], np.int64), I64)
    init("pe", np.array([1, 10, 1, 1], np.int64), I64)
    init("pax", np.array([0, 1, 2, 3], np.int64), I64)
    n("Slice", ["pres", "ps", "pe", "pax"], "pres19")          # [1,9,1,1]
    n("ReduceSum", ["pres19"], "Kf", axes=[1], keepdims=1)     # [1,1,1,1] fp32 = K
    init("one1", np.array([1], np.int64), I64)
    n("Reshape", ["Kf", "one1"], "K")                          # [1] fp32

    # ---- row/col source index map ri[u] = clip(floor(u/K), 0, 3) ----
    # in-footprint (u<3K) -> 0,1,2 ; out-of-footprint (u>=3K) -> 3 (sentinel row/col).
    U = np.arange(30, dtype=np.float32)                        # [30]
    init("U", U, F32)
    n("Div", ["U", "K"], "q")                                 # [30] = u/K
    n("Floor", ["q"], "qf")                                   # [30] = floor(u/K)
    init("zero", np.array(0.0, np.float32), F32)
    init("three", np.array(3.0, np.float32), F32)
    n("Clip", ["qf", "zero", "three"], "ric_f")               # clip [0,3] -> [30]
    n("Cast", ["ric_f"], "ri", to=I64)                        # [30] int64 index

    # ---- gather rows then cols of src -> [1,1,30,30] label plane (uint8) ----
    n("Gather", ["src", "ri"], "g_rows", axis=2)              # [1,1,30,4] uint8
    n("Gather", ["g_rows", "ri"], "L", axis=3)                # [1,1,30,30] uint8

    # ---- final one-hot into the FREE bool output ----
    ar = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("ar", ar, U8)
    n("Equal", ["L", "ar"], "output")                         # [1,10,30,30] bool

    graph = helper.make_graph(
        nodes, "task289", [
            helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", B, [1, 10, 30, 30])], inits)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 11)],
        ir_version=IR_VERSION)
    return model
