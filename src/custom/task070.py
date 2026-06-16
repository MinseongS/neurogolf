"""task070 (ARC-AGI 32597951) — "recolor the blue cells inside the cyan rectangle to green".

Rule (from the generator task_32597951.py):
  A `size`x`size` grid (size=17) is a PERIODIC tiling of black(0)/blue(1) with period
  (height, width):  cell[r][c] = colors[(r%height)*width + c%width].  A rectangle of shape
  (tall, wide) is placed at (row, col).  In the INPUT, every non-blue cell of that rectangle
  is painted cyan(8) (blue cells of the rectangle stay blue).  In the OUTPUT the rectangle's
  blue cells become green(3) and its non-blue cells become cyan(8) — i.e. relative to the
  input the only change is:  blue(1) cells that lie inside the rectangle -> green(3).
  The rectangle = the bounding box of the cyan(8) cells of the input.

  Verified exactly (0/800 fails):
    box = bbox(cyan);  output = input but with input==1 & box  ->  3.

Encoding (route the 10-ch expansion into the FREE output; never materialize [1,10,H,W]):
  Work on the active 17x17 canvas.
    cyan   = input[:, 8:9, 0:17, 0:17]                      [1,1,17,17] f16
    rowhas = ReduceMax(cyan, axis=col) -> [1,1,17,1]        (row r contains a cyan)
    colhas = ReduceMax(cyan, axis=row) -> [1,1,1,17]
  bbox-as-mask WITHOUT scalar argmin/argmax (boolean prefix/suffix OR via triangular matmul):
    row in box  <=>  (some cyan row <= r) AND (some cyan row >= r)
                  =  (LowTri @ rowhas > 0) AND (UpTri @ rowhas > 0)
    LowTri[r,k]=1 iff k<=r (prefix), UpTri[r,k]=1 iff k>=r (suffix).
    col in box analogously (multiply on the right by the transposed triangulars).
    box = row_in[1,1,17,1] AND col_in[1,1,1,17]   (broadcast)             [1,1,17,17]
    green = box AND (input blue plane)            blue = input[:,1:2,0:17,0:17]
  Pad green (uint8) to 30x30, cast to bool cond, then
    output = Where(cond, green_onehot[1,10,1,1], input)
  green_onehot is the FIXED one-hot of color 3 (a [1,10,1,1] init).  Where broadcasts the
  [1,1,30,30] cond and [1,10,1,1] value against the FREE [1,10,30,30] input, so the output is
  the only 10-channel tensor and it is free.  Dominant intermediate is the [1,1,30,30] padded
  uint8 mask (~900B); everything else lives on the 17x17 canvas.
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

W = 17  # active canvas (grid is always 17x17 for this task)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- cyan plane (channel 8) on the 17x17 canvas ------------------------
    init("cy_s", np.array([8, 0, 0], np.int64), np.int64)
    init("cy_e", np.array([9, W, W], np.int64), np.int64)
    init("cy_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "cy_s", "cy_e", "cy_ax"], "cyan_f32")  # [1,1,W,W] f32

    # ---- row/col occupancy of cyan (reduce the fp32 slice directly) --------
    n("ReduceMax", ["cyan_f32"], "rowhas_f", axes=[3], keepdims=1)  # [1,1,W,1] f32
    n("ReduceMax", ["cyan_f32"], "colhas_f", axes=[2], keepdims=1)  # [1,1,1,W] f32
    n("Cast", ["rowhas_f"], "rowhas", to=F16)  # [1,1,W,1] f16 {0,1}
    n("Cast", ["colhas_f"], "colhas", to=F16)  # [1,1,1,W] f16 {0,1}

    # ---- triangular prefix/suffix matrices ---------------------------------
    # Only two distinct matrices are needed: tril.T == triu, so the col side
    # (right-multiply) reuses them with swapped roles.
    LowTri = np.tril(np.ones((W, W), np.float16))  # LowTri[r,k]=1 iff k<=r
    UpTri = np.triu(np.ones((W, W), np.float16))   # UpTri[r,k]=1  iff k>=r
    init("LowTri", LowTri.reshape(1, 1, W, W), np.float16)
    init("UpTri", UpTri.reshape(1, 1, W, W), np.float16)

    # row side:  pref_r[r] = sum_k [k<=r]*rowhas[k] ; suf_r[r] likewise [k>=r]
    n("MatMul", ["LowTri", "rowhas"], "pref_r_s")  # [1,1,W,1]
    n("MatMul", ["UpTri", "rowhas"], "suf_r_s")    # [1,1,W,1]
    # col side:  colhas[1,1,1,W] @ M[1,1,W,W] -> [1,1,1,W].
    #   pref_c[c] = sum_k colhas[k]*[k<=c]  needs M[k,c]=[k<=c]=UpTri[k,c]
    #   suf_c[c]  = sum_k colhas[k]*[k>=c]  needs M[k,c]=[k>=c]=LowTri[k,c]
    n("MatMul", ["colhas", "UpTri"], "pref_c_s")   # [1,1,1,W]
    n("MatMul", ["colhas", "LowTri"], "suf_c_s")   # [1,1,1,W]

    init("ZEROH", np.array(0.0, np.float16), np.float16)
    n("Greater", ["pref_r_s", "ZEROH"], "pref_r")  # [1,1,W,1] bool
    n("Greater", ["suf_r_s", "ZEROH"], "suf_r")
    n("Greater", ["pref_c_s", "ZEROH"], "pref_c")  # [1,1,1,W] bool
    n("Greater", ["suf_c_s", "ZEROH"], "suf_c")

    n("And", ["pref_r", "suf_r"], "row_in")   # [1,1,W,1] bool
    n("And", ["pref_c", "suf_c"], "col_in")   # [1,1,1,W] bool
    n("And", ["row_in", "col_in"], "box")     # [1,1,W,W] bool (broadcast)

    # ---- blue plane (channel 1) on the 17x17 canvas ------------------------
    init("bl_s", np.array([1, 0, 0], np.int64), np.int64)
    init("bl_e", np.array([2, W, W], np.int64), np.int64)
    n("Slice", ["input", "bl_s", "bl_e", "cy_ax"], "blue_f32")  # [1,1,W,W] f32
    n("Cast", ["blue_f32"], "blue_b", to=BOOL)                  # [1,1,W,W] bool

    n("And", ["box", "blue_b"], "green_b")  # [1,1,W,W] bool

    # ---- pad to 30x30 (uint8) then -> bool cond ----------------------------
    n("Cast", ["green_b"], "green_u8", to=U8)  # [1,1,W,W] uint8 {0,1}
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), np.int64)
    init("ZEROU8", np.array(0, np.uint8), np.uint8)
    n("Pad", ["green_u8", "pads", "ZEROU8"], "green30", mode="constant")  # [1,1,30,30] u8
    n("Cast", ["green30"], "cond", to=BOOL)  # [1,1,30,30] bool

    # ---- green one-hot (fixed color 3) -------------------------------------
    oh = np.zeros((1, 10, 1, 1), np.float32)
    oh[0, 3, 0, 0] = 1.0
    init("green_oh", oh, np.float32)  # [1,10,1,1] f32

    # ---- output = Where(cond, green_oh, input) : FREE [1,10,30,30] ----------
    n("Where", ["cond", "green_oh", "input"], "output")

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task070", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
