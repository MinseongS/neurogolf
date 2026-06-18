"""Task 123 (ARC-AGI 539a4f51) — diagonal colour band, 2x tile with wrap.

Rule (from the generator):
  colors = list of L colours (L in {4,5}; every colour 1..9).
  INPUT  (5x5):  grid[r][c] = colors[max(r,c)]  for max(r,c) < L, else 0.
                 -> the input main DIAGONAL is exactly colors[0..L-1] then 0s;
                    L = number of non-zero diagonal cells.
  OUTPUT (10x10): output[r][c] = colors[ max(r,c) % L ].
                 (all output cells are 1..9 -> the 10x10 footprint never has bg;
                  the 30x30 canvas outside the 10x10 is background colour 0.)

Recovery (closed-form, no detection):
  * Slice the input to its 5x5 active region and collapse the 10 colour
    channels to one colour-index plane colf5 = sum_k k*input_k  (5x5 fp32).
  * dcol[k] = colf5[k][k]  (gather the 5 diagonal cells -> length-5 vector;
    dcol = colors padded with 0).  L = #(dcol > 0).
  * The output value at (r,c) depends only on max(r,c): build the fixed 10x10
    "max(r,c)" index plane, take it mod L (fp32 Mod, integer-exact for these
    small ints), and Gather dcol with it -> the 10x10 label plane lab.
  * Pad lab to 30x30 with 0 (off-footprint = background) and route the 10-ch
    expansion into the FREE bool output via Equal(lab_uint8, arange[1,10,1,1]).

Memory floor: every intermediate is <=100 elems (5x5 colf, 10x10 label); the
fixed 10x10 max-plane is a cheap int initializer.  No 30x30 working plane at all.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
B = TensorProto.BOOL
I64 = TensorProto.INT64


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- column 0, rows 0..4 = colors[0..L-1] (grid[r][0]=colors[max(0,r)]) ----
    init("ss", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("se", np.array([1, 10, 5, 1], np.int64), np.int64)
    init("sax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "ss", "se", "sax"], "col0")  # [1,10,5,1] fp32 (200B)

    # ---- colour-index plane = sum_k k*input_k  (1x1 conv) ----
    w = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("w", w, np.float32)
    n("Conv", ["col0", "w"], "colf")                  # [1,1,5,1] fp32 (20B)

    # ---- dcol[k] = colors[k] (0 for k>=L) ----
    init("flat5", np.array([5], np.int64), np.int64)
    n("Reshape", ["colf", "flat5"], "dcol")           # [5] fp32 = colors padded 0

    # ---- L = number of non-zero diagonal entries ----
    init("zero", np.array(0.0, np.float32), np.float32)
    n("Greater", ["dcol", "zero"], "dnz")            # [5] bool
    n("Cast", ["dnz"], "dnz_f", to=F32)              # [5] fp32
    n("ReduceSum", ["dnz_f"], "L", keepdims=0)       # scalar fp32 (=L in {4,5})

    # ---- band->colour vector: bandlabel[m] = dcol[ m % L ], m=0..9 (tiny) ----
    marange = np.arange(10, dtype=np.float32)          # [10]
    init("marange", marange, np.float32)
    n("Mod", ["marange", "L"], "bidx10_f", fmod=1)     # [10] fp32, integer-exact
    n("Cast", ["bidx10_f"], "bidx10", to=TensorProto.INT32)  # [10] int32
    n("Gather", ["dcol", "bidx10"], "bandlabel_f")     # [10] fp32 = colours per band
    n("Cast", ["bandlabel_f"], "bandlabel", to=U8)     # [10] uint8 (tiny)

    # ---- 10x10 label plane: lab[r,c] = bandlabel[ max(r,c) ] (const index) ----
    R, C = 10, 10
    rr = np.arange(R).reshape(R, 1)
    cc = np.arange(C).reshape(1, C)
    maxidx = np.maximum(rr, cc).astype(np.int32)       # [10,10] in 0..9 (const)
    init("maxidx", maxidx, np.int32)
    n("Gather", ["bandlabel", "maxidx"], "lab10f")     # [10,10] uint8 (colours)
    init("s4", np.array([1, 1, 10, 10], np.int64), np.int64)
    n("Reshape", ["lab10f", "s4"], "lab10")            # [1,1,10,10] uint8

    # ---- pad to 30x30 with sentinel 10 (off-footprint = all channels off;
    # the 10x10 output footprint has no background cells) ----
    init("padv", np.array(10, np.uint8), np.uint8)
    init("pads", np.array([0, 0, 0, 0, 0, 0, 20, 20], np.int64), np.int64)
    n("Pad", ["lab10", "pads", "padv"], "L30", mode="constant")  # [1,1,30,30] uint8

    # ---- route 10-ch expansion into FREE bool output ----
    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["L30", "chan"], "output")            # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task123", [x], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
