"""Task 301 (beb8660c): rebuild a right-aligned color staircase.

ARC-GEN rule: input has `num_colors` horizontal bars; the bar for color index
i has length i+1, so a color's pixel count n_c uniquely identifies its index.
Cyan (channel 8) is always the last color: a full-width bar of length
width = num_colors. The output footprint is the rectangle (col < n8) & (r <= r8)
where n8 = cyan count = width and r8 = grid row of cyan = height-1. Inside it
each colour c (>0) is a right-aligned bar of length n_c at output row
Rc = n_c + (r8 - n8); the cells of that row left of the bar are background 0.

Per-row factoring (the floor-break): every output row r holds at most one
colour (the rows Rc are distinct because the bar lengths n_c are distinct), so
the whole picture is described by two 1-D row vectors:

    rowcolor[r] = the colour c with Rc == r          (0 if none)
    rowN[r]     = that colour's bar length n_c        (0 if none)
    split[r]    = n8 - rowN[r]   (bar occupies cols split..n8-1)

A single uint8 label map L[1,1,30,30] is then:
    L[r,col] = 10                      if r > r8 or col >= n8      (outside)
             = rowcolor[r]             if col >= split[r] and rowcolor[r] > 0
             = 0                       otherwise                   (background)
and  output = Equal(L, arange[1,10,1,1])  writes straight into the free BOOL
output (opset 11).  No [1,9,30,30] / [1,10,30,30] stack is ever materialised;
the only ~900B tensor is L itself.  All values are small integers, fp16-exact.
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

    # ---- index ramps / constants ----
    init("I", np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1), np.float32)
    init("J", np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30), np.float32)
    init("half", np.array(0.5, dtype=np.float32), np.float32)
    init("c8_st", np.array([8], dtype=np.int64), np.int64)
    init("c8_en", np.array([9], dtype=np.int64), np.int64)
    init("c1_ax", np.array([1], dtype=np.int64), np.int64)
    init("c1_10st", np.array([1], dtype=np.int64), np.int64)
    init("c1_10en", np.array([10], dtype=np.int64), np.int64)
    init("cidx9", np.arange(1, 10, dtype=np.float32).reshape(1, 9, 1, 1),
         np.float32)                                     # colour index 1..9

    # ---- per-channel pixel counts n_c; n8 = cyan count = width ----
    n("ReduceSum", ["input"], "ncnt", axes=[2, 3], keepdims=1)   # [1,10,1,1]
    n("Slice", ["ncnt", "c8_st", "c8_en", "c1_ax"], "n8")        # [1,1,1,1]

    # ---- r8 = grid row of cyan: row-occupancy of channel 8 ----
    # (ReduceSum over cols gives [1,10,30,1]=1200B, cheaper than slicing the
    #  full 30x30 channel-8 plane which would be 3600B float.)
    n("ReduceSum", ["input"], "rocc", axes=[3], keepdims=1)      # [1,10,30,1]
    n("Slice", ["rocc", "c8_st", "c8_en", "c1_ax"], "occ8")      # [1,1,30,1]
    n("Greater", ["occ8", "half"], "ind8_b")
    n("Cast", ["ind8_b"], "ind8", to=TensorProto.FLOAT)
    n("Mul", ["ind8", "I"], "iI")
    n("ReduceSum", ["iI"], "r8", axes=[2], keepdims=1)           # [1,1,1,1]

    # ---- colours 1..9: output row Rc and bar length n_c ----
    n("Slice", ["ncnt", "c1_10st", "c1_10en", "c1_ax"], "ncnt9")  # [1,9,1,1]
    n("Sub", ["r8", "n8"], "delta")                              # [1,1,1,1]
    n("Add", ["ncnt9", "delta"], "Rc9")                          # [1,9,1,1]
    # rowsel[c,r] = (I == Rc[c])  but only where the colour exists (n_c>0)
    n("Cast", ["I"], "Ii", to=TensorProto.INT32)
    n("Cast", ["Rc9"], "Rci9", to=TensorProto.INT32)
    n("Equal", ["Ii", "Rci9"], "rowsel_b")                       # [1,9,30,1] bool
    n("Greater", ["ncnt9", "half"], "exist9")                    # [1,9,1,1] bool
    n("And", ["rowsel_b", "exist9"], "rowsel")                   # [1,9,30,1] bool
    n("Cast", ["rowsel"], "rowselF", to=TensorProto.FLOAT16)     # [1,9,30,1] f16

    # rowcolor[r] = sum_c cidx[c]*rowsel[c,r] ; rowN[r] = sum_c n_c*rowsel
    # (fp16: every value is a small integer, so exact; halves the [1,9,30,1]s)
    n("Cast", ["cidx9"], "cidx9h", to=TensorProto.FLOAT16)
    n("Cast", ["ncnt9"], "ncnt9h", to=TensorProto.FLOAT16)
    n("Mul", ["rowselF", "cidx9h"], "colsel")                    # [1,9,30,1] f16
    n("ReduceSum", ["colsel"], "rowcolorH", axes=[1], keepdims=1)  # [1,1,30,1] f16
    n("Cast", ["rowcolorH"], "rowcolor", to=TensorProto.FLOAT)
    n("Mul", ["rowselF", "ncnt9h"], "nsel")                      # [1,9,30,1] f16
    n("ReduceSum", ["nsel"], "rowNH", axes=[1], keepdims=1)      # [1,1,30,1] f16
    n("Cast", ["rowNH"], "rowN", to=TensorProto.FLOAT)
    n("Sub", ["n8", "rowN"], "split")                            # [1,1,30,1] = n8-n_c

    # ---- rectangle mask inrect = (r <= r8) & (col < n8) ----
    # 1-D row/col masks first, AND into a single 30x30 plane.
    init("mhalf", np.array(-0.5, np.float32), np.float32)
    n("Sub", ["r8", "I"], "r8mI")                                # [1,1,30,1]
    n("Greater", ["r8mI", "mhalf"], "rleR8")                     # [1,1,30,1] bool
    n("Less", ["J", "n8"], "ltN8")                               # [1,1,1,30] bool
    n("And", ["rleR8", "ltN8"], "inrect")                        # [1,1,30,30] bool

    # ---- bar = inrect & (col >= split[r]) ----
    # split = n8 for colourless rows, so the bar collapses there (col>=n8 but
    # col<n8 in inrect => empty) and the whole row stays background -- no
    # separate hasColor test needed.  col >= split  <=>  col > split-1.
    init("one", np.array(1.0, np.float32), np.float32)
    n("Sub", ["split", "one"], "split_m")
    n("Greater", ["J", "split_m"], "geSplit")                    # [1,1,30,30] bool
    n("And", ["inrect", "geSplit"], "inbar")                     # [1,1,30,30] bool

    # ---- uint8 label map L ----
    n("Cast", ["rowcolor"], "rowcolU", to=TensorProto.UINT8)     # [1,1,30,1] u8
    init("u0", np.array(0, np.uint8), np.uint8)
    init("u10", np.array(10, np.uint8), np.uint8)
    n("Where", ["inrect", "u0", "u10"], "Lrect")                 # 0 in-rect else 10
    n("Where", ["inbar", "rowcolU", "Lrect"], "L")               # bar -> colour

    # ---- final Equal into free BOOL output ----
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                          # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task301", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
