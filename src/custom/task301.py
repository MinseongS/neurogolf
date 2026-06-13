"""Task 301 (beb8660c): rebuild a right-aligned color staircase.

ARC-GEN rule: input has `num_colors` horizontal bars; the bar for color index
i has length i+1, so a color's pixel count n_c uniquely identifies its index.
Cyan (channel 8) is always the last color: a full-width bar of length
width = num_colors. The output footprint is the rectangle (c < n8) & (r <= r8)
where n8 = cyan count = width and r8 = grid row of cyan = height-1. Inside it
each color c (>0) is a right-aligned bar of length n_c at row n_c-1+gap.

Let n_c = pixel count of channel c (= bar length), n8 = cyan count = width,
r8 = row of cyan, delta = r8 - n8 (= gap-1), Rc = n_c + delta (output row).

Colored channel c (>0):  out[c,r,col] = (r == Rc) & (n8-n_c <= col < n8).
Background channel 0:     out[0,r,col] = (r+col < r8) & (col < n8) & (r <= r8),
  i.e. the part of the output rectangle left of the staircase.

Graph: build colored channels 1..9 directly as a [1,9,30,30] bool (rowmask AND
colmask, sliced to drop channel 0 BEFORE the AND so no [1,10,30,30] is ever
materialized), plus the channel-0 triangle [1,1,30,30] bool, Concat, Cast into
`output`. All other intermediates are 1-D / tiny; values stay integer.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ..builders import _model


def build(task):
    inits = []
    nodes = []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    # index ramps
    init("I", np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1), np.float32)
    init("J", np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30), np.float32)
    init("half", np.array(0.5, dtype=np.float32), np.float32)
    init("one", np.array(1.0, dtype=np.float32), np.float32)
    init("c8_st", np.array([8], dtype=np.int64), np.int64)
    init("c8_en", np.array([9], dtype=np.int64), np.int64)
    init("c1_ax", np.array([1], dtype=np.int64), np.int64)
    init("c1_10st", np.array([1], dtype=np.int64), np.int64)
    init("c1_10en", np.array([10], dtype=np.int64), np.int64)

    # per-channel pixel counts n_c -> [1,10,1,1]; n8 = cyan count -> [1,1,1,1]
    n("ReduceSum", ["input"], "ncnt", axes=[2, 3], keepdims=1)
    n("Slice", ["ncnt", "c8_st", "c8_en", "c1_ax"], "n8")

    # r8 = row of cyan: row-occupancy of channel 8 dotted with row index I
    n("ReduceSum", ["input"], "rocc", axes=[3], keepdims=1)        # [1,10,30,1]
    n("Slice", ["rocc", "c8_st", "c8_en", "c1_ax"], "occ8")        # [1,1,30,1]
    n("Greater", ["occ8", "half"], "ind8_b")
    n("Cast", ["ind8_b"], "ind8", to=onnx.TensorProto.FLOAT)
    n("Mul", ["ind8", "I"], "iI")
    n("ReduceSum", ["iI"], "r8", axes=[2], keepdims=1)             # [1,1,1,1]

    # colored channels 1..9 only ---------------------------------------
    n("Slice", ["ncnt", "c1_10st", "c1_10en", "c1_ax"], "ncnt9")   # [1,9,1,1]
    # output row Rc = n_c + (r8 - n8) ; row mask (I == Rc)
    n("Sub", ["r8", "n8"], "delta")                                # [1,1,1,1]
    n("Add", ["ncnt9", "delta"], "Rc9")                            # [1,9,1,1]
    n("Cast", ["I"], "Ii", to=onnx.TensorProto.INT32)
    n("Cast", ["Rc9"], "Rci9", to=onnx.TensorProto.INT32)
    n("Equal", ["Ii", "Rci9"], "rowmask9")                         # bool [1,9,30,1]
    # col mask (n8 - n_c) <= J < n8
    n("Sub", ["n8", "ncnt9"], "lower9")                            # [1,9,1,1]
    n("Sub", ["lower9", "one"], "lowm1_9")
    n("Greater", ["J", "lowm1_9"], "ge9")                          # bool [1,9,1,30]
    n("Less", ["J", "n8"], "lt")                                   # bool [1,1,1,30]
    n("And", ["ge9", "lt"], "colmask9")                            # bool [1,9,1,30]
    n("And", ["rowmask9", "colmask9"], "colored9")                 # bool [1,9,30,30]

    # background channel 0: (J < min(n8, r8 - I)) -----------------------
    n("Sub", ["r8", "I"], "r8mI")                                  # [1,1,30,1]
    nodes.append(helper.make_node("Min", ["n8", "r8mI"], ["U"]))   # [1,1,30,1]
    n("Less", ["J", "U"], "ch0")                                   # bool [1,1,30,30]

    # assemble -> output
    nodes.append(helper.make_node("Concat", ["ch0", "colored9"], ["cat"], axis=1))
    n("Cast", ["cat"], "output", to=onnx.TensorProto.FLOAT)
    return _model(nodes, inits)
