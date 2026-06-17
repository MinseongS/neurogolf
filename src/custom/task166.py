"""Task 166 (ARC-AGI 6d75e8bb): fill the cyan bounding box with red.

Rule (invariant to the generator's final flip/xpose since both grid and output
are transformed identically): the input contains cyan(8) horizontal strips
forming a left-aligned box, plus one extra cyan "marker" pixel inside the same
column band.  The input only ever uses background(0) and cyan(8).  The output
fills the axis-aligned bounding rectangle of all cyan cells with red(2), wherever
the cell is background; cyan cells stay cyan.  Verified exact on all 266 stored
examples and 0/30000 fresh: pred = red on (bbox(cyan) & not cyan), cyan kept.

Because every box ROW contains a strip (length >= 1) and every box COLUMN is hit
by the longest strip, the bounding box is exactly the OUTER PRODUCT of the cyan
row-occupancy and col-occupancy -- no prefix/suffix-OR scan is needed:
    redmask[r,c] = rowhas(r) AND colhas(c) AND (input[r,c] != cyan).
Then  output = Where(redmask, red_onehot, input):  a red cell is always background
in the input, and cyan/background/off-grid cells fall to the free `input` branch.

⭐ BOUNDED-ACTIVE-REGION crop (escape (3)): the generator bounds grid size to
<=14x14 and brow+n-1 <= height-2 <= 12, bcol+maxlen-1 <= width-2 <= 12, so EVERY
cyan cell sits in rows/cols 0..12.  We therefore do all per-cell work on a fixed
13x13 window (verified max cyan row/col = 12 over 30000 fresh instances), turning
the 3600B fp32 cyan plane into a 676B slice and the bool masks into 169B each.
Only the final Where condition must be padded back to 30x30 to broadcast against
the [1,10,30,30] input (the two unavoidable 30x30 bool/uint8 planes dominate).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

K = 13  # fixed crop window: all cyan cells provably lie in rows/cols 0..12


def build(task):
    inits = []
    nodes = []
    vinfos = []

    def init(name, arr, dtype=np.float32):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    def vi(name, dtype, shape):
        vinfos.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    # cyan = input channel 8, cropped to the fixed 13x13 active window (676B fp32).
    init("cy_s", np.array([8, 0, 0], np.int64), np.int64)
    init("cy_e", np.array([9, K, K], np.int64), np.int64)
    init("cy_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "cy_s", "cy_e", "cy_ax"], "cyanf")  # [1,1,K,K] f32 {0,1}
    vi("cyanf", TensorProto.FLOAT, [1, 1, K, K])

    init("zero", np.array(0.0, np.float32))
    init("half", np.array(0.5, np.float32))

    # row / col occupancy reductions on the small float window -> tiny 1-D tensors.
    n("ReduceSum", ["cyanf"], "rc", axes=[3], keepdims=1)   # [1,1,K,1] f32
    vi("rc", TensorProto.FLOAT, [1, 1, K, 1])
    n("ReduceSum", ["cyanf"], "cc", axes=[2], keepdims=1)   # [1,1,1,K] f32
    vi("cc", TensorProto.FLOAT, [1, 1, 1, K])

    n("Greater", ["rc", "zero"], "rowhas")                  # [1,1,K,1] bool
    vi("rowhas", TensorProto.BOOL, [1, 1, K, 1])
    n("Greater", ["cc", "zero"], "colhas")                  # [1,1,1,K] bool
    vi("colhas", TensorProto.BOOL, [1, 1, 1, K])

    # bounding box = outer product of row/col occupancy (every box row & col is hit).
    n("And", ["rowhas", "colhas"], "box")                   # [1,1,K,K] bool
    vi("box", TensorProto.BOOL, [1, 1, K, K])

    # redmask = box & not-cyan (cyan cells stay cyan via the Where input branch).
    n("Less", ["cyanf", "half"], "notcyan")                 # [1,1,K,K] bool
    vi("notcyan", TensorProto.BOOL, [1, 1, K, K])
    n("And", ["box", "notcyan"], "redmask")                 # [1,1,K,K] bool
    vi("redmask", TensorProto.BOOL, [1, 1, K, K])

    # pad the 13x13 mask back to 30x30 (uint8, then cast to bool) so the Where
    # condition broadcasts against the [1,10,30,30] input.
    n("Cast", ["redmask"], "redu8", to=TensorProto.UINT8)   # [1,1,K,K] u8
    vi("redu8", TensorProto.UINT8, [1, 1, K, K])
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - K, 30 - K], np.int64), np.int64)
    init("zeru8", np.array(0, np.uint8), np.uint8)
    n("Pad", ["redu8", "pads", "zeru8"], "pad30", mode="constant")  # [1,1,30,30] u8
    vi("pad30", TensorProto.UINT8, [1, 1, 30, 30])
    n("Cast", ["pad30"], "cond", to=TensorProto.BOOL)       # [1,1,30,30] bool
    vi("cond", TensorProto.BOOL, [1, 1, 30, 30])

    # output = cond ? red_onehot : input  (free [1,10,30,30] output).
    red = np.zeros((1, 10, 1, 1), np.float32)
    red[0, 2, 0, 0] = 1.0   # red = channel 2
    init("red", red)
    n("Where", ["cond", "red", "input"], "output")

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task166", [x], [y], inits, value_info=vinfos)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
