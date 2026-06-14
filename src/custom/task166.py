"""Task 166 (ARC-AGI 6d75e8bb): fill the cyan bounding box with red.

Rule (invariant to the generator's final flip/xpose since both grid and output
are transformed identically): the input contains cyan(8) strips forming a box.
The output fills the axis-aligned bounding rectangle of all cyan cells with
red(2), then overlays cyan back on top of the original cyan cells. Verified
exact on all 266 stored examples: pred = red on box, cyan where input cyan, 0
elsewhere == output.

Because the input only ever uses colors 0 (background) and 8 (cyan), every
non-cyan in-box cell is background, so the transform is simply:
    output = redmask ? red_onehot : input
where redmask[r,c] = (row r has a cyan cell) AND (col c has a cyan cell) AND
(cell (r,c) is not cyan). The bounding box is the outer product of the
row-occupancy and col-occupancy of cyan, so the rule is separable.

Graph: cyan = input channel 8 -> row/col reductions -> bool masks ->
box = rowhas & colhas; redmask = box & ~cyan; one Where into the free output.
Everything is bool/1-D; the only canvas-sized intermediates are bool (900B).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..builders import _model


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

    # cyan = channel 8 of input. Conv emits float, immediately threshold to a
    # bool [1,1,30,30] (900B) so no float canvas lingers.
    Wcyan = np.zeros((1, 10, 1, 1), np.float32)
    Wcyan[0, 8, 0, 0] = 1.0
    init("Wcyan", Wcyan)
    init("half", np.array(0.5, np.float32))
    n("Conv", ["input", "Wcyan"], "cyanf")                 # [1,1,30,30] f32
    vi("cyanf", TensorProto.FLOAT, [1, 1, 30, 30])

    init("zero", np.array(0.0, np.float32))
    # row / col occupancy reductions on the float canvas -> tiny 1-D tensors.
    n("ReduceSum", ["cyanf"], "rc", axes=[3], keepdims=1)   # [1,1,30,1] f32
    vi("rc", TensorProto.FLOAT, [1, 1, 30, 1])
    n("ReduceSum", ["cyanf"], "cc", axes=[2], keepdims=1)   # [1,1,1,30] f32
    vi("cc", TensorProto.FLOAT, [1, 1, 1, 30])

    n("Greater", ["rc", "zero"], "rowhas")                 # [1,1,30,1] bool
    vi("rowhas", TensorProto.BOOL, [1, 1, 30, 1])
    n("Greater", ["cc", "zero"], "colhas")                 # [1,1,1,30] bool
    vi("colhas", TensorProto.BOOL, [1, 1, 1, 30])

    # bounding box = outer product of row/col occupancy.
    n("And", ["rowhas", "colhas"], "box")                  # [1,1,30,30] bool
    vi("box", TensorProto.BOOL, [1, 1, 30, 30])

    # redmask = box & not-cyan (not-cyan via Less on the float canvas).
    n("Less", ["cyanf", "half"], "notcyan")                # [1,1,30,30] bool
    vi("notcyan", TensorProto.BOOL, [1, 1, 30, 30])
    n("And", ["box", "notcyan"], "redmask")                # [1,1,30,30] bool
    vi("redmask", TensorProto.BOOL, [1, 1, 30, 30])

    # output = redmask ? red_onehot : input. A red cell is always background in
    # input, so overwriting the whole channel stack with the red one-hot is safe;
    # the [1,1,30,30] bool condition broadcasts over channels (900B).
    red = np.zeros((1, 10, 1, 1), np.float32)
    red[0, 2, 0, 0] = 1.0   # red = channel 2
    init("red", red)
    n("Where", ["redmask", "red", "input"], "output")

    return _model(nodes, inits, vinfos)
