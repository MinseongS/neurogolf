"""Task 279 (ARC-AGI b2862040) — restore closed blue boxes to cyan.

Generator rule:

* The true scene contains closed cyan(8) boxes and open blue(1) boxes on
  maroon(9), with occasional one-pixel barnacles.
* The input maps every cyan pixel to blue, so both open and closed boxes appear
  blue.
* The output restores only closed boxes to cyan; open boxes and barnacles remain
  blue.

A blue pixel belongs to a closed box iff its 4-connected blue component encloses
a maroon pocket.  On the 16x16 active crop the graph repeatedly erodes the blue
mask with a 3x3 "degree" QLinearConv (a cell survives only with >=2 orthogonal
neighbours), which peels open paths from their endpoints while leaving closed
loops intact.  The surviving loop cores are then grown two masked steps back
inside the original blue mask to recover attached tails.  The same degree kernel
is reused for the dilation (any set neighbour lights the cell), so only one 3x3
kernel is stored.  The resulting mask is padded to 30x30 and used to select
cyan(8) versus the original input.
"""

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ..harness import IR_VERSION


S = 30
CROP = 16
ERODE = 8
GROW = 2


def build(task):
    inits = []
    nodes = []
    seen = set()

    def init(name, arr, dt):
        if name in seen:
            return name
        seen.add(name)
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    U8 = TensorProto.UINT8

    init("slice_starts", np.array([0, 1, 0, 0], np.int64), np.int64)
    init("slice_ends", np.array([1, 2, CROP, CROP], np.int64), np.int64)
    init("zero_u8", np.array(0, np.uint8), np.uint8)
    init("scale_f", np.array(1.0, np.float32), np.float32)
    init("degree_y_scale", np.array(9.0, np.float32), np.float32)
    degree = np.array([[0, 1, 0], [1, 3, 1], [0, 1, 0]], np.uint8).reshape(1, 1, 3, 3)
    init("degree_kernel_u8", degree, np.uint8)

    # Crop the blue channel to the 16x16 active region and quantise to uint8.
    n("Slice", ["input", "slice_starts", "slice_ends"], "blue_f32")
    n("Cast", ["blue_f32"], "blue", to=U8)

    # Erode: a cell survives iff 3*center + sum(orth neighbours) >= 9, i.e. it has
    # at least two orthogonal neighbours.  Open paths peel from their endpoints;
    # closed loops (every cell has two neighbours) survive untouched.
    cur = "blue"
    for i in range(ERODE):
        cur = n(
            "QLinearConv",
            ["blue" if i == 0 else cur, "scale_f", "zero_u8", "degree_kernel_u8",
             "scale_f", "zero_u8", "degree_y_scale", "zero_u8"],
            f"core_{i}",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )

    # Regrow the surviving loop cores back into the original blue mask to recover
    # tails.  y_scale=1 makes the same degree kernel act as a 4-connected dilation
    # (any set cell in the plus-neighbourhood lights the centre).
    for i in range(GROW):
        neigh = n(
            "QLinearConv",
            [cur, "scale_f", "zero_u8", "degree_kernel_u8", "scale_f", "zero_u8", "scale_f", "zero_u8"],
            f"neighbor_{i}",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
        cur = n("Min", [neigh, "blue"], f"flood_{i}")

    n("Greater", [cur, "zero_u8"], "cond_crop")
    init("cond_pads", np.array([0, 0, 0, 0, 0, 0, S - CROP, S - CROP], np.int64), np.int64)
    n("Pad", ["cond_crop", "cond_pads"], "cond", mode="constant")

    cyan = np.zeros((1, 10, 1, 1), np.float32)
    cyan[:, 8, :, :] = 1.0
    init("cyan_vec", cyan, np.float32)
    n("Where", ["cond", "cyan_vec", "input"], "output")

    x = helper.make_tensor_value_info("input", F, [1, 10, S, S])
    y = helper.make_tensor_value_info("output", F, [1, 10, S, S])
    graph = helper.make_graph(nodes, "task279", [x], [y], inits)
    return helper.make_model(
        graph,
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 21)],
    )
