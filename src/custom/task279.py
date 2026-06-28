"""Task 279 (ARC-AGI b2862040) — restore closed blue boxes to cyan.

Generator rule:

* The true scene contains closed cyan(8) boxes and open blue(1) boxes on
  maroon(9), with occasional one-pixel barnacles.
* The input maps every cyan pixel to blue, so both open and closed boxes appear
  blue.
* The output restores only closed boxes to cyan; open boxes and barnacles remain
  blue.

This builder mirrors the compact deployed graph in source form.  On the 16x16
active crop it repeatedly erodes the blue mask with a 3x3 all-ones QLinearConv,
then grows the surviving closed-box cores back two cross-neighbour steps inside
the original blue mask.  The resulting mask is padded to 30x30 and used to
select cyan(8) versus the original input.
"""

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ..harness import IR_VERSION


S = 30
CROP = 16


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
    B = TensorProto.BOOL

    init("slice_starts", np.array([0, 1, 0, 0], np.int64), np.int64)
    init("slice_ends", np.array([1, 2, CROP, CROP], np.int64), np.int64)
    init("zero_u8", np.array(0, np.uint8), np.uint8)
    init("scale_f", np.array(1.0, np.float32), np.float32)
    init("degree_y_scale", np.array(9.0, np.float32), np.float32)
    degree = np.array([[0, 1, 0], [1, 3, 1], [0, 1, 0]], np.uint8).reshape(1, 1, 3, 3)
    init("degree_kernel_u8", degree, np.uint8)
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8).reshape(1, 1, 3, 3)
    init("cross_kernel_u8", cross, np.uint8)

    n("Slice", ["input", "slice_starts", "slice_ends"], "blue_f32")
    n("Cast", ["blue_f32"], "blue", to=U8)

    cur = "blue"
    for i in range(8):
        cur = n(
            "QLinearConv",
            [cur, "scale_f", "zero_u8", "degree_kernel_u8", "scale_f", "zero_u8", "degree_y_scale", "zero_u8"],
            f"core_{i}",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )

    for i in range(2):
        neigh = n(
            "QLinearConv",
            [cur, "scale_f", "zero_u8", "cross_kernel_u8", "scale_f", "zero_u8", "scale_f", "zero_u8"],
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
