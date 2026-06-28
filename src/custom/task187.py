"""Task 187 (7b6016b9): cropped label-map flood fill for red enclosed holes.

Rule from the generator:
- preserve every non-black drawing cell from the input;
- black background connected to the outside becomes green(3);
- black enclosed box interiors become red(2);
- off-grid cells are all-false.

The generator emits grids with height/width in 20..25, so the flood-fill only
needs to run on a 25x25 crop.  The final label map is padded back to the harness
30x30 canvas with sentinel label 10 before the free Equal expansion.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..harness import IR_VERSION


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    U8 = TensorProto.UINT8
    B = TensorProto.BOOL

    init("Wlabel", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    init("zero_f", np.array(0.0, np.float32), np.float32)
    init("zero_u8", np.array(0, np.uint8), np.uint8)
    init("one_u8", np.array(1, np.uint8), np.uint8)
    init("two_u8", np.array(2, np.uint8), np.uint8)
    init("three_u8", np.array(3, np.uint8), np.uint8)
    init("ten_u8", np.array(10, np.uint8), np.uint8)
    init("levels", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    init("crop_st", np.array([0, 0], np.int64), np.int64)
    init("crop_en", np.array([25, 25], np.int64), np.int64)
    init("axes_hw", np.array([2, 3], np.int64), np.int64)
    init("out_pad", np.array([0, 0, 0, 0, 0, 0, 5, 5], np.int64), np.int64)

    n("ReduceMax", ["input"], "row_any_f", axes=[1, 3], keepdims=1)
    n("ReduceMax", ["input"], "col_any_f", axes=[1, 2], keepdims=1)
    n("Greater", ["row_any_f", "zero_f"], "row_any")
    n("Greater", ["col_any_f", "zero_f"], "col_any")
    n("And", ["row_any", "col_any"], "ingrid30")
    n("Slice", ["ingrid30", "crop_st", "crop_en", "axes_hw"], "ingrid")

    n("Conv", ["input", "Wlabel"], "label_f30")
    n("Cast", ["label_f30"], "label_u830", to=U8)
    n("Slice", ["label_u830", "crop_st", "crop_en", "axes_hw"], "label_u8")
    n("Equal", ["label_u8", "zero_u8"], "is_zero")
    n("And", ["is_zero", "ingrid"], "black")
    n("Cast", ["black"], "black_u8", to=U8)

    n("Not", ["ingrid"], "invalid")
    n("Cast", ["invalid"], "invalid_u8", to=U8)
    n("MaxPool", ["invalid_u8"], "invalid_dil", kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])
    border = np.zeros((1, 1, 25, 25), dtype=np.uint8)
    border[:, :, 0, :] = 1
    border[:, :, :, 0] = 1
    border[:, :, 24, :] = 1
    border[:, :, :, 24] = 1
    init("border", border, np.uint8)
    n("Max", ["invalid_dil", "border"], "edge")
    n("Min", ["black_u8", "edge"], "reach0")

    prev = "reach0"
    for i in range(1, 15):
        n("MaxPool", [prev], f"mp{i}", kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])
        n("Min", [f"mp{i}", "black_u8"], f"reach{i}")
        prev = f"reach{i}"

    n("Equal", [prev, "one_u8"], "reachable")
    n("Where", ["reachable", "three_u8", "two_u8"], "black_label")
    n("Where", ["black", "black_label", "label_u8"], "in_label")
    n("Where", ["ingrid", "in_label", "ten_u8"], "out_label25")
    n("Pad", ["out_label25", "out_pad", "ten_u8"], "out_label", mode="constant")
    n("Equal", ["out_label", "levels"], "output")

    graph = helper.make_graph(
        nodes,
        "task187_label_flood",
        [helper.make_tensor_value_info("input", F, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", B, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = IR_VERSION
    return model
