"""Task 146 — select the unique asymmetric 3x3 block from a 9x3 stack."""

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ..harness import IR_VERSION


def build(task):
    inits = []

    def init(name, arr, dt):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))

    # For each non-background channel k, this weighted checksum is zero iff the
    # 3x3 block is symmetric across the main diagonal. Channel 0 is unused here.
    base = np.array([[0, -1, -17], [1, 0, -289], [17, 289, 0]], np.float32)
    w = np.zeros((1, 10, 3, 3), np.float32)
    for k in range(1, 10):
        w[0, k] = k * base
    init("w_checksum", w, np.float32)
    init("zero_f32", np.array([0.0], np.float32), np.float32)
    init("axes_chw", np.array([1, 2, 3], np.int32), np.int32)
    init("zero_i32", np.array([0], np.int32), np.int32)
    init("one_i32", np.array([1], np.int32), np.int32)
    init("three_i32", np.array([3], np.int32), np.int32)
    init("six_i32", np.array([6], np.int32), np.int32)
    init("ten_i32", np.array([10], np.int32), np.int32)
    init("pads_out", np.array([0, 1, 0, 0, 0, 0, 27, 27], np.int64), np.int64)

    nodes = [
        helper.make_node("Conv", ["input", "w_checksum"], ["checks"], kernel_shape=[3, 3], pads=[-3, 0, -21, 0], strides=[3, 30]),
        helper.make_node("Split", ["checks"], ["b1_check", "b2_check"], axis=2, split=[1, 1]),
        helper.make_node("Equal", ["b1_check", "zero_f32"], ["b1_eq0"]),
        helper.make_node("Equal", ["b2_check", "zero_f32"], ["b2_eq0"]),
        helper.make_node("Squeeze", ["b1_eq0"], ["b1_sym"], axes=[1, 2, 3]),
        helper.make_node("Squeeze", ["b2_eq0"], ["b2_sym"], axes=[1, 2, 3]),
        helper.make_node("Where", ["b1_sym", "zero_i32", "three_i32"], ["row_if_not_b2"]),
        helper.make_node("Where", ["b2_sym", "row_if_not_b2", "six_i32"], ["row"]),
        helper.make_node("Add", ["row", "three_i32"], ["row_end"]),
        helper.make_node("Concat", ["one_i32", "row", "zero_i32"], ["starts_out"], axis=0),
        helper.make_node("Concat", ["ten_i32", "row_end", "three_i32"], ["ends_out"], axis=0),
        helper.make_node("Slice", ["input", "starts_out", "ends_out", "axes_chw"], ["selected"]),
        helper.make_node("Pad", ["selected", "pads_out"], ["output"], mode="constant"),
    ]
    graph = helper.make_graph(
        nodes,
        "task146",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        inits,
    )
    graph.value_info.append(helper.make_tensor_value_info("selected", TensorProto.FLOAT, [1, 9, 3, 3]))
    return helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 11)])
