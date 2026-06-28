"""Task 179 — transpose rows/columns."""

from onnx import TensorProto, helper

from ..harness import IR_VERSION


def build(task):
    node = helper.make_node("Transpose", ["input"], ["output"], perm=[0, 1, 3, 2])
    graph = helper.make_graph(
        [node],
        "task179",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [],
    )
    return helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 11)])
