"""Task 067 — one-node repeated-block crop.

The input is three side-by-side copies of a size x size grid, with the middle
copy optionally vertically flipped.  The output is the left copy only.

Use one Einsum with the input twice:

  output[b,k,r,c] = input[b,k,r,c] * sum_{l,d} input[b,l,c,d]

The summed second operand is positive exactly when row c is inside the active
grid, i.e. when c < size.  That gives the needed column keep mask without any
initializer or counted intermediate tensor.
"""

from onnx import TensorProto, helper

from ..harness import IR_VERSION


def build(task):
    node = helper.make_node(
        "Einsum",
        ["input", "input"],
        ["output"],
        equation="bkrc,blcd->bkrc",
    )
    graph = helper.make_graph(
        [node],
        "task067",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [],
    )
    return helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 12)])
