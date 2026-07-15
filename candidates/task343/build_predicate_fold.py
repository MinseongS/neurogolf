"""Fold task343's singleton period predicates without changing semantics."""

from pathlib import Path

import onnx
from onnx import TensorProto, helper

from build_exact_signature_min import build as build_exact_signature_min


OUTPUT = Path(__file__).with_name("predicate_fold.onnx")


def build(_task=None) -> onnx.ModelProto:
    model = build_exact_signature_min()
    graph = model.graph
    graph.name = "task343_predicate_fold"
    producers = {output: node for node in graph.node for output in node.output}
    nodes = [
        producers["column_left"],
        producers["column_right"],
        producers["column_8"],
        producers["column_11"],
        producers["p3_eq"],
        helper.make_node(
            "Where",
            ["p3_eq", "period3_u8", "period4_u8"],
            ["p3_period_choices"],
        ),
        helper.make_node(
            "ReduceMax",
            ["p3_period_choices"],
            ["period_low_u8"],
            axes=[0, 1, 2, 3],
            keepdims=0,
        ),
        producers["p4_col4"],
        producers["p4_col0"],
        producers["p4_eq4"],
        producers["p4_eq8"],
        producers["p4_eq"],
        producers["visible_le8_tensor"],
        producers["visible_le11_tensor"],
        helper.make_node(
            "Where",
            ["p4_eq", "period4_u8", "period6_u8"],
            ["period_mid_tensor_u8"],
        ),
        helper.make_node(
            "Where",
            ["visible_le11_tensor", "period_mid_tensor_u8", "period8_u8"],
            ["period_high_tensor_u8"],
        ),
        helper.make_node(
            "Where",
            ["visible_le8_tensor", "period_low_u8", "period_high_tensor_u8"],
            ["period_tensor_u8"],
        ),
        helper.make_node(
            "Squeeze",
            ["period_tensor_u8", "squeeze_all_axes"],
            ["period_u8"],
        ),
        helper.make_node(
            "Cast", ["period_u8"], ["period"], to=TensorProto.INT32
        ),
        producers["final_cols"],
        producers["output"],
    ]
    del graph.node[:]
    graph.node.extend(nodes)
    del graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    return onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )


if __name__ == "__main__":
    onnx.save(build(), OUTPUT)
    print(OUTPUT)
