"""Share task343's p4 equality with a compact low-period predicate."""

from pathlib import Path

import onnx
from onnx import TensorProto, helper

from build_predicate_fold import build as build_predicate_fold


OUTPUT = Path(__file__).with_name("joint_period_min.onnx")
KEPT_INITIALIZERS = {
    "signature_kernel",
    "period3_u8",
    "period4_u8",
    "period6_u8",
    "period8_u8",
    "final_mod_inputs",
}


def _column_probe(column: int, output: str) -> onnx.NodeProto:
    return helper.make_node(
        "Conv",
        ["input", "signature_kernel"],
        [output],
        name=f"signature_{output}",
        pads=[-2, -column, -25, -(29 - column)],
    )


def build(_task=None) -> onnx.ModelProto:
    model = build_predicate_fold()
    graph = model.graph
    graph.name = "task343_joint_period_min"
    producers = {output: node for node in graph.node for output in node.output}

    nodes = [
        producers["column_8"],
        producers["column_11"],
        producers["p4_col4"],
        producers["p4_col0"],
        _column_probe(1, "low_col1"),
        _column_probe(5, "low_col5"),
        producers["p4_eq4"],
        producers["p4_eq8"],
        producers["p4_eq"],
        helper.make_node("Equal", ["low_col5", "low_col1"], ["low_eq15"]),
        helper.make_node("And", ["p4_eq4", "low_eq15"], ["low_is4"]),
        helper.make_node(
            "Where",
            ["low_is4", "period4_u8", "period3_u8"],
            ["period_low_tensor_u8"],
        ),
        helper.make_node(
            "Cast", ["column_8"], ["visible_gt8_tensor"], to=TensorProto.BOOL
        ),
        helper.make_node(
            "Cast", ["column_11"], ["visible_gt11_tensor"], to=TensorProto.BOOL
        ),
        helper.make_node(
            "Where",
            ["p4_eq", "period4_u8", "period6_u8"],
            ["period_mid_tensor_u8"],
        ),
        helper.make_node(
            "Where",
            ["visible_gt11_tensor", "period8_u8", "period_mid_tensor_u8"],
            ["period_high_tensor_u8"],
        ),
        helper.make_node(
            "Where",
            ["visible_gt8_tensor", "period_high_tensor_u8", "period_low_tensor_u8"],
            ["period_tensor_u8"],
        ),
        helper.make_node(
            "ReduceMax",
            ["period_tensor_u8"],
            ["period_u8"],
            axes=[0, 1, 2],
            keepdims=0,
        ),
        helper.make_node("Cast", ["period_u8"], ["period"], to=TensorProto.INT32),
        producers["final_cols"],
        producers["output"],
    ]
    del graph.node[:]
    graph.node.extend(nodes)
    kept = [init for init in graph.initializer if init.name in KEPT_INITIALIZERS]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    del graph.value_info[:]

    onnx.checker.check_model(model, full_check=True)
    return onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )


if __name__ == "__main__":
    onnx.save(build(), OUTPUT)
    print(OUTPUT)
