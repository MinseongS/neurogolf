"""Use task343's always-present bottom row as the period signature."""

from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from build_predicate_fold import build as build_predicate_fold


OUTPUT = Path(__file__).with_name("bottom_signature.onnx")


def build(_task=None) -> onnx.ModelProto:
    model = build_predicate_fold()
    graph = model.graph
    graph.name = "task343_bottom_signature"
    for index, init in enumerate(graph.initializer):
        if init.name == "signature_kernel":
            graph.initializer[index].CopyFrom(
                numpy_helper.from_array(
                    np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1),
                    name="signature_kernel",
                )
            )
            break
    for node in graph.node:
        if (
            node.op_type != "Conv"
            or list(node.input) != ["input", "signature_kernel"]
        ):
            continue
        attrs = {
            attr.name: helper.get_attribute_value(attr)
            for attr in node.attribute
        }
        pads = list(attrs["pads"])
        pads[0] = -4
        node.CopyFrom(
            helper.make_node(
                "Conv",
                list(node.input),
                list(node.output),
                name=node.name,
                pads=pads,
            )
        )
    del graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    return onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )


if __name__ == "__main__":
    onnx.save(build(), OUTPUT)
    print(OUTPUT)
