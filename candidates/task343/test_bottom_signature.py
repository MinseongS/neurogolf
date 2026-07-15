from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


CANDIDATE = Path(__file__).with_name("bottom_signature.onnx")


def test_bottom_signature_graph_contract() -> None:
    assert CANDIDATE.exists(), f"missing candidate: {CANDIDATE}"
    inferred = onnx.shape_inference.infer_shapes(
        onnx.load(CANDIDATE), strict_mode=True, data_prop=True
    )
    onnx.checker.check_model(inferred, full_check=True)
    initializers = {
        init.name: numpy_helper.to_array(init)
        for init in inferred.graph.initializer
    }
    np.testing.assert_array_equal(
        initializers["signature_kernel"],
        np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1),
    )
    signature_convs = [
        node for node in inferred.graph.node
        if node.op_type == "Conv"
        and list(node.input) == ["input", "signature_kernel"]
    ]
    assert len(signature_convs) == 6
    for node in signature_convs:
        attrs = {
            attr.name: helper.get_attribute_value(attr)
            for attr in node.attribute
        }
        assert list(attrs["pads"])[0] == -4
        assert list(attrs["pads"])[2] == -25
