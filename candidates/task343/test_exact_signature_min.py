from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE / "exact_signature_min.onnx"


def test_exact_signature_min_graph_contract() -> None:
    assert CANDIDATE.exists(), f"missing candidate: {CANDIDATE}"
    model = onnx.load(CANDIDATE)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    producers = {
        output: node for node in inferred.graph.node for output in node.output
    }
    expected_pads = {
        "p4_col0": [-2, 0, -25, -29],
        "p4_col4": [-2, -4, -25, -25],
    }
    for output, pads in expected_pads.items():
        node = producers[output]
        attrs = {
            attr.name: helper.get_attribute_value(attr) for attr in node.attribute
        }
        assert node.op_type == "Conv"
        assert list(node.input) == ["input", "signature_kernel"]
        assert list(attrs["pads"]) == pads

    initializers = {
        init.name: numpy_helper.to_array(init) for init in inferred.graph.initializer
    }
    assert "idx0" not in initializers
    assert "idx1" not in initializers
    np.testing.assert_array_equal(
        initializers["squeeze_all_axes"], np.array([0, 1, 2], dtype=np.int64)
    )
    for name in ("period3_u8", "period4_u8", "period6_u8", "period8_u8"):
        assert initializers[name].shape == (1,)

    final_mod = producers["final_cols"]
    attrs = {
        attr.name: helper.get_attribute_value(attr) for attr in final_mod.attribute
    }
    assert final_mod.op_type == "Mod"
    assert attrs["fmod"] == 1
