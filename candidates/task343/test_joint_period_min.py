from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


CANDIDATE = Path(__file__).with_name("joint_period_min.onnx")


def _attrs(node):
    return {
        attr.name: helper.get_attribute_value(attr)
        for attr in node.attribute
    }


def test_joint_period_min_graph_contract() -> None:
    assert CANDIDATE.exists(), f"missing candidate: {CANDIDATE}"
    inferred = onnx.shape_inference.infer_shapes(
        onnx.load(CANDIDATE), strict_mode=True, data_prop=True
    )
    onnx.checker.check_model(inferred, full_check=True)
    producers = {
        output: node for node in inferred.graph.node for output in node.output
    }

    expected_pads = {
        "column_8": [-2, -8, -25, -21],
        "column_11": [-2, -11, -25, -18],
        "p4_col0": [-2, 0, -25, -29],
        "p4_col4": [-2, -4, -25, -25],
        "low_col1": [-2, -1, -25, -28],
        "low_col5": [-2, -5, -25, -24],
    }
    for output, pads in expected_pads.items():
        node = producers[output]
        assert node.op_type == "Conv"
        assert list(node.input) == ["input", "signature_kernel"]
        assert list(_attrs(node)["pads"]) == pads

    assert list(producers["low_eq15"].input) == ["low_col5", "low_col1"]
    assert list(producers["low_is4"].input) == ["p4_eq4", "low_eq15"]
    assert list(producers["period_low_tensor_u8"].input) == [
        "low_is4", "period4_u8", "period3_u8"
    ]
    assert producers["visible_gt8_tensor"].op_type == "Cast"
    assert producers["visible_gt11_tensor"].op_type == "Cast"
    assert _attrs(producers["visible_gt8_tensor"])["to"] == TensorProto.BOOL
    assert _attrs(producers["visible_gt11_tensor"])["to"] == TensorProto.BOOL
    assert list(producers["period_high_tensor_u8"].input) == [
        "visible_gt11_tensor", "period8_u8", "period_mid_tensor_u8"
    ]
    assert list(producers["period_tensor_u8"].input) == [
        "visible_gt8_tensor", "period_high_tensor_u8", "period_low_tensor_u8"
    ]
    assert producers["period_u8"].op_type == "ReduceMax"
    assert list(producers["period_u8"].input) == ["period_tensor_u8"]
    assert _attrs(producers["period_u8"])["axes"] == [0, 1, 2]
    assert _attrs(producers["period_u8"])["keepdims"] == 0

    removed = {
        "column_left", "column_right", "p3_eq", "p3_period_choices",
        "period_low_u8", "visible_le8_tensor", "visible_le11_tensor",
    }
    assert removed.isdisjoint(producers)

    initializers = {
        init.name: numpy_helper.to_array(init)
        for init in inferred.graph.initializer
    }
    assert set(initializers) == {
        "signature_kernel", "period3_u8", "period4_u8",
        "period6_u8", "period8_u8", "final_mod_inputs",
    }
    assert sum(array.size for array in initializers.values()) == 64
    np.testing.assert_array_equal(
        initializers["final_mod_inputs"],
        np.concatenate(
            [np.arange(15, dtype=np.int32), -np.ones(15, dtype=np.int32)]
        ),
    )
