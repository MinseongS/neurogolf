from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


CANDIDATE = Path(__file__).with_name("predicate_fold.onnx")


def _attrs(node):
    return {
        attr.name: helper.get_attribute_value(attr)
        for attr in node.attribute
    }


def test_predicate_fold_graph_contract() -> None:
    assert CANDIDATE.exists(), f"missing candidate: {CANDIDATE}"
    inferred = onnx.shape_inference.infer_shapes(
        onnx.load(CANDIDATE), strict_mode=True, data_prop=True
    )
    onnx.checker.check_model(inferred, full_check=True)
    producers = {
        output: node for node in inferred.graph.node for output in node.output
    }
    assert list(producers["p3_period_choices"].input) == [
        "p3_eq", "period3_u8", "period4_u8"
    ]
    assert producers["p3_period_choices"].op_type == "Where"
    assert producers["period_low_u8"].op_type == "ReduceMax"
    assert _attrs(producers["period_low_u8"])["axes"] == [0, 1, 2, 3]
    assert list(producers["period_mid_tensor_u8"].input) == [
        "p4_eq", "period4_u8", "period6_u8"
    ]
    assert list(producers["period_high_tensor_u8"].input) == [
        "visible_le11_tensor", "period_mid_tensor_u8", "period8_u8"
    ]
    assert list(producers["period_tensor_u8"].input) == [
        "visible_le8_tensor", "period_low_u8", "period_high_tensor_u8"
    ]
    assert producers["period_u8"].op_type == "Squeeze"
    assert list(producers["period_u8"].input) == [
        "period_tensor_u8", "squeeze_all_axes"
    ]
    removed = {
        "p3_eq_u8", "p3_valid_u8", "p3_valid",
        "p4_valid", "visible_le8", "visible_le11",
    }
    assert removed.isdisjoint(producers)
    initializers = {
        init.name: numpy_helper.to_array(init)
        for init in inferred.graph.initializer
    }
    np.testing.assert_array_equal(
        initializers["squeeze_all_axes"], np.array([0, 1, 2], dtype=np.int64)
    )
