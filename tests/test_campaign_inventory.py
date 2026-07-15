import math

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tools.campaign_inventory import analyze_model, point_gain


def toy_model():
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 4, 4])
    w = numpy_helper.from_array(np.ones((3, 1, 1, 1), dtype=np.float32), name="W")
    conv = helper.make_node("Conv", ["input", "W"], ["plane"], name="conv")
    reduce = helper.make_node("ReduceSum", ["plane"], ["output"], name="reduce", axes=[1], keepdims=1)
    graph = helper.make_graph([conv, reduce], "toy", [x], [y], [w])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def test_point_gain_uses_neurogolf_log_formula():
    assert point_gain(1000, 100) == pytest.approx(math.log(1000 / 900))


def test_analyzer_counts_intermediate_bytes_but_initializer_elements():
    row = analyze_model(1, toy_model(), {"cost": 1000, "points": 18.0, "sha256": "abc"})
    assert row["largest_tensor"]["name"] == "plane"
    assert row["largest_tensor"]["saving"] == 3 * 4 * 4 * 4
    assert row["largest_param_bank"]["name"] == "W"
    assert row["largest_param_bank"]["saving"] == 3
