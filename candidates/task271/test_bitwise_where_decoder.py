from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from neurogolf.gate import eval_isolated
from neurogolf.scans.minmerge import static_cost


BASELINE = Path(__file__).with_name("encoded_score_payload.onnx")
BASELINE_SHA256 = "9f7303ea8e393536079fba14c87fec7b5e0908cf7b95917af2b3eb0b35f2e37d"
EXPECTED_OPS = [
    "Conv",
    "MaxPool",
    "Cast",
    "BitwiseAnd",
    "Cast",
    "Where",
    "ConvInteger",
]


def _encoded_kernel() -> np.ndarray:
    weight = np.zeros((1, 10, 3, 3), dtype=np.float32)
    weight[0, 0] = -4096.0
    for bit in range(9):
        weight[0, 1].reshape(-1)[bit] = 512.0 + (1 << bit)
    return weight


def _payload_masks() -> np.ndarray:
    return (1 << np.arange(9, dtype=np.uint16)).reshape(1, 1, 3, 3)


def _signed_render_weight() -> np.ndarray:
    weight = np.zeros((10, 1, 1, 1), dtype=np.int8)
    weight[1, 0, 0, 0] = 1
    weight[8, 0, 0, 0] = -1
    return weight


def _session(model: onnx.ModelProto) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def _assert_all_positions(session: ort.InferenceSession) -> None:
    patch_code = 0b101011001
    expected_patch = np.array(
        [(patch_code >> bit) & 1 for bit in range(9)], dtype=np.uint8
    ).reshape(3, 3)
    expected = np.zeros((1, 10, 30, 30), dtype=bool)
    expected[0, 1, :3, :3] = expected_patch == 1
    expected[0, 8, :3, :3] = expected_patch == 0

    for flat_index in range(49):
        row, col = divmod(flat_index, 7)
        input_tensor = np.zeros((1, 10, 30, 30), dtype=np.float32)
        input_tensor[0, 0, :9, :9] = 1.0
        for patch_row in range(3):
            for patch_col in range(3):
                input_tensor[0, 0, row + patch_row, col + patch_col] = 0.0
                channel = 1 if expected_patch[patch_row, patch_col] else 8
                input_tensor[0, channel, row + patch_row, col + patch_col] = 1.0
        actual = session.run(["output"], {"input": input_tensor})[0] > 0
        np.testing.assert_array_equal(actual, expected)


def _assert_all_payloads() -> None:
    batch_size = 512
    graph = helper.make_graph(
        [
            helper.make_node(
                "BitwiseAnd", ["encoded", "payload_masks"], ["masked_u16"]
            ),
            helper.make_node(
                "Cast", ["masked_u16"], ["payload_bits"], to=TensorProto.BOOL
            ),
            helper.make_node(
                "Where", ["payload_bits", "two_u8", "zero_u8"], ["blue2"]
            ),
            helper.make_node(
                "ConvInteger",
                ["blue2", "render_weight_signed", "one_u8"],
                ["output"],
                kernel_shape=[1, 1],
                pads=[0, 0, 27, 27],
            ),
        ],
        "task271_decoder_exhaustive",
        [
            helper.make_tensor_value_info(
                "encoded", TensorProto.UINT16, [batch_size, 1, 1, 1]
            )
        ],
        [
            helper.make_tensor_value_info(
                "output", TensorProto.INT32, [batch_size, 10, 30, 30]
            )
        ],
        [
            numpy_helper.from_array(_payload_masks(), "payload_masks"),
            numpy_helper.from_array(np.array([2], dtype=np.uint8), "two_u8"),
            numpy_helper.from_array(np.array([0], dtype=np.uint8), "zero_u8"),
            numpy_helper.from_array(np.array([1], dtype=np.uint8), "one_u8"),
            numpy_helper.from_array(
                _signed_render_weight(), "render_weight_signed"
            ),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
    )
    onnx.checker.check_model(model, full_check=True)
    session = _session(model)

    codes = np.arange(batch_size, dtype=np.uint16).reshape(batch_size, 1, 1, 1)
    actual = session.run(["output"], {"encoded": codes})[0] > 0
    expected = np.zeros((batch_size, 10, 30, 30), dtype=bool)
    for code in range(batch_size):
        patch = ((code >> np.arange(9)) & 1).reshape(3, 3)
        expected[code, 1, :3, :3] = patch == 1
        expected[code, 8, :3, :3] = patch == 0
    np.testing.assert_array_equal(actual, expected)
    assert not actual[:, :, 3:, :].any()
    assert not actual[:, :, :, 3:].any()


def test_bitwise_where_decoder_is_exact_and_cheaper(tmp_path):
    assert hashlib.sha256(BASELINE.read_bytes()).hexdigest() == BASELINE_SHA256

    builder_path = Path(__file__).with_name("build_bitwise_where_decoder.py")
    assert builder_path.exists(), "task271 bitwise/Where builder is missing"
    spec = importlib.util.spec_from_file_location(
        "task271_bitwise_where_decoder", builder_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    candidate = tmp_path / "bitwise_where_decoder.onnx"
    module.build_candidate(BASELINE, candidate)
    model = onnx.load(candidate)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)

    default_opset = next(
        opset.version
        for opset in model.opset_import
        if opset.domain in ("", "ai.onnx")
    )
    assert default_opset == 18
    assert [node.op_type for node in model.graph.node] == EXPECTED_OPS

    conv, pool, cast_code, bitwise, cast_bits, where, renderer = model.graph.node
    assert list(conv.input) == ["input", "score_payload_kernel"]
    assert list(conv.output) == ["scores"]
    conv_attrs = {
        attr.name: helper.get_attribute_value(attr) for attr in conv.attribute
    }
    assert conv_attrs == {"kernel_shape": [3, 3], "pads": [0, 0, -21, -21]}
    assert list(pool.input) == ["scores"]
    assert list(pool.output) == ["max_score"]
    pool_attrs = {
        attr.name: helper.get_attribute_value(attr) for attr in pool.attribute
    }
    assert pool_attrs == {"kernel_shape": [7, 7]}
    assert list(cast_code.input) == ["max_score"]
    assert helper.get_attribute_value(cast_code.attribute[0]) == TensorProto.UINT16
    assert list(bitwise.input) == ["encoded_u16", "payload_masks"]
    assert list(cast_bits.input) == ["masked_u16"]
    assert helper.get_attribute_value(cast_bits.attribute[0]) == TensorProto.BOOL
    assert list(where.input) == ["payload_bits", "two_u8", "zero_u8"]
    assert list(renderer.input) == ["blue2", "render_weight_signed", "one_u8"]
    renderer_attrs = {
        attr.name: helper.get_attribute_value(attr) for attr in renderer.attribute
    }
    assert renderer_attrs == {
        "kernel_shape": [1, 1],
        "pads": [0, 0, 27, 27],
    }
    assert not {"Mod", "GreaterOrEqual", "BitShift"} & {
        node.op_type for node in model.graph.node
    }

    initializers = {
        init.name: numpy_helper.to_array(init) for init in model.graph.initializer
    }
    assert set(initializers) == {
        "score_payload_kernel",
        "payload_masks",
        "two_u8",
        "zero_u8",
        "one_u8",
        "render_weight_signed",
    }
    np.testing.assert_array_equal(
        initializers["score_payload_kernel"], _encoded_kernel()
    )
    np.testing.assert_array_equal(initializers["payload_masks"], _payload_masks())
    np.testing.assert_array_equal(initializers["two_u8"], [2])
    np.testing.assert_array_equal(initializers["zero_u8"], [0])
    np.testing.assert_array_equal(initializers["one_u8"], [1])
    np.testing.assert_array_equal(
        initializers["render_weight_signed"], _signed_render_weight()
    )
    assert model.graph.output[0].type.tensor_type.elem_type == TensorProto.INT32

    _assert_all_positions(_session(model))
    _assert_all_payloads()

    assert static_cost(candidate) == 350
    result = eval_isolated(candidate, 271)
    assert result["ok"] and result["pass"] == 267 and result["fail"] == 0, result
    assert result["memory"] == 238 and result["params"] == 112, result

    persistent = Path(__file__).with_name("bitwise_where_decoder.onnx")
    assert candidate.read_bytes() == persistent.read_bytes()
    second_pass = tmp_path / "second_pass.onnx"
    module.build_candidate(candidate, second_pass)
    assert second_pass.read_bytes() == candidate.read_bytes()
