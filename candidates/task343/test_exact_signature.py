from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE / "exact_signature.onnx"


def _onehot(grid: np.ndarray) -> np.ndarray:
    result = np.zeros((1, 10, 30, 30), dtype=np.float32)
    rows, cols = np.indices(grid.shape)
    result[0, grid, rows, cols] = 1.0
    return result


def test_base10_signature_is_injective_and_fp32_exact() -> None:
    codes = {
        100 * top + 10 * middle + bottom
        for top in range(10)
        for middle in range(10)
        for bottom in range(10)
    }
    assert len(codes) == 1000
    assert min(codes) == 0
    assert max(codes) == 999
    np.testing.assert_array_equal(
        np.asarray(sorted(codes), dtype=np.float32),
        np.arange(1000, dtype=np.float32),
    )


def test_exact_signature_handles_visible7_period3() -> None:
    assert CANDIDATE.exists(), f"missing candidate: {CANDIDATE}"
    input_grid = np.zeros((5, 15), dtype=np.int64)
    output_grid = np.zeros((5, 15), dtype=np.int64)
    columns = ((1,), (2, 2), (2, 2))
    for col in range(15):
        colors = columns[col % 3]
        for delta, color in enumerate(colors):
            output_grid[4 - delta, col] = color
            if col < 7:
                input_grid[4 - delta, col] = color

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(CANDIDATE), options, providers=["CPUExecutionProvider"]
    )
    actual = session.run(["output"], {"input": _onehot(input_grid)})[0] > 0
    expected = _onehot(output_grid) > 0
    np.testing.assert_array_equal(actual, expected)


def test_exact_signature_graph_contract() -> None:
    assert CANDIDATE.exists(), f"missing candidate: {CANDIDATE}"
    model = onnx.load(CANDIDATE)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    initializers = {
        init.name: numpy_helper.to_array(init) for init in inferred.graph.initializer
    }

    expected_kernel = np.zeros((1, 10, 3, 1), dtype=np.float32)
    expected_kernel[0, :, 0, 0] = 100 * np.arange(10, dtype=np.float32)
    expected_kernel[0, :, 1, 0] = 10 * np.arange(10, dtype=np.float32)
    expected_kernel[0, :, 2, 0] = np.arange(10, dtype=np.float32)
    np.testing.assert_array_equal(initializers["signature_kernel"], expected_kernel)
    np.testing.assert_array_equal(
        initializers["squeeze_all_axes"], np.array([0, 1, 2, 3], dtype=np.int64)
    )
    assert all(all(dim > 0 for dim in init.dims) for init in inferred.graph.initializer)
    np.testing.assert_array_equal(
        initializers["final_mod_inputs"],
        np.array(list(range(15)) + [-1] * 15, dtype=np.int32),
    )

    producers = {
        output: node for node in inferred.graph.node for output in node.output
    }
    expected_pads = {
        "column_left": [-2, 0, -25, -27],
        "column_right": [-2, -3, -25, -24],
        "column_8": [-2, -8, -25, -21],
        "column_11": [-2, -11, -25, -18],
    }
    for output, pads in expected_pads.items():
        node = producers[output]
        attrs = {
            attr.name: helper.get_attribute_value(attr) for attr in node.attribute
        }
        assert node.op_type == "Conv"
        assert list(node.input) == ["input", "signature_kernel"]
        assert list(attrs["pads"]) == pads

    assert producers["visible_le8"].op_type == "Squeeze"
    assert producers["visible_le11"].op_type == "Squeeze"
    assert producers["p4_valid"].op_type == "Squeeze"
    assert producers["period_u8"].op_type == "Where"
    period_info = next(
        info for info in inferred.graph.value_info if info.name == "period_u8"
    )
    assert period_info.type.tensor_type.elem_type == TensorProto.UINT8

    final_mod = producers["final_cols"]
    attrs = {
        attr.name: helper.get_attribute_value(attr) for attr in final_mod.attribute
    }
    assert final_mod.op_type == "Mod"
    assert attrs["fmod"] == 1
    assert "chosen_cols" not in producers
    assert all(node.op_type != "Concat" for node in inferred.graph.node)
