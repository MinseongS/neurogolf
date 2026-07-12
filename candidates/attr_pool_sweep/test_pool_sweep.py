import numpy as np
import onnxruntime as ort

from candidates.attr_pool_sweep.pool_sweep import (
    AxisConfig,
    build_model,
    find_pool_configs,
    grid_to_tensor,
    pool1d,
    pool2d,
)


def test_pool1d_identity_and_asymmetric_directional_expansion():
    source = np.zeros((1, 7), dtype=bool)
    source[0, 2] = True

    identity = pool1d(source, AxisConfig(kernel=1, dilation=1, pad_before=0))
    assert np.array_equal(identity, source)

    expanded = pool1d(source, AxisConfig(kernel=3, dilation=1, pad_before=2))
    expected = np.zeros_like(source)
    expected[0, 2:5] = True
    assert np.array_equal(expanded, expected)


def test_pool1d_dilation_preserves_spacing():
    source = np.zeros((1, 7), dtype=bool)
    source[0, 2] = True
    actual = pool1d(source, AxisConfig(kernel=3, dilation=2, pad_before=2))
    expected = np.zeros_like(source)
    expected[0, [0, 2, 4]] = True
    assert np.array_equal(actual, expected)


def test_pool2d_rectangular_configuration():
    source = np.zeros((1, 7, 7), dtype=bool)
    source[0, 2, 3] = True
    actual = pool2d(
        source,
        AxisConfig(kernel=3, dilation=1, pad_before=2),
        AxisConfig(kernel=2, dilation=1, pad_before=0),
    )
    expected = np.zeros_like(source)
    expected[0, 2:5, 2:4] = True
    assert np.array_equal(actual, expected)


def test_find_pool_configs_recovers_shared_config_and_rejects_recoloring():
    row = AxisConfig(kernel=3, dilation=1, pad_before=2)
    col = AxisConfig(kernel=2, dilation=1, pad_before=0)
    examples = []
    for r, c in [(2, 3), (4, 1)]:
        source = np.zeros((2, 7, 7), dtype=bool)
        source[1, r, c] = True
        examples.append((source, pool2d(source, row, col)))

    hits = find_pool_configs(examples)
    assert (row, col) in hits

    recolored = [(examples[0][0], examples[0][1][::-1])]
    assert find_pool_configs(recolored) == []


def test_grid_padding_and_onnx_model_match_numpy_pool():
    tensor = grid_to_tensor([[0, 2], [3, 0]])
    assert tensor.shape == (10, 30, 30)
    assert tensor[0, 0, 0] and tensor[2, 0, 1] and tensor[3, 1, 0]
    assert not tensor[:, 5, 5].any()

    row = AxisConfig(kernel=3, dilation=2, pad_before=2)
    col = AxisConfig(kernel=2, dilation=1, pad_before=1)
    model = build_model(row, col)
    session = ort.InferenceSession(model.SerializeToString())
    source = np.zeros((1, 10, 30, 30), dtype=np.float32)
    source[0, 2, 8, 9] = 1.0
    actual = session.run(["output"], {"input": source})[0] > 0
    expected = pool2d(source[0] > 0, row, col)[None]
    assert np.array_equal(actual, expected)


def test_grid_larger_than_official_canvas_is_skipped():
    assert grid_to_tensor([[0] * 31]) is None
