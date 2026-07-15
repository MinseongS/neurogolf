"""Acceptance contract for task124's exact cost686 composite ladder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, numpy_helper

from neurogolf.scoring import convert_to_numpy, evaluate, load_task

from build_cost686_composite import build


ROOT = Path(__file__).resolve().parents[2]
INCUMBENT = ROOT / "submission/overfit_nets/task124.onnx"
EXPECTED = {
    "conservative": (659, 69, 728),
    "routing_u8": (633, 68, 701),
    "routing_i32": (644, 68, 712),
    "primary": (623, 63, 686),
    "primary_i32": (634, 63, 697),
}


def _initializer(model: onnx.ModelProto, name: str) -> np.ndarray:
    item = next(value for value in model.graph.initializer if value.name == name)
    return numpy_helper.to_array(item)


def _producer_map(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {output: node for node in model.graph.node for output in node.output}


def _value_info(model: onnx.ModelProto, name: str) -> onnx.ValueInfoProto:
    values = [*model.graph.input, *model.graph.value_info, *model.graph.output]
    return next(value for value in values if value.name == name)


def _shape(model: onnx.ModelProto, name: str) -> list[int]:
    value = _value_info(model, name)
    return [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]


def _dtype(model: onnx.ModelProto, name: str) -> int:
    return _value_info(model, name).type.tensor_type.elem_type


def _attribute_ints(node: onnx.NodeProto, name: str) -> list[int]:
    attribute = next(item for item in node.attribute if item.name == name)
    return list(attribute.ints)


def _ort_session(model_or_path: onnx.ModelProto | Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    payload = (
        model_or_path.SerializeToString()
        if isinstance(model_or_path, onnx.ModelProto)
        else model_or_path.read_bytes()
    )
    return ort.InferenceSession(
        payload, options, providers=["CPUExecutionProvider"]
    )


@pytest.mark.parametrize("stage", EXPECTED)
def test_stage_is_checker_legal_and_has_exact_cost(
    stage: str, tmp_path: Path
) -> None:
    candidate = build(stage=stage)
    onnx.checker.check_model(candidate, full_check=True)
    candidate = onnx.shape_inference.infer_shapes(candidate, strict_mode=True)
    path = tmp_path / f"{stage}.onnx"
    onnx.save(candidate, path)
    result = evaluate(path, load_task(124), keep_failures=True)
    memory, params, cost = EXPECTED[stage]
    assert (result["fail"], result["memory"], result["params"]) == (
        0,
        memory,
        params,
    ), result
    assert result["memory"] + result["params"] == cost


def test_primary_deletes_all_priced_intermediates() -> None:
    candidate = onnx.shape_inference.infer_shapes(
        build(stage="primary"), strict_mode=True
    )
    nodes = _producer_map(candidate)
    initializer_names = {item.name for item in candidate.graph.initializer}

    assert nodes["output"].op_type == "QLinearConv"
    assert candidate.graph.output[0].type.tensor_type.elem_type == TensorProto.UINT8
    assert list(nodes["ng_centered_mask"].input) == [
        "centered5",
        "bottom_row_0",
        "bottom_row_1",
        "bottom_row_2",
        "bottom_row_3",
        "bottom_row_4",
    ]
    assert nodes["centered5"].op_type == "QuantizeLinear"
    assert not {
        "fg5",
        "bottom_fg",
        "fg02_u8",
        "left_cols",
        "p3_rows_b",
        "bottom_start_0",
        "bottom_start_1",
        "bottom_start_2",
        "bottom_start_3",
        "bottom_start_4",
    }.intersection(nodes)
    assert not {"shift_kernel", "p3b_idx", "hash_code"}.intersection(
        initializer_names
    )
    assert sum(node.op_type == "QLinearConv" for node in candidate.graph.node) == 3
    assert not any(node.op_type == "QLinearMatMul" for node in candidate.graph.node)
    assert not any(node.op_type == "Pad" for node in candidate.graph.node)

    assert nodes["p3_hash_a"].op_type == "QLinearConv"
    assert nodes["p3_hash_b"].op_type == "QLinearConv"
    assert _attribute_ints(nodes["p3_hash_a"], "pads") == [0, 0, 0, -4]
    assert _attribute_ints(nodes["p3_hash_b"], "pads") == [-4, 0, 0, -4]
    assert _shape(candidate, "p3_hash_a") == [1, 1, 1, 1]
    assert _shape(candidate, "p3_hash_b") == [1, 1, 1, 1]

    assert _shape(candidate, "shift_scalar") == []
    assert _dtype(candidate, "shift_scalar") == TensorProto.UINT8
    assert _shape(candidate, "is_p3_scalar") == []
    assert _dtype(candidate, "is_p3_scalar") == TensorProto.BOOL
    assert _shape(candidate, "candidate_u8") == []
    assert _dtype(candidate, "candidate_u8") == TensorProto.UINT8
    assert _shape(candidate, "candidate_i32") == []
    assert _dtype(candidate, "candidate_i32") == TensorProto.INT32
    assert _shape(candidate, "source_offset") == [5]
    for index in range(5):
        assert _shape(candidate, f"bottom_start_vec_{index}") == [1]
        assert _dtype(candidate, f"bottom_start_vec_{index}") == TensorProto.INT32


def test_primary_quantized_polarity_is_exact() -> None:
    candidate = build(stage="primary")
    assert float(_initializer(candidate, "ng_half_scale")) == 0.5
    assert int(_initializer(candidate, "ng_x_zero_point")) == 1
    assert int(_initializer(candidate, "ng_y_zero_point")) == 0
    assert np.array_equal(
        _initializer(candidate, "ng_mask_base").reshape(-1),
        np.asarray([2] + [1] * 9, dtype=np.uint8),
    )
    assert np.array_equal(
        _initializer(candidate, "ng_background8"),
        np.full((1, 1, 1, 8), 2, dtype=np.uint8),
    )
    assert np.array_equal(
        _initializer(candidate, "ng_zero_update"),
        np.zeros((1, 1, 1, 1), dtype=np.uint8),
    )
    assert np.array_equal(
        _initializer(candidate, "hash6"),
        np.asarray([1, 2, 4, 8, 16, 32], dtype=np.uint8).reshape(
            1, 1, 1, 6
        ),
    )

    raw = np.asarray([0.0, 1.0], dtype=np.float32)
    stored_mask = np.rint(raw / np.float32(0.5)).astype(np.uint8)
    assert np.array_equal(stored_mask, np.asarray([0, 2], dtype=np.uint8))

    for colour in range(1, 10):
        stored_weights = np.asarray([2] + [1] * 9, dtype=np.int16)
        stored_weights[colour] = 0
        effective = stored_weights - 1
        assert effective[0] == 1
        assert effective[colour] == -1
        assert np.count_nonzero(effective) == 2


def test_six_column_hash_and_controller_match_all_generator_states() -> None:
    weights = np.asarray([1, 2, 4, 8, 16, 32], dtype=np.int64)
    cases = 0

    for tall in (2, 3):
        for wide in (1, 2, 3):
            cells = [(row, col) for row in range(tall) for col in range(wide)]
            for bits in range(1, 1 << len(cells)):
                sprite = [
                    cells[index]
                    for index in range(len(cells))
                    if bits >> index & 1
                ]
                if {row for row, _ in sprite} != set(range(tall)):
                    continue
                if {col for _, col in sprite} != set(range(wide)):
                    continue
                for offset in range(4):
                    for diag in ((0,) if tall == 3 else (0, 1)):
                        foreground_rows = []
                        for target_row in (0, 1, 2, 4):
                            mask = np.zeros(10, dtype=np.uint8)
                            for repeat in range(10):
                                for sprite_row, sprite_col in sprite:
                                    row = repeat * tall + sprite_row
                                    col = (
                                        repeat * (wide - 1) * diag
                                        + sprite_col
                                        + offset
                                    )
                                    if row == target_row and 0 <= col < 10:
                                        mask[col] = 2
                            foreground_rows.append(mask)

                        row0, row1, row2, row4 = foreground_rows
                        inverted = [2 - row for row in foreground_rows]
                        inv0, inv1, inv2, inv4 = inverted
                        left0 = int(np.argmin(inv0))
                        left2 = int(np.argmin(inv2))
                        shift_u8 = (left2 - left0) % 256
                        full_equal = np.array_equal(row1, row4)
                        hash1 = int(inv1[:6].astype(np.int64) @ weights)
                        hash4 = int(inv4[:6].astype(np.int64) @ weights)
                        hash_equal = hash1 == hash4
                        full_candidate = 3 if full_equal else shift_u8
                        hash_candidate = 3 if hash_equal else shift_u8

                        assert full_equal == hash_equal
                        assert full_candidate == hash_candidate
                        assert max(hash1, hash4) <= 126
                        cases += 1

    assert cases == 1_428

    period1 = np.full((5, 10), 2, dtype=np.uint8)
    period1[:, 2] = 0
    left0 = int(np.argmin(period1[0]))
    left2 = int(np.argmin(period1[2]))
    assert (left2 - left0) % 256 == 0


def test_all_stages_are_bundled_raw_exact_and_off_grid_zero() -> None:
    incumbent_session = _ort_session(INCUMBENT)
    stage_sessions = {stage: _ort_session(build(stage=stage)) for stage in EXPECTED}
    examples = load_task(124)
    bundled = examples["train"] + examples["test"] + examples.get("arc-gen", [])

    for index, example in enumerate(bundled):
        batch = convert_to_numpy(example)
        old = incumbent_session.run(["output"], {"input": batch["input"]})[0]
        for stage, session in stage_sessions.items():
            raw = session.run(["output"], {"input": batch["input"]})[0]
            assert np.array_equal(raw, old), f"{stage} raw mismatch at {index}"
            assert np.count_nonzero(raw[..., 10:, :]) == 0, (
                f"{stage} row off-grid at {index}"
            )
            assert np.count_nonzero(raw[..., :, 10:]) == 0, (
                f"{stage} column off-grid at {index}"
            )
