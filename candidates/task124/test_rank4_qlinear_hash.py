"""Acceptance tests for task124's rank-4 bank and exact QLinear row hash."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

from neurogolf.scoring import convert_to_numpy, evaluate, load_task

from build_rank4_qlinear_hash import build


ROOT = Path(__file__).resolve().parents[2]
INCUMBENT = ROOT / "submission/overfit_nets/task124.onnx"
CANDIDATE = Path(__file__).with_name("rank4_qlinear_hash.onnx")


def _initializer(model: onnx.ModelProto, name: str) -> np.ndarray:
    value = next(item for item in model.graph.initializer if item.name == name)
    return numpy_helper.to_array(value)


def _shape(model: onnx.ModelProto, name: str) -> list[int]:
    values = [*model.graph.input, *model.graph.value_info, *model.graph.output]
    value = next(item for item in values if item.name == name)
    return [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]


def _axis(node: onnx.NodeProto) -> int:
    return next(attribute.i for attribute in node.attribute if attribute.name == "axis")


def test_rank4_qlinear_hash_structure_and_exact_cost() -> None:
    candidate = build()
    onnx.checker.check_model(candidate, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(candidate, strict_mode=True)
    onnx.checker.check_model(inferred, full_check=True)

    nodes = {output: node for node in candidate.graph.node for output in node.output}
    assert sum(node.op_type == "QLinearMatMul" for node in candidate.graph.node) == 2
    qmm_inputs = [
        "ng_scale", "ng_y_zero_point", "hash_code", "ng_scale",
        "ng_y_zero_point", "ng_scale", "ng_y_zero_point",
    ]
    assert list(nodes["p3_hash_a"].input) == ["p3_rows_a", *qmm_inputs]
    assert list(nodes["p3_hash_b"].input) == ["p3_rows_b", *qmm_inputs]
    assert nodes["is_p3"].op_type == "Equal"

    assert nodes["fg_pad4d"].op_type == "Concat"
    assert list(nodes["fg_pad4d"].input) == [
        "false8", "row0_4d", "false8", "p3_rows_a", "row2_4d",
    ]
    assert _axis(nodes["fg_pad4d"]) == 3
    assert nodes["bottom_fg"].op_type == "Concat"
    assert _axis(nodes["bottom_fg"]) == 2
    assert list(nodes["bottom_fg"].input) == [f"bottom_row_{i}" for i in range(5)]
    assert _axis(nodes["bottom_start4_0"]) == 4

    removed = {
        "row0_flat", "row1_flat", "row2_flat", "fg_pad_flat", "bottom_flat",
        "p3_equal", "p3_equal_u8", "p3_all_equal",
    }
    assert not removed.intersection(nodes)
    assert nodes["output"].op_type == "QLinearConv"
    assert candidate.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.UINT8

    expected_shapes = {
        "p3_rows_a": [1, 1, 1, 10],
        "p3_rows_b": [1, 1, 1, 10],
        "p3_hash_a": [1, 1, 1, 1],
        "p3_hash_b": [1, 1, 1, 1],
        "is_p3": [1, 1, 1, 1],
        "fg_pad4d": [1, 1, 1, 46],
        "bottom_fg": [1, 1, 5, 10],
        "output": [1, 10, 30, 30],
    }
    expected_shapes.update({f"bottom_row_{i}": [1, 1, 1, 10] for i in range(5)})
    for name, expected in expected_shapes.items():
        assert _shape(inferred, name) == expected, name

    onnx.save(candidate, CANDIDATE)
    result = evaluate(CANDIDATE, load_task(124), keep_failures=True)
    assert result["fail"] == 0, result
    assert result["memory"] == 779, result
    assert result["params"] == 70, result
    assert result["memory"] + result["params"] == 849, result


def test_quantized_tail_and_hash_initializers_are_exact() -> None:
    candidate = build()
    nodes = {output: node for node in candidate.graph.node for output in node.output}

    assert np.array_equal(
        _initializer(candidate, "hash_code").reshape(-1),
        np.asarray([1, 2, 4, 8, 16, 32, 0, 0, 0, 0], dtype=np.uint8),
    )
    assert _initializer(candidate, "false8").shape == (1, 1, 1, 8)
    assert np.array_equal(_initializer(candidate, "p3b_idx"), np.asarray([4]))
    assert int(_initializer(candidate, "ng_x_zero_point")) == 1
    assert int(_initializer(candidate, "ng_y_zero_point")) == 0
    assert int(_initializer(candidate, "ng_two").reshape(-1)[0]) == 2
    assert np.array_equal(
        _initializer(candidate, "ng_mask_base").reshape(-1),
        np.asarray([0] + [1] * 9, dtype=np.uint8),
    )
    assert list(nodes["output"].input) == [
        "ng_centered_mask", "ng_scale", "ng_x_zero_point",
        "ng_wmask", "ng_scale", "ng_x_zero_point",
        "ng_scale", "ng_y_zero_point",
    ]
    assert not any(node.op_type == "Pad" for node in candidate.graph.node)


def test_rank4_qlinear_hash_is_raw_exact_and_off_grid_zero() -> None:
    candidate = build()
    onnx.save(candidate, CANDIDATE)
    incumbent_session = ort.InferenceSession(
        INCUMBENT.read_bytes(), providers=["CPUExecutionProvider"]
    )
    candidate_session = ort.InferenceSession(
        CANDIDATE.read_bytes(), providers=["CPUExecutionProvider"]
    )
    examples = load_task(124)
    bundled = examples["train"] + examples["test"] + examples.get("arc-gen", [])

    for index, example in enumerate(bundled):
        batch = convert_to_numpy(example)
        old = incumbent_session.run(["output"], {"input": batch["input"]})[0]
        raw = candidate_session.run(["output"], {"input": batch["input"]})[0]
        assert np.array_equal(raw, old), f"raw A/B mismatch at {index}"
        assert np.count_nonzero(raw[..., 10:, :]) == 0, f"row off-grid at {index}"
        assert np.count_nonzero(raw[..., :, 10:]) == 0, f"column off-grid at {index}"


def test_six_bit_hash_matches_full_row_equality_on_generator_domain() -> None:
    weights = np.asarray([1, 2, 4, 8, 16, 32, 0, 0, 0, 0], dtype=np.int64)
    cases = 0

    for tall in (2, 3):
        for wide in (1, 2, 3):
            cells = [(row, col) for row in range(tall) for col in range(wide)]
            for bits in range(1, 1 << len(cells)):
                sprite = [cells[index] for index in range(len(cells)) if bits >> index & 1]
                if {row for row, _ in sprite} != set(range(tall)):
                    continue
                if {col for _, col in sprite} != set(range(wide)):
                    continue
                for offset in range(4):
                    for diag in ((0,) if tall == 3 else (0, 1)):
                        compared = []
                        for target_row in (1, 4):
                            mask = np.zeros(10, dtype=np.uint8)
                            for repeat in range(10):
                                for sprite_row, sprite_col in sprite:
                                    row = repeat * tall + sprite_row
                                    col = repeat * (wide - 1) * diag + sprite_col + offset
                                    if row == target_row and 0 <= col < 10:
                                        mask[col] = 2
                            compared.append(mask)
                        row1, row4 = compared
                        row1_hash = int(row1.astype(np.int64) @ weights)
                        row4_hash = int(row4.astype(np.int64) @ weights)
                        assert np.array_equal(row1, row4) == (row1_hash == row4_hash)
                        assert max(row1_hash, row4_hash) <= 126
                        cases += 1

    assert cases == 1_428
