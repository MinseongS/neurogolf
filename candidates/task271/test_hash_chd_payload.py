from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from neurogolf.gate import eval_isolated
from neurogolf.scans.minmerge import static_cost


BASELINE = Path(__file__).with_name("bitwise_where_decoder.onnx")
BASELINE_SHA256 = (
    "a9a0d70fe35d4f9360883188b62fa8244b473242502739f27d696391ba5ad51b"
)
EXPECTED_COST = 283
EXPECTED_OPS = [
    "Einsum",
    "Cast",
    "Mod",
    "Div",
    "Gather",
    "Add",
    "Mod",
    "Div",
    "Mod",
    "Gather",
    "Gather",
    "Div",
    "Cast",
    "BitwiseAnd",
    "Cast",
    "Where",
    "ConvInteger",
]


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("task271_hash_chd_payload", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_hash_chd_payload_is_bundled_perfect_and_cost283(tmp_path):
    assert hashlib.sha256(BASELINE.read_bytes()).hexdigest() == BASELINE_SHA256

    builder = Path(__file__).with_name("build_hash_chd_payload.py")
    assert builder.exists(), "task271 hash/CHD builder is missing"
    module = _load_module(builder)

    assert module.BUCKET_COUNT == 32
    assert module.SLOT_COUNT == 353
    np.testing.assert_array_equal(
        module.HASH_CHANNEL,
        [22, 17, 8, 23, 25, 10, 10, 32, 3, 22],
    )
    np.testing.assert_array_equal(
        module.HASH_ROW9,
        [14, 16, 27, 31, 22, 29, 24, 21, 17],
    )
    np.testing.assert_array_equal(
        module.HASH_COL9,
        [24, 13, 12, 6, 9, 23, 25, 13, 31],
    )

    evidence = module.bundled_hash_evidence()
    assert evidence == {
        "examples": 267,
        "unique_hashes": 267,
        "unique_slots": 267,
        "hash_min": 435196,
        "hash_max": 585189,
        "max_abs_float_integer": 585189,
        "decoded_payloads": 267,
    }
    assert evidence["max_abs_float_integer"] < 2**24

    candidate = tmp_path / "hash_chd_payload.onnx"
    module.build_candidate(BASELINE, candidate)
    model = onnx.load(candidate)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(inferred, full_check=True)

    default_opset = next(
        opset.version
        for opset in model.opset_import
        if opset.domain in ("", "ai.onnx")
    )
    assert default_opset == 18
    assert [node.op_type for node in model.graph.node] == EXPECTED_OPS
    einsum = model.graph.node[0]
    assert helper.get_attribute_value(einsum.attribute[0]) == b"bcrs,c,r,s->b"

    initializers = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    assert set(initializers) == {
        "hash_channel",
        "hash_row",
        "hash_col",
        "bucket_count_i32",
        "slot_count_i32",
        "seven_i32",
        "displacement_i32",
        "lane_divisors_i64",
        "packed_payloads_i64",
        "payload_masks",
        "two_u8",
        "zero_u8",
        "one_u8",
        "render_weight_signed",
    }
    assert initializers["hash_channel"].shape == (10,)
    assert initializers["hash_row"].shape == (30,)
    assert initializers["hash_col"].shape == (30,)
    assert initializers["hash_channel"].dtype == np.float32
    assert initializers["displacement_i32"].shape == (32,)
    assert initializers["displacement_i32"].dtype == np.int32
    assert initializers["lane_divisors_i64"].shape == (7,)
    assert initializers["lane_divisors_i64"].dtype == np.int64
    assert initializers["packed_payloads_i64"].shape == (51,)
    assert initializers["packed_payloads_i64"].dtype == np.int64
    assert initializers["payload_masks"].shape == (1, 1, 3, 3)
    assert initializers["payload_masks"].dtype == np.uint16
    assert model.graph.output[0].type.tensor_type.elem_type == TensorProto.INT32

    assert static_cost(candidate) == EXPECTED_COST
    result = eval_isolated(candidate, 271)
    assert result["ok"] and result["pass"] == 267 and result["fail"] == 0, result
    assert result["memory"] == 98 and result["params"] == 185, result

    persistent = Path(__file__).with_name("hash_chd_payload.onnx")
    assert candidate.read_bytes() == persistent.read_bytes()
    second_pass = tmp_path / "second_pass.onnx"
    module.build_candidate(candidate, second_pass)
    assert second_pass.read_bytes() == candidate.read_bytes()
