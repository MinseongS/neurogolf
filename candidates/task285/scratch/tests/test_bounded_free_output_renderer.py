#!/usr/bin/env python3
"""Acceptance tests for task285's bounded FREE-output renderer."""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import onnx
import pytest
from onnx import TensorProto

from neurogolf.scans.public_autopsy import profile
from neurogolf.scoring import calculate_params


ROOT = pathlib.Path(__file__).resolve().parents[4]
BUILDER = ROOT / "candidates/task285/build_bounded_free_output_renderer.py"
MODEL = ROOT / "candidates/task285/bounded_free_output_renderer.onnx"


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "task285_bounded_free_output_builder", BUILDER
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def value_infos(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    return {
        value.name: value
        for value in list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    }


def value_shape(model: onnx.ModelProto, name: str) -> list[int]:
    tensor_type = value_infos(model)[name].type.tensor_type
    return [dim.dim_value for dim in tensor_type.shape.dim]


def value_types(model: onnx.ModelProto) -> dict[str, int]:
    return {
        name: value.type.tensor_type.elem_type
        for name, value in value_infos(model).items()
    }


def largest_counted_bytes(model: onnx.ModelProto) -> int:
    dtype_bytes = {
        TensorProto.FLOAT: 4,
        TensorProto.UINT8: 1,
        TensorProto.INT8: 1,
        TensorProto.UINT16: 2,
        TensorProto.INT16: 2,
        TensorProto.INT32: 4,
        TensorProto.INT64: 8,
        TensorProto.BOOL: 1,
        TensorProto.FLOAT16: 2,
        TensorProto.DOUBLE: 8,
        TensorProto.UINT32: 4,
        TensorProto.UINT64: 8,
    }
    free = {value.name for value in model.graph.input} | {
        value.name for value in model.graph.output
    }
    largest = 0
    for name, value in value_infos(model).items():
        if name in free:
            continue
        tensor_type = value.type.tensor_type
        shape = [dim.dim_value for dim in tensor_type.shape.dim]
        elements = int(np.prod(shape)) if shape else 1
        largest = max(largest, elements * dtype_bytes[tensor_type.elem_type])
    return largest


def test_bounded_free_output_builder_exists():
    assert BUILDER.exists(), "bounded FREE-output renderer builder is not implemented"


def test_background_stamp_delta_replaces_only_the_destination_colour():
    classes = np.arange(10, dtype=np.float32)
    baseline = 1.0 - classes**2
    for colour in range(10):
        delta = colour * (2.0 * classes - colour)
        actual = baseline + delta
        assert np.array_equal(actual > 0, classes == colour)


@pytest.mark.skipif(not MODEL.exists(), reason="candidate has not been built yet")
def test_bounded_model_has_bounded_two_stage_contractions():
    inferred = onnx.shape_inference.infer_shapes(onnx.load(MODEL), strict_mode=True)
    einsums = [node for node in inferred.graph.node if node.op_type == "Einsum"]
    assert [len(node.input) for node in einsums[-2:]] == [12, 19]
    assert value_shape(inferred, "source_shape") == [1, 3, 5, 5]
    assert largest_counted_bytes(inferred) <= 3_600
    assert load_builder().estimate_largest_intermediate(inferred) <= 1_000_000


@pytest.mark.skipif(not MODEL.exists(), reason="candidate has not been built yet")
def test_bounded_model_reaches_next_point_one_budget():
    model = onnx.load(MODEL)
    result = profile(MODEL)
    assert result is not None
    params = calculate_params(model)
    assert params is not None
    assert int(result["memory_static"]) + params <= 14_511


@pytest.mark.skipif(not MODEL.exists(), reason="candidate has not been built yet")
def test_all_topk_inputs_are_float16_or_float32():
    inferred = onnx.shape_inference.infer_shapes(onnx.load(MODEL), strict_mode=True)
    types = value_types(inferred)
    for node in inferred.graph.node:
        if node.op_type == "TopK":
            assert types[node.input[0]] in {TensorProto.FLOAT16, TensorProto.FLOAT}
