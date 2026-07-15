#!/usr/bin/env python3
"""Acceptance tests for task285's exact connectivity/reshape fold."""

from __future__ import annotations

import pathlib

import onnx
import pytest
from onnx import TensorProto

from neurogolf.scans.public_autopsy import profile
from neurogolf.scoring import calculate_params, evaluate, load_task


ROOT = pathlib.Path(__file__).resolve().parents[4]
BUILDER = ROOT / "candidates/task285/build_connectivity_fold_candidate.py"
MODEL = ROOT / "candidates/task285/connectivity_fold_candidate.onnx"
INCUMBENT_COST = 16_038


def test_connectivity_fold_builder_exists():
    assert BUILDER.exists(), "connectivity fold builder is not implemented"


@pytest.mark.skipif(not MODEL.exists(), reason="candidate has not been built yet")
def test_connectivity_fold_replaces_only_exact_shape_operations():
    inferred = onnx.shape_inference.infer_shapes(onnx.load(MODEL), strict_mode=True)
    by_output = {
        output: node for node in inferred.graph.node for output in node.output
    }
    assert by_output["root_cell"].op_type == "Slice"
    assert by_output["p1_block"].op_type == "Expand"
    assert by_output["p1"].op_type == "Pad"
    assert by_output["e"].op_type == "Transpose"
    assert by_output["f"].op_type == "Transpose"
    assert by_output["mflat"].op_type == "Flatten"
    initializer_names = {initializer.name for initializer in inferred.graph.initializer}
    assert not {"shape_seed", "shape_3x1", "shape_3x25"} & initializer_names


@pytest.mark.skipif(not MODEL.exists(), reason="candidate has not been built yet")
def test_connectivity_fold_is_strictly_cheaper():
    model = onnx.load(MODEL)
    result = profile(MODEL)
    assert result is not None
    params = calculate_params(model)
    assert params is not None
    assert int(result["memory_static"]) + params < INCUMBENT_COST


@pytest.mark.skipif(not MODEL.exists(), reason="candidate has not been built yet")
def test_connectivity_fold_keeps_float_topk_inputs():
    inferred = onnx.shape_inference.infer_shapes(onnx.load(MODEL), strict_mode=True)
    types = {
        value.name: value.type.tensor_type.elem_type
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    types.update(
        {initializer.name: initializer.data_type for initializer in inferred.graph.initializer}
    )
    for node in inferred.graph.node:
        if node.op_type == "TopK":
            assert types[node.input[0]] in {TensorProto.FLOAT16, TensorProto.FLOAT}


@pytest.mark.skipif(not MODEL.exists(), reason="candidate has not been built yet")
def test_connectivity_fold_passes_all_bundled_examples():
    result = evaluate(MODEL, load_task(285), keep_failures=True)
    assert result["ok"], result
    assert result["pass"] == 265
    assert result["fail"] == 0
    assert result["memory"] + result["params"] == 15_982
