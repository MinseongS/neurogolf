#!/usr/bin/env python3
"""Acceptance tests for task285's redundant target-sentinel fold."""

from __future__ import annotations

import pathlib

import onnx
from onnx import TensorProto

from neurogolf.scans.public_autopsy import profile
from neurogolf.scoring import calculate_params, evaluate, load_task


ROOT = pathlib.Path(__file__).resolve().parents[4]
BUILDER = ROOT / "candidates/task285/build_target_sentinel_fold.py"
MODEL = ROOT / "candidates/task285/target_sentinel_fold_candidate.onnx"


def test_target_sentinel_builder_exists():
    assert BUILDER.exists(), "target-sentinel fold builder is not implemented"


def test_target_sentinel_fold_structure_and_cost():
    inferred = onnx.shape_inference.infer_shapes(onnx.load(MODEL), strict_mode=True)
    by_output = {
        output: node for node in inferred.graph.node for output in node.output
    }
    assert not {"target_base", "target10", "target_cap", "v81_safe"} & set(
        by_output
    )
    assert list(by_output["newg"].input) == ["g", "tidx", "v81"]
    assert "i89" not in {
        initializer.name for initializer in inferred.graph.initializer
    }
    types = {
        value.name: value.type.tensor_type.elem_type
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    for node in inferred.graph.node:
        if node.op_type == "TopK":
            assert types[node.input[0]] in {TensorProto.FLOAT16, TensorProto.FLOAT}
    result = profile(MODEL)
    params = calculate_params(inferred)
    assert result is not None and params is not None
    assert int(result["memory_static"]) == 15_424
    assert params == 233
    assert int(result["memory_static"]) + params == 15_657


def test_target_sentinel_fold_passes_bundled():
    result = evaluate(MODEL, load_task(285), keep_failures=True)
    assert result["ok"], result
    assert result["pass"] == 265
    assert result["fail"] == 0
    assert result["memory"] + result["params"] == 15_657
