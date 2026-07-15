#!/usr/bin/env python3
"""Acceptance tests for task285's selected-index rank-adapter fold."""

from __future__ import annotations

import pathlib

import onnx

from neurogolf.scans.public_autopsy import profile
from neurogolf.scoring import calculate_params, evaluate, load_task


ROOT = pathlib.Path(__file__).resolve().parents[4]
BUILDER = ROOT / "candidates/task285/build_selected_index_fold.py"
MODEL = ROOT / "candidates/task285/selected_index_fold_candidate.onnx"


def test_selected_index_builder_exists():
    assert BUILDER.exists(), "selected-index fold builder is not implemented"


def test_selected_index_fold_structure_cost_and_bundled():
    inferred = onnx.shape_inference.infer_shapes(onnx.load(MODEL), strict_mode=True)
    by_output = {
        output: node for node in inferred.graph.node for output in node.output
    }
    assert "si2" not in by_output
    assert list(by_output["a_flat"].input) == ["t", "si"]
    assert list(by_output["acol_flat"].input) == ["c", "si"]
    assert by_output["a"].op_type == "Unsqueeze"
    assert by_output["acol"].op_type == "Unsqueeze"
    result = profile(MODEL)
    params = calculate_params(inferred)
    assert result is not None and params is not None
    assert int(result["memory_static"]) == 15_415
    assert params == 233
    assert int(result["memory_static"]) + params == 15_648
    evaluated = evaluate(MODEL, load_task(285), keep_failures=True)
    assert evaluated["pass"] == 265
    assert evaluated["fail"] == 0
