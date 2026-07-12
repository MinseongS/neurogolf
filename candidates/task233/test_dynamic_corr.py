import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from src.custom.task233 import build as build_baseline
from src.harness import convert_to_numpy, evaluate, run_network, sanitize_model

from candidates.task233.build_dynamic_corr import build


MAIN_ROOT = Path("/Users/minseong/project/neurogolf")


def _session(model: onnx.ModelProto) -> ort.InferenceSession:
    cloned = onnx.ModelProto.FromString(model.SerializeToString())
    sanitized = sanitize_model(cloned)
    assert sanitized is not None
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def _bundled_examples():
    with (MAIN_ROOT / "data" / "task233.json").open() as handle:
        task = json.load(handle)
    return task["train"] + task["test"] + task["arc-gen"]


def _bundled_task():
    with (MAIN_ROOT / "data" / "task233.json").open() as handle:
        return json.load(handle)


def test_dynamic_correlation_replaces_hash_lookup_and_preserves_bundled_outputs():
    candidate = build()

    qconvs = [node for node in candidate.graph.node if node.op_type == "QLinearConv"]
    assert len(qconvs) == 1
    assert qconvs[0].input[3] not in {
        initializer.name for initializer in candidate.graph.initializer
    }

    maxpools = [node for node in candidate.graph.node if node.op_type == "MaxPool"]
    assert any(len(node.output) == 2 for node in maxpools)
    assert not any(node.op_type == "MatMul" for node in candidate.graph.node)
    assert not any(
        list(initializer.dims) == [1, 512]
        for initializer in candidate.graph.initializer
    )

    baseline_session = _session(build_baseline(None))
    candidate_session = _session(candidate)
    for example in _bundled_examples():
        converted = convert_to_numpy(example)
        assert converted is not None
        baseline = run_network(baseline_session, converted["input"])
        actual = run_network(candidate_session, converted["input"])
        assert np.array_equal(actual, baseline)


def test_dynamic_correlation_clears_histogram_cost_target():
    result = evaluate(build(), _bundled_task(), keep_failures=True)
    assert result["error"] is None, result
    assert result["fail"] == 0, result
    assert result["memory"] + result["params"] < 24_703, result
    assert result["memory"] + result["params"] < 22_026, result
