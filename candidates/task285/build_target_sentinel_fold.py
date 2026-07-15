#!/usr/bin/env python3
"""Remove task285's redundant target-bound sentinel suffix."""

from __future__ import annotations

import hashlib
import pathlib

import onnx

from neurogolf.scans.public_autopsy import profile
from neurogolf.scoring import calculate_params


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "submission/overfit_nets/task285.onnx"
OUTPUT = pathlib.Path(__file__).resolve().parent / "target_sentinel_fold_candidate.onnx"
SOURCE_SHA256 = "dc509d96955eea40d6984648934e9b345293057aa61b6f2bf0d575106fa09120"
EXPECTED_COST = 15_657
REMOVED = {"target_base", "target10", "target_cap", "v81_safe"}


def clone_node(node: onnx.NodeProto) -> onnx.NodeProto:
    copied = onnx.NodeProto()
    copied.CopyFrom(node)
    return copied


def build_model() -> onnx.ModelProto:
    source_bytes = SOURCE.read_bytes()
    actual_sha = hashlib.sha256(source_bytes).hexdigest()
    if actual_sha != SOURCE_SHA256:
        raise ValueError(f"unrecognized task285 source sha256={actual_sha}")

    model = onnx.load_model_from_string(source_bytes)
    nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if set(node.output) & REMOVED:
            continue
        copied = clone_node(node)
        if "newg" in copied.output:
            for index, name in enumerate(copied.input):
                if name == "v81_safe":
                    copied.input[index] = "v81"
        nodes.append(copied)

    del model.graph.node[:]
    model.graph.node.extend(nodes)

    used = {name for node in model.graph.node for name in node.input}
    initializers: list[onnx.TensorProto] = []
    for initializer in model.graph.initializer:
        if initializer.name in used:
            copied = onnx.TensorProto()
            copied.CopyFrom(initializer)
            initializers.append(copied)
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    del model.graph.value_info[:]
    model.graph.name = "task285_redundant_target_sentinel_fold"
    model.producer_name = "task285-target-sentinel-fold"

    onnx.checker.check_model(model, full_check=True)
    return onnx.shape_inference.infer_shapes(model, strict_mode=True)


def main() -> None:
    model = build_model()
    onnx.save(model, OUTPUT)
    result = profile(OUTPUT)
    params = calculate_params(model)
    assert result is not None and params is not None
    cost = int(result["memory_static"]) + params
    print(f"saved {OUTPUT}")
    print(f"memory={result['memory_static']} params={params} cost={cost}")
    if cost != EXPECTED_COST:
        raise AssertionError(f"unexpected cost {cost} != {EXPECTED_COST}")


if __name__ == "__main__":
    main()
