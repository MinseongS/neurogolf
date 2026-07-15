#!/usr/bin/env python3
"""Build an exact strict-cheaper task285 connectivity/reshape fold.

The first connectivity dilation starts from only ``mem2d[..., 0, 0]``.  Its
5x5 MaxPool with two-cell symmetric padding is exactly a top-left 3x3 block,
so the counted 75-byte seed mask and 75-byte pool output can be replaced by a
3-byte Slice, 27-byte Expand, and 75-byte Pad.  Two 2-D reshapes become default
Transpose nodes, and the 4-D-to-2-D reshape becomes default Flatten.
"""

from __future__ import annotations

import hashlib
import pathlib

import numpy as np
import onnx
from onnx import helper, numpy_helper

from neurogolf.scans.public_autopsy import profile
from neurogolf.scoring import calculate_params


HERE = pathlib.Path(__file__).resolve().parent
SOURCE = HERE / "sparse_renderer_candidate.onnx"
OUTPUT = HERE / "connectivity_fold_candidate.onnx"
SOURCE_SHA256 = "b201dd621c751a7a9a9352cf7fe6fbc6ac9d35c5c701857ee06953c037b148d4"
INCUMBENT_COST = 16_038
EXPECTED_COST = 15_982


def clone_node(node: onnx.NodeProto) -> onnx.NodeProto:
    copied = onnx.NodeProto()
    copied.CopyFrom(node)
    return copied


def build_model() -> onnx.ModelProto:
    source_bytes = SOURCE.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != SOURCE_SHA256:
        raise ValueError(
            f"unrecognized task285 sparse seed sha256={source_sha256}; "
            f"expected {SOURCE_SHA256}"
        )

    model = onnx.load_model_from_string(source_bytes)
    graph = model.graph
    nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        outputs = set(node.output)
        if outputs & {"e", "f"}:
            nodes.append(helper.make_node("Transpose", [node.input[0]], list(node.output)))
        elif "m0" in outputs:
            nodes.extend(
                [
                    helper.make_node(
                        "Slice",
                        ["mem2d", "seed_starts", "seed_ends", "seed_axes"],
                        ["root_cell"],
                    ),
                    helper.make_node(
                        "Expand", ["root_cell", "seed_block_shape"], ["p1_block"]
                    ),
                    helper.make_node("Pad", ["p1_block", "seed_block_pads"], ["p1"]),
                ]
            )
        elif "p1" in outputs:
            continue
        elif "mflat" in outputs:
            nodes.append(helper.make_node("Flatten", [node.input[0]], list(node.output)))
        else:
            nodes.append(clone_node(node))

    new_initializers = {
        "seed_starts": np.array([0, 0], dtype=np.int64),
        "seed_ends": np.array([1, 1], dtype=np.int64),
        "seed_axes": np.array([2, 3], dtype=np.int64),
        "seed_block_shape": np.array([3, 1, 3, 3], dtype=np.int64),
        "seed_block_pads": np.array([0, 0, 0, 0, 0, 0, 2, 2], dtype=np.int64),
    }

    del graph.node[:]
    graph.node.extend(nodes)
    used = {name for node in graph.node for name in node.input}
    kept_initializers: list[onnx.TensorProto] = []
    for old_initializer in graph.initializer:
        if old_initializer.name in used and old_initializer.name not in new_initializers:
            copied = onnx.TensorProto()
            copied.CopyFrom(old_initializer)
            kept_initializers.append(copied)
    kept_initializers.extend(
        numpy_helper.from_array(value, name=name)
        for name, value in new_initializers.items()
        if name in used
    )
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)
    del graph.value_info[:]
    graph.name = "task285_exact_connectivity_seed_fold"
    model.producer_name = "task285-connectivity-fold"

    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    inferred_by_output = {
        output: node for node in inferred.graph.node for output in node.output
    }
    if inferred_by_output["p1"].op_type != "Pad":
        raise AssertionError("first connectivity dilation was not folded")
    return inferred


def main() -> None:
    model = build_model()
    onnx.save(model, OUTPUT)
    result = profile(OUTPUT)
    assert result is not None
    params = calculate_params(model)
    assert params is not None
    memory = int(result["memory_static"])
    cost = memory + params
    print(f"saved {OUTPUT}")
    print(f"memory={memory} params={params} cost={cost}")
    if cost != EXPECTED_COST:
        raise AssertionError(f"unexpected candidate cost {cost} != {EXPECTED_COST}")
    if cost >= INCUMBENT_COST:
        raise AssertionError(f"candidate is not strictly cheaper than {INCUMBENT_COST}")


if __name__ == "__main__":
    main()
