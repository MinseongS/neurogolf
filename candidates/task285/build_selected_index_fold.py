#!/usr/bin/env python3
"""Fold task285's shared selected-index rank adapter into compact gathers."""

from __future__ import annotations

import pathlib

import onnx
from onnx import helper

from neurogolf.scans.public_autopsy import profile
from neurogolf.scoring import calculate_params


HERE = pathlib.Path(__file__).resolve().parent
SOURCE = HERE / "target_sentinel_fold_candidate.onnx"
OUTPUT = HERE / "selected_index_fold_candidate.onnx"
EXPECTED_SOURCE_COST = 15_657
EXPECTED_COST = 15_648


def clone_node(node: onnx.NodeProto) -> onnx.NodeProto:
    copied = onnx.NodeProto()
    copied.CopyFrom(node)
    return copied


def build_model() -> onnx.ModelProto:
    model = onnx.load(SOURCE)
    source_profile = profile(SOURCE)
    source_params = calculate_params(model)
    assert source_profile is not None and source_params is not None
    assert (
        int(source_profile["memory_static"]) + source_params == EXPECTED_SOURCE_COST
    )

    nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        outputs = set(node.output)
        if "si2" in outputs:
            continue
        if "a" in outputs:
            nodes.extend(
                [
                    helper.make_node("Gather", ["t", "si"], ["a_flat"]),
                    helper.make_node("Unsqueeze", ["a_flat", "axs1"], ["a"]),
                ]
            )
        elif "acol" in outputs:
            nodes.extend(
                [
                    helper.make_node("Gather", ["c", "si"], ["acol_flat"]),
                    helper.make_node(
                        "Unsqueeze", ["acol_flat", "axs1"], ["acol"]
                    ),
                ]
            )
        else:
            nodes.append(clone_node(node))

    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.value_info[:]
    model.graph.name = "task285_selected_index_rank_adapter_fold"
    model.producer_name = "task285-selected-index-fold"

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
    assert cost == EXPECTED_COST


if __name__ == "__main__":
    main()
