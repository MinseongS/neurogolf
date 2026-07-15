"""Build task124's exact inverse-quantized composite candidate ladder."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASELINE_SHA256 = "4e4bafbb3d65046a1ec08a211de6c9951705b613777a2a3ece9f4c73f6041b25"
STAGES = {"conservative", "routing_u8", "routing_i32", "primary", "primary_i32"}
OUTPUT = Path(__file__).with_name("cost686_primary.onnx")


def _tensor(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value), name)


def _build_conservative(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph
    initializers: list[onnx.TensorProto] = []
    for source in graph.initializer:
        if source.name == "row02_idx":
            initializers.append(
                _tensor("row0_idx", np.asarray([0], dtype=np.int64))
            )
        elif source.name == "shift_kernel":
            continue
        elif source.name == "false8":
            initializers.append(
                _tensor(
                    "ng_background8",
                    np.full((1, 1, 1, 8), 2, dtype=np.uint8),
                )
            )
        elif source.name == "ng_mask_base":
            initializers.append(
                _tensor(
                    "ng_mask_base",
                    np.asarray([2] + [1] * 9, dtype=np.uint8).reshape(
                        10, 1, 1, 1
                    ),
                )
            )
        elif source.name == "ng_two":
            initializers.append(
                _tensor(
                    "ng_zero_update",
                    np.zeros((1, 1, 1, 1), dtype=np.uint8),
                )
            )
        else:
            initializers.append(
                onnx.TensorProto.FromString(source.SerializeToString())
            )
    initializers.extend(
        [
            _tensor("row2_idx", np.asarray([2], dtype=np.int64)),
            _tensor("ng_half_scale", np.asarray(0.5, dtype=np.float32)),
        ]
    )
    del graph.initializer[:]
    graph.initializer.extend(initializers)

    removed_outputs = {
        "fg5",
        "centered5",
        "fg02_u8",
        "left_cols",
        "shift",
        "row0_4d",
        "row2_4d",
        "fg_pad4d",
        "bottom_fg",
        "ng_centered_mask",
    }
    nodes: list[onnx.NodeProto] = []
    for source in graph.node:
        outputs = set(source.output)
        if outputs.intersection(removed_outputs):
            if "centered5" in outputs:
                nodes.extend(
                    [
                        helper.make_node(
                            "QuantizeLinear",
                            ["ch0_first5", "ng_half_scale", ""],
                            ["centered5"],
                        ),
                        helper.make_node(
                            "Gather",
                            ["centered5", "row0_idx"],
                            ["row0_4d"],
                            axis=2,
                        ),
                        helper.make_node(
                            "ArgMin",
                            ["row0_4d"],
                            ["left0_i64"],
                            axis=3,
                            keepdims=0,
                        ),
                        helper.make_node(
                            "Gather",
                            ["centered5", "row2_idx"],
                            ["row2_4d"],
                            axis=2,
                        ),
                        helper.make_node(
                            "ArgMin",
                            ["row2_4d"],
                            ["left2_i64"],
                            axis=3,
                            keepdims=0,
                        ),
                        helper.make_node(
                            "Sub", ["left2_i64", "left0_i64"], ["shift"]
                        ),
                    ]
                )
            elif "fg_pad4d" in outputs:
                nodes.append(
                    helper.make_node(
                        "Concat",
                        [
                            "ng_background8",
                            "row0_4d",
                            "ng_background8",
                            "p3_rows_a",
                            "row2_4d",
                        ],
                        ["fg_pad4d"],
                        axis=3,
                    )
                )
            elif "ng_centered_mask" in outputs:
                nodes.append(
                    helper.make_node(
                        "Concat",
                        ["centered5"]
                        + [f"bottom_row_{index}" for index in range(5)],
                        ["ng_centered_mask"],
                        axis=2,
                    )
                )
            continue

        node = onnx.NodeProto.FromString(source.SerializeToString())
        for index, name in enumerate(node.input):
            if name == "ng_two":
                node.input[index] = "ng_zero_update"
        nodes.append(node)

    del graph.node[:]
    graph.node.extend(nodes)
    del graph.value_info[:]
    for index in range(5):
        graph.value_info.append(
            helper.make_tensor_value_info(
                f"bottom_row_{index}", TensorProto.UINT8, [1, 1, 1, 10]
            )
        )
    graph.value_info.append(
        helper.make_tensor_value_info(
            "fg_pad4d", TensorProto.UINT8, [1, 1, 1, 46]
        )
    )
    return model


def build(task=None, *, stage: str = "primary") -> onnx.ModelProto:
    if stage not in STAGES:
        raise ValueError(f"unknown task124 stage: {stage}")
    if stage != "conservative":
        raise NotImplementedError(f"task124 stage is not implemented yet: {stage}")

    from src.custom.task124 import build as build_source

    model = build_source(task)
    source_sha256 = hashlib.sha256(model.SerializeToString()).hexdigest()
    assert source_sha256 == BASELINE_SHA256, "task124 source baseline changed"
    model = _build_conservative(model)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    onnx.save(build(), OUTPUT)
    print(f"saved {OUTPUT}")
