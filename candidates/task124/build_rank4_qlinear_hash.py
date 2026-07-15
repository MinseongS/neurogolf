"""Build task124's rank-4 row bank and exact QLinear row-hash fold."""

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
BASELINE_SHA256 = "49ececad3443d478c5f9b3e335f8ced4df82aa25648f89e85bacd985b8737632"
ADOPTED_SHA256 = "4e4bafbb3d65046a1ec08a211de6c9951705b613777a2a3ece9f4c73f6041b25"
OUTPUT = Path(__file__).with_name("rank4_qlinear_hash.onnx")


def build(task=None) -> onnx.ModelProto:
    from src.custom.task124 import build as build_source

    model = build_source(task)
    source_sha256 = hashlib.sha256(model.SerializeToString()).hexdigest()
    if ADOPTED_SHA256 and source_sha256 == ADOPTED_SHA256:
        return model
    assert source_sha256 == BASELINE_SHA256, "task124 source baseline changed"
    graph = model.graph

    rewritten_initializers: list[onnx.TensorProto] = []
    for initializer in graph.initializer:
        if initializer.name == "row_flat_shape":
            continue
        if initializer.name == "p3b_idx":
            rewritten_initializers.append(
                numpy_helper.from_array(np.asarray([4], dtype=np.int64), "p3b_idx")
            )
        elif initializer.name == "false8":
            rewritten_initializers.append(
                numpy_helper.from_array(
                    np.zeros((1, 1, 1, 8), dtype=np.uint8), "false8"
                )
            )
        elif initializer.name == "slice_axes":
            rewritten_initializers.append(
                numpy_helper.from_array(np.asarray([3], dtype=np.int32), "slice_axes")
            )
        else:
            rewritten_initializers.append(initializer)
    rewritten_initializers.append(
        numpy_helper.from_array(
            np.asarray([1, 2, 4, 8, 16, 32, 0, 0, 0, 0], dtype=np.uint8).reshape(10, 1),
            "hash_code",
        )
    )
    del graph.initializer[:]
    graph.initializer.extend(rewritten_initializers)

    removed_outputs = {
        "p3_equal",
        "p3_equal_u8",
        "p3_all_equal",
        "is_p3",
        "row0_flat",
        "row1_flat",
        "row2_flat",
        "fg_pad_flat",
        "bottom_row_0",
        "bottom_row_1",
        "bottom_row_2",
        "bottom_row_3",
        "bottom_row_4",
        "bottom_flat",
        "bottom_fg",
    }
    qlinear_inputs = [
        "ng_scale",
        "ng_y_zero_point",
        "hash_code",
        "ng_scale",
        "ng_y_zero_point",
        "ng_scale",
        "ng_y_zero_point",
    ]
    rewritten_nodes: list[onnx.NodeProto] = []
    for source_node in graph.node:
        outputs = set(source_node.output)
        if outputs.intersection(removed_outputs):
            continue

        node = onnx.NodeProto.FromString(source_node.SerializeToString())
        if "bottom_start4_0" in outputs:
            axis = next(attribute for attribute in node.attribute if attribute.name == "axis")
            axis.i = 4
        rewritten_nodes.append(node)

        if "p3_rows_b" in outputs:
            rewritten_nodes.extend([
                helper.make_node(
                    "QLinearMatMul",
                    ["p3_rows_a", *qlinear_inputs],
                    ["p3_hash_a"],
                ),
                helper.make_node(
                    "QLinearMatMul",
                    ["p3_rows_b", *qlinear_inputs],
                    ["p3_hash_b"],
                ),
                helper.make_node("Equal", ["p3_hash_a", "p3_hash_b"], ["is_p3"]),
            ])

        if "row2_4d" in outputs:
            rewritten_nodes.append(
                helper.make_node(
                    "Concat",
                    ["false8", "row0_4d", "false8", "p3_rows_a", "row2_4d"],
                    ["fg_pad4d"],
                    axis=3,
                )
            )
            bottom_rows = [f"bottom_row_{index}" for index in range(5)]
            for index, bottom_row in enumerate(bottom_rows):
                rewritten_nodes.append(
                    helper.make_node(
                        "Slice",
                        [
                            "fg_pad4d",
                            f"bottom_start_{index}",
                            f"bottom_end_{index}",
                            "slice_axes",
                            "slice_steps",
                        ],
                        [bottom_row],
                    )
                )
            rewritten_nodes.append(
                helper.make_node("Concat", bottom_rows, ["bottom_fg"], axis=2)
            )

    del graph.node[:]
    graph.node.extend(rewritten_nodes)
    del graph.value_info[:]
    for index in range(5):
        graph.value_info.append(
            helper.make_tensor_value_info(
                f"bottom_row_{index}", TensorProto.UINT8, [1, 1, 1, 10]
            )
        )
    graph.value_info.extend([
        helper.make_tensor_value_info("fg_pad4d", TensorProto.UINT8, [1, 1, 1, 46]),
        helper.make_tensor_value_info(
            "bottom_fg", TensorProto.UINT8, [1, 1, 5, 10]
        ),
    ])

    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    onnx.save(build(), OUTPUT)
    print(f"saved {OUTPUT}")
