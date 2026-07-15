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
STAGES = {"conservative", "routing_i32", "routing_mod", "primary", "primary_i32"}
OUTPUTS = {
    "conservative": Path(__file__).with_name("cost728_conservative.onnx"),
    "routing_i32": Path(__file__).with_name("cost712_routing_i32.onnx"),
    "routing_mod": Path(__file__).with_name("cost707_routing_mod.onnx"),
    "primary": Path(__file__).with_name("cost692_primary.onnx"),
    "primary_i32": Path(__file__).with_name("cost697_primary_i32.onnx"),
}


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


def _build_scalar_route(
    model: onnx.ModelProto, *, use_uint8: bool
) -> onnx.ModelProto:
    graph = model.graph
    scalar_dtype = np.uint8 if use_uint8 else np.int32
    scalar_tensor_type = TensorProto.UINT8 if use_uint8 else TensorProto.INT32
    three_name = "three_u8" if use_uint8 else "three_i32"

    initializers: list[onnx.TensorProto] = []
    for source in graph.initializer:
        if source.name == "three_i64":
            initializers.append(
                _tensor(three_name, np.asarray(3, dtype=scalar_dtype))
            )
        elif source.name == "source_offsets":
            initializers.append(
                _tensor(
                    "source_offsets",
                    numpy_helper.to_array(source).reshape(4, 5),
                )
            )
        elif source.name == "slice_start_shape":
            continue
        else:
            initializers.append(
                onnx.TensorProto.FromString(source.SerializeToString())
            )
    del graph.initializer[:]
    graph.initializer.extend(initializers)

    nodes: list[onnx.NodeProto] = []
    for source in graph.node:
        outputs = set(source.output)
        if "shift" in outputs:
            nodes.extend(
                [
                    helper.make_node(
                        "Cast",
                        ["left0_i64"],
                        ["left0_scalar"],
                        to=scalar_tensor_type,
                    ),
                    helper.make_node(
                        "Cast",
                        ["left2_i64"],
                        ["left2_scalar"],
                        to=scalar_tensor_type,
                    ),
                    helper.make_node(
                        "Sub",
                        ["left2_scalar", "left0_scalar"],
                        ["shift_rank3"],
                    ),
                    helper.make_node(
                        "Squeeze",
                        ["shift_rank3"],
                        ["shift_scalar"],
                        axes=[0, 1, 2],
                    ),
                ]
            )
            continue
        if "is_p3" in outputs:
            nodes.append(
                onnx.NodeProto.FromString(source.SerializeToString())
            )
            nodes.append(
                helper.make_node(
                    "Squeeze",
                    ["is_p3"],
                    ["is_p3_scalar"],
                    axes=[0, 1, 2, 3],
                )
            )
            continue
        if "candidate" in outputs:
            route_output = "candidate_u8" if use_uint8 else "candidate_i32"
            nodes.append(
                helper.make_node(
                    "Where",
                    ["is_p3_scalar", three_name, "shift_scalar"],
                    [route_output],
                )
            )
            if use_uint8:
                nodes.append(
                    helper.make_node(
                        "Cast",
                        ["candidate_u8"],
                        ["candidate_i32"],
                        to=TensorProto.INT32,
                    )
                )
            continue
        if "source_offset" in outputs:
            nodes.append(
                helper.make_node(
                    "Gather",
                    ["source_offsets", "candidate_i32"],
                    ["source_offset"],
                    axis=0,
                )
            )
            continue
        if any(name.startswith("bottom_start4_") for name in outputs):
            nodes.append(
                helper.make_node(
                    "Split",
                    ["source_offset"],
                    [f"bottom_start_vec_{index}" for index in range(5)],
                    axis=0,
                    split=[1] * 5,
                )
            )
            continue
        if any(name.startswith("bottom_start_") for name in outputs):
            continue

        node = onnx.NodeProto.FromString(source.SerializeToString())
        for index, name in enumerate(node.input):
            if name.startswith("bottom_start_"):
                suffix = name.removeprefix("bottom_start_")
                node.input[index] = f"bottom_start_vec_{suffix}"
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


def _build_mod_route(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph
    initializers: list[onnx.TensorProto] = []
    for source in graph.initializer:
        if source.name == "three_i64":
            continue
        if source.name == "source_offsets":
            offsets = numpy_helper.to_array(source).reshape(4, 5)
            initializers.append(
                _tensor("source_offsets", offsets[[2, 3, 0, 1]])
            )
        elif source.name == "slice_start_shape":
            continue
        else:
            initializers.append(
                onnx.TensorProto.FromString(source.SerializeToString())
            )
    initializers.extend(
        [
            _tensor(
                "ng_diff_weight",
                np.asarray([0, 2], dtype=np.uint8).reshape(2, 1),
            ),
            _tensor(
                "ng_diff_y_zero_point", np.asarray(2, dtype=np.uint8)
            ),
            _tensor("ng_four_u8", np.asarray(4, dtype=np.uint8)),
        ]
    )
    del graph.initializer[:]
    graph.initializer.extend(initializers)

    nodes: list[onnx.NodeProto] = []
    for source in graph.node:
        outputs = set(source.output)
        if "shift" in outputs:
            nodes.extend(
                [
                    helper.make_node(
                        "Cast",
                        ["left0_i64"],
                        ["left0_u8"],
                        to=TensorProto.UINT8,
                    ),
                    helper.make_node(
                        "Cast",
                        ["left2_i64"],
                        ["left2_u8"],
                        to=TensorProto.UINT8,
                    ),
                    helper.make_node(
                        "Concat",
                        ["left0_u8", "left2_u8"],
                        ["left02_u8"],
                        axis=2,
                    ),
                    helper.make_node(
                        "QLinearMatMul",
                        [
                            "left02_u8",
                            "ng_scale",
                            "ng_y_zero_point",
                            "ng_diff_weight",
                            "ng_scale",
                            "ng_x_zero_point",
                            "ng_scale",
                            "ng_diff_y_zero_point",
                        ],
                        ["shift_rank3_u8"],
                    ),
                    helper.make_node(
                        "Squeeze",
                        ["shift_rank3_u8"],
                        ["shift_route_u8"],
                        axes=[0, 1, 2],
                    ),
                ]
            )
            continue
        if "is_p3" in outputs:
            nodes.append(
                onnx.NodeProto.FromString(source.SerializeToString())
            )
            nodes.append(
                helper.make_node(
                    "Squeeze",
                    ["is_p3"],
                    ["is_p3_scalar"],
                    axes=[0, 1, 2, 3],
                )
            )
            continue
        if "candidate" in outputs:
            nodes.extend(
                [
                    helper.make_node(
                        "Where",
                        [
                            "is_p3_scalar",
                            "ng_x_zero_point",
                            "shift_route_u8",
                        ],
                        ["candidate_u8"],
                    ),
                    helper.make_node(
                        "Mod",
                        ["candidate_u8", "ng_four_u8"],
                        ["candidate_mod_u8"],
                    ),
                    helper.make_node(
                        "Cast",
                        ["candidate_mod_u8"],
                        ["candidate_i32"],
                        to=TensorProto.INT32,
                    ),
                ]
            )
            continue
        if "source_offset" in outputs:
            nodes.append(
                helper.make_node(
                    "Gather",
                    ["source_offsets", "candidate_i32"],
                    ["source_offset"],
                    axis=0,
                )
            )
            continue
        if any(name.startswith("bottom_start4_") for name in outputs):
            nodes.append(
                helper.make_node(
                    "Split",
                    ["source_offset"],
                    [f"bottom_start_vec_{index}" for index in range(5)],
                    axis=0,
                    split=[1] * 5,
                )
            )
            continue
        if any(name.startswith("bottom_start_") for name in outputs):
            continue

        node = onnx.NodeProto.FromString(source.SerializeToString())
        for index, name in enumerate(node.input):
            if name.startswith("bottom_start_"):
                suffix = name.removeprefix("bottom_start_")
                node.input[index] = f"bottom_start_vec_{suffix}"
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


def _build_qlinear_hash(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph
    initializers = [
        onnx.TensorProto.FromString(source.SerializeToString())
        for source in graph.initializer
        if source.name not in {"p3b_idx", "hash_code"}
    ]
    initializers.append(
        _tensor(
            "hash6",
            np.asarray([1, 2, 4, 8, 16, 32], dtype=np.uint8).reshape(
                1, 1, 1, 6
            ),
        )
    )
    del graph.initializer[:]
    graph.initializer.extend(initializers)

    nodes: list[onnx.NodeProto] = []
    for source in graph.node:
        outputs = set(source.output)
        if "p3_rows_b" in outputs:
            continue
        if "p3_hash_a" in outputs:
            nodes.extend(
                [
                    helper.make_node(
                        "QLinearConv",
                        [
                            "p3_rows_a",
                            "ng_scale",
                            "ng_y_zero_point",
                            "hash6",
                            "ng_scale",
                            "ng_y_zero_point",
                            "ng_scale",
                            "ng_y_zero_point",
                        ],
                        ["p3_hash_a"],
                        pads=[0, 0, 0, -4],
                    ),
                    helper.make_node(
                        "QLinearConv",
                        [
                            "centered5",
                            "ng_scale",
                            "ng_y_zero_point",
                            "hash6",
                            "ng_scale",
                            "ng_y_zero_point",
                            "ng_scale",
                            "ng_y_zero_point",
                        ],
                        ["p3_hash_b"],
                        pads=[-4, 0, 0, -4],
                    ),
                    helper.make_node(
                        "Equal",
                        ["p3_hash_a", "p3_hash_b"],
                        ["is_p3"],
                    ),
                ]
            )
            continue
        if outputs.intersection({"p3_hash_b", "is_p3"}):
            continue
        nodes.append(onnx.NodeProto.FromString(source.SerializeToString()))

    del graph.node[:]
    graph.node.extend(nodes)
    del graph.value_info[:]
    for index in range(5):
        graph.value_info.append(
            helper.make_tensor_value_info(
                f"bottom_row_{index}", TensorProto.UINT8, [1, 1, 1, 10]
            )
        )
    graph.value_info.extend(
        [
            helper.make_tensor_value_info(
                "fg_pad4d", TensorProto.UINT8, [1, 1, 1, 46]
            ),
            helper.make_tensor_value_info(
                "p3_hash_a", TensorProto.UINT8, [1, 1, 1, 1]
            ),
            helper.make_tensor_value_info(
                "p3_hash_b", TensorProto.UINT8, [1, 1, 1, 1]
            ),
        ]
    )
    return model


def build(task=None, *, stage: str = "primary") -> onnx.ModelProto:
    if stage not in STAGES:
        raise ValueError(f"unknown task124 stage: {stage}")

    from src.custom.task124 import build as build_source

    model = build_source(task)
    source_sha256 = hashlib.sha256(model.SerializeToString()).hexdigest()
    assert source_sha256 == BASELINE_SHA256, "task124 source baseline changed"
    model = _build_conservative(model)
    if stage in {"routing_i32", "primary_i32"}:
        model = _build_scalar_route(
            model, use_uint8=False
        )
    elif stage in {"routing_mod", "primary"}:
        model = _build_mod_route(model)
    if stage in {"primary", "primary_i32"}:
        model = _build_qlinear_hash(model)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    for stage, output in OUTPUTS.items():
        onnx.save(build(stage=stage), output)
        print(f"saved {output}")
