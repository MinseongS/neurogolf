"""Decode task271's score payload with UINT16 BitwiseAnd and UINT8 Where."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


DEFAULT_INCUMBENT = Path(__file__).with_name("encoded_score_payload.onnx")
DEFAULT_OUTPUT = Path(__file__).with_name("bitwise_where_decoder.onnx")
BASELINE_SHA256 = "9f7303ea8e393536079fba14c87fec7b5e0908cf7b95917af2b3eb0b35f2e37d"
BASELINE_OPS = [
    "Conv",
    "MaxPool",
    "Cast",
    "Mod",
    "GreaterOrEqual",
    "Cast",
    "BitShift",
    "ConvInteger",
]
FOLDED_OPS = [
    "Conv",
    "MaxPool",
    "Cast",
    "BitwiseAnd",
    "Cast",
    "Where",
    "ConvInteger",
]


def _score_payload_kernel() -> np.ndarray:
    weight = np.zeros((1, 10, 3, 3), dtype=np.float32)
    weight[0, 0] = -4096.0
    for bit in range(9):
        weight[0, 1].reshape(-1)[bit] = 512.0 + (1 << bit)
    return weight


def _payload_masks() -> np.ndarray:
    return (1 << np.arange(9, dtype=np.uint16)).reshape(1, 1, 3, 3)


def _signed_render_weight() -> np.ndarray:
    weight = np.zeros((10, 1, 1, 1), dtype=np.int8)
    weight[1, 0, 0, 0] = 1
    weight[8, 0, 0, 0] = -1
    return weight


def _default_opset(model: onnx.ModelProto) -> int:
    return next(
        opset.version
        for opset in model.opset_import
        if opset.domain in ("", "ai.onnx")
    )


def _attrs(node: onnx.NodeProto) -> dict[str, object]:
    return {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }


def _validate_folded(model: onnx.ModelProto) -> None:
    graph = model.graph
    if _default_opset(model) != 18:
        raise ValueError("folded task271 graph must use opset 18")
    if [node.op_type for node in graph.node] != FOLDED_OPS:
        raise ValueError("folded task271 graph has unexpected nodes")
    if graph.name != "task271_bitwise_where_decoder":
        raise ValueError("folded task271 graph has unexpected name")

    conv, pool, cast_code, bitwise, cast_bits, where, renderer = graph.node
    if list(conv.input) != ["input", "score_payload_kernel"]:
        raise ValueError("folded task271 graph has unexpected score inputs")
    if list(conv.output) != ["scores"] or _attrs(conv) != {
        "kernel_shape": [3, 3],
        "pads": [0, 0, -21, -21],
    }:
        raise ValueError("folded task271 graph has unexpected score geometry")
    if list(pool.input) != ["scores"] or list(pool.output) != ["max_score"]:
        raise ValueError("folded task271 graph has unexpected MaxPool wiring")
    if _attrs(pool) != {"kernel_shape": [7, 7]}:
        raise ValueError("folded task271 graph has unexpected MaxPool geometry")
    expected_io = [
        (["max_score"], ["encoded_u16"]),
        (["encoded_u16", "payload_masks"], ["masked_u16"]),
        (["masked_u16"], ["payload_bits"]),
        (["payload_bits", "two_u8", "zero_u8"], ["blue2"]),
        (
            ["blue2", "render_weight_signed", "one_u8"],
            ["output"],
        ),
    ]
    for node, (inputs, outputs) in zip(
        [cast_code, bitwise, cast_bits, where, renderer], expected_io
    ):
        if list(node.input) != inputs or list(node.output) != outputs:
            raise ValueError(
                f"folded task271 graph has unexpected {node.op_type} wiring"
            )
    if _attrs(cast_code) != {"to": TensorProto.UINT16}:
        raise ValueError("folded task271 graph has unexpected score cast")
    if _attrs(bitwise):
        raise ValueError("folded task271 graph has unexpected BitwiseAnd attributes")
    if _attrs(cast_bits) != {"to": TensorProto.BOOL}:
        raise ValueError("folded task271 graph has unexpected payload cast")
    if _attrs(where):
        raise ValueError("folded task271 graph has unexpected Where attributes")
    if _attrs(renderer) != {
        "kernel_shape": [1, 1],
        "pads": [0, 0, 27, 27],
    }:
        raise ValueError("folded task271 graph has unexpected renderer geometry")

    initializers = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in graph.initializer
    }
    if set(initializers) != {
        "score_payload_kernel",
        "payload_masks",
        "two_u8",
        "zero_u8",
        "one_u8",
        "render_weight_signed",
    }:
        raise ValueError("folded task271 graph has unexpected initializers")
    expected = {
        "score_payload_kernel": _score_payload_kernel(),
        "payload_masks": _payload_masks(),
        "two_u8": np.array([2], dtype=np.uint8),
        "zero_u8": np.array([0], dtype=np.uint8),
        "one_u8": np.array([1], dtype=np.uint8),
        "render_weight_signed": _signed_render_weight(),
    }
    for name, array in expected.items():
        if not np.array_equal(initializers.get(name), array):
            raise ValueError(f"folded task271 graph has unexpected {name}")
    if graph.output[0].type.tensor_type.elem_type != TensorProto.INT32:
        raise ValueError("folded task271 graph has unexpected output dtype")

    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)


def build_candidate(incumbent: Path, output: Path) -> None:
    model = onnx.load(incumbent)
    graph = model.graph
    ops = [node.op_type for node in graph.node]
    if ops == FOLDED_OPS:
        _validate_folded(model)
        output.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, output)
        return

    if hashlib.sha256(incumbent.read_bytes()).hexdigest() != BASELINE_SHA256:
        raise ValueError("task271 input is not the pinned cost366 baseline")
    if ops != BASELINE_OPS or _default_opset(model) != 12:
        raise ValueError("unexpected task271 cost366 baseline")

    initializers = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in graph.initializer
    }
    if not np.array_equal(
        initializers.get("score_payload_kernel"), _score_payload_kernel()
    ):
        raise ValueError("unexpected task271 score payload kernel")
    if not np.array_equal(
        initializers.get("render_weight_signed"), _signed_render_weight()
    ):
        raise ValueError("unexpected task271 signed renderer")
    if not np.array_equal(
        initializers.get("one_u8"), np.array([1], dtype=np.uint8)
    ):
        raise ValueError("unexpected task271 renderer zero-point")

    del graph.initializer[:]
    graph.initializer.extend(
        [
            numpy_helper.from_array(
                _score_payload_kernel(), "score_payload_kernel"
            ),
            numpy_helper.from_array(_payload_masks(), "payload_masks"),
            numpy_helper.from_array(np.array([2], dtype=np.uint8), "two_u8"),
            numpy_helper.from_array(np.array([0], dtype=np.uint8), "zero_u8"),
            numpy_helper.from_array(np.array([1], dtype=np.uint8), "one_u8"),
            numpy_helper.from_array(
                _signed_render_weight(), "render_weight_signed"
            ),
        ]
    )

    del graph.node[:]
    graph.node.extend(
        [
            helper.make_node(
                "Conv",
                ["input", "score_payload_kernel"],
                ["scores"],
                kernel_shape=[3, 3],
                pads=[0, 0, -21, -21],
            ),
            helper.make_node(
                "MaxPool", ["scores"], ["max_score"], kernel_shape=[7, 7]
            ),
            helper.make_node(
                "Cast", ["max_score"], ["encoded_u16"], to=TensorProto.UINT16
            ),
            helper.make_node(
                "BitwiseAnd",
                ["encoded_u16", "payload_masks"],
                ["masked_u16"],
            ),
            helper.make_node(
                "Cast", ["masked_u16"], ["payload_bits"], to=TensorProto.BOOL
            ),
            helper.make_node(
                "Where",
                ["payload_bits", "two_u8", "zero_u8"],
                ["blue2"],
            ),
            helper.make_node(
                "ConvInteger",
                ["blue2", "render_weight_signed", "one_u8"],
                ["output"],
                kernel_shape=[1, 1],
                pads=[0, 0, 27, 27],
            ),
        ]
    )

    del graph.value_info[:]
    graph.value_info.extend(
        [
            helper.make_tensor_value_info(
                "scores", TensorProto.FLOAT, [1, 1, 7, 7]
            ),
            helper.make_tensor_value_info(
                "max_score", TensorProto.FLOAT, [1, 1, 1, 1]
            ),
            helper.make_tensor_value_info(
                "encoded_u16", TensorProto.UINT16, [1, 1, 1, 1]
            ),
            helper.make_tensor_value_info(
                "masked_u16", TensorProto.UINT16, [1, 1, 3, 3]
            ),
            helper.make_tensor_value_info(
                "payload_bits", TensorProto.BOOL, [1, 1, 3, 3]
            ),
            helper.make_tensor_value_info(
                "blue2", TensorProto.UINT8, [1, 1, 3, 3]
            ),
        ]
    )
    graph.name = "task271_bitwise_where_decoder"
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            opset.version = 18

    _validate_folded(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent", type=Path, default=DEFAULT_INCUMBENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_candidate(args.incumbent, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
