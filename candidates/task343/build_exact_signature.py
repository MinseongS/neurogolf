"""Exact task343 rewrite using base-10 column signatures and one fmod index carrier."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[2]
INCUMBENT = ROOT / "submission/overfit_nets/task343.onnx"
OUTPUT = Path(__file__).with_name("exact_signature.onnx")


def _init(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(value, name=name)


def _signature_conv(output: str, pads: list[int]) -> onnx.NodeProto:
    return helper.make_node(
        "Conv",
        ["input", "signature_kernel"],
        [output],
        name=f"signature_{output}",
        pads=pads,
    )


def build(_task=None) -> onnx.ModelProto:
    model = onnx.load(INCUMBENT)
    graph = model.graph
    graph.name = "task343_exact_signature_fmod"

    nodes = [
        _signature_conv("column_left", [-2, 0, -25, -27]),
        _signature_conv("column_right", [-2, -3, -25, -24]),
        _signature_conv("column_8", [-2, -8, -25, -21]),
        _signature_conv("column_11", [-2, -11, -25, -18]),
        # Exact period-3 consistency over all three arbitrary row colours.
        helper.make_node(
            "Equal", ["column_right", "column_left"], ["p3_eq"]
        ),
        helper.make_node(
            "Cast", ["p3_eq"], ["p3_eq_u8"], to=TensorProto.UINT8
        ),
        helper.make_node(
            "ReduceMin",
            ["p3_eq_u8"],
            ["p3_valid_u8"],
            axes=[0, 1, 2, 3],
            keepdims=0,
        ),
        helper.make_node(
            "Cast", ["p3_valid_u8"], ["p3_valid"], to=TensorProto.BOOL
        ),
        # Exact period-4 consistency: col4==col0 and col8==col0.
        helper.make_node(
            "Gather", ["column_right", "idx1"], ["p4_col4"], axis=3
        ),
        helper.make_node(
            "Gather", ["column_left", "idx0"], ["p4_col0"], axis=3
        ),
        helper.make_node("Equal", ["p4_col4", "p4_col0"], ["p4_eq4"]),
        helper.make_node("Equal", ["column_8", "p4_col0"], ["p4_eq8"]),
        helper.make_node("And", ["p4_eq4", "p4_eq8"], ["p4_eq"]),
        helper.make_node(
            "Squeeze", ["p4_eq", "squeeze_all_axes"], ["p4_valid"]
        ),
        # A zero signature means the relevant visible-prefix column is background.
        helper.make_node(
            "Equal", ["column_8", "zero_f"], ["visible_le8_tensor"]
        ),
        helper.make_node(
            "Squeeze",
            ["visible_le8_tensor", "squeeze_all_axes"],
            ["visible_le8"],
        ),
        helper.make_node(
            "Equal", ["column_11", "zero_f"], ["visible_le11_tensor"]
        ),
        helper.make_node(
            "Squeeze",
            ["visible_le11_tensor", "squeeze_all_axes"],
            ["visible_le11"],
        ),
        # Preserve the incumbent's exact visibility/p3/p4 decision tree in uint8.
        helper.make_node(
            "Where",
            ["p3_valid", "period3_u8", "period4_u8"],
            ["period_low_u8"],
        ),
        helper.make_node(
            "Where",
            ["p4_valid", "period4_u8", "period6_u8"],
            ["period_mid_u8"],
        ),
        helper.make_node(
            "Where",
            ["visible_le11", "period_mid_u8", "period8_u8"],
            ["period_high_u8"],
        ),
        helper.make_node(
            "Where",
            ["visible_le8", "period_low_u8", "period_high_u8"],
            ["period_u8"],
        ),
        helper.make_node(
            "Cast", ["period_u8"], ["period"], to=TensorProto.INT32
        ),
        # C fmod keeps -1 negative, so the padded half gathers input column29.
        helper.make_node(
            "Mod",
            ["final_mod_inputs", "period"],
            ["final_cols"],
            fmod=1,
        ),
        helper.make_node(
            "Gather", ["input", "final_cols"], ["output"], axis=3
        ),
    ]
    del graph.node[:]
    graph.node.extend(nodes)

    kernel = np.zeros((1, 10, 3, 1), dtype=np.float32)
    colors = np.arange(10, dtype=np.float32)
    kernel[0, :, 0, 0] = 100.0 * colors
    kernel[0, :, 1, 0] = 10.0 * colors
    kernel[0, :, 2, 0] = colors
    initializers = [
        _init("signature_kernel", kernel),
        _init("idx1", np.array(1, dtype=np.int32)),
        _init("idx0", np.array(0, dtype=np.int32)),
        _init("squeeze_all_axes", np.array([0, 1, 2, 3], dtype=np.int64)),
        _init("zero_f", np.array(0.0, dtype=np.float32)),
        _init("period3_u8", np.array(3, dtype=np.uint8)),
        _init("period4_u8", np.array(4, dtype=np.uint8)),
        _init("period6_u8", np.array(6, dtype=np.uint8)),
        _init("period8_u8", np.array(8, dtype=np.uint8)),
        _init(
            "final_mod_inputs",
            np.array(list(range(15)) + [-1] * 15, dtype=np.int32),
        ),
    ]
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    del graph.value_info[:]

    onnx.checker.check_model(model, full_check=True)
    return onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )


if __name__ == "__main__":
    candidate = build()
    onnx.save(candidate, OUTPUT)
    print(OUTPUT)
