"""Replace task233's hash-table placement lookup with dynamic correlation."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper, numpy_helper

from src.custom.task233 import build as build_baseline


REMOVED_OUTPUTS = {
    "redf",
    "shash",
    "holf",
    "hsh",
    "hshr",
    "hshf",
    "tbl",
    "shi",
    "posg",
    "pos0",
}


def _initializer(name: str, value, dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def _dynamic_correlation_nodes() -> list[onnx.NodeProto]:
    return [
        helper.make_node("Reshape", ["red", "dc_shape_5133"], ["dc_red4b"], name="dc_red4b"),
        helper.make_node("Cast", ["dc_red4b"], ["dc_red4u"], name="dc_red4u", to=TP.UINT8),
        helper.make_node(
            "Slice",
            ["dc_red4u", "dc_rev_starts", "dc_rev_ends", "dc_rev_axes", "dc_rev_steps"],
            ["dc_red4ur"],
            name="dc_red4ur",
        ),
        helper.make_node("Mul", ["dc_red4ur", "dc_16u"], ["dc_weight"], name="dc_weight"),
        helper.make_node("Cast", ["red"], ["dc_redf"], name="dc_redf", to=TP.FLOAT16),
        helper.make_node(
            "ReduceSum", ["dc_redf", "ax1"], ["dc_count51"], name="dc_count51", keepdims=1
        ),
        helper.make_node(
            "Reshape", ["dc_count51", "dc_shape_1511"], ["dc_count"], name="dc_count"
        ),
        helper.make_node(
            "Slice",
            ["holm", "dc_rev_starts", "dc_rev_ends", "dc_rev_axes", "dc_rev_steps"],
            ["dc_holmr"],
            name="dc_holmr",
        ),
        helper.make_node(
            "QLinearConv",
            [
                "dc_holmr",
                "dc_one",
                "dc_zero",
                "dc_weight",
                "dc_one",
                "dc_1u",
                "dc_one",
                "dc_128u",
            ],
            ["dc_corr"],
            name="dc_corr",
            kernel_shape=[3, 3],
        ),
        helper.make_node(
            "MaxPool",
            ["dc_corr"],
            ["dc_max", "dc_idx"],
            name="dc_max",
            kernel_shape=[18, 18],
        ),
        helper.make_node("Cast", ["dc_max"], ["dc_maxf"], name="dc_maxf", to=TP.FLOAT16),
        helper.make_node("Mul", ["dc_count", "dc_15f"], ["dc_count15"], name="dc_count15"),
        helper.make_node("Add", ["dc_count15", "dc_128f"], ["dc_exact"], name="dc_exact"),
        helper.make_node("Equal", ["dc_maxf", "dc_exact"], ["dc_full"], name="dc_full"),
        helper.make_node("Mod", ["dc_idx", "dc_324"], ["dc_pos64"], name="dc_pos64", fmod=0),
        helper.make_node("Sub", ["dc_323", "dc_pos64"], ["dc_unrev64"], name="dc_unrev64"),
        helper.make_node("Cast", ["dc_unrev64"], ["dc_posf"], name="dc_posf", to=TP.FLOAT16),
        helper.make_node(
            "Where", ["dc_full", "dc_posf", "c4096"], ["dc_hitposf"], name="dc_hitposf"
        ),
        helper.make_node("Reshape", ["dc_hitposf", "sh51"], ["pos0"], name="pos0"),
    ]


def _dynamic_correlation_value_infos() -> list[onnx.ValueInfoProto]:
    return [
        helper.make_tensor_value_info("dc_red4b", TP.BOOL, [5, 1, 3, 3]),
        helper.make_tensor_value_info("dc_red4u", TP.UINT8, [5, 1, 3, 3]),
        helper.make_tensor_value_info("dc_red4ur", TP.UINT8, [5, 1, 3, 3]),
        helper.make_tensor_value_info("dc_weight", TP.UINT8, [5, 1, 3, 3]),
        helper.make_tensor_value_info("dc_redf", TP.FLOAT16, [5, 9]),
        helper.make_tensor_value_info("dc_count51", TP.FLOAT16, [5, 1]),
        helper.make_tensor_value_info("dc_count", TP.FLOAT16, [1, 5, 1, 1]),
        helper.make_tensor_value_info("dc_holmr", TP.UINT8, [1, 1, 20, 20]),
        helper.make_tensor_value_info("dc_corr", TP.UINT8, [1, 5, 18, 18]),
        helper.make_tensor_value_info("dc_max", TP.UINT8, [1, 5, 1, 1]),
        helper.make_tensor_value_info("dc_idx", TP.INT64, [1, 5, 1, 1]),
        helper.make_tensor_value_info("dc_maxf", TP.FLOAT16, [1, 5, 1, 1]),
        helper.make_tensor_value_info("dc_count15", TP.FLOAT16, [1, 5, 1, 1]),
        helper.make_tensor_value_info("dc_exact", TP.FLOAT16, [1, 5, 1, 1]),
        helper.make_tensor_value_info("dc_full", TP.BOOL, [1, 5, 1, 1]),
        helper.make_tensor_value_info("dc_pos64", TP.INT64, [1, 5, 1, 1]),
        helper.make_tensor_value_info("dc_unrev64", TP.INT64, [1, 5, 1, 1]),
        helper.make_tensor_value_info("dc_posf", TP.FLOAT16, [1, 5, 1, 1]),
        helper.make_tensor_value_info("dc_hitposf", TP.FLOAT16, [1, 5, 1, 1]),
        helper.make_tensor_value_info("pos0", TP.FLOAT16, [5, 1]),
    ]


def build(task=None) -> onnx.ModelProto:
    model = onnx.ModelProto.FromString(build_baseline(task).SerializeToString())
    graph = model.graph

    rewritten_nodes: list[onnx.NodeProto] = []
    inserted = False
    for node in graph.node:
        if any(output in REMOVED_OUTPUTS for output in node.output):
            continue
        rewritten_nodes.append(node)
        if "holm" in node.output:
            rewritten_nodes.extend(_dynamic_correlation_nodes())
            inserted = True
    if not inserted:
        raise ValueError("task233 baseline no longer exposes holm")

    del graph.node[:]
    graph.node.extend(rewritten_nodes)

    graph.initializer.extend(
        [
            _initializer("dc_shape_5133", [5, 1, 3, 3], np.int64),
            _initializer("dc_shape_1511", [1, 5, 1, 1], np.int64),
            _initializer("dc_one", 1.0, np.float32),
            _initializer("dc_zero", 0, np.uint8),
            _initializer("dc_1u", 1, np.uint8),
            _initializer("dc_16u", 16, np.uint8),
            _initializer("dc_128u", 128, np.uint8),
            _initializer("dc_15f", 15, np.float16),
            _initializer("dc_128f", 128, np.float16),
            _initializer("dc_324", 324, np.int64),
            _initializer("dc_323", 323, np.int64),
            _initializer("dc_rev_starts", [-1, -1], np.int64),
            _initializer("dc_rev_ends", [np.iinfo(np.int64).min, np.iinfo(np.int64).min], np.int64),
            _initializer("dc_rev_axes", [2, 3], np.int64),
            _initializer("dc_rev_steps", [-1, -1], np.int64),
        ]
    )

    kept_value_infos = [vi for vi in graph.value_info if vi.name not in REMOVED_OUTPUTS]
    del graph.value_info[:]
    graph.value_info.extend(kept_value_infos)
    graph.value_info.extend(_dynamic_correlation_value_infos())

    used_inputs = {name for node in graph.node for name in node.input if name}
    kept_initializers = [init for init in graph.initializer if init.name in used_inputs]
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)

    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def write_candidate(path: Path | str = Path("candidates/task233/cand_dynamic_corr.onnx")) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build(), destination)
    return destination


if __name__ == "__main__":
    print(write_candidate())
