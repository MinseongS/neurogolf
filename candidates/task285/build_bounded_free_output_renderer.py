#!/usr/bin/env python3
"""Build task285's bounded sparse-pivot FREE-output renderer.

The exact deployed pivot/orientation prefix is retained.  A small Fourier
contraction reconstructs a 3x5x5 source-shape mask, then a second contraction
applies the four affine reflections directly to the free graph output.  The
stamp branch uses the exact label delta ``h * (2*z - h)`` on background cells,
so it replaces class zero with class ``h`` without a dense label-grid carrier.
"""

from __future__ import annotations

import hashlib
import math
import pathlib
import re

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from neurogolf.scans.public_autopsy import DTYPE_BYTES, profile
from neurogolf.scoring import calculate_params


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = pathlib.Path(__file__).resolve().parent / "sparse_renderer_candidate.onnx"
OUTPUT = pathlib.Path(__file__).resolve().parent / "bounded_free_output_renderer.onnx"
SOURCE_SHA256 = "b201dd621c751a7a9a9352cf7fe6fbc6ac9d35c5c701857ee06953c037b148d4"
PERIOD = 31
FREQUENCIES = (PERIOD + 1) // 2
MAX_COST = 14_511
MAX_COUNTED_BYTES = 3_600
MAX_CONTRACTION_ELEMENTS = 1_000_000


def clone_node(node: onnx.NodeProto) -> onnx.NodeProto:
    copied = onnx.NodeProto()
    copied.CopyFrom(node)
    return copied


def trig_embedding(size: int) -> np.ndarray:
    """Return [position, frequency, cos/sin] real Fourier features."""
    position = np.arange(size, dtype=np.float64)[:, None]
    frequency = np.arange(FREQUENCIES, dtype=np.float64)[None, :]
    angle = 2.0 * math.pi * position * frequency / PERIOD
    return np.stack([np.cos(angle), np.sin(angle)], axis=-1).astype(np.float32)


def relation_core() -> np.ndarray:
    """Core for coord=pivot-offset (side 0) or pivot+1+offset (side 1)."""
    core = np.zeros((2, FREQUENCIES, 2, 2, 2), dtype=np.float32)
    core[0, 0, 0, 0, 0] = 1.0 / PERIOD
    core[1, 0, 0, 0, 0] = 1.0 / PERIOD
    for frequency in range(1, FREQUENCIES):
        scale = 2.0 / PERIOD
        core[0, frequency, 0, 0, 0] = scale
        core[0, frequency, 1, 1, 0] = scale
        core[0, frequency, 1, 0, 1] = -scale
        core[0, frequency, 0, 1, 1] = scale

        theta = 2.0 * math.pi * frequency / PERIOD
        cosine, sine = math.cos(theta), math.sin(theta)
        core[1, frequency, 0, 0, 0] = scale * cosine
        core[1, frequency, 1, 1, 0] = scale * cosine
        core[1, frequency, 1, 0, 1] = scale * cosine
        core[1, frequency, 0, 1, 1] = -scale * cosine
        core[1, frequency, 1, 0, 0] = scale * sine
        core[1, frequency, 0, 1, 0] = -scale * sine
        core[1, frequency, 0, 0, 1] = -scale * sine
        core[1, frequency, 1, 1, 1] = -scale * sine
    return core


def output_relation_core() -> np.ndarray:
    """Branch 0 is spatially neutral; branch 1 is the affine delta."""
    core = np.zeros((2, *relation_core().shape), dtype=np.float32)
    core[0, 0, 0, 0, 0, 0] = 1.0
    core[1] = relation_core()
    return core


def materialize_relation_delta() -> np.ndarray:
    e30 = trig_embedding(30)
    e5 = trig_embedding(5)
    return np.einsum("xmp,bmq,imr,smpqr->sbix", e30, e30, e5, relation_core())


def initializer(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value), name=name)


def _attribute_text(node: onnx.NodeProto, name: str) -> str:
    attribute = next(attribute for attribute in node.attribute if attribute.name == name)
    return attribute.s.decode("utf-8")


def _shape_map(model: onnx.ModelProto) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        init.name: tuple(int(dim) for dim in init.dims)
        for init in model.graph.initializer
    }
    for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        tensor_type = value.type.tensor_type
        shapes[value.name] = tuple(int(dim.dim_value) for dim in tensor_type.shape.dim)
    return shapes


def estimate_largest_intermediate(model: onnx.ModelProto) -> int:
    """Return NumPy greedy-path's largest intermediate across every Einsum."""
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    shapes = _shape_map(inferred)
    largest = 0
    for node in inferred.graph.node:
        if node.op_type != "Einsum":
            continue
        arrays = [np.empty(shapes[name], dtype=np.float32) for name in node.input]
        _, report = np.einsum_path(
            _attribute_text(node, "equation"), *arrays, optimize="greedy"
        )
        match = re.search(r"Largest intermediate:\s*([0-9.eE+]+)\s*elements", report)
        if match is None:
            raise AssertionError(f"could not parse einsum path report:\n{report}")
        largest = max(largest, int(math.ceil(float(match.group(1)))))
    return largest


def _value_info_map(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    return {
        value.name: value
        for value in list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    }


def counted_tensor_inventory(model: onnx.ModelProto) -> list[tuple[str, int]]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    infos = _value_info_map(inferred)
    free = {value.name for value in inferred.graph.input} | {
        value.name for value in inferred.graph.output
    }
    rows: list[tuple[str, int]] = []
    for node in inferred.graph.node:
        for output in node.output:
            if output in free:
                continue
            tensor_type = infos[output].type.tensor_type
            shape = [int(dim.dim_value) for dim in tensor_type.shape.dim]
            elements = int(np.prod(shape)) if shape else 1
            rows.append((output, elements * DTYPE_BYTES[int(tensor_type.elem_type)]))
    return rows


def _assert_float_topk_inputs(model: onnx.ModelProto) -> None:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    infos = _value_info_map(inferred)
    for node in inferred.graph.node:
        if node.op_type == "TopK":
            dtype = infos[node.input[0]].type.tensor_type.elem_type
            if dtype not in {TensorProto.FLOAT16, TensorProto.FLOAT}:
                raise AssertionError(
                    f"forbidden TopK input {node.input[0]} dtype={dtype}"
                )


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
    terminal_prefix = next(
        index for index, node in enumerate(graph.node) if "f" in node.output
    )
    nodes = [clone_node(node) for node in graph.node[: terminal_prefix + 1]]

    e30 = trig_embedding(30)
    e5 = trig_embedding(5)
    k_source = relation_core()
    k_output = output_relation_core()

    input_mixer = np.zeros((2, 10, 10), dtype=np.float32)
    input_mixer[0] = 2.0 * np.eye(10, dtype=np.float32) - 1.0
    input_mixer[1, :, 0] = 1.0

    colour_core = np.zeros((2, 10, 2, 2), dtype=np.float32)
    colour_core[0, :, 0, 0] = 1.0
    colour_core[1, :, 1, 0] = 2.0 * np.arange(10, dtype=np.float32)
    colour_core[1, :, 1, 1] = -1.0

    local_gate = np.ones((2, 5, 5), dtype=np.float32)
    local_gate[0] = 0.0
    local_gate[0, 0, 0] = 1.0
    local_gate[1, 0, 0] = 0.0

    pivot_gate = np.ones((2, 3), dtype=np.float32)
    pivot_gate[0, 1:] = 0.0
    quadrant_row = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    quadrant_col = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )

    nodes.extend(
        [
            helper.make_node("Less", ["e", "f0"], ["e_negative"]),
            helper.make_node("Less", ["f", "f0"], ["f_negative"]),
            helper.make_node("Cast", ["e_negative"], ["e_bit_2d"], to=TensorProto.INT32),
            helper.make_node("Cast", ["f_negative"], ["f_bit_2d"], to=TensorProto.INT32),
            helper.make_node("Reshape", ["e_bit_2d", "shape3"], ["e_bit"]),
            helper.make_node("Reshape", ["f_bit_2d", "shape3"], ["f_bit"]),
            helper.make_node("Reshape", ["a", "shape3"], ["a_flat"]),
            helper.make_node("Mul", ["f_bit", "i30"], ["pivot_row_offset"]),
            helper.make_node("Sub", ["a_flat", "e_bit"], ["pivot_without_col"]),
            helper.make_node(
                "Sub", ["pivot_without_col", "pivot_row_offset"], ["pivot_flat"]
            ),
            helper.make_node("Div", ["pivot_flat", "i30"], ["pivot_row"]),
            helper.make_node("Mod", ["pivot_flat", "i30"], ["pivot_col"]),
            helper.make_node("Gather", ["e30", "pivot_row"], ["pivot_row_feature"]),
            helper.make_node("Gather", ["e30", "pivot_col"], ["pivot_col_feature"]),
            helper.make_node(
                "OneHot", ["f_bit", "depth2", "onehot_values"], ["orientation_row"]
            ),
            helper.make_node(
                "OneHot", ["e_bit", "depth2", "onehot_values"], ["orientation_col"]
            ),
            helper.make_node("Reshape", ["av3", "shape3"], ["valid_bool"]),
            helper.make_node("Cast", ["valid_bool"], ["valid"], to=TensorProto.FLOAT),
            helper.make_node(
                "OneHot", ["acol", "depth10", "onehot_values"], ["root_colour"]
            ),
            helper.make_node("Unsqueeze", ["pivot_flat", "axs1"], ["pivot_column"]),
            helper.make_node(
                "Add", ["pivot_column", "anchor_offsets"], ["anchor_indices"]
            ),
            helper.make_node("Gather", ["g", "anchor_indices"], ["destination_labels"]),
            helper.make_node(
                "Cast", ["destination_labels"], ["destination_labels_f"], to=TensorProto.FLOAT
            ),
            helper.make_node(
                "Unsqueeze", ["destination_labels_f", "axs2"], ["destination_label_column"]
            ),
            helper.make_node(
                "Concat",
                ["destination_ones", "destination_label_column"],
                ["destination_colour_feature"],
                axis=2,
            ),
            helper.make_node(
                "Einsum",
                [
                    "root_colour",
                    "input",
                    "orientation_row",
                    "orientation_col",
                    "pivot_row_feature",
                    "e5",
                    "e30",
                    "k_source",
                    "pivot_col_feature",
                    "e5",
                    "e30",
                    "k_source",
                ],
                ["source_shape"],
                equation=(
                    "pds,nsEF,pu,pv,pma,imb,Emc,umabc,"
                    "pMd,jMe,FMf,vMdef->npij"
                ),
                name="bounded_source_shape",
            ),
            helper.make_node(
                "Einsum",
                [
                    "source_shape",
                    "input",
                    "input_mixer",
                    "valid",
                    "pivot_gate",
                    "local_gate",
                    "destination_colour_feature",
                    "destination_colour_feature",
                    "colour_core",
                    "quadrant_row",
                    "quadrant_col",
                    "pivot_row_feature",
                    "e5",
                    "e30",
                    "k_output",
                    "pivot_col_feature",
                    "e5",
                    "e30",
                    "k_output",
                ],
                ["output"],
                equation=(
                    "npij,nkyx,tzk,p,tp,tij,pqb,pqr,tzbr,qU,qV,"
                    "pma,imb,ymc,tUmabc,pMd,jMe,xMf,tVMdef->nzyx"
                ),
                name="bounded_free_output",
            ),
        ]
    )

    new_initializers = {
        "e30": e30,
        "e5": e5,
        "k_source": k_source,
        "k_output": k_output,
        "input_mixer": input_mixer,
        "colour_core": colour_core,
        "local_gate": local_gate,
        "pivot_gate": pivot_gate,
        "quadrant_row": quadrant_row,
        "quadrant_col": quadrant_col,
        "i30": np.array(30, dtype=np.int32),
        "shape3": np.array([3], dtype=np.int64),
        "depth2": np.array(2, dtype=np.int64),
        "depth10": np.array(10, dtype=np.int64),
        "onehot_values": np.array([0.0, 1.0], dtype=np.float32),
        "anchor_offsets": np.array([0, 1, 30, 31], dtype=np.int32),
        "destination_ones": np.ones((3, 4, 1), dtype=np.float32),
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
        initializer(name, value)
        for name, value in new_initializers.items()
        if name in used
    )
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)
    del graph.value_info[:]
    del graph.output[:]
    graph.output.append(
        helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    )
    graph.name = "task285_bounded_sparse_pivot_free_output"
    model.producer_name = "task285-bounded-free-output"

    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    _assert_float_topk_inputs(inferred)
    arities = [len(node.input) for node in inferred.graph.node if node.op_type == "Einsum"]
    if arities[-2:] != [12, 19]:
        raise AssertionError(f"unexpected Einsum arities: {arities}")
    inventory = counted_tensor_inventory(inferred)
    largest_counted = max(size for _, size in inventory)
    if largest_counted > MAX_COUNTED_BYTES:
        raise AssertionError(
            f"counted tensor exceeds {MAX_COUNTED_BYTES}B: {largest_counted}B"
        )
    largest_contraction = estimate_largest_intermediate(inferred)
    if largest_contraction > MAX_CONTRACTION_ELEMENTS:
        raise AssertionError(
            f"contraction estimate {largest_contraction} exceeds "
            f"{MAX_CONTRACTION_ELEMENTS} elements"
        )
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
    for name, size in counted_tensor_inventory(model):
        print(f"tensor {name} {size}B")
    arities = [len(node.input) for node in model.graph.node if node.op_type == "Einsum"]
    print(f"memory={memory} params={params} cost={cost}")
    print(f"einsum_arities={arities[-2:]}")
    print(f"largest_contraction_elements={estimate_largest_intermediate(model)}")
    if cost > MAX_COST:
        raise AssertionError(f"candidate cost {cost} exceeds target {MAX_COST}")


if __name__ == "__main__":
    main()
