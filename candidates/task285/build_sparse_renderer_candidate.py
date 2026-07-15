#!/usr/bin/env python3
"""Build task285 with a quadrant-aware direct 5x5 sparse renderer.

The incumbent pivot extractor already returns the visible pivot root ``a`` and
the two quadrant signs ``e``/``f``.  Its renderer nevertheless gathers a 5x9
strip because it keeps only one orientation axis in ``var``.  This lowering
uses both signs, gathers the exact 5x5 quadrant, and emits the three reflected
destinations with one bounded three-operand Einsum over 3x9 slots.
"""

from __future__ import annotations

import hashlib
import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from neurogolf.scans.public_autopsy import profile
from neurogolf.scoring import calculate_params


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "submission/overfit_nets/task285.onnx"
OUTPUT = pathlib.Path(__file__).resolve().parent / "sparse_renderer_candidate.onnx"
PIVOT_OUTPUT = pathlib.Path(__file__).resolve().parent / "sparse_pivot_slots.onnx"
SOURCE_SHA256 = "33f21e87ce2a622b56995d27d50915ea4f6e7e12888d0a1877a211f1cb6c7b87"
FINAL_SHA256 = "b201dd621c751a7a9a9352cf7fe6fbc6ac9d35c5c701857ee06953c037b148d4"


def clone_node(node: onnx.NodeProto) -> onnx.NodeProto:
    copied = onnx.NodeProto()
    copied.CopyFrom(node)
    return copied


def build_model() -> onnx.ModelProto:
    source_bytes = SOURCE.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    # A later authorized adoption may advance the deployed graph beyond this
    # historical endpoint.  Rebuild this stage from its own immutable artifact
    # instead of rejecting the newer deployment as a corrupt seed.
    if source_sha256 not in {SOURCE_SHA256, FINAL_SHA256} and OUTPUT.exists():
        historical_bytes = OUTPUT.read_bytes()
        historical_sha256 = hashlib.sha256(historical_bytes).hexdigest()
        if historical_sha256 == FINAL_SHA256:
            source_bytes = historical_bytes
            source_sha256 = historical_sha256
    if source_sha256 == FINAL_SHA256:
        final_model = onnx.load_model_from_string(source_bytes)
        onnx.checker.check_model(final_model, full_check=True)
        onnx.shape_inference.infer_shapes(final_model, strict_mode=True)
        return final_model
    if source_sha256 != SOURCE_SHA256:
        raise ValueError(
            f"unrecognized task285 SOURCE sha256={source_sha256}; "
            f"expected seed {SOURCE_SHA256} or final {FINAL_SHA256}"
        )

    model = onnx.load_model_from_string(source_bytes)
    graph = model.graph
    by_output = {output: node for node in graph.node for output in node.output}

    # Keep the incumbent through pivot root/colour extraction, then decode the
    # two quadrant signs with sparse arithmetic instead of an 8x4 fp16 MatMul.
    terminal_prefix = next(
        index for index, node in enumerate(graph.node) if "acol" in node.output
    )
    nodes = [clone_node(node) for node in graph.node[: terminal_prefix + 1]]
    nodes.append(clone_node(by_output["seedm"]))
    split_outputs = [f"direction_{index}" for index in range(8)]
    nodes.extend(
        [
            helper.make_node("Split", ["seedm"], split_outputs, axis=0, num_outputs=8),
            helper.make_node("Sub", ["direction_1", "direction_0"], ["e_primary_i"]),
            helper.make_node("Sub", ["direction_3", "direction_2"], ["f_primary_i"]),
            helper.make_node("Add", ["direction_5", "direction_7"], ["dx_positive"]),
            helper.make_node("Add", ["direction_4", "direction_6"], ["dx_negative"]),
            helper.make_node("Sub", ["dx_positive", "dx_negative"], ["dx_i"]),
            helper.make_node("Add", ["direction_6", "direction_7"], ["dy_positive"]),
            helper.make_node("Add", ["direction_4", "direction_5"], ["dy_negative"]),
            helper.make_node("Sub", ["dy_positive", "dy_negative"], ["dy_i"]),
            helper.make_node("Cast", ["e_primary_i"], ["e_primary"], to=TensorProto.FLOAT16),
            helper.make_node("Cast", ["f_primary_i"], ["f_primary"], to=TensorProto.FLOAT16),
            helper.make_node("Cast", ["dx_i"], ["dx"], to=TensorProto.FLOAT16),
            helper.make_node("Cast", ["dy_i"], ["dy"], to=TensorProto.FLOAT16),
            helper.make_node("Equal", ["e_primary", "f0"], ["ze0"]),
            helper.make_node("Where", ["ze0", "dx", "e_primary"], ["e_row"]),
            helper.make_node("Equal", ["f_primary", "f0"], ["zf0"]),
            helper.make_node("Where", ["zf0", "dy", "f_primary"], ["f_row"]),
            helper.make_node("Reshape", ["e_row", "shape_3x1"], ["e"]),
            helper.make_node("Reshape", ["f_row", "shape_3x1"], ["f"]),
        ]
    )

    ij = np.indices((5, 5), dtype=np.float16).reshape(2, 25)
    local_coeff = np.stack((-ij[1], -30.0 * ij[0]), axis=0).astype(np.int32)
    seed = np.zeros((1, 1, 5, 5), dtype=np.uint8)
    seed[0, 0, 0, 0] = 1

    coord_features = np.zeros((25, 2, 2), dtype=np.float16)
    coord_features[:, 0, 0] = 1.0
    coord_features[:, 0, 1] = ij[1]
    coord_features[:, 1, 0] = 1.0
    coord_features[:, 1, 1] = ij[0]
    coord_core = np.zeros((2, 2, 3), dtype=np.float16)
    coord_core[0, 0] = [1.0, 0.0, 1.0]
    coord_core[0, 1] = [1.0, -1.0, 1.0]
    coord_core[1, 0] = [0.0, 30.0, 30.0]
    coord_core[1, 1] = [-30.0, 30.0, 30.0]

    new_initializers = {
        "local_coeff": local_coeff,
        "shape_seed": seed,
        "shape_3x1x5x5": np.array([3, 1, 5, 5], dtype=np.int64),
        "shape_3x1": np.array([3, 1], dtype=np.int64),
        "shape_3x25": np.array([3, 25], dtype=np.int64),
        "coord_features": coord_features,
        "coord_core": coord_core,
    }

    nodes.extend(
        [
            helper.make_node("Concat", ["e", "f"], ["orient"], axis=1),
            helper.make_node("Cast", ["orient"], ["orient_i"], to=TensorProto.INT32),
            helper.make_node("MatMul", ["orient_i", "local_coeff"], ["woff"]),
            helper.make_node("Add", ["woff", "a"], ["widx"]),
            helper.make_node("Gather", ["g", "widx"], ["wcol"]),
            helper.make_node("Equal", ["wcol", "acol"], ["memb"]),
            helper.make_node("Cast", ["memb"], ["membu"], to=TensorProto.UINT8),
            helper.make_node("Reshape", ["membu", "shape_3x1x5x5"], ["mem2d"]),
            helper.make_node("Mul", ["mem2d", "shape_seed"], ["m0"]),
            helper.make_node(
                "MaxPool",
                ["m0"],
                ["p1"],
                kernel_shape=[5, 5],
                pads=[2, 2, 2, 2],
            ),
            helper.make_node("Mul", ["p1", "mem2d"], ["m1"]),
            helper.make_node(
                "MaxPool",
                ["m1"],
                ["p2"],
                kernel_shape=[5, 5],
                pads=[2, 2, 2, 2],
            ),
            helper.make_node("Mul", ["p2", "mem2d"], ["m2"]),
            helper.make_node("Reshape", ["m2", "shape_3x25"], ["mflat"]),
            helper.make_node("Cast", ["mflat"], ["mf16"], to=TensorProto.FLOAT16),
            clone_node(by_output["mv"]),
            clone_node(by_output["memv"]),
            clone_node(by_output["memv3"]),
            helper.make_node("Gather", ["coord_features", "mi"], ["slot_coord"]),
            helper.make_node(
                "Einsum",
                ["orient", "slot_coord", "coord_core"],
                ["dest_offset"],
                equation="pa,pqab,abd->pqd",
            ),
            helper.make_node("Cast", ["a"], ["af"], to=TensorProto.FLOAT16),
            helper.make_node("Unsqueeze", ["af", "axs2"], ["a3"]),
            helper.make_node("Sum", ["dest_offset", "a3"], ["t3"]),
            helper.make_node("Reshape", ["t3", "sh81"], ["t3r"]),
            helper.make_node("Cast", ["t3r"], ["tidx"], to=TensorProto.INT32),
            clone_node(by_output["tf2"]),
        ]
    )
    for output in (
        "ei",
        "f30i",
        "he",
        "hv",
        "hd",
        "ch0",
        "cv0",
        "cd0",
        "chu",
        "cvu",
        "cdu",
        "vals",
        "cond",
        "cond_i8",
        "cond10",
        "cond_cap",
        "values",
        "v81",
        "target_base",
        "target10",
        "target_cap",
        "v81_safe",
        "newg",
        "out2d",
        "output",
    ):
        nodes.append(clone_node(by_output[output]))

    del graph.node[:]
    graph.node.extend(nodes)

    used = {name for node in graph.node for name in node.input}
    kept = []
    for initializer in graph.initializer:
        if initializer.name in used:
            copied = onnx.TensorProto()
            copied.CopyFrom(initializer)
            kept.append(copied)
    kept.extend(
        numpy_helper.from_array(value, name=name)
        for name, value in new_initializers.items()
    )
    del graph.initializer[:]
    graph.initializer.extend(kept)
    del graph.value_info[:]

    graph.name = "task285_direct_5x5_sparse_renderer"
    model.producer_name = "task285-direct-5x5"
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def build_pivot_probe() -> onnx.ModelProto:
    """Return a pivot-only diagnostic model with one INT32 [3,8] output.

    Slot columns are ``row, col, quadrant, c00, c01, c10, c11, valid``.
    Invalid padded slots are ignored by the certificate test via ``valid``.
    """
    model = build_model()
    graph = model.graph
    terminal = next(index for index, node in enumerate(graph.node) if "f" in node.output)
    kept_nodes = [clone_node(node) for node in graph.node[: terminal + 1]]
    kept_nodes.extend(
        [
            helper.make_node("Less", ["e", "f0"], ["e_negative"]),
            helper.make_node("Less", ["f", "f0"], ["f_negative"]),
            helper.make_node("Cast", ["e_negative"], ["e_bit"], to=TensorProto.INT32),
            helper.make_node("Cast", ["f_negative"], ["f_bit"], to=TensorProto.INT32),
            helper.make_node("Mul", ["f_bit", "probe_i30"], ["pivot_row_offset"]),
            helper.make_node("Sub", ["a", "e_bit"], ["pivot_without_col"]),
            helper.make_node(
                "Sub", ["pivot_without_col", "pivot_row_offset"], ["pivot_flat"]
            ),
            helper.make_node("Div", ["pivot_flat", "probe_i30"], ["pivot_row"]),
            helper.make_node("Mod", ["pivot_flat", "probe_i30"], ["pivot_col"]),
            helper.make_node("Mul", ["f_bit", "probe_i2"], ["quadrant_row_bit"]),
            helper.make_node("Add", ["e_bit", "quadrant_row_bit"], ["quadrant"]),
            helper.make_node(
                "Add", ["pivot_flat", "probe_anchor_offsets"], ["anchor_indices"]
            ),
            helper.make_node("Gather", ["g", "anchor_indices"], ["anchor_colours"]),
            helper.make_node(
                "Cast", ["anchor_colours"], ["anchor_colours_i"], to=TensorProto.INT32
            ),
            helper.make_node("Reshape", ["av3", "shape_3x1"], ["valid_bool"]),
            helper.make_node("Cast", ["valid_bool"], ["valid"], to=TensorProto.INT32),
            helper.make_node(
                "Concat",
                [
                    "pivot_row",
                    "pivot_col",
                    "quadrant",
                    "anchor_colours_i",
                    "valid",
                ],
                ["pivot_slots"],
                axis=1,
            ),
        ]
    )
    del graph.node[:]
    graph.node.extend(kept_nodes)
    graph.initializer.extend(
        [
            numpy_helper.from_array(np.array(30, dtype=np.int32), "probe_i30"),
            numpy_helper.from_array(np.array(2, dtype=np.int32), "probe_i2"),
            numpy_helper.from_array(
                np.array([0, 1, 30, 31], dtype=np.int32), "probe_anchor_offsets"
            ),
        ]
    )
    used = {name for node in graph.node for name in node.input}
    kept_initializers = []
    for initializer in graph.initializer:
        if initializer.name in used:
            copied = onnx.TensorProto()
            copied.CopyFrom(initializer)
            kept_initializers.append(copied)
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)
    del graph.output[:]
    graph.output.append(
        helper.make_tensor_value_info("pivot_slots", TensorProto.INT32, [3, 8])
    )
    del graph.value_info[:]
    graph.name = "task285_sparse_pivot_slots"
    model.producer_name = "task285-sparse-pivot-probe"
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def main() -> None:
    model = build_model()
    onnx.save(model, OUTPUT)
    onnx.save(build_pivot_probe(), PIVOT_OUTPUT)
    result = profile(OUTPUT)
    assert result is not None
    params = calculate_params(model)
    assert params is not None
    memory = int(result["memory_static"])
    print(f"saved {OUTPUT}")
    print(f"saved {PIVOT_OUTPUT}")
    print(f"memory={memory} params={params} cost={memory + params}")


if __name__ == "__main__":
    main()
