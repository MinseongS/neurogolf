"""Task 023 — greedy stencil decomposition of a gray shape.

The input is a 9x11 gray mask.  Decompose it into repeated primitive stencils:
a 2x2 block (painted colour 8), a horizontal 1x3 bar and a vertical 3x1 bar
(painted colour 2).  The ONNX graph greedily extracts non-overlapping stencil
anchors and routes their coverage into the two output colours.
"""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper

from ._exact import model, tensor


def _candidate_kernel() -> np.ndarray:
    k = np.zeros((3, 1, 3, 3), dtype=np.float16)
    k[0, 0, :2, :2] = 1
    k[1, 0, 0, :] = 1
    k[2, 0, :, 0] = 1
    return k


def _coverage_kernels() -> tuple[np.ndarray, np.ndarray]:
    k8 = np.zeros((1, 3, 3, 3), dtype=np.float16)
    k2 = np.zeros((1, 3, 3, 3), dtype=np.float16)
    k8[0, 0, 1:, 1:] = 1
    k2[0, 1, 2, :] = 1
    k2[0, 2, :, 2] = 1
    return k8, k2


def build(task):
    k_cand = _candidate_kernel()
    k_cov8, k_cov2 = _coverage_kernels()
    inits = [
        tensor("s5", np.array([0, 5, 0, 0], dtype=np.int64)),
        tensor("e5", np.array([1, 6, 9, 11], dtype=np.int64)),
        tensor("s0", np.array([0, 0, 0, 0], dtype=np.int64)),
        tensor("e0", np.array([1, 1, 9, 11], dtype=np.int64)),
        tensor("k_cand", k_cand),
        tensor("k_cov8", k_cov8),
        tensor("k_cov2", k_cov2),
        tensor("thr_cand", np.array([3.5, 2.5, 2.5], dtype=np.float16).reshape(1, 3, 1, 1)),
        tensor("half", np.array(0.5, dtype=np.float16)),
        tensor("one", np.array(1.0, dtype=np.float16)),
        tensor("zero", np.array(0.0, dtype=np.float16)),
        tensor("pads", np.array([0, 0, 0, 0, 0, 0, 21, 19], dtype=np.int64)),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "s5", "e5"], ["x0"]),
        helper.make_node("Cast", ["x0"], ["x"], to=TensorProto.FLOAT16),
        helper.make_node("Conv", ["x", "k_cand"], ["cand_sum"], pads=[0, 0, 2, 2]),
        helper.make_node("Greater", ["cand_sum", "thr_cand"], ["cand_b"]),
        helper.make_node("Cast", ["cand_b"], ["C0"], to=TensorProto.FLOAT16),
    ]

    acc = None
    current = "C0"
    for i in range(4):
        nodes.extend(
            [
                helper.make_node("ConvTranspose", [current, "k_cand"], [f"cov{i}"], pads=[0, 0, 2, 2]),
                helper.make_node("Equal", [f"cov{i}", "one"], [f"eq{i}"]),
                helper.make_node("Where", [f"eq{i}", "x", "zero"], [f"uniq{i}"]),
                helper.make_node("Conv", [f"uniq{i}", "k_cand"], [f"hit{i}"], pads=[0, 0, 2, 2]),
                helper.make_node("Greater", [f"hit{i}", "half"], [f"hit_b{i}"]),
                helper.make_node("Where", [f"hit_b{i}", current, "zero"], [f"new{i}"]),
            ]
        )
        if acc is None:
            acc = "new0"
        else:
            nodes.append(helper.make_node("Add", [acc, f"new{i}"], [f"acc{i}"]))
            acc = f"acc{i}"
        nodes.extend(
            [
                helper.make_node("ConvTranspose", [acc, "k_cand"], [f"paint{i}"], pads=[0, 0, 2, 2]),
                helper.make_node("Conv", [f"paint{i}", "k_cand"], [f"overlap{i}"], pads=[0, 0, 2, 2]),
                helper.make_node("Greater", [f"overlap{i}", "half"], [f"overlap_b{i}"]),
                helper.make_node("Where", [f"overlap_b{i}", "zero", current], [f"C{i + 1}"]),
            ]
        )
        current = f"C{i + 1}"

    nodes.extend(
        [
            helper.make_node("Add", [acc, current], ["acc4"]),
            helper.make_node("Conv", ["acc4", "k_cov8"], ["out8_sum"], pads=[2, 2, 0, 0]),
            helper.make_node("Conv", ["acc4", "k_cov2"], ["out2_sum"], pads=[2, 2, 0, 0]),
            helper.make_node("Slice", ["input", "s0", "e0"], ["in0"]),
            helper.make_node("Cast", ["in0"], ["bg"], to=TensorProto.FLOAT16),
            helper.make_node("Sub", ["bg", "bg"], ["zch"]),
            helper.make_node(
                "Concat",
                ["bg", "zch", "out2_sum", "zch", "zch", "zch", "zch", "zch", "out8_sum", "zch"],
                ["crop_out"],
                axis=1,
            ),
            helper.make_node("Pad", ["crop_out", "pads", "zero"], ["output"], mode="constant"),
        ]
    )
    return model("task023", nodes, inits, output_dtype=TensorProto.FLOAT16, opset=11)
