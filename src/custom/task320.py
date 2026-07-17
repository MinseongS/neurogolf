"""Task 320 — factored sign routing in a free-output Einsum.

Four red runs are summarized by their lengths.  A coordinate polynomial marks
the lower half of each run, while factored channel polynomials preserve every
non-red colour and route red to red/cyan without a dense 2x10x10 table.
"""

import numpy as np
from onnx import TensorProto, helper

from ._exact import model, tensor


def _route_factors() -> tuple[np.ndarray, np.ndarray]:
    channels = np.arange(10, dtype=np.float32)
    distance_from_red = (channels - np.float32(2)) ** 2
    left = np.zeros((2, 10, 3), dtype=np.float32)
    right = np.zeros((2, 10, 3), dtype=np.float32)

    # For k != 2 this is (k-2)^2 * (0.5 - (k-o)^2): positive only at o=k.
    left[0, :, 0] = distance_from_red * (np.float32(0.5) - channels**2)
    left[0, :, 1] = np.float32(2) * channels * distance_from_red
    left[0, :, 2] = -distance_from_red
    right[0, :, 0] = np.float32(1)
    right[0, :, 1] = channels
    right[0, :, 2] = channels**2

    # Red contributes -Q to channel 2 and +Q to channel 8.
    left[1, 2, 0] = np.float32(1)
    right[1, 2, 0] = np.float32(-1)
    right[1, 8, 0] = np.float32(1)
    return left, right


def _polynomial_factors() -> tuple[np.ndarray, np.ndarray]:
    left = np.zeros((2, 2, 2), dtype=np.float32)
    left[0, 0, 0] = np.float32(1)
    left[1, 1, 0] = np.float32(2)
    left[1, 0, 1] = np.float32(1)

    right = np.zeros((6, 4, 2), dtype=np.float32)
    right[0, 0, 0] = np.float32(1)
    right[:, :, 1] = np.array(
        [
            [-1.5, 0, 0, 0],
            [2.1875, -1.4791666269302368, 0.3125, -0.02083333395421505],
            [-2.1875, 2.9375, -0.8125, 0.0625],
            [1.3125, -1.9375, 0.6875, -0.0625],
            [-0.3125, 0.4791666567325592, -0.1875, 0.02083333395421505],
            [-2, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    return left, right


def build(task):
    coordinates = np.arange(30, dtype=np.float32)
    route_left, route_right = _route_factors()
    poly_left, poly_right = _polynomial_factors()
    inits = [
        tensor("slice_starts", np.array([0, 2, 1, 1], dtype=np.int64)),
        tensor("slice_ends", np.array([1, 3, 11, 9], dtype=np.int64)),
        tensor("slice_steps", np.array([1, 1, 1, 2], dtype=np.int64)),
        tensor("one", np.ones(1, dtype=np.float32)),
        tensor("row_powers", np.stack([np.ones(30), coordinates], axis=1).astype(np.float32)),
        tensor(
            "col_powers",
            np.stack([np.ones(30), coordinates, coordinates**2, coordinates**3], axis=1).astype(
                np.float32
            ),
        ),
        tensor("route_left", route_left),
        tensor("route_right", route_right),
        tensor("poly_left", poly_left),
        tensor("poly_right", poly_right),
    ]
    nodes = [
        helper.make_node(
            "Slice", ["input", "slice_starts", "slice_ends", "", "slice_steps"], ["lanes"]
        ),
        helper.make_node("ReduceSum", ["lanes"], ["lengths"], axes=[0, 1, 2], keepdims=0),
        helper.make_node("ReduceMax", ["lengths"], ["max_length"], axes=[0], keepdims=1),
        helper.make_node("Concat", ["one", "lengths", "max_length"], ["state"], axis=0),
        helper.make_node(
            "Einsum",
            [
                "input",
                "route_left",
                "route_right",
                "poly_left",
                "poly_right",
                "state",
                "row_powers",
                "col_powers",
            ],
            ["output"],
            equation="bkrc,tku,tou,tiv,sjv,s,ri,cj->borc",
        ),
    ]
    return model(
        "task320_factored_sign_routing",
        nodes,
        inits,
        output_dtype=TensorProto.FLOAT,
        opset=12,
    )
