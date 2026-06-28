"""Task 230 — source-owned sparse convolution.

The transform is a single padded 3x3 convolution.  Most channels are identity;
channels 0..4 also receive a sparse signed stencil from channel 5.
"""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper

from ._exact import model, tensor


def build(task):
    w = np.zeros((10, 10, 3, 3), dtype=np.float32)
    for ch in range(10):
        w[ch, ch, 1, 1] = 1.0

    w[0, 5] = np.array(
        [
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
        ],
        dtype=np.float32,
    )
    w[1, 5, 1, 2] = -0.5
    w[1, 5, 2, 1] = -0.5
    w[1, 5, 2, 2] = 0.5
    w[2, 5, 1, 0] = -0.5
    w[2, 5, 2, 0] = 0.5
    w[2, 5, 2, 1] = -0.5
    w[3, 5, 0, 1] = -0.5
    w[3, 5, 0, 2] = 0.5
    w[3, 5, 1, 2] = -0.5
    w[4, 5, 0, 0] = 0.5
    w[4, 5, 0, 1] = -0.5
    w[4, 5, 1, 0] = -0.5

    nodes = [
        helper.make_node("Conv", ["input", "conv_weight"], ["output"], pads=[1, 1, 1, 1]),
    ]
    return model(
        "task230",
        nodes,
        [tensor("conv_weight", w)],
        output_dtype=TensorProto.FLOAT,
        opset=17,
    )
