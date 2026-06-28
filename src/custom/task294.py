"""Task 294 — 3x3 gray erosion into red.

For gray cells (channel 5), a full 3x3 gray block turns the center red
(channel 2).  Other gray cells remain gray.  Background is preserved through
channel 0 identity.
"""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper

from ._exact import model, tensor


def build(task):
    w = np.zeros((10, 10, 3, 3), dtype=np.float32)
    b = np.zeros((10,), dtype=np.float32)

    w[0, 0, 1, 1] = 1.0
    w[2, 5, :, :] = 1.0
    b[2] = -8.0

    w[5, 5, :, :] = -1.0
    w[5, 5, 1, 1] = 8.0

    nodes = [
        helper.make_node(
            "Conv",
            ["input", "score_kernel", "score_bias"],
            ["output"],
            pads=[1, 1, 1, 1],
        ),
    ]
    return model(
        "task294",
        nodes,
        [tensor("score_kernel", w), tensor("score_bias", b)],
        output_dtype=TensorProto.FLOAT,
        opset=11,
    )
