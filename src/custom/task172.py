"""Task 172 — source-owned row-gather transform.

The live graph is a single Gather over the height axis.  Keep the semantic
mapping in source instead of depending on networks/task172.onnx.
"""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper

from ._exact import model, tensor


def build(task):
    idx = np.array(
        [0, 1, 2, 2, 1, 0] + [3] * 24,
        dtype=np.int64,
    )
    nodes = [
        helper.make_node("Gather", ["input", "idx"], ["output"], axis=2),
    ]
    return model(
        "task172",
        nodes,
        [tensor("idx", idx)],
        output_dtype=TensorProto.FLOAT,
        opset=11,
    )
