"""Task 329 — exact shared colour/spatial core route at cost 105.

The rational feature ``F(x)=[32*(x+0.5), 0.5-x]`` and swap matrix ``J``
satisfy ``F(x)J=F(-x)``.  One shared ``2 x 2 x 2`` core therefore supplies
both reflected colour factors and both dynamic spatial roots.  The same
``J`` exchanges the preserve/background spatial branches inside the FREE
output Einsum, leaving 93 parameters and 12 bytes of intermediates.
"""

from __future__ import annotations

import numpy as np
from onnx import helper

from ._exact import model, node, tensor


def _arrays() -> dict[str, np.ndarray]:
    counts = np.array([9.0, 25.0, 49.0, 81.0], dtype=np.float32)
    colours = np.arange(10, dtype=np.float32)
    colour_features = np.stack(
        [
            np.float32(32.0) * (colours + np.float32(0.5)),
            np.float32(0.5) - colours,
        ],
        axis=1,
    ).astype(np.float32)
    shared_core = np.array(
        [
            [
                [1.0 / 1024.0, -1.0 / 32.0],
                [3.0 / 32.0, 1.0],
            ],
            [[0.0, 1.25], [0.0, 40.0]],
        ],
        dtype=np.float32,
    )
    swap = np.array(
        [[0.0, 1.0 / 32.0], [32.0, 0.0]], dtype=np.float32
    )
    route_r = np.empty((2, 30), dtype=np.float32)
    for column in range(30):
        if 1 <= column <= 4:
            count = counts[column - 1]
            z = (
                np.float32(3.0) * count - np.float32(32.0)
            ) / (
                np.float32(-1280.0)
                * (count - np.float32(32.0))
            )
        else:
            z = np.float32(1.0) / np.float32(5.0)
        route_r[0, column] = np.float32(1024.0) * z
        route_r[1, column] = (
            np.float32(-1.0) / np.float32(20.0)
            - route_r[0, column] / np.float32(32.0)
        )
    return {
        "colour_features": colour_features,
        "shared_core": shared_core,
        "swap": swap,
        "one_state": -np.ones(
            (1, 1, 1, 1), dtype=np.float32
        ),
        "route_r": route_r,
    }


def build(task):
    arrays = _arrays()
    inits = [tensor(name, array) for name, array in arrays.items()]
    nodes = [
        node(
            "ReduceSum",
            ["input"],
            ["cell_count_keep"],
            attrs=[("keepdims", 1)],
        ),
        node(
            "Concat",
            ["cell_count_keep", "one_state"],
            ["state_vec"],
            attrs=[("axis", 1)],
        ),
        node(
            "Einsum",
            [
                "input",
                "colour_features",
                "shared_core",
                "colour_features",
                "colour_features",
                "swap",
                "shared_core",
                "colour_features",
                "swap",
                "swap",
                "shared_core",
                "state_vec",
                "route_r",
                "shared_core",
                "state_vec",
                "route_r",
            ],
            ["output"],
            name="shared_colour_spatial_core_output",
            attrs=[
                (
                    "equation",
                    "bkhw,ka,tad,cd,ke,eg,tgi,cf,fi,tu,upq,bqmn,pw,"
                    "urs,bsxy,rw->bchw",
                )
            ],
        ),
    ]
    value_infos = [
        helper.make_tensor_value_info(
            "cell_count_keep", 1, [1, 1, 1, 1]
        ),
        helper.make_tensor_value_info("state_vec", 1, [1, 2, 1, 1]),
    ]
    return model(
        "task329_regime",
        nodes,
        inits,
        output_dtype=1,
        opset=18,
        value_infos=value_infos,
        ir_version=10,
    )
