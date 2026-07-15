"""Task 329 — exact shared-parabola route at cost 190.

The output score is ``E(k,c) + D(s,w) * P(c)`` with
``E(k,c)=0.25-(k-c)^2`` and ``D(s,w)=(w-(s+1))^2``.  Input and output
colours share the same feature ``[1,x,x^2]``; the two colour forms therefore
need only a 2x3x3 bilinear core.  The spatial terms share the exact rank-three
feature ``[1,w,w^2]``.
"""

from __future__ import annotations

import numpy as np
from onnx import helper

from ._exact import model, node, tensor


def _arrays() -> dict[str, np.ndarray]:
    epsilon = np.float32(0.25)
    background_scale = np.float32(400.0)

    colours = np.arange(10, dtype=np.float32)
    colour_features = np.stack(
        [np.ones(10, dtype=np.float32), colours, colours**2], axis=1
    )

    route_a = np.zeros((2, 3, 3), dtype=np.float32)
    route_a[0, 0, 0] = epsilon
    route_a[0, 0, 2] = -1.0
    route_a[0, 1, 1] = 2.0
    route_a[0, 2, 0] = -1.0
    route_a[1, 0, 0] = background_scale * epsilon
    route_a[1, 0, 2] = -background_scale

    route_l = np.zeros((4, 2, 3), dtype=np.float32)
    for size_state in range(4):
        middle = np.float32(size_state + 1)
        route_l[size_state, 0] = [1.0, 0.0, 0.0]
        route_l[size_state, 1] = [middle**2, -2.0 * middle, 1.0]

    columns = np.arange(30, dtype=np.float32)
    route_r = np.stack(
        [np.ones(30, dtype=np.float32), columns, columns**2], axis=0
    )
    return {
        "counts4_f": np.array([9.0, 25.0, 49.0, 81.0], dtype=np.float32),
        "colour_features": colour_features,
        "route_a": route_a,
        "route_l": route_l,
        "route_r": route_r,
    }


def build(task):
    arrays = _arrays()
    inits = [tensor(name, array) for name, array in arrays.items()]
    nodes = [
        node("ReduceSum", ["input"], ["cell_count_f"], attrs=[("keepdims", 0)]),
        node("Equal", ["cell_count_f", "counts4_f"], ["sel_b"]),
        node("Cast", ["sel_b"], ["sel_f"], attrs=[("to", 1)]),
        node(
            "Einsum",
            [
                "input",
                "colour_features",
                "route_a",
                "colour_features",
                "route_l",
                "route_r",
                "sel_f",
            ],
            ["output"],
            name="shared_bilinear_colour_rank3_spatial_output",
            attrs=[("equation", "bkhw,ka,tad,cd,str,rw,s->bchw")],
        ),
    ]
    value_infos = [
        helper.make_tensor_value_info("cell_count_f", 1, []),
        helper.make_tensor_value_info("sel_b", 9, [4]),
        helper.make_tensor_value_info("sel_f", 1, [4]),
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
