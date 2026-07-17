"""Task 215 — period-three, polynomial-colour single-Einsum implementation.

The three columns of ``basis`` are exact coordinate residues modulo three.
Their unweighted sum selects every input column, producing the positive scalar
``grid_width`` on each in-grid row and zero off-grid.  The scorer only observes
output signs, so that scalar replaces the old fixed-column selector.

The same basis routes source rows to output rows with matching residues.  The
rank-three colour polynomial ``(1000k+1)*(1-2(f-k)^2)`` is already a strict
separator for background k=0 and every foreground colour, so no independent
background spatial route is required.  The complete graph has 171 parameters,
zero counted intermediates, and a generator-complete minimum output margin 30.
"""

import numpy as np
from onnx import TensorProto, helper

from ._exact import model, tensor


def build(task):
    basis = np.zeros((30, 3), dtype=np.float32)
    basis[np.arange(30), np.arange(30) % 3] = 1
    route = np.zeros((3, 1, 3), dtype=np.float32)
    route[:, 0, :] = np.eye(3, dtype=np.float32)

    f = np.arange(10, dtype=np.float32)
    k = np.arange(10, dtype=np.float32)
    z = np.float32(1000) * k + np.float32(1)
    a_factor = np.ones((1, 3), dtype=np.float32)
    f_factor = np.stack(
        [np.float32(0.5) - f * f, f, np.ones_like(f)], axis=1
    )
    k_factor = np.stack(
        [2 * z, 4 * k * z, -2 * k * k * z], axis=1
    ).astype(np.float32)

    inits = [
        tensor("basis", basis),
        tensor("row_route", route),
        tensor("a_factor", a_factor),
        tensor("f_factor", f_factor),
        tensor("k_factor", k_factor),
        tensor("col_route", route),
    ]
    nodes = [
        helper.make_node(
            "Einsum",
            [
                "input",
                "basis",
                "basis",
                "row_route",
                "a_factor",
                "f_factor",
                "k_factor",
                "basis",
                "col_route",
                "input",
            ],
            ["output"],
            equation="burv,vq,rx,xap,aj,fj,kj,hw,wap,bkhc->bfrc",
        )
    ]
    return model(
        "task215_spatial_rank3",
        nodes,
        inits,
        output_dtype=TensorProto.FLOAT,
        opset=14,
    )
