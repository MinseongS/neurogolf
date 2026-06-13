"""task215 (ARC-AGI 8eb1be9a) — vertical period-3 tiling of a 3-row band.

Rule: the input holds exactly one 3-row-tall colored band (rows offset..offset+2,
offset in {3,4}). The output tiles that band vertically with period 3 across the
whole grid. The band has exactly one row in each residue class mod 3, so the
transform is the fixed row-mixing matrix M[r,b] = (r%3 == b%3) applied along the
height axis — BUT only the 3 band rows may act as a source (empty grid rows carry
a background one-hot that would otherwise leak), and output rows r >= H (grid
height) must stay empty.

We fold both per-instance masks into M:
    Mdyn[r,b] = (r%3 == b%3) * rowmask[r] * srcmask[b]
  rowmask[r] = 1 if row r is inside the grid (any one-hot occupancy)
  srcmask[b] = 1 if input row b contains a colored pixel (channels 1..9)
so the single MatMul(Mdyn, input) writes straight into the free `output`.
"""

import numpy as np
import onnx

from ..harness import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS


def build(task):
    nodes, inits = [], []

    def init(name, arr, dtype=np.float32):
        inits.append(onnx.numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(onnx.helper.make_node(op, ins, [out], **attrs))
        return out

    # Fixed period-3 row-mixing matrix M[r,b] = (r%3 == b%3).
    M = (np.arange(30)[:, None] % 3 == np.arange(30)[None, :] % 3).astype(np.float32)
    init("M", M)
    init("zero", np.array(0.0, np.float32))

    # Per-row total one-hot occupancy: W for rows inside the grid, 0 outside.
    n("ReduceSum", ["input"], "occ", axes=[1, 3], keepdims=0)        # [1,30]
    # Per-row background (channel-0) count.
    init("c0_start", np.array([0], np.int64), np.int64)
    init("c0_end", np.array([1], np.int64), np.int64)
    init("c0_axis", np.array([1], np.int64), np.int64)
    n("Slice", ["input", "c0_start", "c0_end", "c0_axis"], "ch0")    # [1,1,30,30]
    n("ReduceSum", ["ch0"], "bg", axes=[1, 3], keepdims=0)           # [1,30]

    # rowmask[r] = occ > 0 (inside grid); srcmask[b] = colored count > 0.
    n("Greater", ["occ", "zero"], "rowmask_b")
    n("Cast", ["rowmask_b"], "rowmask_f", to=int(DATA_TYPE))         # [1,30]
    n("Sub", ["occ", "bg"], "colored")                              # [1,30]
    n("Greater", ["colored", "zero"], "srcmask_b")
    n("Cast", ["srcmask_b"], "srcmask", to=int(DATA_TYPE))          # [1,30]  (axis b)

    # rowmask as a column vector [30,1] to scale M's rows.
    init("colshape", np.array([30, 1], np.int64), np.int64)
    n("Reshape", ["rowmask_f", "colshape"], "rowmask")              # [30,1]

    # Mdyn[r,b] = M[r,b] * rowmask[r] * srcmask[b]; final MatMul -> output (free).
    n("Mul", ["M", "rowmask"], "Mr")                               # [30,30]
    n("Mul", ["Mr", "srcmask"], "Mdyn")                            # [30,30] (srcmask bcast over b)
    n("MatMul", ["Mdyn", "input"], "output")                       # [1,10,30,30]

    x = onnx.helper.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = onnx.helper.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = onnx.helper.make_graph(nodes, "task215", [x], [y], inits)
    return onnx.helper.make_model(graph, ir_version=IR_VERSION,
                                  opset_imports=OPSET_IMPORTS)
