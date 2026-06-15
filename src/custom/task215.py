"""task215 (ARC-AGI 8eb1be9a) — vertical period-3 tiling of a 3-row band.

Rule: the input holds exactly one 3-row-tall colored band (rows offset..offset+2,
offset in {3,4}). The output tiles that band vertically with period 3 across the
whole grid. The band has exactly one row in each residue class mod 3, so output
row r is a copy of the band row whose residue equals r % 3 — for every output
row r that is inside the grid (r < H); rows r >= H stay empty.

Memory floor-break (row Gather, no [30,30] mixing matrix):
  We never build a 30x30 transform. Instead we compute, for each residue
  rho in {0,1,2}, the single band row srcrow[rho] that carries it, then a length
  30 source-index vector  idx[r] = srcrow[r % 3]  (redirected to an empty edge
  row 29 when r >= H), and Gather the input rows straight into the free output.

    bandmask[b] = 1 if input row b has a colored pixel        (width conv, [30])
    srcrow[rho] = sum_b R[rho,b] * bandmask[b] * b            R[rho,b]=(b%3==rho)
    idx[r]      = srcrow[r % 3]  if r < H  else 29            (29 is empty)
    output      = Gather(input, idx, axis=2)                  [1,10,30,30] (free)

  Only length-30 / length-3 vectors are materialised; the sole 2-D tensor is the
  free output.
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

    F = onnx.TensorProto.FLOAT

    # --- per-row colored count (channels 1..9) via one width-30 conv -> [1,1,30,1]
    Wc = np.zeros((1, 10, 1, 30), np.float32); Wc[0, 1:, 0, :] = 1.0
    init("Wc", Wc, np.float32)
    n("Conv", ["input", "Wc"], "col4")                             # [1,1,30,1]
    n("Squeeze", ["col4"], "colored", axes=[0, 1, 3])              # [30] float
    init("zero_f", np.array(0.0, np.float32), np.float32)
    n("Greater", ["colored", "zero_f"], "band_b")                  # [30] bool
    n("Cast", ["band_b"], "bandmask", to=F)                        # [30] 0/1

    # --- per-row occupancy (any channel) -> grid height test rowmask[r] = r < H
    Wo = np.zeros((1, 10, 1, 30), np.float32); Wo[0, :, 0, :] = 1.0
    init("Wo", Wo, np.float32)
    n("Conv", ["input", "Wo"], "occ4")                             # [1,1,30,1]
    n("Squeeze", ["occ4"], "occ", axes=[0, 1, 3])                  # [30] float
    n("Greater", ["occ", "zero_f"], "rowmask_b")                   # [30] bool r<H

    # --- srcrow[rho] = sum_b (b%3==rho) * bandmask[b] * b ---------------------
    R = (np.arange(30)[None, :] % 3 == np.arange(3)[:, None]).astype(np.float32)
    init("R", R, np.float32)                                       # [3,30]
    init("ramp", np.arange(30, dtype=np.float32), np.float32)      # [30] = b
    n("Mul", ["bandmask", "ramp"], "bw")                           # [30] = b on band
    n("Mul", ["R", "bw"], "Rbw")                                   # [3,30]
    n("ReduceSum", ["Rbw"], "srcrow", axes=[1], keepdims=0)        # [3] float

    # --- idx[r] = srcrow[r % 3], redirected to empty row 29 when r >= H -------
    init("rmod3", (np.arange(30) % 3).astype(np.int64), np.int64)  # [30]
    n("Gather", ["srcrow", "rmod3"], "idx_f", axis=0)              # [30] float
    init("c29", np.full(30, 29.0, np.float32), np.float32)
    n("Where", ["rowmask_b", "idx_f", "c29"], "idx_sel")          # [30] float
    n("Cast", ["idx_sel"], "idx", to=onnx.TensorProto.INT32)      # [30] int32

    # --- Gather input rows straight into the free output ----------------------
    n("Gather", ["input", "idx"], "output", axis=2)               # [1,10,30,30]

    x = onnx.helper.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = onnx.helper.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = onnx.helper.make_graph(nodes, "task215", [x], [y], inits)
    return onnx.helper.make_model(graph, ir_version=IR_VERSION,
                                  opset_imports=OPSET_IMPORTS)
