"""Task 376 (eb281b96): vertical reflect-tiling of a zigzag wave.

Rule (from the ARC-GEN generator): input is a wave pattern of height
H = stretch + 2 (stretch in 1..4) and width always 17; output has the same
width and height 4*p + 1 where p = H - 1, and output row r equals input row
reflect(r) with reflection period 2p: m(r) = p - |(r mod 2p) - p|.

Graph: every grid cell is one-hot (color 0 included), so the total one-hot
count is 17*H; row-select = (sum - 51)/17 = H - 3 indexes a precomputed
[4,30] table of row index maps (rows past the output height point at canvas
row 29, which is always all-zero since H <= 6).  Final
Gather(input, idx, axis=2) writes `output` directly.  Intermediates: four
scalars + one [30] int32 vector; 136B memory + 122 params.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ..builders import _model


def _index_table():
    """IDX [4,30] int32: row maps for p = 2..5 (stretch 1..4)."""
    rows = []
    for p in range(2, 6):
        i = np.arange(30)
        m = p - np.abs((i % (2 * p)) - p)
        rows.append(np.where(i <= 4 * p, m, 29).astype(np.int32))
    return np.array(rows, np.int32)


def build(task):
    inits = []
    nodes = []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    init("IDX", _index_table(), np.int32)                 # [4,30]
    init("c51", np.array(51.0, dtype=np.float32), np.float32)
    init("c17", np.array(17.0, dtype=np.float32), np.float32)

    # total one-hot count = 17*H  ->  row = H-3 in {0..3}
    n("ReduceSum", ["input"], "tot", keepdims=1)          # [1,1,1,1] f32
    n("Sub", ["tot", "c51"], "t51")                       # 17*(H-3), exact
    n("Div", ["t51", "c17"], "rowf")                      # H-3, exact
    n("Cast", ["rowf"], "row4", to=onnx.TensorProto.INT32)
    n("Squeeze", ["row4"], "row", axes=[0, 1, 2, 3])      # scalar int32
    n("Gather", ["IDX", "row"], "idx", axis=0)            # [30] int32
    n("Gather", ["input", "idx"], "output", axis=2)
    return _model(nodes, inits)
