"""Task 061 (29ec7d0e): restore the (r*c) % mod multiplication-table grid.

Rule (from ARC-GEN generator): the grid is 18x18; every cell holds
`(r*c) % mod + 1` with `mod` a single per-instance integer in {4..9}.
The INPUT additionally blacks out (value 0) a handful of random rectangles;
the OUTPUT is the *clean* table with NO blackouts.  So the entire output is a
deterministic function of one scalar, `mod` — there is nothing spatial to
detect beyond recovering that scalar.

Approach (label-map, but the spatial work is a pure lookup):

1.  Recover `mod`.  The grid always contains every colour 1..mod (an 18x18
    range of r*c hits every residue 0..mod-1), and never a colour > mod, so
    `mod = max colour present`.  From the one-hot input, presence of colour k
    is `pres[k] = max over (r,c) of channel k`; `mod = max_k k*pres[k]`.
    (Verified: input.max() == mod on 2000 fresh instances.)

2.  Look up the answer.  A precomputed table `TAB[m-4][r][c] = (r*c)%m+1`
    of shape [6,18,18] (uint8) holds the clean grid for every possible mod.
    Gather row `mod-4` -> the full 18x18 label plane, Pad to 30x30 with the
    sentinel 10 (the output grid is only 18x18; cells outside it are
    all-channels-off, and 10 matches no channel 0..9), then the
    final `Equal(L, arange[0..9])` writes straight into the FREE bool `output`.

Dominant intermediate is the padded label plane L[1,1,30,30] uint8 = 900 B;
everything else is scalars / 1-D aggregates.  No [1,10,30,30] is ever built.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

SIZE = 18          # grid side
MODS = list(range(4, 10))  # possible mod values 4..9


def _table():
    """[6,18,18] uint8: TAB[m-4][r][c] = (r*c) % m + 1."""
    tab = np.zeros((len(MODS), SIZE, SIZE), np.uint8)
    for i, m in enumerate(MODS):
        for r in range(SIZE):
            for c in range(SIZE):
                tab[i, r, c] = (r * c) % m + 1
    return tab


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- constants ----
    init("TAB", _table(), np.uint8)                                   # [6,18,18]
    init("colk", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1),
         np.float32)                                                  # 0..9
    init("four", np.array(4, dtype=np.int32), np.int32)               # mod offset
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    # Pad 18x18 -> 30x30 (bottom/right).  The output grid is only 18x18; cells
    # outside it are all-channels-off, so fill with sentinel 10 (matches no
    # channel 0..9 in the final Equal -> all-false there).
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - SIZE, 30 - SIZE], np.int64),
         np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)

    # ---- recover mod (scalar) ----
    # presence of colour k anywhere on the grid -> [1,10,1,1]
    n("ReduceMax", ["input"], "pres", axes=[2, 3], keepdims=1)        # [1,10,1,1]
    n("Mul", ["pres", "colk"], "kparts")                             # k where present
    n("ReduceMax", ["kparts"], "modf", keepdims=0)                   # scalar = mod
    n("Cast", ["modf"], "modi", to=TensorProto.INT32)                # int32 mod
    n("Sub", ["modi", "four"], "idx")                                # 0..5 table row

    # ---- look up the clean 18x18 label plane ----
    n("Gather", ["TAB", "idx"], "L18", axis=0)                       # [18,18] uint8
    # reshape to [1,1,18,18] for Pad / Equal broadcast
    init("shp", np.array([1, 1, SIZE, SIZE], np.int64), np.int64)
    n("Reshape", ["L18", "shp"], "L18r")                             # [1,1,18,18]
    n("Pad", ["L18r", "padpads", "padval"], "Lp", mode="constant")   # [1,1,30,30]

    # ---- final Equal into the free BOOL output ----
    n("Equal", ["Lp", "chan"], "output")                             # -> BOOL output

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
