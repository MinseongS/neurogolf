"""Task 061 (29ec7d0e): restore the (r*c) % mod multiplication-table grid.

Rule (from ARC-GEN generator): the grid is 18x18; every cell holds
`(r*c) % mod + 1` with `mod` a single per-instance integer in {4..9}.  The
INPUT additionally blacks out (value 0) a handful of random rectangles; the
OUTPUT is the *clean* table with NO blackouts.  So the ENTIRE output is a
deterministic function of one scalar, `mod` — there is nothing spatial to
detect beyond recovering that scalar.

Construction (a label map, but the only computed scalar is `mod`):

1.  Recover `mod`.  The grid always contains every colour 1..mod (the 18x18
    range of r*c hits every residue 0..mod-1) and never a colour > mod, so
    `mod = max colour present`.  From the one-hot input, presence of colour k
    is `pres[k] = max over (r,c) of channel k`; `mod = max_k k*pres[k]`.
    (Verified: input.max() == mod on 2000 fresh instances.)

2.  Recompute the table arithmetically (no big lookup table).  A fixed plane
    `PROD[1,1,18,18] = r*c` (fp16, all values <= 289 < 2048 so exact) is reduced
    mod `mod` with a single `Mod(fmod=1)`.  The remainder `(r*c) % mod` is in
    0..8.

3.  Write into the FREE output.  Cast the remainder to uint8, `Pad` it to
    30x30 (the grid is only 18x18; outside cells are all-channels-off, so the
    pad value 254 deliberately matches nothing), then the final `Equal` against
    `chan-1 = [255,0,1,..,8]` writes straight into the BOOL `output`.  Comparing
    against `chan-1` folds the "+1" colour offset into the constant, so no extra
    add/plane is needed: output channel k is hot where `(r*c)%mod == k-1`,
    i.e. colour `k = (r*c)%mod + 1`.  Channel 0's constant is 255, which the
    remainder (0..8) never equals, so the background channel is always off.

Memory: the only sizeable intermediates are the padded 30x30 uint8 label plane
(900 B, the irreducible gateway into the free Equal) and the 18x18 fp16 Mod
result (648 B) + its uint8 cast (324 B).  No [1,10,30,30] is ever built and the
lookup table is just a 324-element PROD param.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

SIZE = 18  # grid side


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
    prod = np.array([[r * c for c in range(SIZE)] for r in range(SIZE)],
                    np.float16).reshape(1, 1, SIZE, SIZE)
    init("PROD", prod, np.float16)                                   # r*c plane
    init("colk", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1),
         np.float32)                                                 # 0..9
    # chan-1 = [255,0,1,..,8]: channel k matches remainder k-1 (colour k); the
    # 255 for channel 0 is never produced by the remainder (0..8), so the
    # background channel stays off everywhere.
    chanm1 = ((np.arange(10) - 1) % 256).astype(np.uint8).reshape(1, 10, 1, 1)
    init("chanm1", chanm1, np.uint8)
    # Pad 18x18 -> 30x30 (bottom/right) with sentinel 254 (matches no channel).
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - SIZE, 30 - SIZE], np.int64),
         np.int64)
    init("padval", np.array(254, np.uint8), np.uint8)

    # ---- recover mod (scalar) ----
    n("ReduceMax", ["input"], "pres", axes=[2, 3], keepdims=1)       # [1,10,1,1]
    n("Mul", ["pres", "colk"], "kparts")                             # k where present
    n("ReduceMax", ["kparts"], "modf", keepdims=0)                   # scalar = mod
    n("Cast", ["modf"], "modh", to=TensorProto.FLOAT16)              # fp16 mod

    # ---- remainder plane (r*c) % mod, then into the free Equal ----
    n("Mod", ["PROD", "modh"], "rem", fmod=1)                        # [1,1,18,18] fp16
    n("Cast", ["rem"], "remu", to=TensorProto.UINT8)                 # uint8 0..8
    n("Pad", ["remu", "padpads", "padval"], "Lp", mode="constant")   # [1,1,30,30]
    n("Equal", ["Lp", "chanm1"], "output")                           # -> BOOL output

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
