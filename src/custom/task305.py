"""Task 305 (c3f564a4): restore the (r+c) % colors diagonal-stripe grid.

Rule (from ARC-GEN generator): the grid is 16x16; every cell holds
`(r + c) % colors + 1` with `colors` a single per-instance integer in {4..9}.
The INPUT additionally blacks out (value 0) a handful of random rectangles;
the OUTPUT is the clean stripe pattern with NO blackouts.  So the ENTIRE
output is a deterministic function of one scalar, `colors` — there is nothing
spatial to detect beyond recovering that scalar.

Construction (a label map; the only computed scalar is `colors`):

1.  Recover `colors`.  The validator guarantees every row/col shows all
    `colors` colours, and values never exceed `colors`, so
    `colors = max colour present`.  From the one-hot input, presence of colour
    k is `pres[k] = max over (r,c) of channel k`; `colors = max_k k*pres[k]`.

2.  Recompute the stripe arithmetically.  A fixed plane `SUM[1,1,16,16] = r+c`
    (fp16, values <= 30 < 2048 so exact) is reduced mod `colors` with one
    `Mod(fmod=1)`.  The remainder `(r+c) % colors` is in 0..colors-1.

3.  Write into the FREE output.  Cast the remainder to uint8, `Pad` it to
    30x30 with sentinel 254 (off-grid cells are all-channels-off in the target,
    254 matches nothing), then the final `Equal` against `chan-1 =
    [255,0,1,..,8]` writes straight into the BOOL `output`.  Comparing against
    `chan-1` folds the "+1" colour offset into the constant: output channel k
    is hot where `(r+c)%colors == k-1`, i.e. colour `k = (r+c)%colors + 1`.
    Channel 0's constant is 255, which the remainder never equals, so the
    background channel stays off everywhere.

Memory: only sizeable intermediates are the padded 30x30 uint8 label plane
(900 B, irreducible gateway into the free Equal) and the 16x16 fp16 Mod result
(512 B) + its uint8 cast (256 B).  No [1,10,30,30] is ever built.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

SIZE = 16  # grid side


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
    ssum = np.array([[r + c for c in range(SIZE)] for r in range(SIZE)],
                    np.float16).reshape(1, 1, SIZE, SIZE)
    init("SUM", ssum, np.float16)                                   # r+c plane
    init("colk", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1),
         np.float32)                                                 # 0..9
    # chan-1 = [255,0,1,..,8]: channel k matches remainder k-1 (colour k); the
    # 255 for channel 0 is never produced by the remainder, so the background
    # channel stays off everywhere.
    chanm1 = ((np.arange(10) - 1) % 256).astype(np.uint8).reshape(1, 10, 1, 1)
    init("chanm1", chanm1, np.uint8)
    # Pad 16x16 -> 30x30 (bottom/right) with sentinel 254 (matches no channel).
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - SIZE, 30 - SIZE], np.int64),
         np.int64)
    init("padval", np.array(254, np.uint8), np.uint8)

    # ---- recover colors (scalar) ----
    n("ReduceMax", ["input"], "pres", axes=[2, 3], keepdims=1)       # [1,10,1,1]
    n("Mul", ["pres", "colk"], "kparts")                             # k where present
    n("ReduceMax", ["kparts"], "colf", keepdims=0)                   # scalar = colors
    n("Cast", ["colf"], "colh", to=TensorProto.FLOAT16)              # fp16 colors

    # ---- remainder plane (r+c) % colors, then into the free Equal ----
    n("Mod", ["SUM", "colh"], "rem", fmod=1)                         # [1,1,16,16] fp16
    n("Cast", ["rem"], "remu", to=TensorProto.UINT8)                 # uint8 0..colors-1
    n("Pad", ["remu", "padpads", "padval"], "Lp", mode="constant")   # [1,1,30,30]
    n("Equal", ["Lp", "chanm1"], "output")                           # -> BOOL output

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
