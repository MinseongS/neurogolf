"""Task 369 (ARC-AGI e8593010) — colour each black blob by its component size.

Rule (from the generator): a `size`x`size` grid (default size=10) on a gray (5)
background holds non-overlapping, mutually-separated black blobs of size 1, 2 or
3 (single pixel / 2-cell domino / 3-cell line or L). Because the generator's
overlap check forbids two distinct blobs from touching (4-adjacency), every
4-connected black component IS exactly one blob, so its size is well defined.
The OUTPUT recolours each black cell by `colour = 4 - component_size`:
    size 1 -> colour 3,  size 2 -> colour 2,  size 3 -> colour 1.
Gray background stays gray (5); off-grid stays unset.

Closed-form local detection (no flood-fill needed because blobs are <=3 cells &
separated): let deg = #black 4-neighbours of a black cell.
    size-1 cell: deg == 0
    size-2 cell: deg == 1 and NO neighbour has deg>=2
    size-3 cell: deg >= 2  OR  a neighbour has deg>=2
(In a 3-line the middle has deg 2; in an L the corner has deg 2; endpoints get
caught by "neighbour has deg>=2". A domino's two cells both have deg 1 with no
heavy neighbour.)

The active grid is a fixed 10x10 in the top-left, so we slice the black channel
to [1,1,10,10] and do all work there (100-elem planes, tiny). The final 10-way
one-hot expansion is routed into the FREE bool output via Pad(sentinel)->Equal.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
I32 = TensorProto.INT32
U8 = TensorProto.UINT8
B = TensorProto.BOOL

W = 10  # active grid extent


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, list(ins), [out], **attrs))
        return out

    # All full-canvas working planes are fp16 (100-elem, 200B each).
    # ---- black mask, sliced to the active 10x10 grid (channel 0), cast fp16 ----
    init("blk_starts", np.array([0, 0, 0], np.int64), np.int64)
    init("blk_ends", np.array([1, W, W], np.int64), np.int64)
    init("blk_axes", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "blk_starts", "blk_ends", "blk_axes"], "black32")  # [1,1,10,10] fp32
    n("Cast", ["black32"], "black", to=F16)

    # ---- deg = #black 4-neighbours (plus-shaped conv, no self) ----
    plus = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float16).reshape(1, 1, 3, 3)
    init("plusK", plus, np.float16)
    n("Conv", ["black", "plusK"], "deg", pads=[1, 1, 1, 1])     # [1,1,10,10] fp16

    # ---- heavy = BLACK AND (deg >= 2) -> interior/corner of a size-3 blob.
    # The "black AND" is load-bearing: a GRAY cell can have deg>=2 (borders two
    # black cells of nearby blobs) and would otherwise leak through neighbor_heavy
    # onto an adjacent domino, mislabelling it size-3.
    init("c1_5", np.array([1.5], np.float16), np.float16)
    n("Greater", ["deg", "c1_5"], "deg2_b")                     # bool
    n("Cast", ["deg2_b"], "deg2", to=F16)
    n("Mul", ["black", "deg2"], "heavy")                        # {0,1} fp16

    # ---- neighbour_heavy raw = #heavy 4-neighbours (>=1 -> has heavy nbr) ----
    n("Conv", ["heavy", "plusK"], "nh_raw", pads=[1, 1, 1, 1])

    # is3 = black AND (heavy OR neighbor_heavy) = black AND (heavy + nh_raw >= 1)
    init("c0_5", np.array([0.5], np.float16), np.float16)
    n("Add", ["heavy", "nh_raw"], "heavy_or_raw")
    n("Greater", ["heavy_or_raw", "c0_5"], "is3_pre_b")
    n("Cast", ["is3_pre_b"], "is3_pre", to=F16)
    n("Mul", ["black", "is3_pre"], "is3")                       # {0,1}

    # is1 = black AND deg==0  -> deg < 0.5 (deg=0 on a black cell == isolated dot)
    n("Less", ["deg", "c0_5"], "deg0_b")
    n("Cast", ["deg0_b"], "deg0", to=F16)
    n("Mul", ["black", "deg0"], "single")                      # isolated black pixels {0,1}

    # Closed-form label = 5 - 3*black + single - is3   (derived):
    #   gray  (black=0)              -> 5
    #   size1 (single=1, is3=0)      -> 3
    #   size2 (single=0, is3=0)      -> 2
    #   size3 (single=0, is3=1)      -> 1
    init("c3", np.array([3.0], np.float16), np.float16)
    init("c5", np.array([5.0], np.float16), np.float16)
    n("Mul", ["black", "c3"], "three_black")
    n("Sub", ["c5", "three_black"], "p1")                      # 5 - 3*black
    n("Add", ["p1", "single"], "p2")                           # + single
    n("Sub", ["p2", "is3"], "label_f")                         # - is3  -> {1,2,3,5}
    n("Cast", ["label_f"], "label_u8", to=U8)

    # ---- pad to 30x30 with off-grid sentinel 99 (uint8), then Equal -> output ----
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), np.int64)
    init("sent", np.array(99, np.uint8), np.uint8)
    n("Pad", ["label_u8", "pads", "sent"], "label30", mode="constant")  # [1,1,30,30] uint8
    init("arange10", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["label30", "arange10"], "output")              # [1,10,30,30] bool

    graph = helper.make_graph(
        nodes, "task369",
        [helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", B, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
