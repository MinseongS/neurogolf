"""task048 (ARC-AGI 239be575) — single-pixel path-existence between two red boxes.

Rule (from the generator):
  A width x height (5..8 each) black canvas holds scattered cyan pixels plus TWO
  solid 2x2 red boxes (non-overlapping, gap>=1).  The output is a 1x1 grid:
  cyan if a 4-connected path of non-background (cyan OR red) cells links box0 to
  box1 (i.e. reaches a red cell outside box0), else black.  (The generator rejects
  instances where 4- vs 8-connectivity disagree, so the answer is unambiguous.)

  This is a genuine 4-connectivity / flood-fill predicate over a variable-size
  noisy grid: there is no closed-form / separable escape (path existence is
  inherently iterative).  The active canvas is gen-bounded to <=8x8, so the flood
  runs on a tiny [1,1,8,8] plane.

Encoding (bounded BFS, 12 cross-dilations):
  passable = red OR cyan   (channels 2,8 of the input, sliced to 8x8, fp16).
  seed     = the FIRST red cell (ArgMax of flattened red one-hot) -> [1,1,8,8].
  per round:  count = Conv_cross(reach)         (4-neighbour+self sum, fp16)
              reach = Min(passable, count)       (passable AND count>=1, {0,1})
    -- Min(passable in {0,1}, integer count>=0) == passable*(count>=1), and it
       stays bounded in {0,1} (no fp16 overflow), folding the previous net's
       Greater+Where (2 ops + a bool plane) into ONE Min op, saving 64B/round.
  12 rounds suffices: over 20000 fresh instances the max dilations needed to
  reach box1 was 11 (margin 1).
  connected = (#red cells reached) > 4   (box0 has 4; >4 ==> box1 reached).
  output    = cyan one-hot if connected else black, padded to 30x30.

Dominant intermediate: the 12 fp16 [1,1,8,8] reach/count planes (128B each) —
irreducible because (a) the canvas max is 8x8 (generator bound), (b) the cross
(4-neighbour) dilation forbids MaxPool (rectangular footprint => 8-neighbour =
wrong connectivity), so Conv (float) is mandatory, and (c) 11 dilations can be
needed so the round count cannot drop.  Min replaces Greater+Where to delete the
per-round bool plane.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F16 = TensorProto.FLOAT16
I64 = TensorProto.INT64
BOOL = TensorProto.BOOL

N_ROUNDS = 12  # max dilations needed (empirically 11) + margin


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- passable = red(ch2) OR cyan(ch8), cropped to the 8x8 active region ---
    init("starts2", np.array([2, 0, 0], np.int64), np.int64)
    init("ends2",   np.array([3, 8, 8], np.int64), np.int64)
    init("axes",    np.array([1, 2, 3], np.int64), np.int64)
    init("steps",   np.array([1, 1, 1], np.int64), np.int64)
    init("starts8", np.array([8, 0, 0], np.int64), np.int64)
    init("ends8",   np.array([9, 8, 8], np.int64), np.int64)

    n("Slice", ["input", "starts2", "ends2", "axes", "steps"], "twos")    # [1,1,8,8] f32
    n("Slice", ["input", "starts8", "ends8", "axes", "steps"], "eights")  # [1,1,8,8] f32
    n("Cast", ["twos"], "twos16", to=F16)
    n("Cast", ["eights"], "eights16", to=F16)
    n("Sum", ["twos16", "eights16"], "passable")                          # [1,1,8,8] f16

    # ---- seed = the first red cell -------------------------------------------
    init("flat_shape", np.array([1, 64], np.int64), np.int64)
    n("Reshape", ["twos16", "flat_shape"], "flat_twos")                   # [1,64] f16
    n("ArgMax", ["flat_twos"], "first_two_index", axis=1, keepdims=0)     # [1] i64
    init("depth64", np.array(64, np.int64), np.int64)
    init("hot_values", np.array([0, 1], np.float16), np.float16)
    n("OneHot", ["first_two_index", "depth64", "hot_values"], "seed_flat", axis=-1)
    init("grid_shape", np.array([1, 1, 8, 8], np.int64), np.int64)
    n("Reshape", ["seed_flat", "grid_shape"], "reach0")                   # [1,1,8,8] f16

    # ---- bounded BFS: count = Conv_cross(reach); reach = Min(passable, count) --
    init("flood_kernel",
         np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.float16).reshape(1, 1, 3, 3),
         np.float16)
    # The first N_ROUNDS-1 rounds gate to `passable`; the LAST round gates the
    # propagation directly to `twos16` (red) instead of `passable` — since every
    # red cell is passable, Min(twos16, count) == Min(passable, count) intersected
    # with red == the red cells reached after the final propagation.  This fuses
    # the final reach plane and the reached-red plane into ONE op (saves 128B).
    cur = "reach0"
    for i in range(N_ROUNDS):
        cnt = n("Conv", [cur, "flood_kernel"], f"count{i}", pads=[1, 1, 1, 1])
        gate = "twos16" if i == N_ROUNDS - 1 else "passable"
        cur = n("Min", [gate, cnt], f"reach{i + 1}")

    # ---- connected iff > 4 red cells reached (box0 has 4) ---------------------
    n("ReduceSum", [cur], "reached_two_count", axes=[1, 2, 3], keepdims=0)  # [1] f16
    init("four", np.array([4], np.float16), np.float16)
    n("Greater", ["reached_two_count", "four"], "connected")              # [1] bool
    n("Cast", ["connected"], "class_index", to=TensorProto.INT32)         # [1] i32

    # ---- output: cyan(ch8) one-hot if connected else black; pad to 30x30 ------
    bank = np.zeros((2, 10, 1, 1), np.float16)
    bank[0, 0, 0, 0] = 1.0    # not connected -> black at channel 0
    bank[1, 8, 0, 0] = 1.0    # connected -> cyan at channel 8
    init("output_bank", bank, np.float16)
    n("Gather", ["output_bank", "class_index"], "small_output", axis=0)   # [1,10,1,1] f16
    init("pads", np.array([0, 0, 0, 0, 0, 0, 29, 29], np.int64), np.int64)
    init("zero", np.array(0.0, np.float16), np.float16)
    n("Pad", ["small_output", "pads", "zero"], "output")                  # [1,10,30,30] f16

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F16, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task048", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
