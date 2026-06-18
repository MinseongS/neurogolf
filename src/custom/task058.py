"""Task 058 (ARC-AGI 28e73c20) — draw an inward rectangular SPIRAL.

Rule (from the generator): the input is a blank size x size grid (size 5..20,
all background).  The OUTPUT is a deterministic green (3) spiral that winds
inward from the top-left corner.  The pattern is a pure function of N = size;
the input carries NO information except its dimension N (the harness pads the
grid into a fixed 1x10x30x30 tensor, so the N x N in-grid block is exactly the
cells where input channel-0 == 1).

CLOSED FORM (verified exact for every N in 5..20):
  L = min(r, c, N-1-r, N-1-c)                      # ring layer
  special = (r == L+1) AND (c == L)                # the per-ring connector cell
  green = (L even) XOR special
  plus, when N % 4 == 2, force green at (N/2, N/2-1)  # even-center correction

Every even ring is a full ring minus its `special` cell; every odd ring is empty
except its `special` cell (the connector that threads the spiral inward).

MEMORY: N <= 20, so the whole pattern lives in the top-left 20x20 block; we do
all the per-cell algebra on a 20x20 canvas (400 cells, 2.25x smaller than 30x30)
then Pad the colour-index label up to 30x30 with an off-grid sentinel and route
the 10-channel one-hot into the FREE bool output via Equal vs arange.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
I32 = TensorProto.INT32
B = TensorProto.BOOL

W = 20  # working canvas (max grid size)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- in-grid block (input channel 0 == 1 on the N x N top-left) ----
    # Slice channel 0 AND crop spatially to the 20x20 working window in one go.
    init("s_starts", np.array([0, 0, 0], np.int64), np.int64)
    init("s_ends", np.array([1, W, W], np.int64), np.int64)
    init("s_axes", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "s_starts", "s_ends", "s_axes"], "ingrid")  # [1,1,20,20] fp32

    # ---- N scalar = max per-row count of in-grid cells ----
    n("ReduceSum", ["ingrid"], "rowcnt", axes=[3], keepdims=1)   # [1,1,20,1]
    n("ReduceMax", ["rowcnt"], "Nf", axes=[2], keepdims=1)       # [1,1,1,1] = N

    # ---- coordinate ramps (1-D, broadcast) — all fp16 (N<=20 exact in fp16) ----
    n("Cast", ["Nf"], "Nh", to=F16)                             # N scalar fp16
    init("rramp", np.arange(W, dtype=np.float16).reshape(1, 1, W, 1), np.float16)
    init("cramp", np.arange(W, dtype=np.float16).reshape(1, 1, 1, W), np.float16)

    init("one_f", np.array([[[[1.0]]]], np.float16), np.float16)
    n("Sub", ["Nh", "one_f"], "Nm1")                            # N-1
    n("Sub", ["Nm1", "rramp"], "distR")                         # [1,1,20,1]
    n("Sub", ["Nm1", "cramp"], "distC")                         # [1,1,1,20]

    # L = min(r, c, distR, distC)  — ONE variadic Min, one 20x20 fp16 plane
    n("Min", ["rramp", "cramp", "distR", "distC"], "L")         # [1,1,20,20] fp16

    # special cell is exactly (r,c) = (L+1, L), i.e. c==L AND r==c+1.
    # r==c+1 is a separable 1-D comparison (no full L+1 plane needed).
    n("Add", ["cramp", "one_f"], "cp1")                         # [1,1,1,20] fp16
    n("Equal", ["rramp", "cp1"], "eqr")                         # r==c+1 (broadcast)
    n("Equal", ["cramp", "L"], "eqc")                           # c==L
    n("And", ["eqr", "eqc"], "special")

    # Leven = (L mod 2 == 0)
    init("two_f", np.array([[[[2.0]]]], np.float16), np.float16)
    init("zero_f", np.array([[[[0.0]]]], np.float16), np.float16)
    n("Mod", ["L", "two_f"], "Lmod", fmod=1)
    n("Equal", ["Lmod", "zero_f"], "Leven")

    # green = Leven XOR special
    n("Xor", ["Leven", "special"], "green0")

    # ---- even-center correction: if N % 4 == 2, force green at (N/2, N/2-1) ----
    init("four_f", np.array([[[[4.0]]]], np.float16), np.float16)
    init("two_b", np.array([[[[2.0]]]], np.float16), np.float16)
    n("Mod", ["Nh", "four_f"], "Nmod4", fmod=1)
    n("Equal", ["Nmod4", "two_b"], "isN2")                      # [1,1,1,1] bool
    init("half_f", np.array([[[[0.5]]]], np.float16), np.float16)
    n("Mul", ["Nh", "half_f"], "halfN")                         # N/2
    n("Sub", ["halfN", "one_f"], "halfNm1")                     # N/2 - 1
    n("Equal", ["rramp", "halfN"], "eqcr")                      # [1,1,20,1] bool
    n("Equal", ["cramp", "halfNm1"], "eqcc")                    # [1,1,1,20] bool
    # And the two 1-D row/col conditions with the scalar N%4==2 gate; both Ands
    # below have a 1-D operand so the only 20x20 bool plane is the final fixcell.
    n("And", ["eqcr", "isN2"], "rowfix")                        # [1,1,20,1] bool
    n("And", ["rowfix", "eqcc"], "fixcell")                     # [1,1,20,20] bool
    n("Or", ["green0", "fixcell"], "green")                     # [1,1,20,20] bool

    # ---- label plane (20x20, fp16): green -> 3, else 0 ; off-grid -> sentinel.
    #      fp16 halves every label plane; Pad+Equal both run on fp16. ----
    init("three_h", np.array([[[[3.0]]]], np.float16), np.float16)
    init("zero_h", np.array([[[[0.0]]]], np.float16), np.float16)
    n("Where", ["green", "three_h", "zero_h"], "ingrid_lab")    # [1,1,20,20] fp16

    # in-grid mask (bool) to set off-grid cells inside the 20x20 window to sentinel
    init("half2", np.array([[[[0.5]]]], np.float32), np.float32)
    n("Greater", ["ingrid", "half2"], "ingrid_b")               # [1,1,20,20] bool
    init("sent_h", np.array([[[[99.0]]]], np.float16), np.float16)
    n("Where", ["ingrid_b", "ingrid_lab", "sent_h"], "lab20")   # off-grid -> 99

    # ---- Pad label 20x20 -> 30x30 with sentinel 99, then route one-hot ----
    pads = np.array([0, 0, 0, 0, 0, 0, 10, 10], np.int64)
    init("pads", pads, np.int64)
    init("sentv", np.array(99, np.float16), np.float16)
    n("Pad", ["lab20", "pads", "sentv"], "lab", mode="constant")  # [1,1,30,30] fp16

    init("arange", np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1), np.float16)
    n("Equal", ["lab", "arange"], "output")                     # [1,10,30,30] bool

    out_vi = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    in_vi = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task058", [in_vi], [out_vi], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
