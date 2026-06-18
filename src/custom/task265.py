"""Task 265 (ARC-AGI a8d7556c) — paint qualifying 2x2 black holes red.

Rule (size=18 grid of gray(5)/black(0)): two stripe passes paint 2x2 all-black
holes red(2), order-dependent (pass0 mutates the grid that pass1 reads):
  pass0 (downstripe): paint hole (r,c) unless a horizontal side-column pair
    (col c-1 or c+2, both rows) is black.
  pass1 (sidestripe), on the pass-0-reded grid: a hole counts only if all four
    cells are STILL black; paint unless a vertical side-row pair (row r-1 or r+2,
    both cols) is still-black.

Memory-golfed encoding (mem 11916, was 17100):
  - cpb = vertical-pair-black indicator via a HALF-weighted 2x1 conv (sums to
    {0,0.5,1}) + Floor -> {0,1} fp16, no bool round-trip.
  - hole = banded conv [-1,1,1,-1] over the pair indicator + bias -1, then Relu:
    score = (cpb[c]+cpb[c+1]) - (cpb[c-1]+cpb[c+2]) - 1 equals 1 ONLY at a hole
    and <=0 elsewhere, so Relu yields the {0,1} hole indicator directly (this
    one banded conv folds the 2x2-empty test AND both side-pair fail tests).
  - red0 = 2x2-dilation of pass0 holes; blk1f (still-black, fp16) = Where(red0,0,
    blkf) — no separate Not/And bool planes; pass1 mirrors pass0 on horizontal
    pairs; final red = red0 | red1.
  - label L (uint8 0/2/5, off-grid sentinel 10) -> Pad to 30x30 -> Equal(arange)
    into the FREE bool output. Only fp32 plane is the 18x18 channel-0 slice.
"""
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

W = 18


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT16

    init("half", np.array(0.5, np.float16), np.float16)
    # half-weighted pair kernels: sum in {0, 0.5, 1}; Floor -> {0,0,1} indicator
    init("k21", np.full((1, 1, 2, 1), 0.5, np.float16), np.float16)  # vert pair
    init("k12", np.full((1, 1, 1, 2), 0.5, np.float16), np.float16)  # horiz pair
    init("k22", np.ones((1, 1, 2, 2), np.float16), np.float16)  # expand
    # banded hole+fail kernels [-1,1,1,-1] with bias -1 then Relu -> {0,1}:
    #   score = (cpb[c]+cpb[c+1]) - (cpb[c-1]+cpb[c+2]) - 1; ==1 only at a hole,
    #   <=0 everywhere else, so Relu yields the {0,1} hole indicator in fp16 with
    #   no bool round-trip.
    init("kh0", np.array([-1., 1., 1., -1.], np.float16).reshape(1, 1, 1, 4),
         np.float16)
    init("kh1", np.array([-1., 1., 1., -1.], np.float16).reshape(1, 1, 4, 1),
         np.float16)
    init("biasm1", np.array([-1.], np.float16), np.float16)

    # ---- in-grid black plane (channel 0) of the 18x18 corner ----
    init("s_st", np.array([0, 0, 0], np.int64), np.int64)
    init("s_en", np.array([1, W, W], np.int64), np.int64)
    init("s_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "s_st", "s_en", "s_ax"], "blk32")
    n("Cast", ["blk32"], "blkf", to=F)
    n("Greater", ["blkf", "half"], "blk")

    # ============ PASS 0 (downstripe) ============
    # cpb[r,c] = blk[r,c] & blk[r+1,c]  (vertical pair, pad bottom)
    # half-sum in {0,0.5,1}; Floor -> {0,0,1} indicator fp16 (no bool round trip)
    n("Conv", ["blkf", "k21"], "cpb_s", pads=[0, 0, 1, 0])
    n("Floor", ["cpb_s"], "cpbf")
    # h0f[r,c] = hole indicator (Conv with bias -1 then Relu -> {0,1} fp16)
    n("Conv", ["cpbf", "kh0", "biasm1"], "h0_s", pads=[0, 1, 0, 2])
    n("Relu", ["h0_s"], "h0f")
    # expand holes -> red0 cells (2x2 ones, pad top/left)
    n("Conv", ["h0f", "k22"], "red0_s", pads=[1, 1, 0, 0])
    n("Greater", ["red0_s", "half"], "red0")

    # ============ PASS 1 (sidestripe) on pass-0-mutated grid ============
    # blk1f = black-still as fp16: zero out cells painted red in pass 0.
    # (red0 cells are always originally black, so this == blk AND NOT red0.)
    init("zerof", np.array(0.0, np.float16), np.float16)
    n("Where", ["red0", "zerof", "blkf"], "blk1f")
    # rpb1[r,c] = blk1[r,c] & blk1[r,c+1]  (horizontal pair, pad right)
    n("Conv", ["blk1f", "k12"], "rpb1_s", pads=[0, 0, 0, 1])
    n("Floor", ["rpb1_s"], "rpb1f")
    # h1f[r,c] = pass-1 hole indicator (Conv bias -1 then Relu -> {0,1} fp16)
    n("Conv", ["rpb1f", "kh1", "biasm1"], "h1_s", pads=[1, 0, 2, 0])
    n("Relu", ["h1_s"], "h1f")
    n("Conv", ["h1f", "k22"], "red1_s", pads=[1, 1, 0, 0])
    n("Greater", ["red1_s", "half"], "red1")

    # ---- final red cells ----
    n("Or", ["red0", "red1"], "red")

    # ---- label map L18 (uint8): red 2, gray 5, black 0 ----
    init("v0", np.array(0, np.uint8), np.uint8)
    init("v2", np.array(2, np.uint8), np.uint8)
    init("v5", np.array(5, np.uint8), np.uint8)
    n("Where", ["blk", "v0", "v5"], "base")
    n("Where", ["red", "v2", "base"], "L18")

    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64),
         np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    n("Pad", ["L18", "padpads", "padval"], "L", mode="constant")
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task265", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
