"""Task 323 (ARC-AGI d06dbe63): stamp a fixed gray staircase at the cyan marker.

Rule: the 13x13 input has exactly one cyan(8) pixel at (row,col). The output keeps
that cyan pixel and draws a gray(5) staircase path emanating in two point-symmetric
directions ((-1,+1) up-right and (+1,-1) down-left). Each branch alternates 2 vertical
then 2 horizontal unit steps, producing a 45-degree staircase, clipped at the grid edge.

The staircase, RELATIVE to the marker, is a FIXED set of (dr,dc) offsets (independent
of position) -- a fixed-shape stamp. So this is a marker-stamp task:

  cyan = input channel 8, cropped to the 13x13 active grid -> fp32 [1,1,13,13]
  resp = Conv(cyan_fp16, K) SAME pad=12, kernel 25x25 where K[12-dr,12-dc]=1 for every
         staircase offset (dr,dc). Because cyan is a single 1 (and 0 off-grid), resp is
         1 exactly at the staircase cells, auto-clipped to the 13x13 active region.
  condK = resp > 0  -> bool [1,1,13,13]; opset-13 Pad (accepts bool) -> cond [1,1,30,30]
  output = Where(cond, gray_onehot[1,10,1,1], input)   -> FREE 10-ch output.

The only full-canvas intermediate is the 900B bool cond; everything upstream is 13x13.
The 625-elem stamp kernel is the dominant cost and is structural (the staircase can reach
+-12 from a corner marker, so a 25x25 kernel is required). Mem 2421, params 656 -> 16.97,
beating the prior conv-stamp net (16.58) which built a 1690B 10-channel Concat at 13x13.

Cyan survives because the Where condition only covers gray cells (the marker itself is
not a staircase offset), so the marker passes through via the `input` branch. Off-grid
canvas (rows/cols 13..29) stays bg: cyan is 0 there so the conv response is 0.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL


def _model13(nodes, inits, vinfos):
    """opset-13 single-in/out model (opset 13 Pad accepts bool + pads-as-input)."""
    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    graph = helper.make_graph(list(nodes), "graph", [x], [y], list(inits),
                              value_info=list(vinfos))
    return helper.make_model(graph, ir_version=8,
                             opset_imports=[helper.make_opsetid("", 13)])

K = 13   # active canvas
PAD = 12  # kernel half-size -> kernel is (2*PAD+1)=25


def _gray_offsets():
    """Replicate the generator's two-branch staircase as a set of (dr,dc) offsets."""
    offs = set()
    for dr, dc in [(-1, 1), (1, -1)]:
        v, h, r, c = 2, 0, 0, 0
        while True:
            if v:
                r, v = r + dr, v - 1
                if r < -PAD or r > PAD:
                    break
                offs.add((r, c))
                if not v:
                    h = 2
            else:
                c, h = c + dc, h - 1
                if c < -PAD or c > PAD:
                    break
                offs.add((r, c))
                if not h:
                    v = 2
    return offs


def build(task):
    inits = []
    nodes = []
    vinfos = []

    def init(name, arr, dtype=np.float32):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    def vi(name, dtype, shape):
        vinfos.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    # cyan = channel 8, cropped to KxK -> fp32 [1,1,K,K]
    init("c_s", np.array([0, 8, 0, 0], np.int64), np.int64)
    init("c_e", np.array([1, 9, K, K], np.int64), np.int64)
    init("c_ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "c_s", "c_e", "c_ax"], "cyan")
    vi("cyan", F32, [1, 1, K, K])
    # cast the marker plane to fp16 so the whole conv runs in fp16 (no fp32 resp plane).
    n("Cast", ["cyan"], "cyan16", to=TensorProto.FLOAT16)
    vi("cyan16", TensorProto.FLOAT16, [1, 1, K, K])

    # staircase stamp kernel: K[0, 0, 12-dr, 12-dc] = 1 for every offset (dr,dc)
    ks = 2 * PAD + 1
    ker = np.zeros((1, 1, ks, ks), np.float16)
    for dr, dc in _gray_offsets():
        ker[0, 0, PAD - dr, PAD - dc] = 1.0
    init("stamp", ker, np.float16)

    # SAME conv (pad=12 each side) -> fp16 [1,1,K,K], clipped to the 13x13 active grid so
    # staircase pixels that run off the grid are correctly dropped (the generator breaks at
    # the edge). A wider conv would mis-stamp those into the off-grid 30x30 margin.
    n("Conv", ["cyan16", "stamp"], "resp16",
      kernel_shape=[ks, ks], pads=[PAD, PAD, PAD, PAD], strides=[1, 1])
    vi("resp16", TensorProto.FLOAT16, [1, 1, K, K])

    # threshold at 13x13 -> bool (169B), then pad the BOOL mask to 30x30 (900B).
    # opset-13 Pad accepts bool (pads given as an input tensor), keeping the only
    # full-canvas plane a 900B bool (vs an 1800B fp16 30x30 or a 1690B 10-ch Concat).
    init("zero16", np.array(0.0, np.float16), np.float16)
    n("Greater", ["resp16", "zero16"], "condK")
    vi("condK", BOOL, [1, 1, K, K])
    init("padspec", np.array([0, 0, 0, 0, 0, 0, 30 - K, 30 - K], np.int64), np.int64)
    n("Pad", ["condK", "padspec"], "cond", mode="constant")
    vi("cond", BOOL, [1, 1, 30, 30])

    # output = cond ? gray_onehot : input  (cyan marker survives via the input branch;
    # off-grid 30x30 margin stays bg because condK is False there after the Pad).
    gray = np.zeros((1, 10, 1, 1), np.float32)
    gray[0, 5, 0, 0] = 1.0
    init("gray", gray)
    n("Where", ["cond", "gray", "input"], "output")

    return _model13(nodes, inits, vinfos)
