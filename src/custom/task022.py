"""task022 (ARC-AGI 137eaa0f) — overlay the 3x3 neighbourhoods of 4 gray centres.

Rule (from generator task_137eaa0f.py, size=11, minisize=3):
  An 11x11 input carries 4 "centre" cells at well-separated positions (>=4 apart).
  Each centre is painted GRAY(5) unless its colour is 0 (then it stays background).
  Around each centre, the 8 non-centre cells of a 3x3 window are (partly) painted with
  that centre's colour; the 8 output positions (rows[i],cols[i]) cover the 3x3-minus-
  centre exactly once, each owned by one centre via idxs[i].  Pixel colours are drawn
  from random_colors(4) with 5 remapped to 0, so a coloured pixel is NEVER gray.

  The OUTPUT is the 3x3 reconstruction: output[1][1]=GRAY(5), and for each offset
  (dr,dc) in {-1,0,1}^2 minus (0,0) the value is whichever centre's colour sits at that
  offset.  Centres are >=4 apart so the four 3x3 windows are pairwise disjoint in the
  grid -> overlaying them tiles the 3x3 with no collisions.

  Verified exactly (0/3000 fresh) as a CORRELATION:
      G    = (input == gray)                  gray-centre mask
      colf = sum_{k!=5} k * input_k           coloured-pixel value plane (centres excluded)
      out[1+dr][1+dc] = sum_{i,j} G[i,j] * colf[i+dr, j+dc]
  Each offset has at most one nonzero contribution (disjoint windows) so the sum recovers
  the unique colour.  Centre (dr,dc)=(0,0) is overwritten with GRAY.

Encoding (pts 15.78, mem 9985, params 129, fresh 200/200):
  - G = Slice channel 5 cropped to 11x11, cast fp16.
  - colf = 1x1 Conv on the FREE [1,10,30,30] input with weights [0,1,2,3,4,0,6,7,8,9]
    (the one fp32 entry plane, [1,1,30,30]=3600B), cropped to 11x11 and cast fp16.
  - im2col the 8 NON-centre neighbour shifts in ONE Conv with one-hot kernels
    K[o,dr+1,dc+1]=1 (pad=1) -> shifted[1,8,11,11]; convout[o,r,c]=colf[r+dr,c+dc].
  - correlate with G: Mul(shifted, G) + ReduceSum over space -> [1,8] (the 8 values).
  - insert GRAY(5) at the centre (Slice/Concat) -> [1,9] -> reshape [1,1,3,3] value plane.
  - cast uint8, Pad to 30x30 with sentinel 255 (off-grid matches no colour -> all-zero).
  - output = Equal(L_u8[1,1,30,30], arange[0..9][1,10,1,1]) -> BOOL [1,10,30,30] (FREE).
  Dominant intermediate: the fp32 colf entry plane [1,1,30,30]=3600B (irreducible: the
  channel-collapse Conv on the free input is full-canvas; cropping the 10 input channels
  to 11x11 first would cost 4840B).  shifted/prod fp16 [1,8,11,11]=1936B each.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I64 = TensorProto.INT64

N = 11  # active grid size


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- gray-centre mask G = channel 5, cropped to 11x11 in ONE Slice -----
    init("g_s", np.array([0, 5, 0, 0], np.int64), np.int64)
    init("g_e", np.array([1, 6, N, N], np.int64), np.int64)
    init("g_ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "g_s", "g_e", "g_ax"], "Gf")  # [1,1,11,11] f32
    n("Cast", ["Gf"], "G", to=F16)                      # [1,1,11,11] f16

    # ---- colf = sum_{k!=5} k * input_k  via 1x1 Conv on the FREE input ------
    # Conv on the full free [1,10,30,30] input avoids materialising a 10-ch crop.
    w = np.array([0, 1, 2, 3, 4, 0, 6, 7, 8, 9], np.float32).reshape(1, 10, 1, 1)
    init("Wcol", w, np.float32)
    n("Conv", ["input", "Wcol"], "colf30")  # [1,1,30,30] f32 (entry plane)
    # crop to 11x11 and go fp16 for all downstream full-canvas ops
    init("cc_s", np.array([0, 0], np.int64), np.int64)
    init("cc_e", np.array([N, N], np.int64), np.int64)
    init("cc_ax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["colf30", "cc_s", "cc_e", "cc_ax"], "colff")  # [1,1,11,11] f32
    n("Cast", ["colff"], "colf", to=F16)                       # f16

    # ---- im2col the 8 NON-centre neighbour shifts via ONE Conv -------------
    # Centre (0,0) is overwritten by GRAY, so only 8 shifts are needed.
    # K[o, dr+1, dc+1] = 1  =>  convout[o,r,c] = colf[r+dr, c+dc]  (pad=1).
    offs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    K = np.zeros((8, 1, 3, 3), np.float16)
    for o, (dr, dc) in enumerate(offs):
        K[o, 0, dr + 1, dc + 1] = 1.0
    init("Kshift", K, np.float16)
    n("Conv", ["colf", "Kshift"], "shifted", pads=[1, 1, 1, 1])  # [1,8,11,11] f16

    # ---- correlate with G: out[o] = sum_{r,c} shifted[o,r,c]*G[r,c] --------
    n("Mul", ["shifted", "G"], "prod")               # [1,8,11,11] f16 (G broadcasts)
    n("ReduceSum", ["prod"], "vals8f", axes=[2, 3], keepdims=0)  # [1,8] f16
    n("Cast", ["vals8f"], "vals8", to=F32)           # f32 for exact compose

    # ---- insert GRAY(5) at the centre -> 9 values, reshape to 3x3 ----------
    # offs order = (0,0),(0,1),(0,2),(1,0),(1,2),(2,0),(2,1),(2,2); centre is index 4.
    init("v_s4", np.array([0], np.int64), np.int64)
    init("v_e4", np.array([4], np.int64), np.int64)
    init("v_ax", np.array([1], np.int64), np.int64)
    n("Slice", ["vals8", "v_s4", "v_e4", "v_ax"], "vpre")   # [1,4] (0,0)..(1,0)
    init("v_s5", np.array([4], np.int64), np.int64)
    init("v_e5", np.array([8], np.int64), np.int64)
    n("Slice", ["vals8", "v_s5", "v_e5", "v_ax"], "vpost")  # [1,4] (1,2)..(2,2)
    init("gray1", np.array([[5.0]], np.float32), np.float32)  # [1,1]
    n("Concat", ["vpre", "gray1", "vpost"], "vals9", axis=1)  # [1,9] f32

    init("sh33", np.array([1, 1, 3, 3], np.int64), np.int64)
    n("Reshape", ["vals9", "sh33"], "Lval")  # [1,1,3,3] f32

    # ---- cast 3x3 value plane to uint8 FIRST, then pad to 30x30 (uint8 pad) -
    n("Cast", ["Lval"], "L33u8", to=U8)  # [1,1,3,3] uint8
    init("pad30", np.array([0, 0, 0, 0, 0, 0, 27, 27], np.int64), np.int64)
    init("sent", np.array([255], np.uint8), np.uint8)
    n("Pad", ["L33u8", "pad30", "sent"], "Lu8", mode="constant")  # [1,1,30,30] uint8

    # ---- one-hot expansion into the FREE BOOL output -----------------------
    arange = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("ar", arange, np.uint8)
    n("Equal", ["Lu8", "ar"], "output")  # [1,10,30,30] BOOL

    inp_vi = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    out_vi = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task022", [inp_vi], [out_vi], inits)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 11)],
        ir_version=IR_VERSION)
    return model
