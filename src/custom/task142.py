"""Task 142 (ARC-AGI 62c24649): 4-fold mirror of a size-3 grid into 6x6.

Rule (from ARC-GEN generator; all 262 arc-gen instances are 3x3 -> 6x6,
colours 0..3):
  input  = a size-3 grid at the top-left corner (rows/cols 0..2).
  output = a 2*size = 6 square with 4-fold mirror symmetry, top-left corner:
    out[r , c        ] = grid[r][c]
    out[r , 2s-1-c   ] = grid[r][c]   (horizontal mirror)
    out[2s-1-r , c   ] = grid[r][c]   (vertical mirror)
    out[2s-1-r,2s-1-c] = grid[r][c]   (both)
  Everything OFF the 6x6 block is all-zero (the scorer's expected output has
  no background fill -- it compares (output>0) against the one-hot 6x6, with
  every cell outside the 6x6 block zero on every channel).

Encoding (beats the public GridSample-on-30x30 + Pad net, 17.67):
  The public net samples the FULL 30x30 fp32 input -> a 1440B [1,10,6,6] fp32
  mirror block.  We shrink that single dominant intermediate two ways:
    1. colours are only 0..3, so slice the input to channels 0..3 AND the active
       3x3 corner in ONE Slice -> [1,4,3,3] (only 36 elems);
    2. Cast that tiny block to fp16, then GridSample (mode=nearest) the 6x6
       mirror as a [1,4,6,6] fp16 block (288B vs 1440B).
  A final Pad (channels 4..9 with zeros + spatial 6->30) writes straight into
  the free output.  GridSample + fp16 + the 4-channel/3x3 slice keep every
  counted tensor tiny.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

S = 3        # input grid side
B = 2 * S    # output block side = 6
NC = 4       # active colour channels (colours are 0..3)


def build(task):
    inits, nodes = [], []
    vinfos = []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def vi(name, dtype, shape):
        vinfos.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F16 = TensorProto.FLOAT16
    F32 = TensorProto.FLOAT

    # ---- 1. slice input to channels 0..3 AND the 3x3 corner in one Slice ----
    init("s_st", np.array([0, 0, 0], np.int64), np.int64)        # ch0, r0, c0
    init("s_en", np.array([NC, S, S], np.int64), np.int64)       # ch4, r3, c3
    init("s_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "s_st", "s_en", "s_ax"], "g3")          # [1,4,3,3] f32
    vi("g3", F32, [1, NC, S, S])

    # ---- 2. cast the tiny block to fp16 ----
    n("Cast", ["g3"], "g3h", to=F16)                             # [1,4,3,3] f16
    vi("g3h", F16, [1, NC, S, S])

    # ---- 3. GridSample the 6x6 mirror (nearest, align_corners=0) ----
    # output cell (r,c) reads input cell (mr,mc):
    #   mr = r if r < S else (2S-1-r);  mc = c if c < S else (2S-1-c)
    # normalised coord for index i over N=S pixels (align_corners=0):
    #   coord = (2*i + 1)/S - 1
    grid = np.zeros((1, B, B, 2), np.float32)   # last dim = (x->col, y->row)
    for r in range(B):
        mr = r if r < S else (2 * S - 1 - r)
        yv = (2 * mr + 1) / S - 1.0
        for c in range(B):
            mc = c if c < S else (2 * S - 1 - c)
            xv = (2 * mc + 1) / S - 1.0
            grid[0, r, c, 0] = xv
            grid[0, r, c, 1] = yv
    # grid must match input (fp16) dtype for GridSample in ORT
    init("grid", grid, np.float16)
    n("GridSample", ["g3h", "grid"], "gs",
      mode="nearest", align_corners=0, padding_mode="zeros")     # [1,4,6,6] f16
    vi("gs", F16, [1, NC, B, B])

    # ---- 4. Pad: channels (add 6 high) + spatial (6 -> 30), zeros -> output ----
    # pads order for [N,C,H,W]: [n0,c0,h0,w0, n1,c1,h1,w1]
    init("padpads",
         np.array([0, 0, 0, 0, 0, 10 - NC, 30 - B, 30 - B], np.int64), np.int64)
    init("padz", np.array(0.0, np.float16), np.float16)
    n("Pad", ["gs", "padpads", "padz"], "output", mode="constant")  # free output

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F16, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits, value_info=vinfos)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 16)])
