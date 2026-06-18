"""Task 082 (3ac3eb23): split-and-stripe a row-0 pixel pattern.

Rule (from the ARC-GEN generator, verified exact + fresh):
  The input has coloured pixels only in row 0; the grid height is always 6 and the
  width is 5..15.  Each marked column `c` (carrying `color`) becomes a 3-wide vertical
  stripe in the output:
    - row r EVEN (0,2,4): output[r][c]   = color ;  output[r][c-1] = output[r][c+1] = black
    - row r ODD  (1,3,5): output[r][c]   = black ;  output[r][c-1] = output[r][c+1] = color
  i.e. the output is a separable row-parity (x) column-pattern, fully determined by row 0.
  Marked columns are >=3 apart so neighbour stripes never collide.

ENCODING (small-active-canvas + per-column closed form, routed into the FREE output):
  Everything is done on the tiny ACTIVE strip [1,10,1,WORK] (WORK=15, the true max
  content width is col 14) — never a 30x30 plane — then the 2-row parity TEMPLATE is
  Padded to width 30 and a single parity MatMul writes the 10-channel result straight
  into the FREE `output` (zero mem for the 10-ch expansion).

    x32  = Slice(input, row0, cols0..14)              [1,10,1,15] fp32  (the only fp32 plane)
    x    = Cast(x32 -> fp16)                           [1,10,1,15]       (EVEN template = row 0)
    odd  = Conv(x, G) 1x3                              [1,10,1,15]       (ODD  template)
              colored ch 1..9 :  odd[ch,w] = x[ch,w-1] + x[ch,w+1]   (stripe -> neighbours)
              black   ch 0     :  odd[0,w]  = SUM_k x[k,w]  -  SUM_{k>=1}(x[k,w-1]+x[k,w+1])
              (SUM_k x[k,w] is the in-width one-hot mass = the in-grid mask, so ch0 is set
               wherever the cell is in-grid and not a colour neighbour, and 0 off-grid;
               this reconstructs the odd-row background in the SAME conv, no extra plane.)
    tpls = Concat(x, odd, axis=rows)                  [1,10,2,15] fp16  (2-row template)
    tpl  = Pad(tpls, width 15 -> 30)                  [1,10,2,30] fp16
    output = MatMul(P[30,2], tpl)                      -> [1,10,30,30]   (FREE; rows 6..29 = 0)
       P[r,parity]=1 broadcasts each template row by parity into output rows 0..5.

  fp16 throughout (declared output fp16); the harness scores each plane at its declared
  dtype so every working plane is half-width.  No 30x30 working plane is ever materialised.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..builders import _model

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
I64 = TensorProto.INT64

WORK = 15  # active width: grid width <= 15, max content column is 14


def build(task):
    inits, nodes, vinfos = [], [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    def vi(name, dt, shape):
        vinfos.append(helper.make_tensor_value_info(name, dt, shape))
        return name

    # odd-row conv kernel (full channel mix), kernel cols [w-1, w, w+1]
    G = np.zeros((10, 10, 1, 3), np.float16)
    for ch in range(1, 10):                 # coloured channels: neighbour sum
        G[ch, ch, 0, 0] = 1.0
        G[ch, ch, 0, 2] = 1.0
    for k in range(10):                     # ch0 += SUM_k x[k,w]  (in-grid mask)
        G[0, k, 0, 1] = 1.0
    for k in range(1, 10):                  # ch0 -= colour neighbours
        G[0, k, 0, 0] = -1.0
        G[0, k, 0, 2] = -1.0
    init("G", G, np.float16)

    # parity broadcast P [30,2]: rows 0..5 only (6..29 stay zero)
    P = np.zeros((30, 2), np.float16)
    for r in range(6):
        P[r, r % 2] = 1.0
    init("P", P, np.float16)

    # slice row 0 AND crop width to WORK in one Slice
    init("starts", np.array([0, 0], np.int64), np.int64)
    init("ends", np.array([1, WORK], np.int64), np.int64)
    init("axes", np.array([2, 3], np.int64), np.int64)

    n("Slice", ["input", "starts", "ends", "axes"], "x32")
    vi("x32", F32, [1, 10, 1, WORK])
    n("Cast", ["x32"], "x", to=F16)
    vi("x", F16, [1, 10, 1, WORK])
    n("Conv", ["x", "G"], "odd", kernel_shape=[1, 3], pads=[0, 1, 0, 1])
    vi("odd", F16, [1, 10, 1, WORK])
    n("Concat", ["x", "odd"], "tpls", axis=2)
    vi("tpls", F16, [1, 10, 2, WORK])
    n("Pad", ["tpls"], "tpl", mode="constant",
      pads=[0, 0, 0, 0, 0, 0, 0, 30 - WORK])
    vi("tpl", F16, [1, 10, 2, 30])
    n("MatMul", ["P", "tpl"], "output")

    model = _model(nodes, inits, vinfos)
    model.graph.output[0].type.tensor_type.elem_type = F16
    return model
