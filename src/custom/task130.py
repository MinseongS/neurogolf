"""task130 (ARC-AGI 5614dbcf) — "downsample a 3x3 grid of 3x3 colour blocks, ignoring gray noise".

Rule (from the generator, size=3 so the active canvas is a fixed 9x9):
  The 9x9 input is partitioned into a FIXED 3x3 grid of 3x3 blocks.  Each block is EITHER
  a solid axis-aligned box filled with ONE non-gray colour (1..9, excluding gray=5) OR a
  noise block containing only background (0) plus a few scattered gray (5) pixels.  Gray
  pixels may also overwrite a handful of a box's cells.  OUTPUT is the 3x3 grid whose cell
  (R,C) is the colour of the box in block (R,C), or 0 if that block has no box.

  Because gray and background are the only non-box values, a block's box colour =
  (sum of colour-index over the block, EXCLUDING gray & bg) / (count of those cells).
  A box block always has >=1 non-gray box cell so count>0; a noise block has count==0 -> 0.

Encoding (fixed-stride downsample as TWO STRIDED CONVS — no full-size plane ever forms):
  The partition is a fixed 3x3 tiling of 3x3 blocks, so block reductions are a single
  stride-3 Conv with a 3x3 kernel:
    Snum = Conv(input, W_color, stride=3) -> [1,1,10,10]   (block sum of colour-index,
           W_color[k] = k for k!=0,5 else 0)  excludes bg & gray.
    Sden = Conv(input, W_occ,   stride=3) -> [1,1,10,10]   (block count of box cells,
           W_occ[k] = 1 for k!=0,5 else 0)
  colour = round(Snum/Sden) where Sden>0 else 0 (background).  The active output occupies
  the top-left 3x3 of the [10,10] grid; the rest is a noise-only / off-canvas region that
  the generator leaves empty, so it is padded with sentinel 99 (-> all-zero one-hot).
  output = Equal(L, arange[1,10,1,1]) -> BOOL [1,10,30,30].
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL

N = 30
K = 3      # output is 3x3 (size=3)
B = 3      # block size in pixels
G = N // B  # strided-conv grid = 10x10


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- stride-3 block reductions via two 3x3 convs (no [1,1,30,30] plane) --
    # colour weight: k for box colours, 0 for bg(0) & gray(5).
    wc = np.arange(10, dtype=np.float32)
    wc[5] = 0.0
    Wcolor = np.tile(wc.reshape(1, 10, 1, 1), (1, 1, B, B))   # [1,10,3,3]
    init("Wcolor", Wcolor, np.float32)
    # occupancy weight: 1 for box colours, 0 for bg & gray.
    wo = np.ones(10, dtype=np.float32)
    wo[0] = 0.0
    wo[5] = 0.0
    Wocc = np.tile(wo.reshape(1, 10, 1, 1), (1, 1, B, B))     # [1,10,3,3]
    init("Wocc", Wocc, np.float32)

    n("Conv", ["input", "Wcolor"], "Snum", strides=[B, B])   # f32 [1,1,10,10]
    n("Conv", ["input", "Wocc"], "Sden", strides=[B, B])     # f32 [1,1,10,10]

    # ---- colour = round(Snum / Sden), 0 (bg) where the block has no box ------
    n("Cast", ["Snum"], "SnumH", to=F16)                     # f16 [1,1,10,10]
    n("Cast", ["Sden"], "SdenH", to=F16)                     # f16 [1,1,10,10]
    init("ZEROH", np.array(0.0, np.float16), np.float16)
    init("ONEH", np.array(1.0, np.float16), np.float16)
    n("Greater", ["SdenH", "ZEROH"], "valid_b")              # bool [1,1,10,10]
    n("Where", ["valid_b", "SdenH", "ONEH"], "den_safe")     # avoid /0
    n("Div", ["SnumH", "den_safe"], "colr0")                 # f16 [1,1,10,10]
    n("Round", ["colr0"], "colr")                            # exact integer colour
    n("Where", ["valid_b", "colr", "ZEROH"], "Lf")           # 0 (bg) where no box
    n("Cast", ["Lf"], "Lg", to=TensorProto.UINT8)            # uint8 [1,1,10,10]

    # ---- keep the active top-left K x K only -------------------------------
    init("st0", np.array([0, 0], np.int64), np.int64)
    init("en0", np.array([K, K], np.int64), np.int64)
    init("ax23", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["Lg", "st0", "en0", "ax23"], "Lsmall")       # uint8 [1,1,K,K]

    # ---- one-hot the small K x K label, THEN pad into the FREE output -------
    # padding a small uint8 one-hot (90B) keeps the only 30x30 plane in the
    # free output; the off-grid region stays all-zero (background-empty).
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["Lsmall", "chan"], "oh_b")                   # bool [1,10,K,K]
    n("Cast", ["oh_b"], "oh_u", to=TensorProto.UINT8)        # uint8 [1,10,K,K]
    init("ZEROU", np.array(0, np.uint8), np.uint8)
    init("pads", np.array([0, 0, 0, 0, 0, 0, N - K, N - K], np.int64), np.int64)
    n("Pad", ["oh_u", "pads", "ZEROU"], "output", mode="constant")  # uint8 [1,10,30,30]

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task130", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
