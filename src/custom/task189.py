"""Task 189 (ARC-AGI 7c008303): quadrant-legend recolor of a green stamp.

Generator (size is always 6 -> input 9x9, output 6x6):
  A 2x2 legend of four colors sits in a corner, separated from a size x size
  region by a cyan(8) cross at index 2 (a full cyan row and a full cyan column).
  Each green(3) pixel in the size x size region is recolored by the QUADRANT it
  falls in (r,c vs size//2=3 -> one of the four legend cells), then a global
  flip_horiz / flip_vert is applied to BOTH the grid and the output.

Because the flips transform grid and output identically, everything can be read
in the *flipped frame* with no un-flipping:
  out[R][C] (R,C in 0..5) = legend_cell[R//3][C//3]   if green at (R+orr, C+orc)
                          = 0 (background)             otherwise
where the green-block offset orr/orc in {0,3} and the legend corner lr/lc in
{0,7} are fixed by which side of the cyan cross the legend sits on -- i.e. by
flip_vert (full cyan row at index 2 vs 6) and flip_horiz (full cyan col 2 vs 6).
Verified exact on all 266 stored examples and 200/200 fresh arc-gen instances.

Implementation (all integer-valued; exact in float32):
  - vflip = (full-cyan row is at index 6), hflip = (full-cyan col at 6), as 0/1
    scalars from the cyan(8) channel.
  - Flip-indexed (Gather) constant banks select rows/cols from the 30-wide
    canvas: Lr/Lc extract the 2x2 legend corner; Sr/Sc extract the 6x6 green
    block. The 2x2 legend is read into a canvas-index-ordered local block, then
    expanded to the 6x6 quadrant layout by *flip-independent* constants Kr/Kc.
  - L22 = Lr @ input @ Lc                 -> [1,10,2,2] legend colors (one-hot)
    le  = Kr @ L22 @ Kc                    -> [1,10,6,6] per-cell quadrant color
    gb  = Sr @ green @ Sc                  -> [1,1,6,6]  green-block mask
    out6 = gb*(le - onehot0) + onehot0     -> recolor green cells, bg elsewhere
  - Pad the 6x6 result to the 30x30 canvas (free `output`).

The recolor identity: ch0 = gb*(0-1)+1 = 1-gb (background where no green);
ch_k = gb*(le_k-0) = gb*le_k (legend never uses colors 0/3/8, so le_0 == 0).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..builders import _model


def _banks():
    # legend-corner extractors: [2(flip), 2(local), 30] and [2, 30, 2]
    Lr = np.zeros((2, 2, 30), np.float32)
    Lc = np.zeros((2, 30, 2), np.float32)
    # green-block extractors: [2, 6, 30] and [2, 30, 6]
    Sr = np.zeros((2, 6, 30), np.float32)
    Sc = np.zeros((2, 30, 6), np.float32)
    for q in range(2):
        Lr[0, q, 0 + q] = 1.0   # no flip: legend rows 0,1
        Lr[1, q, 7 + q] = 1.0   # flip:    legend rows 7,8
        Lc[0, 0 + q, q] = 1.0
        Lc[1, 7 + q, q] = 1.0
    for R in range(6):
        Sr[0, R, R + 3] = 1.0   # no flip: green rows 3..8
        Sr[1, R, R] = 1.0       # flip:    green rows 0..5
        Sc[0, R + 3, R] = 1.0
        Sc[1, R, R] = 1.0
    # legend 2x2 -> 6x6 quadrant expansion (flip-independent)
    Kr = np.zeros((6, 2), np.float32)
    Kc = np.zeros((2, 6), np.float32)
    for R in range(6):
        Kr[R, R // 3] = 1.0
        Kc[R // 3, R] = 1.0
    return Lr, Lc, Sr, Sc, Kr, Kc


def build(task):
    inits, nodes, vinfos = [], [], []

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

    # --- cyan(8) per-row / per-col counts -> vflip / hflip 0/1 scalars ---
    # A width-30 conv kernel selecting channel 8 sums cyan across each row in one
    # shot (-> [1,1,30,1]); a height-30 kernel does the same per column. This
    # avoids ever materialising a 30x30 cyan canvas.
    Wrow = np.zeros((1, 10, 1, 30), np.float32); Wrow[0, 8, 0, :] = 1.0
    init("Wrow", Wrow)
    Wcol = np.zeros((1, 10, 30, 1), np.float32); Wcol[0, 8, :, 0] = 1.0
    init("Wcol", Wcol)
    n("Conv", ["input", "Wrow"], "rowsum")                  # [1,1,30,1]
    vi("rowsum", TensorProto.FLOAT, [1, 1, 30, 1])
    n("Conv", ["input", "Wcol"], "colsum")                  # [1,1,1,30]
    vi("colsum", TensorProto.FLOAT, [1, 1, 1, 30])

    # The full-cyan line is at index 6 exactly when the flip moved it there, so
    # vflip = (cyan count of row 6 == 9), hflip = (cyan count of col 6 == 9).
    # Gather index 6 off the count vectors, threshold, cast to a scalar index.
    init("idx6", np.array(6, np.int64), dtype=np.int64)
    init("eight5", np.array(8.5, np.float32))
    n("Gather", ["rowsum", "idx6"], "row6", axis=2)         # [1,1,1,1]
    vi("row6", TensorProto.FLOAT, [1, 1, 1, 1])
    n("Gather", ["colsum", "idx6"], "col6", axis=3)         # [1,1,1,1]
    vi("col6", TensorProto.FLOAT, [1, 1, 1, 1])
    n("Greater", ["row6", "eight5"], "vbool")               # [1,1,1,1] bool
    vi("vbool", TensorProto.BOOL, [1, 1, 1, 1])
    n("Greater", ["col6", "eight5"], "hbool")               # [1,1,1,1] bool
    vi("hbool", TensorProto.BOOL, [1, 1, 1, 1])
    n("Cast", ["vbool"], "vflip4", to=TensorProto.INT64)
    vi("vflip4", TensorProto.INT64, [1, 1, 1, 1])
    n("Cast", ["hbool"], "hflip4", to=TensorProto.INT64)
    vi("hflip4", TensorProto.INT64, [1, 1, 1, 1])
    n("ReduceSum", ["vflip4"], "vflip", axes=[0, 1, 2, 3], keepdims=0)  # scalar
    vi("vflip", TensorProto.INT64, [])
    n("ReduceSum", ["hflip4"], "hflip", axes=[0, 1, 2, 3], keepdims=0)  # scalar
    vi("hflip", TensorProto.INT64, [])

    # --- pick flip-specific extractor matrices ---
    Lr, Lc, Sr, Sc, Kr, Kc = _banks()
    init("LrB", Lr); init("LcB", Lc); init("SrB", Sr); init("ScB", Sc)
    init("Kr", Kr, dtype=np.float16); init("Kc", Kc, dtype=np.float16)
    n("Gather", ["LrB", "vflip"], "lr", axis=0)             # [2,30]
    vi("lr", TensorProto.FLOAT, [2, 30])
    n("Gather", ["LcB", "hflip"], "lc", axis=0)             # [30,2]
    vi("lc", TensorProto.FLOAT, [30, 2])
    n("Gather", ["SrB", "vflip"], "sr", axis=0)             # [6,30]
    vi("sr", TensorProto.FLOAT, [6, 30])
    n("Gather", ["ScB", "hflip"], "sc", axis=0)             # [30,6]
    vi("sc", TensorProto.FLOAT, [30, 6])

    # --- legend 2x2 -> per-cell quadrant color le [1,10,6,6] ---
    n("MatMul", ["lr", "input"], "lrow")                   # [1,10,2,30]
    vi("lrow", TensorProto.FLOAT, [1, 10, 2, 30])
    n("MatMul", ["lrow", "lc"], "L22f")                    # [1,10,2,2]
    vi("L22f", TensorProto.FLOAT, [1, 10, 2, 2])
    n("Cast", ["L22f"], "L22", to=TensorProto.FLOAT16)     # fp16 from here
    vi("L22", TensorProto.FLOAT16, [1, 10, 2, 2])
    n("MatMul", ["Kr", "L22"], "lemid")                    # [1,10,6,2] fp16
    vi("lemid", TensorProto.FLOAT16, [1, 10, 6, 2])
    n("MatMul", ["lemid", "Kc"], "le")                     # [1,10,6,6] fp16
    vi("le", TensorProto.FLOAT16, [1, 10, 6, 6])

    # --- green-block mask gb [1,1,6,6] ---
    Wgreen = np.zeros((1, 10, 1, 1), np.float32); Wgreen[0, 3, 0, 0] = 1.0
    init("Wgreen", Wgreen)
    n("Conv", ["input", "Wgreen"], "greenf")               # [1,1,30,30]
    vi("greenf", TensorProto.FLOAT, [1, 1, 30, 30])
    n("MatMul", ["sr", "greenf"], "grow")                  # [1,1,6,30]
    vi("grow", TensorProto.FLOAT, [1, 1, 6, 30])
    n("MatMul", ["grow", "sc"], "gbf")                     # [1,1,6,6]
    vi("gbf", TensorProto.FLOAT, [1, 1, 6, 6])
    n("Cast", ["gbf"], "gb", to=TensorProto.FLOAT16)
    vi("gb", TensorProto.FLOAT16, [1, 1, 6, 6])

    # --- recolor: out6 = gb*(le - onehot0) + onehot0  (fp16) ---
    oh0 = np.zeros((1, 10, 1, 1), np.float16); oh0[0, 0, 0, 0] = 1.0
    init("oh0", oh0, dtype=np.float16)
    n("Sub", ["le", "oh0"], "lediff")                      # [1,10,6,6] fp16
    vi("lediff", TensorProto.FLOAT16, [1, 10, 6, 6])
    n("Mul", ["gb", "lediff"], "scaled")                   # [1,10,6,6] fp16
    vi("scaled", TensorProto.FLOAT16, [1, 10, 6, 6])
    n("Add", ["scaled", "oh0"], "out6h")                   # [1,10,6,6] fp16
    vi("out6h", TensorProto.FLOAT16, [1, 10, 6, 6])
    n("Cast", ["out6h"], "out6", to=TensorProto.FLOAT)     # back to fp32
    vi("out6", TensorProto.FLOAT, [1, 10, 6, 6])

    # --- pad 6x6 -> 30x30 (free output) ---
    n("Pad", ["out6"], "output", mode="constant", value=0.0,
      pads=[0, 0, 0, 0, 0, 0, 24, 24])

    return _model(nodes, inits, vinfos)
