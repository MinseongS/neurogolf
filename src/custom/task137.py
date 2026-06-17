"""Task 137 (5c2c9af4): concentric square rings from 3 diagonal pixels.

Rule (from ARC-GEN generator): input is a size x size grid (size in 20..30)
holding exactly 3 pixels of one colour at rows {row-s, row, row+s} and the
matching diagonal columns (flip only swaps which diagonal). Output: cell (r,c)
inside the grid is `color` iff max(|r-row|, |c-col|) % s == 0, else black;
outside the grid the whole canvas is 0.

Encoding -- the whole problem reduces to 1-D vectors plus exactly TWO
canvas-sized intermediates (a bool comparison + a uint8 index plane):
  - colour scalar: the single colour appears exactly 3x; count==3 one-hot
    -> weighted sum -> uint8 scalar `color`.
  - per-row / per-col pixel counts via two channel+axis Convs (ch0 weight 0)
    rowpix[1,1,30,1], colpix[1,1,1,30]; everything downstream is fp16.
  - row = (Sum r*rowpix)/3 ; s = max(r:rowpix>0) - row ; col = (Sum c*colpix)/3.
  - size = sqrt(total one-hot count) -> g_r = iota_r<size, g_c = iota_c<size.
  - |dr| = |iota_r - row|, |dc| = |iota_c - col| (fp16); fp16 fmod gives the
    per-axis ring bit zr0/zc0 = (|d| mod s == 0), integer-exact (<2048).
  - 1-D index vectors: lr = ring?color:0, then off-grid row -> sentinel 10
    (matches no channel 0..9 so Equal -> all-off -> black); lc symmetric.
  - force the off-grid axis to dominate: dr2 = g_r?|dr|:99, dc2 likewise, so an
    off-grid row makes dr2=99 win the max and route to lr's sentinel 10.
  - dlt = Less(dr2, dc2)  [1,1,30,30] bool        (canvas plane #1)
    Lidx = Where(dlt, lc, lr)  [1,1,30,30] uint8   (canvas plane #2)
  - output = Equal(Lidx, arange(10))  BOOL [1,10,30,30]  (FREE output).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

I32 = onnx.TensorProto.INT32
F32 = onnx.TensorProto.FLOAT
F16 = onnx.TensorProto.FLOAT16
F16NP = np.float16
BOOL = onnx.TensorProto.BOOL


def build(task):
    inits = []
    nodes = []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    # ---- constants ----
    wr = np.zeros((1, 10, 1, 30), np.float32)
    wr[0, 1:, 0, :] = 1.0
    init("Wr", wr, np.float32)                          # per-row pixel count
    init("Wc", wr.reshape(1, 10, 30, 1), np.float32)    # per-col pixel count
    init("iotaR", np.arange(30, dtype=np.float16).reshape(1, 1, 30, 1), F16NP)
    init("iotaC", np.arange(30, dtype=np.float16).reshape(1, 1, 1, 30), F16NP)
    init("chidx", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    init("chanU8", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    init("c0u8", np.array(0, np.uint8), np.uint8)
    init("c10u8", np.array(10, np.uint8), np.uint8)  # off-grid sentinel
    init("c3i", np.array(3, np.int32), np.int32)
    init("c3f", np.array(3.0, np.float16), F16NP)
    init("c0f16", np.array(0.0, np.float16), F16NP)
    init("c99f16", np.array(99.0, np.float16), F16NP)

    # ---- colour scalar: the single colour appears exactly 3 times ----
    n("ReduceSum", ["input"], "chcnt", axes=[2, 3], keepdims=1)  # [1,10,1,1]
    n("Cast", ["chcnt"], "chcnt_i", to=I32)
    n("Equal", ["chcnt_i", "c3i"], "color1h_b")                  # bool [1,10,1,1]
    n("Cast", ["color1h_b"], "color1h", to=F32)                  # 1.0 on colour
    n("Mul", ["color1h", "chidx"], "colwt")                      # k on colour ch
    n("ReduceSum", ["colwt"], "colorS", keepdims=0)              # scalar colour
    n("Cast", ["colorS"], "colorU8", to=onnx.TensorProto.UINT8)  # uint8 scalar

    # ---- center (row, col) and spacing s from the 3 pixels ----
    n("Conv", ["input", "Wr"], "rowpix")               # [1,1,30,1] fp32
    n("Conv", ["input", "Wc"], "colpix")               # [1,1,1,30] fp32
    n("Cast", ["rowpix"], "rp", to=F16)                # [1,1,30,1] fp16
    n("Cast", ["colpix"], "cp", to=F16)                # [1,1,1,30] fp16
    n("Mul", ["rp", "iotaR"], "rprod")                 # [1,1,30,1] fp16
    n("ReduceSum", ["rprod"], "srow", keepdims=0)      # 3*row
    n("ReduceMax", ["rprod"], "mrow", keepdims=0)      # row+s (max occupied row)
    n("Div", ["srow", "c3f"], "rowc")                  # row
    n("Sub", ["mrow", "rowc"], "s_f")                  # s
    n("Mul", ["cp", "iotaC"], "cprod")                 # [1,30]
    n("ReduceSum", ["cprod"], "scol", keepdims=0)      # 3*col
    n("Div", ["scol", "c3f"], "colc")                  # col

    # ---- grid size -> in-grid masks ----
    n("ReduceSum", ["input"], "ntot", keepdims=0)      # size^2 fp32
    n("Sqrt", ["ntot"], "size32")
    n("Cast", ["size32"], "size", to=F16)              # size (<=30, fp16 exact)
    n("Less", ["iotaR", "size"], "g_r")                # [30,1] bool
    n("Less", ["iotaC", "size"], "g_c")                # [1,30] bool

    # ---- bounded |dr|, |dc| ----
    n("Sub", ["iotaR", "rowc"], "drs")
    n("Abs", ["drs"], "dr")                            # [30,1]
    n("Sub", ["iotaC", "colc"], "dcs")
    n("Abs", ["dcs"], "dc")                            # [1,30]

    # ---- ring-on-axis test: dist mod s == 0, gated in-grid ----
    # fp16 fmod is integer-exact for these small positive ints (<2048).
    n("Mod", ["dr", "s_f"], "mr", fmod=1)              # [30,1] fp16
    n("Mod", ["dc", "s_f"], "mc", fmod=1)              # [1,30] fp16
    n("Equal", ["mr", "c0f16"], "zr0")                 # [30,1] bool
    n("Equal", ["mc", "c0f16"], "zc0")                 # [1,30] bool
    # per-axis colour-index VECTORS (1-D): ring->colour, off-grid->sentinel 10.
    # Off-grid sentinel 10 matches no channel 0..9 so Equal -> all-off -> black 0.
    n("Where", ["zr0", "colorU8", "c0u8"], "lr0")      # [.,.,30,1] uint8 ring colour
    n("Where", ["g_r", "lr0", "c10u8"], "lr")          # off-grid row -> 10
    n("Where", ["zc0", "colorU8", "c0u8"], "lc0")      # [.,.,1,30] uint8
    n("Where", ["g_c", "lc0", "c10u8"], "lc")          # off-grid col -> 10

    # ---- bounded distances: force off-grid axis to dominate (->its sentinel) --
    n("Where", ["g_r", "dr", "c99f16"], "dr2")         # off-grid row dist -> 99
    n("Where", ["g_c", "dc", "c99f16"], "dc2")         # off-grid col dist -> 99

    # ---- dominating axis picks which 1-D vector applies (only 2 canvas planes)
    # max distance axis: dr2<dc2 -> column vector (lc), else row vector (lr).
    # off-grid row -> dr2=99 dominates -> picks lr=10; off-grid col symmetric.
    n("Less", ["dr2", "dc2"], "dlt")                   # [.,.,30,30] bool  plane#1
    n("Where", ["dlt", "lc", "lr"], "Lidx")            # [.,.,30,30] uint8 plane#2

    # ---- route 10-ch one-hot into the FREE BOOL output ----
    n("Equal", ["Lidx", "chanU8"], "output")           # BOOL [1,10,30,30] FREE

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task137", [x], [y], inits)
    return helper.make_model(
        g, ir_version=10, opset_imports=[helper.make_opsetid("", 11)])
