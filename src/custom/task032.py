"""task032 (ARC-AGI 1e0a9b12) — per-column gravity, one colour per column.

Rule (from the generator): an s x s grid (s in 4..6) sits at the top-left of the
30x30 canvas.  Each column c holds exactly ONE colour colors[c] painted in cnt[c]
arbitrary rows (colors[c] MAY be 0).  The output drops those cnt[c] cells to the
BOTTOM of the column: output(r,c) is coloured iff in-grid (r<s, c<s) AND
r >= s-cnt[c]; colour = colors[c].

Input is one-hot [1,10,30,30] (channel k = 1 where colour k); off-grid cells are
all-zero; in-grid background cells set ch0=1.

KEY OBSERVATION (scoring is out>0): a colour-0 column writes value 0 to its
active cells == background, so colour-0 columns are INDISTINGUISHABLE from empty
columns in BOTH input and output.  We therefore only ever handle colour>=1
columns; colour-0 columns fall out as all-background automatically.

CROP-TO-ACTIVE: the grid is at most 6x6 at the top-left, so every WORKING plane is
built at WORK=6 (36 cells) instead of 30x30.  The final small one-hot [1,10,6,6]
is Pad'ed back to [1,10,30,30] (uint8 Pad).  We deliberately do NOT slice the input
to [1,10,6,6] (that f32 window costs 1440B); instead the per-column scalars come
from a full-height Conv on the free input, and only the cheap 30-wide vector
outputs are sliced to width 6.

Per-column scalars come from ONE no-pad Conv W[2,10,30,1] -> [1,2,1,30] (240B):
  ch0 weight W[0,k,r]=k          -> colsum[c] = colour * cnt   (per column)
  ch1 weight W[1,k,r]=1 (k>=1)   -> cnt[c]   = # colour>=1 cells in column c
colidx[c] = colsum / max(cnt,1)  (exact: colsum=colour*cnt, small integers).
The extra conv PARAMS (600 vs 120 for a cropped-input conv) cost far less than
materialising the 1440B [1,10,6,6] input window would.

in-grid: incol[c] = ReduceMax(input,[1,2]) sliced to width 6 (off-grid cols -> 0);
s = ReduceSum(incol) (square grid).  Coloured region is ONE 6x6 bool plane:
  botc[r] = ((s-1)-r < 0) ? 99 : (s-1)-r   ([1,1,6,1] vector, folds r<s)
  coloured = Less(botc, cnt2)              ([1,1,6,6] bool)   -- r in [s-cnt, s)
off-grid columns get cnt2=100 -> coloured everywhere but route to sentinel 99.

Route into output with ONE Where (no second full plane), then Pad to 30x30:
  Xonehot[k,c] = (colidx99[c]==k)          ([1,10,1,6])  per-column colour one-hot
  Yonehot[k,r] = (k==0) AND (r<s)          ([1,10,6,1])  bg channel for in-grid rows
  small = Where(coloured, Xonehot, Yonehot) ([1,10,6,6] u8)
  output = Pad(small, ... 99..)            ([1,10,30,30] u8)  (off-grid stays 0)

Dominant intermediate: the [1,10,6,6] f32 input Slice (1440B, Slice preserves the
fp32 input dtype; irreducible since the colour Conv needs all 10 channels at the
6x6 footprint).  Every downstream plane is <=360B.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
B = TensorProto.BOOL

W = 6  # grid is at most 6x6 at the top-left of the 30x30 canvas


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def nd(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- per-column colour-sum + count via ONE no-pad Conv on FULL input ---
    # Full-height kernel W[2,10,30,1] -> [1,2,1,30] (240B, NO 1440B input slice);
    # the extra conv params (600 vs 120 for a cropped conv) are far cheaper than
    # materialising a [1,10,6,6] f32 window.  We then Slice the 30-wide vectors
    # to the 6-wide active window for free.
    cw = np.zeros((2, 10, 30, 1), np.float32)
    for k in range(10):
        cw[0, k, :, 0] = k
        if k >= 1:
            cw[1, k, :, 0] = 1.0
    init("convW", cw, np.float32)
    nd("Conv", ["input", "convW"], "cc30_f32")          # [1,2,1,30] f32
    init("s0", np.array([0], np.int64), np.int64)
    init("s1", np.array([1], np.int64), np.int64)
    init("s2", np.array([2], np.int64), np.int64)
    init("sW", np.array([W], np.int64), np.int64)
    init("ax1", np.array([1], np.int64), np.int64)
    init("ax3", np.array([3], np.int64), np.int64)
    nd("Slice", ["cc30_f32", "s0", "sW", "ax3"], "cc_f32")  # [1,2,1,6] f32
    nd("Cast", ["cc_f32"], "cc", to=F16)                # [1,2,1,6] f16
    nd("Slice", ["cc", "s0", "s1", "ax1"], "colsum")    # [1,1,1,6] f16
    nd("Slice", ["cc", "s1", "s2", "ax1"], "cnt")       # [1,1,1,6] f16
    init("one16", np.array(1.0, np.float16), np.float16)
    nd("Max", ["cnt", "one16"], "cntpos")
    nd("Div", ["colsum", "cntpos"], "colidx")           # [1,1,1,6] f16 = colour

    # ---- in-grid bounds ----------------------------------------------------
    nd("ReduceMax", ["input"], "incol30", axes=[1, 2], keepdims=1)  # [1,1,1,30]
    nd("Slice", ["incol30", "s0", "sW", "ax3"], "incol")  # [1,1,1,6] f32
    nd("ReduceSum", ["incol"], "s_f", axes=[3], keepdims=1)     # [1,1,1,1] = s
    nd("Cast", ["s_f"], "s16", to=F16)
    rr = np.arange(W, dtype=np.float16).reshape(1, 1, W, 1)
    init("rr", rr, np.float16)
    init("half", np.array(0.5, np.float32), np.float32)
    nd("Greater", ["incol", "half"], "colin")           # [1,1,1,6] bool (in-grid col)

    # ---- coloured region (one 6x6 bool plane) ------------------------------
    init("smb16", np.array(-1.0, np.float16), np.float16)
    nd("Add", ["s16", "smb16"], "sm1")                  # s-1
    nd("Sub", ["sm1", "rr"], "botdist")                 # (s-1)-r  [1,1,6,1]
    init("zero16", np.array(0.0, np.float16), np.float16)
    init("big16", np.array(99.0, np.float16), np.float16)
    nd("Less", ["botdist", "zero16"], "botneg")
    nd("Where", ["botneg", "big16", "botdist"], "botc")  # [1,1,6,1] f16
    init("big100", np.array(100.0, np.float16), np.float16)
    nd("Where", ["colin", "cnt", "big100"], "cnt2")     # [1,1,1,6] f16
    nd("Less", ["botc", "cnt2"], "coloured")            # [1,1,6,6] bool

    # ---- route into a small one-hot then Pad to 30x30 ----------------------
    init("f99", np.array(99.0, np.float16), np.float16)
    nd("Where", ["colin", "colidx", "f99"], "colidx99f")  # [1,1,1,6] f16
    nd("Cast", ["colidx99f"], "colidx99", to=U8)        # [1,1,1,6] u8
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    nd("Equal", ["colidx99", "chan"], "Xonehot_b")      # [1,10,1,6] bool
    nd("Cast", ["Xonehot_b"], "Xonehot", to=U8)         # [1,10,1,6] u8
    nd("Less", ["rr", "s16"], "rowin")                  # [1,1,6,1] bool (r<s)
    is_ch0u = np.zeros((1, 10, 1, 1), np.uint8); is_ch0u[0, 0] = 1
    init("is_ch0u", is_ch0u, np.uint8)
    init("u0", np.array(0, np.uint8), np.uint8)
    nd("Where", ["rowin", "is_ch0u", "u0"], "Yonehot")  # [1,10,6,1] u8
    nd("Where", ["coloured", "Xonehot", "Yonehot"], "small")  # [1,10,6,6] u8
    # Pad the 6x6 one-hot back to 30x30; off-grid cells stay 0 (all channels off).
    init("pads",
         np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), np.int64)
    nd("Pad", ["small", "pads"], "output", mode="constant")  # [1,10,30,30] u8

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", U8, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task032", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
