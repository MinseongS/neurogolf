"""Task 397 (ARC-AGI fcc82909) — green shadow under each 2x2 colour box.

Rule (from generator):
  * 2-3 boxes, each a 2x2 block of colours at (brows[idx], bcols[idx]).  Box
    columns are width-2, non-overlapping (sorted bcols, spacing>=0 so two boxes
    may be column-ADJACENT).  Column-adjacent boxes always differ in top row by
    >=2 (generator `touching` guard on |brows[i]-brows[i+1]|).
  * Output = input copied, PLUS a green (colour 3) "shadow" directly below each
    box: a 2-wide x H-tall block starting at row brows+2, where
        H = shadow = len(set(box's 4 colours))   (distinct colour count, 1..4).

Active region: size=10 grid -> cols 0..9, box top<=6, shadow reaches row <=11.
We crop the working canvas to ROWS 0..11 (W rows) x COLS 0..9 (W cols); only the
final shadow mask is padded back to 30x30 for the FREE Where output.

Encoding (per-column vectors; one fp32 occupancy + one fp32 colour-presence
entry, everything else fp16; 10-ch expansion routed into the FREE Where output):
  * occf = Conv(input, ones_ch1..9) -> [1,1,30,30] fp32 occupancy (one-hot=>{0,1}).
  * crop occf to [1,1,WR,WC]; top30[c] = max_r (WR-1-r? no: use 30-r) ... we use
    a row weight (BIG-r) so max over rows gives BIG-toprow per column.
  * colhas[k,c] = colour k present in column c = ReduceMax over rows of input,
    cropped to WC cols [1,10,1,WC].
  * same[c] = cols (c,c+1) are one box (both occupied AND equal top row).
  * merged[k,c] = colhas[k,c] OR same[c]*colhas[k,c+1] OR same[c-1]*colhas[k,c-1]
    -> each column sees the FULL colour set of its box.
  * dcount[c] = #{k>=1 : merged[k,c]>0}  (= shadow height, same for both columns).
  * shadow band: low=top+2, high=top+1+dcount ; band = (r>=low) AND (r<=high)
    computed as Min(r-low, high-r) >= 0 on the small canvas.  Unoccupied columns
    get low>high -> empty.  Pad band (fp16) to 30x30 -> Greater -> bool condition.
  * output = Where(band30, green_onehot[1,10,1,1], input).
All values are small ints, exact in fp16 (<2048).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
I64 = TensorProto.INT64
BOOL = TensorProto.BOOL

N = 30
WR = 12   # working rows 0..11
WC = 10   # working cols 0..9
BIG = 50  # row weight base (BIG - r), keeps top extraction in fp16-exact range


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---------- constants ----------
    occw = np.ones((1, 10, 1, 1), np.float32)
    occw[0, 0, 0, 0] = 0.0
    init("occw", occw, np.float32)
    init("zero", np.array(0.0, np.float32), np.float32)
    init("half16", np.array(0.5, np.float16), np.float16)
    init("neghalf16", np.array(-0.5, np.float16), np.float16)

    # row weight (BIG - r) over WR rows, and plain row ramp (r) over WR rows
    rw = (BIG - np.arange(WR, dtype=np.float16)).reshape(1, 1, WR, 1)
    init("rowwt", rw, np.float16)                          # BIG - r
    rr = np.arange(WR, dtype=np.float16).reshape(1, 1, WR, 1)
    init("rramp", rr, np.float16)                          # r
    init("cLow", np.array(BIG + 2.0, np.float16), np.float16)   # low = BIG+2 - top30
    init("cHigh", np.array(BIG + 1.0, np.float16), np.float16)  # high base = BIG+1 - top30

    # slice params (crop occf to [1,1,WR,WC]; colhas to [1,10,1,WC])
    init("st_occ", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("en_occ", np.array([1, 1, WR, WC], np.int64), np.int64)
    init("ax_occ", np.array([0, 1, 2, 3], np.int64), np.int64)
    init("st_col", np.array([0], np.int64), np.int64)
    init("en_col", np.array([WC], np.int64), np.int64)
    init("ax_col", np.array([3], np.int64), np.int64)

    # column neighbour shifts via Slice+Pad on the WC-wide axis (no big matrices).
    # right neighbour V[c+1]: drop col0, pad zero at end.
    init("st_drop0", np.array([1], np.int64), np.int64)
    init("en_dropL", np.array([WC], np.int64), np.int64)
    init("ax3", np.array([3], np.int64), np.int64)
    init("pad_end", np.array([0, 0, 0, 0, 0, 0, 0, 1], np.int64), np.int64)
    init("pad_beg", np.array([0, 0, 0, 1, 0, 0, 0, 0], np.int64), np.int64)
    init("pad0_16", np.array(0.0, np.float16), np.float16)
    init("st_dropL", np.array([0], np.int64), np.int64)
    init("en_drop0", np.array([WC - 1], np.int64), np.int64)

    chmask = np.ones((1, 10, 1, 1), np.float16)
    chmask[0, 0, 0, 0] = 0.0
    init("chmask", chmask, np.float16)

    gv = np.zeros((1, 10, 1, 1), np.float32)
    gv[0, 3, 0, 0] = 1.0
    init("greenv", gv, np.float32)

    # final pad of the small band [1,1,WR,WC] -> [1,1,30,30]
    init("pad_band", np.array([0, 0, 0, 0, 0, 0, N - WR, N - WC], np.int64), np.int64)
    init("padbig16", np.array(-9.0, np.float16), np.float16)  # padded region -> not selected

    def shiftR(V, tag):    # V[...,c+1]
        n("Slice", [V, "st_drop0", "en_dropL", "ax3"], tag + "_s")
        return n("Pad", [tag + "_s", "pad_end", "pad0_16"], tag, mode="constant")

    def shiftL(V, tag):    # V[...,c-1]
        n("Slice", [V, "st_dropL", "en_drop0", "ax3"], tag + "_s")
        return n("Pad", [tag + "_s", "pad_beg", "pad0_16"], tag, mode="constant")

    # ---------- occupancy entry plane + crop ----------
    n("Conv", ["input", "occw"], "occf")                  # [1,1,30,30] fp32 {0,1}
    n("Slice", ["occf", "st_occ", "en_occ", "ax_occ"], "occs32")  # [1,1,WR,WC]
    n("Cast", ["occs32"], "occs", to=F16)

    # ---------- per-column top (BIG - toprow) ----------
    n("Mul", ["occs", "rowwt"], "occ_rw")                 # [1,1,WR,WC] fp16
    n("ReduceMax", ["occ_rw"], "top30", axes=[2], keepdims=1)  # [1,1,1,WC] = BIG-top
    n("Greater", ["top30", "half16"], "occc_b")           # column occupied?
    n("Cast", ["occc_b"], "occc", to=F16)

    # ---------- per-(channel,column) colour presence ----------
    n("ReduceMax", ["input"], "colhas32", axes=[2], keepdims=1)  # [1,10,1,30] fp32
    n("Slice", ["colhas32", "st_col", "en_col", "ax_col"], "colhasS")  # [1,10,1,WC]
    n("Cast", ["colhasS"], "colhas", to=F16)

    # ---------- same-box adjacency: (c,c+1) equal top AND col c occupied ----------
    # Two adjacent occupied columns with equal top are necessarily ONE box (a
    # column-adjacent foreign box differs in top by >=2).  Equal top with c
    # occupied is sufficient: if c+1 were unoccupied its top30=0 != top30[c]>0.
    shiftR("top30", "top30_r")
    n("Sub", ["top30", "top30_r"], "dtop")
    n("Mul", ["dtop", "dtop"], "dtop2")
    n("Less", ["dtop2", "half16"], "eqtop_b")             # |dtop|<1 -> equal top
    n("Cast", ["eqtop_b"], "eqtop", to=F16)
    n("Mul", ["eqtop", "occc"], "same")                   # [1,1,1,WC] {0,1}
    shiftL("same", "same_l")                              # same[c-1]

    # ---------- merge colour presence across the box's two columns ----------
    shiftR("colhas", "colhas_r")
    shiftL("colhas", "colhas_l")
    n("Mul", ["colhas_r", "same"], "from_r")
    n("Mul", ["colhas_l", "same_l"], "from_l")
    n("Add", ["colhas", "from_r"], "m0")
    n("Add", ["m0", "from_l"], "merged")                  # >=1 where present
    n("Greater", ["merged", "half16"], "mpres_b")
    n("Cast", ["mpres_b"], "mpres", to=F16)

    # ---------- distinct colour count per column ----------
    n("Mul", ["mpres", "chmask"], "mpres1")
    n("ReduceSum", ["mpres1"], "dcount", axes=[1], keepdims=1)  # [1,1,1,WC]

    # ---------- shadow row bounds ----------
    # top = BIG - top30 ; low = top+2 = (BIG+2) - top30 ; high = top+1+dcount.
    n("Sub", ["cLow", "top30"], "low")                    # [1,1,1,WC]
    n("Sub", ["cHigh", "top30"], "tmp_hi")
    n("Add", ["tmp_hi", "dcount"], "high")

    # ---------- band on small canvas: Min(r-low, high-r) >= 0 ----------
    n("Sub", ["rramp", "low"], "t1")                      # [1,1,WR,WC] fp16
    n("Sub", ["high", "rramp"], "t2")
    n("Min", ["t1", "t2"], "mn")                          # >=0 inside band
    # pad to 30x30 with a negative fill so off-region is never selected
    n("Pad", ["mn", "pad_band", "padbig16"], "mn30", mode="constant")  # [1,1,30,30] fp16
    n("Greater", ["mn30", "neghalf16"], "band_b")         # bool [1,1,30,30]

    # ---------- final output ----------
    n("Where", ["band_b", "greenv", "input"], "output")   # [1,10,30,30] fp32 FREE

    x = helper.make_tensor_value_info("input", F32, [1, 10, N, N])
    y = helper.make_tensor_value_info("output", F32, [1, 10, N, N])
    g = helper.make_graph(nodes, "task397", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
