"""Task 025 (ARC-AGI 1a07d186) — move each dot beside its same-colour line.

Rule (from the generator, vertical orientation = xpose 0):
  * Several full vertical LINES sit at columns ``linecols`` with distinct colours
    ``linecolors`` (a colour appears as exactly one full column).  Lines are kept.
  * Scattered DOTS (single coloured cells off the lines) each have a colour.  A
    dot whose colour matches a line is MOVED onto the cell immediately adjacent to
    that line, on the same row, on the side facing the dot:
        out[r][lc-1] = colour   if dot col < lc
        out[r][lc+1] = colour   if dot col > lc
    The dot disappears from its original cell.  A dot whose colour matches NO line
    (an "extra" colour the generator scatters) is simply ERASED.
  * ``xpose`` transposes BOTH input and output, so lines may instead be full ROWS.
    The rule is transpose-equivariant: solve(T(g)) == T(solve(g)).  We compute the
    vertical-orientation label map on the input AND on its transpose, then select
    by which orientation actually has full lines.

Encoding (memory floor-break, label map + final Equal):
  * Per channel k, the line column indicator Lcol[1,10,1,30] = (colcount_k == H)
    where colcount_k = ReduceSum_rows(X), H = grid height (max column height).
  * Dots D = X with line cells removed.  cumge[k,c] = [c >= lc_k] via Lcol @ Utri.
    leftmask = [c < lc_k], rightmask = [c > lc_k].
  * hasleft[r,k] = any dot of colour k left of its line in row r (ReduceMax over
    cols of D*leftmask); hasright analogous.
  * Output planes (all rank-1 separable per channel, disjoint across colours):
        line_k   = ones_row  (x) Lcol_k
        left_k   = hasleft_k (x) shiftR(Lcol_k)   (target col lc-1)
        right_k  = hasright_k(x) shiftL(Lcol_k)   (target col lc+1)
    Label L[1,1,30,30] = sum_k k * (line_k + left_k + right_k).  Off-grid -> 10.
  * BOOL output = Equal(L, arange[1,10,1,1]) (opset 11).
All values are small ints, exact in float32; combined into one uint8 label plane.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    N = 30

    # ---------- constants ----------
    init("half", np.array(0.5, np.float32), np.float32)
    arangek = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("arangek", arangek, np.float32)            # colour weights per channel
    chan_u = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan_u", chan_u, np.uint8)                # final Equal arange
    # upper-tri (inclusive): Utri[a,b] = [a <= b]; cumge[c] = sum_a Lcol[a]*Utri[a,c]
    Utri = np.triu(np.ones((N, N), np.float32))     # [N,N], Utri[a,b]=1 if a<=b
    init("Utri", Utri, np.float32)
    # column shift matrices: (V @ Sr)[c] = V[c+1] -> shiftR (target lc-1)
    Sr = np.zeros((N, N), np.float32)
    for c in range(N - 1):
        Sr[c + 1, c] = 1.0                          # out[c] = in[c+1]
    init("Sr", Sr, np.float32)
    Sl = np.zeros((N, N), np.float32)
    for c in range(1, N):
        Sl[c - 1, c] = 1.0                          # out[c] = in[c-1]
    init("Sl", Sl, np.float32)

    def vertical_branch(X, tag):
        """Build the vertical-orientation label map L[1,1,30,30] (fp32, sentinel
        10 off-grid) from one-hot input X[1,10,30,30]."""
        def t(s):
            return f"{tag}_{s}"

        # in-grid mask + grid height H
        n("ReduceMax", [X], t("ingrid"), axes=[1], keepdims=1)        # [1,1,30,30]
        n("ReduceSum", [t("ingrid")], t("colheight"), axes=[2], keepdims=1)  # [1,1,1,30]
        n("ReduceMax", [t("colheight")], t("H"), axes=[3], keepdims=1)       # [1,1,1,1]

        # per-channel column counts and line indicator
        n("ReduceSum", [X], t("colcnt"), axes=[2], keepdims=1)        # [1,10,1,30]
        n("Sub", [t("colcnt"), t("H")], t("ccmH"))                    # ==0 on lines
        n("Mul", [t("ccmH"), t("ccmH")], t("ccmH2"))                  # 0 iff line
        # Lcol = (colcnt == H) -> (ccmH2 < 0.5)
        n("Less", [t("ccmH2"), "half"], t("Lcol_b"))                  # bool [1,10,1,30]
        n("Cast", [t("Lcol_b")], t("Lcol"), to=F32)                   # fp32 {0,1}
        # also require column to be colored (H>0); H>0 always for in-grid, but a
        # fully-background column would set Lcol for ch0 (k=0, harmless in label).

        # dots: colored cell whose column is NOT a line for that channel
        n("Sub", ["one_f", t("Lcol")], t("notline"))                  # [1,10,1,30]
        n("Mul", [X, t("notline")], t("D"))                           # [1,10,30,30]

        # cumge[k,c] = [c >= lc_k]  via Lcol @ Utri  (contract col axis)
        n("MatMul", [t("Lcol"), "Utri"], t("cumge"))                  # [1,10,1,30]
        # leftmask = [c < lc_k] = 1 - cumge ; rightmask = [c > lc_k] = cumge - Lcol
        n("Sub", ["one_f", t("cumge")], t("leftmask"))
        n("Sub", [t("cumge"), t("Lcol")], t("rightmask"))

        # hasleft[k,r] = max_c D*leftmask ; hasright similar
        n("Mul", [t("D"), t("leftmask")], t("Dleft"))                 # [1,10,30,30]
        n("ReduceMax", [t("Dleft")], t("hasleft"), axes=[3], keepdims=1)   # [1,10,30,1]
        n("Mul", [t("D"), t("rightmask")], t("Dright"))
        n("ReduceMax", [t("Dright")], t("hasright"), axes=[3], keepdims=1)

        # target column indicators: lc-1 = shiftR(Lcol), lc+1 = shiftL(Lcol)
        n("MatMul", [t("Lcol"), "Sr"], t("leftpos"))                  # [1,10,1,30]
        n("MatMul", [t("Lcol"), "Sl"], t("rightpos"))

        # rank-1 output planes per channel (broadcast outer products)
        n("Mul", [t("hasleft"), t("leftpos")], t("leftplane"))        # [1,10,30,30]
        n("Mul", [t("hasright"), t("rightpos")], t("rightplane"))
        # line plane = Lcol broadcast over rows (rows-of-ones (x) Lcol)
        n("Mul", ["rowones", t("Lcol")], t("lineplane"))              # [1,10,30,30]

        n("Add", [t("leftplane"), t("rightplane")], t("lr"))
        n("Add", [t("lr"), t("lineplane")], t("onplanes"))            # [1,10,30,30] {0,1}
        # weight by colour index k and sum over channels -> label (fp32)
        n("Mul", [t("onplanes"), "arangek"], t("weighted"))           # [1,10,30,30]
        n("ReduceSum", [t("weighted")], t("Lcolor"), axes=[1], keepdims=1)  # [1,1,30,30]
        return t("Lcolor"), t("ingrid"), t("H")

    # one-hot helper constants that need value (depend on N)
    init("one_f", np.array(1.0, np.float32), np.float32)
    rowones = np.ones((1, 1, N, 1), np.float32)
    init("rowones", rowones, np.float32)

    # ---------- vertical branch on X ----------
    Lv, ingV, Hv = vertical_branch("input", "v")

    # ---------- horizontal branch: transpose input, solve vertical, transpose back ----------
    n("Transpose", ["input"], "inputT", perm=[0, 1, 3, 2])
    Lh_t, ingH_t, Hh = vertical_branch("inputT", "h")
    n("Transpose", [Lh_t], "Lh", perm=[0, 1, 3, 2])      # back to canvas coords

    # ---------- orientation select ----------
    # vertical orientation iff there exist full COLUMNS, i.e. some column has
    # colcnt (over channels 1..9) == H.  Equivalently the vertical branch found a
    # nonzero label somewhere... simplest robust test: count full columns.
    # colored colcount per column over channels>=1:
    init("zero_ch0", np.concatenate(
        [np.zeros((1, 1, 1, 1)), np.ones((1, 9, 1, 1))], axis=1).astype(np.float32),
        np.float32)
    n("Mul", ["input", "zero_ch0"], "colored")           # ch0 removed [1,10,30,30]
    n("ReduceSum", ["colored"], "ccol", axes=[1, 2], keepdims=1)   # [1,1,1,30]
    # a full vertical line column has ccol == H
    n("Sub", ["ccol", "v_H"], "ccolmH")
    n("Mul", ["ccolmH", "ccolmH"], "ccolmH2")
    n("Less", ["ccolmH2", "half"], "vcolfull_b")         # [1,1,1,30] bool
    n("Cast", ["vcolfull_b"], "vcolfull", to=F32)        # [1,1,1,30] fp32
    n("ReduceMax", ["vcolfull"], "is_vertical_f", axes=[3], keepdims=1)  # [1,1,1,1]
    n("Greater", ["is_vertical_f", "half"], "is_vertical")  # bool

    # select label by orientation
    n("Where", ["is_vertical", Lv, "Lh"], "Lsel")        # [1,1,30,30] fp32

    # ---------- assemble final label with off-grid sentinel ----------
    n("Greater", [ingV, "half"], "ingrid_b")             # bool (orientation-free)
    n("Cast", ["Lsel"], "Lsel_u", to=U8)                 # in-grid colour indices
    init("u10", np.array(10, np.uint8), np.uint8)
    n("Where", ["ingrid_b", "Lsel_u", "u10"], "L")       # [1,1,30,30] uint8

    n("Equal", ["L", "chan_u"], "output")                # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, N, N])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, N, N])
    g = helper.make_graph(nodes, "task025", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
