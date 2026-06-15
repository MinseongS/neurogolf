"""task086 (ARC-AGI 3befdf3e) — "flower" stamp.

Rule (from the generator): up to two non-overlapping "flower" centers are drawn.
Each center has a length L in {1,2}.  In the INPUT a center is an (L+2)x(L+2)
solid block: a 1-cell-thick frame of colors[1] surrounding an LxL core of
colors[0].  In the OUTPUT each center is replaced by a fixed local stamp:
  * the central (L+2)x(L+2) block has its two colours SWAPPED (frame becomes
    colors[0], core becomes colors[1]), and
  * "petals" of colors[1] are added in a fixed pattern extending L cells beyond
    the block (corners skipped).
The two flowers never overlap (generator overlap-guard), so the output is the
union of the two independent stamps.

colors[1] (the frame colour, call it C1) always occupies strictly more cells
than colors[0] (the core colour, C0): per flower the frame has 4L+4 cells vs L^2
core cells.  So C1 = most-frequent nonzero colour, C0 = the other nonzero colour.

Floor-break (detection + correlation-stamp + uint8 label map + final Equal):
  * A = (input == C0) plane;  detect each core's TOP-LEFT anchor for L=1 and L=2
    (a 1x1 isolated c0, resp. a 2x2 c0 block whose TL has no c0 above/left).
  * Stamp = correlate each anchor map with the fixed L-specific offset kernels
    (one for the C0 cells of the stamp, one for the C1 cells) via Conv.  This
    "paints" each stamp into a [1,1,30,30] plane without any 10-channel tensor.
  * Build uint8 label L[1,1,30,30]: C1val where any C1-stamp, C0val where any
    C0-stamp, 0 elsewhere in-grid, 10 off-grid (matches no channel).  Final op
    output = Equal(L, arange[1,10,1,1])  (opset 11, BOOL output).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
B = TensorProto.BOOL
I64 = TensorProto.INT64


def _stamp_offsets(L):
    """Return (c0_offsets, c1_offsets) relative to the core top-left for length L
    by replicating the generator on a lone flower far from any edge."""
    size = 30
    row = col = 12
    c0col, c1col = 5, 8
    out = np.zeros((size, size), int)
    for r in range(row - L - 1, row + 2 * L + 1):
        for c in range(col - L - 1, col + 2 * L + 1):
            if r < row - 1 or r > row + L:
                if c < col - 1 or c > col + L:
                    continue
            out[r][c] = c1col
    for r in range(row - 1, row + L + 1):
        for c in range(col - 1, col + L + 1):
            out[r][c] = c0col
    for r in range(row, row + L):
        for c in range(col, col + L):
            out[r][c] = c1col
    c0off, c1off = [], []
    ys, xs = np.nonzero(out)
    for y, x in zip(ys, xs):
        dr, dc = int(y - row), int(x - col)
        (c0off if out[y, x] == c0col else c1off).append((dr, dc))
    return c0off, c1off


def _stamp_kernel(offsets):
    """Build a Conv kernel K[1,1,2R+1,2R+1] so that Conv(anchor,K,pads=R) gives
    out[i,j] = sum over offsets (dr,dc) of anchor[i-dr, j-dc].
    ONNX Conv is cross-correlation: out[i,j] = sum_{a,b} K[a,b]*anchor[i+a-R,j+b-R].
    Setting a = R - dr, b = R - dc makes the term anchor[i-dr, j-dc]."""
    R = max(max(abs(dr), abs(dc)) for dr, dc in offsets)
    K = np.zeros((1, 1, 2 * R + 1, 2 * R + 1), np.float32)
    for dr, dc in offsets:
        K[0, 0, R - dr, R - dc] = 1.0
    return K, R


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- per-channel cell counts -> C0 / C1 colour channel indices ----------
    n("ReduceSum", ["input"], "cnt0", axes=[2, 3], keepdims=1)  # [1,10,1,1] f32
    # zero out channel 0 (background) so it never wins argmax / min-positive.
    init("notbg", np.array([0] + [1] * 9, np.float32).reshape(1, 10, 1, 1),
         np.float32)
    n("Mul", ["cnt0", "notbg"], "cnt")                          # [1,10,1,1]
    # C1 = (nonzero) channel with the most cells.
    n("ArgMax", ["cnt"], "c1idx", axis=1, keepdims=1)            # [1,1,1,1] int64
    # C0 = channel with the fewest POSITIVE cells.  Set zero-count channels to a
    # large value so they are never the min, but keep channel 0 (always 0) out by
    # adding a big constant where count==0.
    init("BIG", np.array(1e6, np.float32), np.float32)
    init("ZEROF", np.array(0.0, np.float32), np.float32)
    init("ONEF", np.array(1.0, np.float32), np.float32)
    n("Greater", ["cnt", "ZEROF"], "haspix")                    # [1,10,1,1] bool
    n("Cast", ["haspix"], "haspixf", to=F32)
    n("Sub", ["ONEF", "haspixf"], "nopix")      # 1 if no pix else 0
    n("Mul", ["nopix", "BIG"], "pen")           # BIG if no pix else 0
    n("Add", ["cnt", "pen"], "cnt_adj")         # count if pix else count+BIG
    n("ArgMin", ["cnt_adj"], "c0idx", axis=1, keepdims=1)       # [1,1,1,1] int64

    # colour VALUES are the channel indices themselves.
    n("Cast", ["c1idx"], "c1val_u", to=U8)      # [1,1,1,1] uint8 broadcast
    n("Cast", ["c0idx"], "c0val_u", to=U8)
    n("Cast", ["c0idx"], "c0val_f", to=F16)     # [1,1,1,1] fp16 scalar

    # ---- colour-index plane G via Conv(input, arange), cropped to WxW -------
    # The grid is always anchored top-left with size<=12, and every flower's
    # stamp stays inside the grid, so a 12x12 working canvas is sufficient.
    W = 12
    Wg = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("Wg", Wg, np.float32)
    n("Conv", ["input", "Wg"], "Gf30")                          # [1,1,30,30] f32
    # crop G to [1,1,W,W] (top-left)
    init("cstart", np.array([0, 0], np.int64), np.int64)
    init("cend", np.array([W, W], np.int64), np.int64)
    init("caxes", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["Gf30", "cstart", "cend", "caxes"], "Gf")       # [1,1,W,W] f32
    n("Cast", ["Gf"], "G", to=F16)                              # fp16 colour plane
    init("HALF32", np.array(0.5, np.float32), np.float32)
    # in-grid mask (WxW): a cell is in-grid iff it has a colour (G>0) OR it is
    # the background channel-0 (==1).  Slice channel 0 directly (avoids a 30x30
    # reduce).  Off-grid cells (size<12) have all channels 0, so both fail.
    init("ch0s", np.array([0, 0, 0], np.int64), np.int64)
    init("ch0e", np.array([1, W, W], np.int64), np.int64)
    init("ch0ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "ch0s", "ch0e", "ch0ax"], "bg0")       # [1,1,W,W] f32 ch0
    n("Greater", ["bg0", "HALF32"], "isbg_b")
    n("Greater", ["Gf", "HALF32"], "iscol_b")                   # has a colour
    n("Or", ["isbg_b", "iscol_b"], "ing_b")                     # [1,1,W,W] bool
    # A = (G == C0 colour) as fp16 0/1 plane
    n("Sub", ["G", "c0val_f"], "gd")
    n("Abs", ["gd"], "gda")
    init("HALF", np.array(0.5, np.float16), np.float16)
    n("Less", ["gda", "HALF"], "A_b")                           # bool C0 plane
    n("Cast", ["A_b"], "A16", to=F16)                           # fp16 0/1

    # ---- anchor detection from A -------------------------------------------
    # 2x2 block sum anchored at TL (pads bottom-right): blk2[r,c]=sum A[r..r+1,c..c+1]
    init("K22", np.ones((1, 1, 2, 2), np.float16), np.float16)
    n("Conv", ["A16", "K22"], "blk2", pads=[0, 0, 1, 1])
    init("THR35", np.array(3.5, np.float16), np.float16)
    n("Greater", ["blk2", "THR35"], "blk2full_b")               # 2x2 all c0

    # nbr4[i,j] = A[i-1,j]+A[i+1,j]+A[i,j-1]+A[i,j+1]  (orthogonal neighbour count)
    Kplus = np.zeros((1, 1, 3, 3), np.float16)
    for (a, b) in [(0, 1), (2, 1), (1, 0), (1, 2)]:
        Kplus[0, 0, a, b] = 1.0
    init("Kplus", Kplus, np.float16)
    n("Conv", ["A16", "Kplus"], "nbr4", pads=[1, 1, 1, 1])
    n("Less", ["nbr4", "HALF"], "iso_b")            # no orthogonal c0 neighbour

    # nbrUL[i,j] = A[i-1,j] + A[i,j-1]  (up + left neighbour count)
    Kul = np.zeros((1, 1, 3, 3), np.float16)
    Kul[0, 0, 0, 1] = 1.0   # up
    Kul[0, 0, 1, 0] = 1.0   # left
    init("Kul", Kul, np.float16)
    n("Conv", ["A16", "Kul"], "nbrUL", pads=[1, 1, 1, 1])
    n("Less", ["nbrUL", "HALF"], "noUL_b")          # no c0 above and none left

    # anchor1: isolated single c0 cell
    n("And", ["A_b", "iso_b"], "anchor1_b")
    # anchor2: TL of a full 2x2 c0 block (A, full 2x2, nothing above/left)
    n("And", ["blk2full_b", "A_b"], "a2a")
    n("And", ["a2a", "noUL_b"], "anchor2_b")

    n("Cast", ["anchor1_b"], "anchor1", to=F16)
    n("Cast", ["anchor2_b"], "anchor2", to=F16)

    # ---- correlation-stamp each anchor with its fixed offset kernels --------
    c0off1, c1off1 = _stamp_offsets(1)
    c0off2, c1off2 = _stamp_offsets(2)

    def stamp(anchor, offs, tag):
        K, R = _stamp_kernel(offs)
        init("K_" + tag, K.astype(np.float16), np.float16)
        n("Conv", [anchor, "K_" + tag], "s_" + tag, pads=[R, R, R, R])
        n("Greater", ["s_" + tag, "HALF"], "sb_" + tag)
        return "sb_" + tag

    c0s1 = stamp("anchor1", c0off1, "c0_1")
    c1s1 = stamp("anchor1", c1off1, "c1_1")
    c0s2 = stamp("anchor2", c0off2, "c0_2")
    c1s2 = stamp("anchor2", c1off2, "c1_2")

    # combine masks (disjoint across flowers; c0 vs c1 disjoint within a stamp)
    n("Or", [c0s1, c0s2], "c0mask_b")               # cells that should be C0
    n("Or", [c1s1, c1s2], "c1mask_b")               # cells that should be C1

    # ---- assemble uint8 label map ------------------------------------------
    init("V0", np.array(0, np.uint8), np.uint8)
    init("V10", np.array(10, np.uint8), np.uint8)
    # base: 0 in-grid, 10 off-grid (within the WxW crop)
    n("Where", ["ing_b", "V0", "V10"], "Lbase")     # [1,1,W,W] uint8
    # paint C1 cells, then C0 on top (C0/C1 are disjoint so order is irrelevant)
    n("Where", ["c1mask_b", "c1val_u", "Lbase"], "L1")
    n("Where", ["c0mask_b", "c0val_u", "L1"], "Lw")  # [1,1,W,W] uint8
    # pad WxW back to 30x30 with sentinel 10 (off-grid -> matches no channel)
    init("padval", np.array(10, np.uint8), np.uint8)
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), np.int64)
    n("Pad", ["Lw", "pads", "padval"], "L", mode="constant")  # [1,1,30,30] uint8

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")             # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task086", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
