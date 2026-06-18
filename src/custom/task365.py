"""task365 (ARC-AGI e50d258f) — crop the box with the MOST red pixels.

Rule (from the generator):
  A 10x10 grid holds 2-3 solid, gap-separated, axis-aligned rectangles ("boxes")
  filled with blue(1)/cyan(8) plus a few red(2) cells.  Each box has a DISTINCT
  red count drawn from {1,2,3,4}; the output is the box with the MOST reds,
  cropped to its bounding box at the top-left of a fresh grid.

Approach (ONNX, opset 11), everything on the 10x10 active canvas:
  - colf = per-cell colour index, built WITHOUT a 30x30 plane: Conv channels 0..2
    (weights [0,1,2]) on a [1,3,10,10] slice + 8*cyan(ch8).  occ = colf>0;
    red = (colf==2).
  - box-red total at each box's BOTTOM-RIGHT corner via TWO forward segmented
    prefix scans (gated Hillis-Steele doubling, shifts 1,2,4): rH = H-prefix of
    red (= box-row red at the run's right end), then boxred = V-prefix of rH
    (= full box red at the box's BR cell).  Each shift is ONE fp16 MatMul with a
    constant shift matrix; the gate g_d (=occ-run membership, idempotent products)
    stops the prefix at the >=1-cell gaps between boxes.
  - BR corners = occupied cells with no occupied neighbour right/below.  Distinct
    per-box red counts => the BR with max boxred is the UNIQUE winner.
  - winner scalars: BR (r1,c1) from winf row/col selectors; box H,W from tiny 1-D
    run-length scans of the winning row/column occupancy (gathered at r1/c1).
    TL = (r1-H+1, c1-W+1).
  - Gather-shift the winner colf window to the top-left, label (sentinel 10 off
    box), Pad to 30x30, Equal -> free bool one-hot output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
I64 = TensorProto.INT64
B = TensorProto.BOOL

G = 10  # active canvas size


def build(task):
    inits, nodes = [], []
    _seen_init = set()
    _uid = [0]

    def init(name, arr, dtype):
        if name in _seen_init:
            return name
        _seen_init.add(name)
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def uid(tag):
        _uid[0] += 1
        return f"{tag}_{_uid[0]}"

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---------------- derive colf / red / occ on the 10x10 canvas ------------
    init("s0", np.array([0], np.int64), np.int64)
    init("s10", np.array([G], np.int64), np.int64)
    init("ax2", np.array([2], np.int64), np.int64)
    init("ax3", np.array([3], np.int64), np.int64)

    # colf = per-cell colour index on the 10x10 canvas, computed WITHOUT a full
    # 30x30 plane: colours present are {0(bg),1(blue),2(red),8(cyan)}.  Conv the
    # contiguous channels 0..2 (weights [0,1,2]) then add 8*cyan(ch8).
    init("ax123", np.array([1, 2, 3], np.int64), np.int64)
    init("lo_st", np.array([0, 0, 0], np.int64), np.int64)
    init("lo_en", np.array([3, G, G], np.int64), np.int64)
    n("Slice", ["input", "lo_st", "lo_en", "ax123"], "inLo")      # [1,3,10,10] f32
    cw = np.array([0, 1, 2], np.float32).reshape(1, 3, 1, 1)
    init("cw", cw, np.float32)
    n("Conv", ["inLo", "cw"], "colfLo")                           # [1,1,10,10] f32
    init("cy_st", np.array([8, 0, 0], np.int64), np.int64)
    init("cy_en", np.array([9, G, G], np.int64), np.int64)
    n("Slice", ["input", "cy_st", "cy_en", "ax123"], "cyan32")    # [1,1,10,10] f32
    init("eight", np.array(8.0, np.float32), np.float32)
    n("Mul", ["cyan32", "eight"], "cy8")
    n("Add", ["colfLo", "cy8"], "colf32")                         # [1,1,10,10] f32
    # red mask = (colf32 == 2) -> fp16 {0,1}
    init("twoF", np.array(2.0, np.float32), np.float32)
    n("Equal", ["colf32", "twoF"], "red_b")                       # bool [1,1,10,10]
    n("Cast", ["red_b"], "red", to=F16)                           # [1,1,10,10] f16

    # occupancy = colf32 > 0 (background -> colf 0) -> fp16 {0,1}
    init("zeroF", np.array(0.0, np.float32), np.float32)
    n("Greater", ["colf32", "zeroF"], "occ_b0")                   # bool [1,1,10,10]
    n("Cast", ["occ_b0"], "occ", to=F16)                          # f16 {0,1}

    # ---------------- segmented total helpers -------------------------------
    init("zero16", np.array(0.0, np.float16), np.float16)
    init("half16", np.array(0.5, np.float16), np.float16)

    # axis-index initializers used by shift/slice
    init("a2", np.array([2], np.int64), np.int64)
    init("a3", np.array([3], np.int64), np.int64)

    # Shift via a single MatMul with a fp16 shift matrix (one plane, no Concat).
    #   axis-2 (rows) fwd by d:  out = M_d @ v ,  M_d[i,j]=1 iff j==i-d
    #   axis-3 (cols) fwd by d:  out = v @ M_d^T
    # Negative (toward smaller index) uses the transpose (M_d^T = M_{-d}).
    def _shiftmat(d):
        """[G,G] matrix with 1 on (i, i-d). Pre-mult shifts rows; this matrix used
        directly for axis-2 and as the col-shift operand for axis-3 (v @ M)."""
        M = np.zeros((G, G), np.float16)
        for i in range(G):
            j = i - d
            if 0 <= j < G:
                M[i, j] = 1.0
        name = f"M_{d}"
        init(name, M, np.float16)
        return name

    def _do_shift(src, axis, d):
        # axis-2 (rows): out = M_d @ v.  axis-3 (cols): out = v @ M_{-d}
        # because (v @ A)[r,c] = sum_j v[r,j] A[j,c]; choosing A=M_{-d} gives
        # A[j,c]=1 iff j == c-(-d)=c+d ... we want out[r,c]=v[r,c-d] -> A[j,c]=1
        # iff j=c-d -> that is M_d^T = M_{-d}. So col-shift fwd uses M_{-d}.
        if axis == 2:
            return n("MatMul", [_shiftmat(d), src], uid("sh"))
        else:
            return n("MatMul", [src, _shiftmat(-d)], uid("sh"))

    def shift(src, axis, dist, ch=1):
        return _do_shift(src, axis, dist)

    def neg_shift(src, axis, dist, ch=1):
        return _do_shift(src, axis, -dist)

    def gates(occ_, axis, forward):
        """g1,g2,g4 where g_d[i]=1 iff cells i..i-d (toward scan source) all occ.
        occ in {0,1} -> idempotent products let doubling build them cheaply."""
        sh = shift if forward else neg_shift
        g1 = n("Mul", [occ_, sh(occ_, axis, 1)], uid("g1"))
        g2 = n("Mul", [g1, sh(g1, axis, 1)], uid("g2"))
        g4 = n("Mul", [g2, sh(g2, axis, 2)], uid("g4"))
        return g1, g2, g4

    def fwd_scan(val, g, axis, ch):
        """Segmented inclusive forward prefix via gated doubling. Operates on all
        channels of `val` simultaneously (gates broadcast over channels)."""
        g1, g2, g4 = g
        v = val
        for gd, d in [(g1, 1), (g2, 2), (g4, 4)]:
            sv = shift(v, axis, d, ch)
            add = n("Mul", [gd, sv], uid("av"))
            v = n("Add", [v, add], uid("vp"))
        return v

    # ---------------- box red total at BR corners (2 forward scans) ----------
    # Horizontal forward scan of red -> rH (= box-row red at the run's right end).
    gH = gates("occ", 3, True)
    rH = fwd_scan("red", gH, 3, 1)
    # Vertical forward scan of rH -> boxred (= box red total at the box's BR cell).
    gV = gates("occ", 2, True)
    boxred = fwd_scan(rH, gV, 2, 1)

    # ---------------- winning box = BR corner with max box-red ---------------
    # BR corner: occupied with no occupied neighbour to the right or below.
    occR = neg_shift("occ", 3, 1)                                 # occ at col+1
    occD = neg_shift("occ", 2, 1)                                 # occ at row+1
    n("Add", [occR, occD], "nbrs")                                # 0 only at BR
    n("Greater", ["half16", "nbrs"], "noNbr")                     # bool: no occ R/D
    n("And", ["noNbr", "occ_b0"], "isBR")                        # BR corner cells
    n("Cast", ["isBR"], "isBRf", to=F16)
    n("Mul", [boxred, "isBRf"], "brBR")                          # boxred only at BR
    n("ReduceMax", ["brBR"], "M", axes=[2, 3], keepdims=1)       # [1,1,1,1] f16
    # brBR is 0 off BR corners and = box-red at BR corners (>0), so brBR==M (>0)
    # marks exactly the unique winning BR cell.
    n("Equal", ["brBR", "M"], "win")                             # winning BR cell

    # recover scalars at the (unique) winning BR cell
    n("Cast", ["win"], "winf", to=F16)
    rr = np.arange(G, dtype=np.float16).reshape(1, 1, G, 1)
    rc = np.arange(G, dtype=np.float16).reshape(1, 1, 1, G)
    init("rr", rr, np.float16)
    init("rc", rc, np.float16)
    # winning row / column selector vectors (1-D) drive both scalar recovery and
    # the run-length read.
    n("ReduceMax", ["winf"], "wrowm", axes=[3], keepdims=1)      # [1,1,10,1] row sel
    n("ReduceMax", ["winf"], "wcolm", axes=[2], keepdims=1)      # [1,1,1,10] col sel
    n("Mul", ["wrowm", "rr"], "wr_r")                           # [1,1,10,1]
    n("ReduceMax", ["wr_r"], "r1", axes=[2, 3], keepdims=1)      # BR row
    n("Mul", ["wcolm", "rc"], "wc_c")                           # [1,1,1,10]
    n("ReduceMax", ["wc_c"], "c1", axes=[2, 3], keepdims=1)      # BR col
    # occupancy of the winning ROW / COL via a data-dependent Gather (tiny output).
    init("shp1g", np.array([1], np.int64), np.int64)
    n("Cast", ["r1"], "r1f32", to=F32)
    n("Reshape", ["r1f32", "shp1g"], "r1v")
    n("Cast", ["r1v"], "r1i", to=I64)                           # [1]
    n("Cast", ["c1"], "c1f32", to=F32)
    n("Reshape", ["c1f32", "shp1g"], "c1v")
    n("Cast", ["c1v"], "c1i", to=I64)
    n("Gather", ["occ", "r1i"], "occWrow", axis=2)              # [1,1,1,10]
    n("Gather", ["occ", "c1i"], "occWcol", axis=3)              # [1,1,10,1]
    # 1-D run-length forward scans (gated) on these tiny vectors.
    gWrow = gates("occWrow", 3, True)
    widV = fwd_scan("occWrow", gWrow, 3, 1)                      # run length, axis3
    gWcol = gates("occWcol", 2, True)
    heiV = fwd_scan("occWcol", gWcol, 2, 1)                      # run length, axis2
    # read the run length at the BR cell via Gather (at c1 / r1)
    n("Gather", [widV, "c1i"], "Wf", axis=3)                     # [1,1,1,1] width
    n("Gather", [heiV, "r1i"], "Hf", axis=2)                     # [1,1,1,1] height
    init("one16", np.array(1.0, np.float16), np.float16)
    n("Sub", ["r1", "Hf"], "r0a")
    n("Add", ["r0a", "one16"], "r0")                             # TL row = r1-H+1
    n("Sub", ["c1", "Wf"], "c0a2")
    n("Add", ["c0a2", "one16"], "c0")                            # TL col = c1-W+1

    # ---------------- colour content of winner, shifted to top-left ---------
    # colf32 already computed on the 10x10 canvas above.
    # row/col gather indices = arange(WORK) + r0 / c0, clipped
    WORK = 6
    baseW = np.arange(WORK, dtype=np.float32).reshape(1, 1, WORK)
    init("baseWr", baseW.reshape(WORK), np.float32)               # [WORK]
    n("Cast", ["r0"], "r0f", to=F32)
    n("Cast", ["c0"], "c0f", to=F32)
    init("shp1", np.array([1], np.int64), np.int64)
    n("Reshape", ["r0f", "shp1"], "r0s")                          # [1]
    n("Reshape", ["c0f", "shp1"], "c0s")
    n("Add", ["baseWr", "r0s"], "ridxf")                          # [WORK]
    n("Add", ["baseWr", "c0s"], "cidxf")
    init("f0", np.array(0.0, np.float32), np.float32)
    init("f9", np.array(float(G - 1), np.float32), np.float32)
    n("Clip", ["ridxf", "f0", "f9"], "ridxc")
    n("Clip", ["cidxf", "f0", "f9"], "cidxc")
    n("Cast", ["ridxc"], "ridx", to=I64)
    n("Cast", ["cidxc"], "cidx", to=I64)
    n("Gather", ["colf32", "ridx"], "colVr", axis=2)             # [1,1,WORK,10]
    n("Gather", ["colVr", "cidx"], "colW", axis=3)               # [1,1,WORK,WORK] f32

    # box mask on the WORK window: r<H and c<W
    wr = np.arange(WORK, dtype=np.float16).reshape(1, 1, WORK, 1)
    wc = np.arange(WORK, dtype=np.float16).reshape(1, 1, 1, WORK)
    init("wr", wr, np.float16)
    init("wc", wc, np.float16)
    n("Less", ["wr", "Hf"], "rmask")                             # bool [1,1,WORK,1]
    n("Less", ["wc", "Wf"], "cmask")                             # bool [1,1,1,WORK]
    n("And", ["rmask", "cmask"], "boxmask")                      # bool WORKxWORK

    # label = colf inside box else sentinel; outside box -> 10
    n("Cast", ["colW"], "colWu", to=U8)                          # [1,1,WORK,WORK] u8
    init("u10", np.array(10, np.uint8), np.uint8)
    n("Where", ["boxmask", "colWu", "u10"], "Lw")               # u8 WORKxWORK

    # pad to 30x30 with sentinel 10, Equal -> one-hot output
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    n("Pad", ["Lw", "padpads", "u10"], "L", mode="constant")    # [1,1,30,30] u8
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                         # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task365", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
