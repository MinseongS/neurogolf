"""task198 (ARC-AGI 83302e8f) — "permeable line-grid: mark cells reachable through gaps".

Rule (from the generator, NOT a flood-fill — fully local closed-form):
  A size x size cell grid; each cell is minisize x minisize pixels, cells separated by
  single 1-px lines of `color`.  pitch p = minisize+1, actual_size = size*p-1 (<=29).
  Cell interiors default GREEN(3).  "Permeable points" are black(0) pixels sitting ON a
  line (i.e. where the line colour would otherwise be).  A black-on-vertical-line pixel
  (c%p==minisize, and r%p!=minisize) connects cell (R,C) and (R,C+1) -> both YELLOW(4).
  A black-on-horizontal-line pixel (r%p==minisize, c%p!=minisize) connects (R,C),(R+1,C).
  A pixel exactly on a line crossing triggers nothing.  Output keeps the line colour on
  every line position EXCEPT permeable points, which become YELLOW(4); cell interiors are
  YELLOW if their cell got marked else GREEN.

Encoding (cell-space marking via separable selector MatMuls; route 10-ch to FREE output):
  isblack = input ch0 (free slice).  online = positional (r%p==p-1 or c%p==p-1).
  color  = ArgMax of per-channel pixel counts with ch0 (background) masked to 0 (no plane).
  p      = (first full line row index)+1.
  Vertical-gap pixel  Gv = isblack & (c%p==p-1) & (r%p!=p-1)  -> connects horiz neighbours.
  Horizontal-gap pixel Gh = isblack & (r%p==p-1) & (c%p!=p-1) -> connects vert neighbours.
  Downsample gaps to cell-space "wall has gap" via selector MatMuls keyed on cell index:
    Cidx[c]=c//p (col cell), Ridx[r]=r//p (row cell).
    Vertical gap on the line that is the RIGHT wall of cell C lives at col (C+1)*p-1
    i.e. cell-of-col = C.  So per-(row-cell R, col-cell C):
      VgapR[R,C] = any pixel in row-cell R, col-cell C that is a Gv  (marks C and C+1)
      HgapR[R,C] = any pixel in row-cell R, col-cell C that is a Gh  (marks R and R+1)
  cell yellow Y[R,C] = VgapR[R,C] | VgapR[R,C-1] | HgapR[R,C] | HgapR[R-1,C].
  Upsample Y back to pixel space (selector MatMuls), combine with line colour + permeable
  yellow points and route 10-ch expansion into the FREE bool output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64

N = 30
S = 7   # max cells per axis (size in 3..7)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ============== scalars ==============
    # per-channel pixel counts (mask ch0=background)
    n("ReduceSum", ["input"], "cnt0", axes=[2, 3], keepdims=1)   # [1,10,1,1] f32
    mask0 = np.ones((1, 10, 1, 1), np.float32); mask0[0, 0] = 0.0
    init("chmask", mask0, np.float32)
    n("Mul", ["cnt0", "chmask"], "cnt")                          # [1,10,1,1]
    n("ArgMax", ["cnt"], "color_i64", axis=1, keepdims=1)        # [1,1,1,1] int64
    n("Cast", ["color_i64"], "colorF", to=F16)                   # color scalar f16

    # in-grid-black = input ch0 (off-grid is all-zero -> ch0=0, so ch0==1 iff in-grid black)
    init("s0", np.array([0], np.int64), np.int64)
    init("s1", np.array([1], np.int64), np.int64)
    init("ax1", np.array([1], np.int64), np.int64)
    n("Slice", ["input", "s0", "s1", "ax1"], "ch0")              # [1,1,30,30] f32
    init("HALF", np.array(0.5, np.float32), np.float32)
    n("Greater", ["ch0", "HALF"], "isblack_b")                   # bool [1,1,30,30] (in-grid black)
    # anyset = sum over channels (1 in-grid, 0 off-grid) -> grid-extent signal
    n("ReduceSum", ["input"], "anyset", axes=[1], keepdims=1)    # [1,1,30,30] f32 {0,1}
    init("HALF2", np.array(0.5, np.float32), np.float32)
    n("Greater", ["anyset", "HALF2"], "ingrid_pix_b")            # bool [1,1,30,30]

    # ============== recover pitch p ==============
    # colour-index plane only as a SCALAR per row: rowcolorcount = #cells == color per row
    # A full line row has color at all in-grid positions. Use occupancy of the color channel.
    # in-grid extent + per-row nonblack count via DIRECT axis reductions on the FREE
    # input (no full planes).  colany[c] = any in-grid pixel in col c (sum over ch & rows).
    rampR = np.arange(N, dtype=np.float16).reshape(1, 1, N, 1)
    rampC = np.arange(N, dtype=np.float16).reshape(1, 1, 1, N)
    init("rampR", rampR, np.float16)
    init("rampC", rampC, np.float16)
    init("ONEH", np.array(1.0, np.float16), np.float16)
    # column occupancy (any set) -> width; ReduceSum input over ch(1) & rows(2) is [1,1,1,30]
    n("ReduceSum", ["input"], "colset", axes=[1, 2], keepdims=1)  # [1,1,1,30] f32 (count)
    init("ZEROF", np.array(0.0, np.float32), np.float32)
    n("Greater", ["colset", "ZEROF"], "colany_b")                # bool [1,1,1,30]
    n("Cast", ["colany_b"], "colany", to=F16)
    n("Mul", ["colany", "rampC"], "cc")                          # [1,1,1,30]
    n("ReduceMax", ["cc"], "Wm1", axes=[3], keepdims=1)          # [1,1,1,1] = actual_size-1

    # per-row nonblack count = sum over channels 1..9 and over cols (slice off ch0).
    init("c1", np.array([1], np.int64), np.int64)
    init("c10", np.array([10], np.int64), np.int64)
    init("axc", np.array([1], np.int64), np.int64)
    n("Slice", ["input", "c1", "c10", "axc"], "fg")              # [1,9,30,30] f32 (FREE-ish view)
    n("ReduceSum", ["fg"], "rownb32", axes=[1, 3], keepdims=1)   # [1,1,30,1] f32
    n("Cast", ["rownb32"], "rownb", to=F16)                      # [1,1,30,1] f16
    # A line row has many nonblack (>=10); a cell row has <=size-1 (<=6).  Threshold 6.5.
    init("THR", np.array(6.5, np.float16), np.float16)
    n("Greater", ["rownb", "THR"], "isline_r_b")                 # bool [1,1,30,1]
    # first line row index = min r where isline.  ramp masked.
    init("BIG", np.array(99.0, np.float16), np.float16)
    n("Where", ["isline_r_b", rampR_name := "rampR", "BIG"], "liner_or_big")  # [1,1,30,1]
    n("ReduceMin", ["liner_or_big"], "pm1", axes=[2], keepdims=1)  # [1,1,1,1] = p-1
    n("Add", ["pm1", "ONEH"], "pF")                              # p (f16) scalar

    # ============== positional masks (need r%p, c%p) ==============
    # r%p via Mod (fp16 integer-exact for small ints). ramp/p.
    # Build rmodp[r] = r mod p  as [1,1,30,1]; cmodp[c] similarly [1,1,1,30].
    n("Mod", ["rampR", "pF"], "rmodp", fmod=1)                   # [1,1,30,1] f16
    n("Mod", ["rampC", "pF"], "cmodp", fmod=1)                   # [1,1,1,30] f16
    # on horizontal line: r%p == p-1 == pm1
    n("Equal", ["rmodp", "pm1"], "rline_b")                      # bool [1,1,30,1]
    n("Equal", ["cmodp", "pm1"], "cline_b")                      # bool [1,1,1,30]
    n("Not", ["rline_b"], "rnotline_b")                          # bool [1,1,30,1]
    n("Not", ["cline_b"], "cnotline_b")                          # bool [1,1,1,30]

    # Gv = isblack & cline & rnotline  ; Gh = isblack & rline & cnotline
    n("And", ["cline_b", "rnotline_b"], "vsite_b")               # [1,1,30,30] (broadcast)
    n("And", ["rline_b", "cnotline_b"], "hsite_b")               # [1,1,30,30]
    n("And", ["isblack_b", "vsite_b"], "Gv_b")                   # bool [1,1,30,30]
    n("And", ["isblack_b", "hsite_b"], "Gh_b")                   # bool [1,1,30,30]
    n("Cast", ["Gv_b"], "Gv", to=F16)                            # f16 [1,1,30,30]
    n("Cast", ["Gh_b"], "Gh", to=F16)                            # f16 [1,1,30,30]

    # ============== cell index per axis ==============
    # Ridx[r] = floor(r/p), Cidx[c]=floor(c/p).
    n("Div", ["rampR", "pF"], "rdiv")                            # [1,1,30,1] f16
    n("Floor", ["rdiv"], "Ridx")                                 # [1,1,30,1]
    n("Div", ["rampC", "pF"], "cdiv")
    n("Floor", ["cdiv"], "Cidx")                                 # [1,1,1,30]

    # selector matrices: Rsel[R,r] = (Ridx[r]==R) ; Csel[c,C]=(Cidx[c]==C)
    RidxK = np.arange(S, dtype=np.float16).reshape(1, 1, S, 1)   # [1,1,S,1]
    init("RidxK", RidxK, np.float16)
    # Ridx is [1,1,30,1]; reshape to [1,1,1,30] to compare against RidxK[1,1,S,1]
    init("to1130", np.array([1, 1, 1, N], np.int64), np.int64)
    n("Reshape", ["Ridx", "to1130"], "Ridx_r")                  # [1,1,1,30]
    n("Equal", ["Ridx_r", "RidxK"], "Rsel_b")                   # bool [1,1,S,30] (R,r)
    n("Cast", ["Rsel_b"], "Rsel", to=F16)                       # f16 [1,1,S,30]
    # Csel transposed: [1,1,30,S] (c,C)
    CidxK = np.arange(S, dtype=np.float16).reshape(1, 1, 1, S)  # [1,1,1,S]
    init("CidxK", CidxK, np.float16)
    init("to1301", np.array([1, 1, N, 1], np.int64), np.int64)
    n("Reshape", ["Cidx", "to1301"], "Cidx_c")                  # [1,1,30,1]
    n("Equal", ["Cidx_c", "CidxK"], "Csel_b")                   # bool [1,1,30,S] (c,C)
    n("Cast", ["Csel_b"], "Csel", to=F16)                       # f16 [1,1,30,S]

    # downsample: VgapR[R,C] = any Gv pixel in cell (R,C) = (Rsel @ Gv @ Csel) > 0
    n("MatMul", ["Rsel", "Gv"], "vg1")                          # [1,1,S,30] f16
    n("MatMul", ["vg1", "Csel"], "VgapS")                       # [1,1,S,S] f16
    n("MatMul", ["Rsel", "Gh"], "hg1")
    n("MatMul", ["hg1", "Csel"], "HgapS")                       # [1,1,S,S]
    init("ZEROH", np.array(0.0, np.float16), np.float16)
    n("Greater", ["VgapS", "ZEROH"], "Vg_b")                    # bool [1,1,S,S]
    n("Greater", ["HgapS", "ZEROH"], "Hg_b")                    # bool [1,1,S,S]

    # cell yellow Y[R,C] = Vg[R,C] | Vg[R,C-1] | Hg[R,C] | Hg[R-1,C]
    # shifts on the SxS cell grid via small pad+slice.
    n("Cast", ["Vg_b"], "Vg", to=F16)
    n("Cast", ["Hg_b"], "Hg", to=F16)
    # Vg shifted right by one col (C-1): pad left 1, drop last col
    init("pad_left", np.array([0, 0, 0, 1, 0, 0, 0, 0], np.int64), np.int64)
    init("ZP", np.array(0.0, np.float16), np.float16)
    n("Pad", ["Vg", "pad_left", "ZP"], "Vg_padL", mode="constant")  # [1,1,S,S+1]
    init("sl0", np.array([0], np.int64), np.int64)
    init("slS", np.array([S], np.int64), np.int64)
    init("ax3", np.array([3], np.int64), np.int64)
    n("Slice", ["Vg_padL", "sl0", "slS", "ax3"], "VgL")         # [1,1,S,S] = Vg[R,C-1]
    # Hg shifted down by one row (R-1): pad top 1, drop last row
    init("pad_top", np.array([0, 0, 1, 0, 0, 0, 0, 0], np.int64), np.int64)
    n("Pad", ["Hg", "pad_top", "ZP"], "Hg_padT", mode="constant")  # [1,1,S+1,S]
    init("ax2", np.array([2], np.int64), np.int64)
    n("Slice", ["Hg_padT", "sl0", "slS", "ax2"], "HgT")         # [1,1,S,S] = Hg[R-1,C]

    # Y = max of the four
    n("Max", ["Vg", "VgL"], "y1")
    n("Max", ["Hg", "HgT"], "y2")
    n("Max", ["y1", "y2"], "Ycell")                             # f16 [1,1,S,S] {0,1}

    # ============== upsample Y to pixel space ==============
    # Ypix[r,c] = Ycell[Ridx[r], Cidx[c]] = Rsel^T @ Ycell @ Csel^T
    # Rsel is [1,1,S,30] (R,r); Rsel^T = [1,1,30,S].  Build RselT, CselT.
    # RselT[r,R] = Rsel[R,r]; use Transpose.
    n("Transpose", ["Rsel"], "RselT", perm=[0, 1, 3, 2])        # [1,1,30,S]
    n("Transpose", ["Csel"], "CselT", perm=[0, 1, 3, 2])        # [1,1,S,30]
    n("MatMul", ["RselT", "Ycell"], "yp1")                      # [1,1,30,S]
    n("MatMul", ["yp1", "CselT"], "Ypix")                       # [1,1,30,30] f16 {0,1}
    n("Greater", ["Ypix", "ZEROH"], "Ypix_b")                   # bool [1,1,30,30]

    # ============== compose output colour-index plane (small as possible) ==============
    # online = rline OR cline (positional)
    n("Or", ["rline_b", "cline_b"], "online_b")                 # bool [1,1,30,30]
    # permeable yellow point = isblack & online
    n("And", ["isblack_b", "online_b"], "permyellow_b")         # bool [1,1,30,30]
    # in-grid mask: r<=Wm1 and c<=Wm1 (square). off-grid -> background 0.
    n("Not", [n("Greater", ["rampR", "Wm1"], "rog")], "ringrid_b")  # [1,1,30,1]
    n("Not", [n("Greater", ["rampC", "Wm1"], "cog")], "cingrid_b")  # [1,1,1,30]
    n("And", ["ringrid_b", "cingrid_b"], "ingrid_b")            # [1,1,30,30]

    # Build colour-index L (f16) by priority:
    #   if off-grid -> 0
    #   elif online (line) -> permyellow? 4 : color
    #   else (cell interior) -> Ypix? 4 : 3
    init("YEL", np.array(4.0, np.float16), np.float16)
    init("GRN", np.array(3.0, np.float16), np.float16)
    init("ZL", np.array(99.0, np.float16), np.float16)  # off-grid sentinel (no channel hot)
    # interior value
    n("Where", ["Ypix_b", "YEL", "GRN"], "interiorL")          # [1,1,30,30] f16
    # line value: permyellow?4:color  (colorF is [1,1,1,1])
    n("Where", ["permyellow_b", "YEL", "colorF"], "lineL")     # [1,1,30,30] f16
    # combine line vs interior by online
    n("Where", ["online_b", "lineL", "interiorL"], "gridL")    # [1,1,30,30]
    # apply in-grid
    n("Where", ["ingrid_b", "gridL", "ZL"], "L")               # [1,1,30,30] f16

    # ============== route 10-ch expansion to FREE output ==============
    chan = np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1)
    init("chan", chan, np.float16)
    n("Equal", ["L", "chan"], "output")                        # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task198", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
