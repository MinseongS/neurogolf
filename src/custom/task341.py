"""task341 (ARC-AGI d6ad076f) — "cyan bridge between two blocks".

Rule (from the generator, pre-gravity):
  * A SHORT block (colour c0, thickness 2-4, length 4-6) sits near the top.
  * A LONG block / BRIDGE (colour c1, length 6-9) sits near the bottom.  The
    short block's columns are strictly NESTED inside the long block's columns.
  * A vertical CYAN(8) bar fills the gap rows between the two blocks, spanning
    the short block's columns shrunk by one cell on each side (its interior).
  * apply_gravity then transposes / flips the whole figure into one of 4
    cardinal orientations.  So the two blocks may be stacked vertically or
    horizontally, in either order.

  INPUT = the two colour blocks only.  OUTPUT = input + the cyan bridge.

Recovery (fully reduction-based, verified 0 errors over thousands of fresh
instances).  Work entirely on a 10x10 canvas (the grid is always 10x10
anchored top-left).  Let M[1,1,10,10] = "any non-background colour" mask.

  * rowocc / colocc (ReduceMax of M over the perpendicular axis) give the
    occupied row / col extents [rmin,rmax] / [cmin,cmax].  Exactly ONE axis has
    an internal empty gap (the gap axis); detect via
    (extent_length) > (#occupied lines).
  * ROW-GAP branch: gap rows = empty rows inside [rmin,rmax].  Split M into the
    band above the gap and the band below it; each band is one block.  The cyan
    column span = INTERSECTION of the two bands' column extents (short block is
    nested inside long block), shrunk by 1 each side.
  * COL-GAP branch: symmetric (transpose roles of rows/cols).
  * Select the branch by the gap flag; build rowmask[1,1,10,1] &
    colmask[1,1,1,10] index-range gates; cyan = rowmask (x) colmask AND
    background.
  * L[1,1,10,10] uint8 = colour index V + 8*cyan (cyan lands only on background
    where V==0).  Pad to 30x30 with sentinel 15, final Equal(L,[0..9]) -> BOOL
    into the FREE output.  No [1,10,*,*] plane is ever materialised.

Memory: a handful of [1,1,10,10] (=100) planes; the dominant survivor is the
uint8 L padded to [1,1,30,30] (900 B) just before the Equal.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

CW = 10  # working canvas side (grids are always 10x10 top-left)


def build(task):
    inits, nodes = [], []

    def init(name, arr, npdtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=npdtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    U8 = TensorProto.UINT8
    B = TensorProto.BOOL

    # ---- constants ----
    # colour index conv weight [1,10,1,1] = [0,1,..,9]
    init("colW", np.arange(10).reshape(1, 10, 1, 1), np.float32)
    init("half", np.array(0.5), np.float32)
    init("one", np.array(1.0), np.float32)
    init("BIG", np.array(1e4), np.float32)
    init("nBIG", np.array(-1e4), np.float32)
    init("rowidx", np.arange(CW).reshape(1, 1, CW, 1).astype(np.float32), np.float32)
    init("colidx", np.arange(CW).reshape(1, 1, 1, CW).astype(np.float32), np.float32)
    init("chan", np.arange(10).reshape(1, 10, 1, 1), np.uint8)
    init("v8", np.array(8.0), np.float32)
    # crop a [1,1,30,30] plane -> [1,1,CW,CW] via negative Pad
    init("crop", np.array([0, 0, 0, 0, 0, 0, CW - 30, CW - 30], np.int64), np.int64)
    # pad L [1,1,CW,CW] -> [1,1,30,30] with sentinel
    init("padO", np.array([0, 0, 0, 0, 0, 0, 30 - CW, 30 - CW], np.int64), np.int64)
    init("sentU8", np.array(15), np.uint8)

    # ---- colour index V (cropped to CWxCW); colour mask M = (V > 0) ----
    n("Conv", ["input", "colW"], "V32")          # [1,1,30,30] fp32
    n("Pad", ["V32", "crop"], "V")               # [1,1,CW,CW]
    n("Greater", ["V", "half"], "Mb")            # bool colour mask
    n("Cast", ["Mb"], "M", to=F)                 # [1,1,CW,CW] {0,1}

    # ---- 1-D occupancy ----
    n("ReduceMax", ["M"], "rowocc", axes=[3], keepdims=1)   # [1,1,CW,1]
    n("ReduceMax", ["M"], "colocc", axes=[2], keepdims=1)   # [1,1,1,CW]
    n("Greater", ["rowocc", "half"], "rowoccB")
    n("Greater", ["colocc", "half"], "coloccB")

    # occupied row extent [rmin,rmax] and count
    n("Where", ["rowoccB", "rowidx", "BIG"], "rmin_t")
    n("ReduceMin", ["rmin_t"], "rmin", axes=[2, 3], keepdims=1)   # scalar
    n("Where", ["rowoccB", "rowidx", "nBIG"], "rmax_t")
    n("ReduceMax", ["rmax_t"], "rmax", axes=[2, 3], keepdims=1)
    n("ReduceSum", ["rowocc"], "nrows", axes=[2, 3], keepdims=1)
    # gap flag: (rmax-rmin+1) > nrows
    n("Sub", ["rmax", "rmin"], "rext0"); n("Add", ["rext0", "one"], "rext")
    n("Greater", ["rext", "nrows"], "rgapB")                     # scalar bool
    n("Cast", ["rgapB"], "rgap", to=F)

    # occupied col extent
    n("Where", ["coloccB", "colidx", "BIG"], "cmin_t")
    n("ReduceMin", ["cmin_t"], "cmin", axes=[2, 3], keepdims=1)
    n("Where", ["coloccB", "colidx", "nBIG"], "cmax_t")
    n("ReduceMax", ["cmax_t"], "cmax", axes=[2, 3], keepdims=1)

    # ================= ROW-GAP branch =================
    # gap rows = empty rows strictly inside [rmin,rmax]
    n("Less", ["rowidx", "rmin"], "rlt"); n("Not", ["rlt"], "rge")
    n("Greater", ["rowidx", "rmax"], "rgt"); n("Not", ["rgt"], "rle")
    n("And", ["rge", "rle"], "rin")                              # [1,1,CW,1]
    n("Not", ["rowoccB"], "rnocc")
    n("And", ["rin", "rnocc"], "remptyB")                        # gap rows
    n("Where", ["remptyB", "rowidx", "BIG"], "gr0_t")
    n("ReduceMin", ["gr0_t"], "gr0", axes=[2, 3], keepdims=1)    # first gap row
    n("Where", ["remptyB", "rowidx", "nBIG"], "gr1_t")
    n("ReduceMax", ["gr1_t"], "gr1", axes=[2, 3], keepdims=1)    # last gap row
    # split M into band above the gap (rows < gr0) and below (rows > gr1)
    n("Less", ["rowidx", "gr0"], "aboveB")                       # [1,1,CW,1] bool
    n("Cast", ["aboveB"], "above", to=F)
    n("Greater", ["rowidx", "gr1"], "belowB")
    n("Cast", ["belowB"], "below", to=F)
    n("Mul", ["M", "above"], "Mabove")                           # [1,1,CW,CW]
    n("Mul", ["M", "below"], "Mbelow")
    # each band's column occupancy -> column extents
    n("ReduceMax", ["Mabove"], "acol", axes=[2], keepdims=1)     # [1,1,1,CW]
    n("ReduceMax", ["Mbelow"], "bcol", axes=[2], keepdims=1)
    n("Greater", ["acol", "half"], "acolB")
    n("Greater", ["bcol", "half"], "bcolB")
    n("Where", ["acolB", "colidx", "BIG"], "amin_t")
    n("ReduceMin", ["amin_t"], "amin", axes=[2, 3], keepdims=1)
    n("Where", ["acolB", "colidx", "nBIG"], "amax_t")
    n("ReduceMax", ["amax_t"], "amax", axes=[2, 3], keepdims=1)
    n("Where", ["bcolB", "colidx", "BIG"], "bmin_t")
    n("ReduceMin", ["bmin_t"], "bmin", axes=[2, 3], keepdims=1)
    n("Where", ["bcolB", "colidx", "nBIG"], "bmax_t")
    n("ReduceMax", ["bmax_t"], "bmax", axes=[2, 3], keepdims=1)
    # nested intersection, shrunk by 1: c0 = max(amin,bmin)+1, c1 = min(amax,bmax)-1
    n("Max", ["amin", "bmin"], "rg_cmin"); n("Add", ["rg_cmin", "one"], "rg_c0")
    n("Min", ["amax", "bmax"], "rg_cmax"); n("Sub", ["rg_cmax", "one"], "rg_c1")
    # row span of cyan = gap rows themselves: r0=gr0, r1=gr1
    # (rg_r0 = gr0, rg_r1 = gr1)

    # ================= COL-GAP branch (transpose) =================
    n("Less", ["colidx", "cmin"], "clt"); n("Not", ["clt"], "cge")
    n("Greater", ["colidx", "cmax"], "cgt"); n("Not", ["cgt"], "cle")
    n("And", ["cge", "cle"], "cin")
    n("Not", ["coloccB"], "cnocc")
    n("And", ["cin", "cnocc"], "cemptyB")
    n("Where", ["cemptyB", "colidx", "BIG"], "gc0_t")
    n("ReduceMin", ["gc0_t"], "gc0", axes=[2, 3], keepdims=1)
    n("Where", ["cemptyB", "colidx", "nBIG"], "gc1_t")
    n("ReduceMax", ["gc1_t"], "gc1", axes=[2, 3], keepdims=1)
    n("Less", ["colidx", "gc0"], "leftB"); n("Cast", ["leftB"], "left", to=F)
    n("Greater", ["colidx", "gc1"], "rightB"); n("Cast", ["rightB"], "right", to=F)
    n("Mul", ["M", "left"], "Mleft")
    n("Mul", ["M", "right"], "Mright")
    n("ReduceMax", ["Mleft"], "lrow", axes=[3], keepdims=1)      # [1,1,CW,1]
    n("ReduceMax", ["Mright"], "rrow", axes=[3], keepdims=1)
    n("Greater", ["lrow", "half"], "lrowB")
    n("Greater", ["rrow", "half"], "rrowB")
    n("Where", ["lrowB", "rowidx", "BIG"], "lmin_t")
    n("ReduceMin", ["lmin_t"], "lmin", axes=[2, 3], keepdims=1)
    n("Where", ["lrowB", "rowidx", "nBIG"], "lmax_t")
    n("ReduceMax", ["lmax_t"], "lmax", axes=[2, 3], keepdims=1)
    n("Where", ["rrowB", "rowidx", "BIG"], "rmin2_t")
    n("ReduceMin", ["rmin2_t"], "rmin2", axes=[2, 3], keepdims=1)
    n("Where", ["rrowB", "rowidx", "nBIG"], "rmax2_t")
    n("ReduceMax", ["rmax2_t"], "rmax2", axes=[2, 3], keepdims=1)
    n("Max", ["lmin", "rmin2"], "cg_rmin"); n("Add", ["cg_rmin", "one"], "cg_r0")
    n("Min", ["lmax", "rmax2"], "cg_rmax"); n("Sub", ["cg_rmax", "one"], "cg_r1")
    # (cg_c0 = gc0, cg_c1 = gc1)

    # ================= select branch by gap flag =================
    # r0 = rgap ? gr0 : cg_r0 ;  r1 = rgap ? gr1 : cg_r1
    n("Where", ["rgapB", "gr0", "cg_r0"], "r0")
    n("Where", ["rgapB", "gr1", "cg_r1"], "r1")
    n("Where", ["rgapB", "rg_c0", "gc0"], "c0")
    n("Where", ["rgapB", "rg_c1", "gc1"], "c1")

    # ================= build the cyan rectangle =================
    n("Less", ["rowidx", "r0"], "rm_lt"); n("Not", ["rm_lt"], "rm_ge")
    n("Greater", ["rowidx", "r1"], "rm_gt"); n("Not", ["rm_gt"], "rm_le")
    n("And", ["rm_ge", "rm_le"], "rowmaskB")                     # [1,1,CW,1]
    n("Less", ["colidx", "c0"], "cm_lt"); n("Not", ["cm_lt"], "cm_ge")
    n("Greater", ["colidx", "c1"], "cm_gt"); n("Not", ["cm_gt"], "cm_le")
    n("And", ["cm_ge", "cm_le"], "colmaskB")                     # [1,1,1,CW]
    n("And", ["rowmaskB", "colmaskB"], "rectB")                  # [1,1,CW,CW]
    # cyan lands only on background cells (M == 0)
    n("Not", ["Mb"], "bgB")
    n("And", ["rectB", "bgB"], "cyanB")
    n("Cast", ["cyanB"], "cyan", to=F)

    # ================= label map L and final Equal =================
    n("Mul", ["cyan", "v8"], "cyanv")                            # 8 where cyan
    n("Add", ["V", "cyanv"], "Lcol")                             # colour value plane
    # off-grid (outside the 10x10 active grid is background -> ch0; V==0 there;
    # cyan is bounded inside the grid).  Cells that are background & not cyan keep
    # value 0 -> match channel 0, which is correct (output bg).  Pad fills 30x30
    # border with sentinel 15 so those never match a channel.
    n("Cast", ["Lcol"], "Lc", to=U8)                             # [1,1,CW,CW] uint8
    n("Pad", ["Lc", "padO", "sentU8"], "L", mode="constant")     # [1,1,30,30]
    n("Equal", ["L", "chan"], "output")                          # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task341", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
