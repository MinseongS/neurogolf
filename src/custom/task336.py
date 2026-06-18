"""task336 (ARC-AGI d4f3cd78) — fill the gray container and drip out the gap.

Rule (from the generator, verified 0/10000 fresh in numpy):
  size-10 grid drawn with only background(0) and gray(5).  A gray "container" is
  a closed rectangle of gray walls (top+bottom+left+right) EXCEPT a single 1-cell
  gap in ONE wall; apply_gravity puts the gap on any of the 4 walls.  OUTPUT
  keeps the gray walls and ADDS cyan(8): the rectangle INTERIOR is filled, and a
  straight "drip" of cyan flows OUT through the gap to the grid edge.  Every
  change is 0 -> 8.

Closed-form SEPARABLE characterisation (verified 0/10000 fresh):
  gray = channel-5.  Row/col gray-occupancy profiles grow/gcol ([1,1,10,1] /
  [1,1,1,10]) give, by 1-D strict prefix/suffix-OR (exclusive CumSum, >0):
    aboveR/belowR (gray strictly above/below a row), aboveC/belowC (left/right).
    rowin = aboveR & belowR,  colin = aboveC & belowC      (strictly-inside band)
    interior band = rowin & colin                          (rectangle interior)
  Wall lines: toprow = grow & ~aboveR (topmost gray row), botrow/leftcol/rightcol
  analogously (one-hot per axis).  The gap is the lone bg cell on a wall.  Read
  that wall line by a SCALAR Gather (no 2-D plane): wall index = Σ(one-hot·ramp);
  Gather the gray row/col at that index ([1,1,1,10] / [1,1,10,1]); the gap is
  where the line is bg, restricted to the interior span:
    gaptop = (toprow_line < .5) & colin    gapbot/gapleft/gapright analogous.
  The drip extends from that wall OUTWARD to the edge (at-or-beyond the wall):
    vdrip_up = ~aboveR & gaptop      (rows at/above the top wall, gap column)
    vdrip_dn = ~belowR & gapbot
    hdrip_l  = gapleft  & ~aboveC
    hdrip_r  = gapright & ~belowC
  cyan = interior-band | the four drip terms.  Each term is a rank-1 product of
  two 1-D masks; the terms are inherently all-bg (the input interior + gap line
  are empty), so NO bg gate and NO general 2-D scan/flood is needed.

Encoding (tiny 10x10 active canvas):
  ONE fp32 Slice of channel 5 (the only 2-D plane, 400B).  Profiles via
  ReduceMax; the prefix/suffix-OR scans are exclusive CumSum (no matrix params);
  the 4 wall lines are scalar-index Gathers (40B vectors).  Each cyan rank-1
  term is a 100B bool 10x10 plane; cyan = OR.  L10 = Where(cyan,8,Where(gray,5,0)),
  Pad to 30x30 with sentinel 10, Equal(L,arange[0..9]) -> FREE BOOL output.
  Dominant intermediate: the 30x30 uint8 carrier L (900B) + the 10x10 gray slice
  (400B); everything else is <=100B.  mem ~3.4k, params ~56.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

U8 = TensorProto.UINT8
B = TensorProto.BOOL
F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT

WORK = 10


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    init("ax2", np.array(2, np.int64), np.int64)   # CumSum axis = rows
    init("ax3", np.array(3, np.int64), np.int64)   # CumSum axis = cols
    init("half32", np.array(0.5, np.float32), np.float32)
    init("rrow", np.arange(WORK, dtype=np.float16).reshape(1, 1, WORK, 1), np.float16)
    init("rcol", np.arange(WORK, dtype=np.float16).reshape(1, 1, 1, WORK), np.float16)
    init("shape1", np.array([1], np.int64), np.int64)

    init("g_st", np.array([5, 0, 0], np.int64), np.int64)
    init("g_en", np.array([6, WORK, WORK], np.int64), np.int64)
    init("g_ax", np.array([1, 2, 3], np.int64), np.int64)
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    init("u8", np.array(8, np.uint8), np.uint8)
    init("u5", np.array(5, np.uint8), np.uint8)
    init("u0", np.array(0, np.uint8), np.uint8)
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)

    # ---- gray {0,1} on the 10x10 corner ----
    n("Slice", ["input", "g_st", "g_en", "g_ax"], "gray_f32")  # [1,1,10,10] fp32
    n("Greater", ["gray_f32", "half32"], "grayb")

    # ---- row/col gray occupancy profiles (fp32 for CumSum) ----
    n("ReduceMax", ["gray_f32"], "grow", axes=[3], keepdims=1)   # [1,1,10,1] fp32
    n("ReduceMax", ["gray_f32"], "gcol", axes=[2], keepdims=1)   # [1,1,1,10] fp32

    # 1-D strict prefix/suffix-OR scans via exclusive CumSum (>0), no matrix params
    n("CumSum", ["grow", "ax2"], "aboveR_f", exclusive=1, reverse=0)  # gray strictly above
    n("CumSum", ["grow", "ax2"], "belowR_f", exclusive=1, reverse=1)  # gray strictly below
    n("CumSum", ["gcol", "ax3"], "aboveC_f", exclusive=1, reverse=0)  # gray strictly left
    n("CumSum", ["gcol", "ax3"], "belowC_f", exclusive=1, reverse=1)  # gray strictly right
    for nm in ("aboveR", "belowR", "aboveC", "belowC"):
        n("Greater", [nm + "_f", "half32"], nm)

    # ---- interior band (separable, [1,1,10,1] & [1,1,1,10]) ----
    n("And", ["aboveR", "belowR"], "rowin")    # [1,1,10,1]
    n("And", ["aboveC", "belowC"], "colin")    # [1,1,1,10]

    # ---- wall lines (boundary gray rows/cols), all 1-D ----
    n("Not", ["aboveR"], "naR")
    n("Not", ["belowR"], "nbR"); n("Not", ["aboveC"], "naC"); n("Not", ["belowC"], "nbC")
    n("Greater", ["grow", "half32"], "growb")
    n("Greater", ["gcol", "half32"], "gcolb")
    n("And", ["growb", "naR"], "toprow")    # [1,1,10,1] bool one-hot: topmost gray row
    n("And", ["growb", "nbR"], "botrow")
    n("And", ["gcolb", "naC"], "leftcol")   # [1,1,1,10]
    n("And", ["gcolb", "nbC"], "rightcol")

    # ---- gap detection via a scalar Gather of the wall line (NO 2-D plane) ----
    # wall index = Σ (one-hot row · ramp); Gather that gray line; gap = (gray<.5)&span.
    def wall_index(onehot, ramp, idxname):
        # weighted-sum the one-hot wall line to a scalar index, reshape to [1]
        n("Cast", [onehot], idxname + "_f", to=F16)
        n("Mul", [idxname + "_f", ramp], idxname + "_w")
        n("ReduceSum", [idxname + "_w"], idxname + "_s", axes=[1, 2, 3], keepdims=0)
        n("Cast", [idxname + "_s"], idxname + "_i", to=TensorProto.INT64)
        n("Reshape", [idxname + "_i", "shape1"], idxname)      # [1]
        return idxname

    def wall_index_row(onehot, idxname):
        return wall_index(onehot, "rrow", idxname)

    def wall_index_col(onehot, idxname):
        return wall_index(onehot, "rcol", idxname)

    # top/bottom walls -> Gather a row (axis 2) -> [1,1,1,10]; gap cols = bg in span
    wall_index_row("toprow", "ti")
    n("Gather", ["gray_f32", "ti"], "toprow_line", axis=2)   # [1,1,1,10]
    n("Less", ["toprow_line", "half32"], "tl0"); n("And", ["tl0", "colin"], "gaptop")
    wall_index_row("botrow", "bi")
    n("Gather", ["gray_f32", "bi"], "botrow_line", axis=2)
    n("Less", ["botrow_line", "half32"], "bl0"); n("And", ["bl0", "colin"], "gapbot")
    # left/right walls -> Gather a col (axis 3) -> [1,1,10,1]; gap rows = bg in span
    wall_index_col("leftcol", "li")
    n("Gather", ["gray_f32", "li"], "leftcol_line", axis=3)  # [1,1,10,1]
    n("Less", ["leftcol_line", "half32"], "ll0"); n("And", ["ll0", "rowin"], "gapleft")
    wall_index_col("rightcol", "ri")
    n("Gather", ["gray_f32", "ri"], "rightcol_line", axis=3)
    n("Less", ["rightcol_line", "half32"], "rl0"); n("And", ["rl0", "rowin"], "gapright")

    # ---- cyan = bg & ( interior-band  OR  the four rank-1 drip rays ) ----
    n("And", ["rowin", "colin"], "t_band")
    n("And", ["naR", "gaptop"], "t_vu")     # rows at/above top wall, gap column
    n("And", ["nbR", "gapbot"], "t_vd")
    n("And", ["gapleft", "naC"], "t_hl")
    n("And", ["gapright", "nbC"], "t_hr")
    # the band/drip terms are inherently all-bg (interior + gap line are empty in
    # the input), so no extra bg gate is needed (verified 0/10000).
    n("Or", ["t_band", "t_vu"], "c1")
    n("Or", ["c1", "t_vd"], "c2")
    n("Or", ["c2", "t_hl"], "c3")
    n("Or", ["c3", "t_hr"], "cyan")             # [1,1,10,10] bool

    # ---- colour plane: gray->5, cyan->8, else 0 (bg); pad to 30x30 -> Equal ----
    n("Where", ["grayb", "u5", "u0"], "base")   # uint8 5 on gray
    n("Where", ["cyan", "u8", "base"], "L10")   # uint8
    n("Pad", ["L10", "padpads", "padval"], "L", mode="constant")  # [1,1,30,30]
    n("Equal", ["L", "chan"], "output")         # -> FREE BOOL output

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task336", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
