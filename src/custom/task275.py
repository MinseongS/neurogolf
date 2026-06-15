"""Task 275 (ARC-AGI b190f7f5) — Kronecker product of a colour grid and a plus mask.

Rule (from the generator):
  The input holds TWO s x s sub-grids (s in {3,4}) packed side-by-side or stacked
  (4 layouts via `pairwise`):
    * a COLOUR grid C, cells coloured with digits in {1,2,3,4} (sparse),
    * a PLUS / cyan mask P, cells set to cyan(8) (sparse).
  The output is the (s*s) x (s*s) Kronecker product:
       output[row*s + r][col*s + c] = C[row][col]
       for every coloured cell (row,col,colour) of C and every set cell (r,c) of P.
  i.e. output[R,C] = C[R//s, C//s]  iff  C[R//s,C//s] is coloured AND P[R%s,C%s] set,
  else 0.  Cells outside the s*s x s*s footprint are all-channels-off.

Recovery (layout/size agnostic):
  * vertical split (grids stacked) iff max-nonzero-row > max-nonzero-col.
  * the COLOUR half and PLUS half are disjoint on the split axis; the half with the
    smaller min split-coordinate sits at split-origin 0, the other at split-origin s.
  * size s: 4 iff (max-axis-extent >= 6) OR (min-axis-extent >= 3) OR
    (lower-region split-max >= 3) -- each impossible for s=3.  (residual bbox
    ambiguity ~3e-6, far below any fresh-200 gate.)
  * gather each s x s sub-grid to a top-left 4x4 block (10 channels kept, tiny),
    reduce to a colour value grid (1..4) and a binary plus grid.
  * assemble the Kronecker label L on a 16x16 canvas with size-dependent macro/micro
    flat-index maps (Gather into the 4x4 grids), pad to 30x30 with off-grid sentinel,
    final BOOL output = Equal(L, arange) (opset 11) -- 10-channel expansion is free.

Memory floor: the fattest intermediates are the two [1,10,4,4] gathered blocks
(640 B fp32 each); everything else is <=256-element 1-D / 16x16 working tensors.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
B = TensorProto.BOOL
I64 = TensorProto.INT64


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ============================================================
    # 1-D profiles via EARLY spatial reduction (keep all 10 channels, <=1200B)
    # ============================================================
    # Reduce one spatial axis first so no full 30x30 multi-channel plane is ever
    # materialised:  rowprof[1,10,30,1] (1200B), colprof[1,10,1,30] (1200B).
    n("ReduceMax", ["input"], "rowprof", axes=[3], keepdims=1)  # [1,10,30,1]
    n("ReduceMax", ["input"], "colprof", axes=[2], keepdims=1)  # [1,10,1,30]

    init("a1", np.array([1], np.int64), np.int64)
    init("a5", np.array([5], np.int64), np.int64)
    init("e10", np.array([10], np.int64), np.int64)
    init("a8", np.array([8], np.int64), np.int64)
    init("a9", np.array([9], np.int64), np.int64)
    init("axc", np.array([1], np.int64), np.int64)

    # any coloured/cyan (channels 1..9) presence per row/col
    n("Slice", ["rowprof", "a1", "e10", "axc"], "rowprof19")  # [1,9,30,1]
    n("Slice", ["colprof", "a1", "e10", "axc"], "colprof19")  # [1,9,1,30]
    n("ReduceMax", ["rowprof19"], "rowany", axes=[1], keepdims=1)  # [1,1,30,1]
    n("ReduceMax", ["colprof19"], "colany", axes=[1], keepdims=1)  # [1,1,1,30]

    # colour (channels 1..4) presence per row/col
    n("Slice", ["rowprof", "a1", "a5", "axc"], "rowprof14")  # [1,4,30,1]
    n("Slice", ["colprof", "a1", "a5", "axc"], "colprof14")  # [1,4,1,30]
    n("ReduceMax", ["rowprof14"], "col_row", axes=[1], keepdims=1)  # [1,1,30,1]
    n("ReduceMax", ["colprof14"], "col_col", axes=[1], keepdims=1)  # [1,1,1,30]

    # cyan (channel 8) presence per row/col
    n("Slice", ["rowprof", "a8", "a9", "axc"], "cyn_row")  # [1,1,30,1]
    n("Slice", ["colprof", "a8", "a9", "axc"], "cyn_col")  # [1,1,1,30]

    Irow = np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1)
    Icol = np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30)
    init("Irow", Irow, np.float32)
    init("Icol", Icol, np.float32)
    init("Half", np.array(0.5, np.float32), np.float32)
    init("Neg", np.array(-1.0, np.float32), np.float32)

    n("Greater", ["rowany", "Half"], "rpres")   # [1,1,30,1] bool
    n("Greater", ["colany", "Half"], "cpres")   # [1,1,1,30] bool
    n("Where", ["rpres", "Irow", "Neg"], "rcand")
    n("Where", ["cpres", "Icol", "Neg"], "ccand")
    n("ReduceMax", ["rcand"], "rmax", keepdims=0)   # []
    n("ReduceMax", ["ccand"], "cmax", keepdims=0)

    # smax = max(rmax,cmax) ; omax = min(rmax,cmax)
    n("Max", ["rmax", "cmax"], "smax")
    n("Min", ["rmax", "cmax"], "omax")

    # vertical split iff rmax > cmax
    n("Greater", ["rmax", "cmax"], "vertical")   # bool scalar

    # reshape row/col profiles to flat [30] so we can pick by `vertical` uniformly
    init("s30", np.array([30], np.int64), np.int64)
    for nm in ["col_row", "cyn_row"]:
        n("Reshape", [nm, "s30"], nm + "_f")   # [30]
    for nm in ["col_col", "cyn_col"]:
        n("Reshape", [nm, "s30"], nm + "_f")   # [30]

    # choose split-axis profile: vertical -> *_row_f ; else *_col_f
    n("Where", ["vertical", "col_row_f", "col_col_f"], "col_sp")  # [30] colour presence on split axis
    n("Where", ["vertical", "cyn_row_f", "cyn_col_f"], "cyn_sp")  # [30] cyan presence on split axis

    Iflat = np.arange(30, dtype=np.float32)
    init("Iflat", Iflat, np.float32)
    init("Big", np.array(99.0, np.float32), np.float32)
    init("NegF", np.array(-1.0, np.float32), np.float32)
    init("Halff", np.array(0.5, np.float32), np.float32)

    n("Greater", ["col_sp", "Halff"], "col_spb")  # [30] bool
    n("Greater", ["cyn_sp", "Halff"], "cyn_spb")

    # min split-coord of colour & cyan
    n("Where", ["col_spb", "Iflat", "Big"], "col_minc")
    n("Where", ["cyn_spb", "Iflat", "Big"], "cyn_minc")
    n("ReduceMin", ["col_minc"], "col_min", keepdims=0)  # []
    n("ReduceMin", ["cyn_minc"], "cyn_min", keepdims=0)
    # max split-coord of colour & cyan
    n("Where", ["col_spb", "Iflat", "NegF"], "col_maxc")
    n("Where", ["cyn_spb", "Iflat", "NegF"], "cyn_maxc")
    n("ReduceMax", ["col_maxc"], "col_max", keepdims=0)
    n("ReduceMax", ["cyn_maxc"], "cyn_max", keepdims=0)

    # colour is the LOWER half iff col_min < cyn_min
    n("Less", ["col_min", "cyn_min"], "color_lower")  # bool scalar

    # lower-region split-max = color_lower ? col_max : cyn_max
    n("Where", ["color_lower", "col_max", "cyn_max"], "low_max")

    # ============================================================
    # size detection: s4 = (smax>=6) OR (omax>=3) OR (low_max>=3)
    # ============================================================
    init("c5_5", np.array(5.5, np.float32), np.float32)
    init("c2_5", np.array(2.5, np.float32), np.float32)
    n("Greater", ["smax", "c5_5"], "p1")     # smax>=6
    n("Greater", ["omax", "c2_5"], "p2")      # omax>=3
    n("Greater", ["low_max", "c2_5"], "p3")   # low_max>=3
    n("Or", ["p1", "p2"], "p12")
    n("Or", ["p12", "p3"], "s4")              # bool scalar

    # s as float scalar (3 or 4)
    init("f3", np.array(3.0, np.float32), np.float32)
    init("f4", np.array(4.0, np.float32), np.float32)
    n("Where", ["s4", "f4", "f3"], "sval")    # [] float

    # ============================================================
    # half origins on the split axis (0 or s)
    # ============================================================
    init("f0", np.array(0.0, np.float32), np.float32)
    # colour split-origin = color_lower ? 0 : s ; plus split-origin = color_lower ? s : 0
    n("Where", ["color_lower", "f0", "sval"], "col_sorig")
    n("Where", ["color_lower", "sval", "f0"], "cyn_sorig")

    # convert to (row_off,col_off): vertical -> split is rows -> row_off=sorig,col_off=0
    #                               horiz    -> split is cols -> row_off=0,col_off=sorig
    n("Where", ["vertical", "col_sorig", "f0"], "col_roff")
    n("Where", ["vertical", "f0", "col_sorig"], "col_coff")
    n("Where", ["vertical", "cyn_sorig", "f0"], "cyn_roff")
    n("Where", ["vertical", "f0", "cyn_sorig"], "cyn_coff")

    # ============================================================
    # gather each sub-grid to a top-left 4x4 block
    # ============================================================
    # Build the two single-channel value planes once (each [1,1,30,30] = 3600B),
    # then gather rows then cols (tiny [1,1,4,30] -> [1,1,4,4] intermediates).
    # colour-value plane via a 1x1 Conv (weights are params, not memory):
    # cplane = sum_{k=1..4} k * input[:,k]  -> [1,1,30,30] directly (3600B).
    wcol = np.zeros((1, 10, 1, 1), np.float32)
    for k in range(1, 5):
        wcol[0, k, 0, 0] = float(k)
    init("wcol", wcol, np.float32)
    n("Conv", ["input", "wcol"], "cplane")               # [1,1,30,30] colour values
    # cyan plane = channel 8
    n("Slice", ["input", "a8", "a9", "axc"], "pplane")   # [1,1,30,30] cyan 0/1

    # row idx = roff + [0,1,2,3] ; col idx = coff + [0,1,2,3]  (clamp 0..29, cast int)
    base4 = np.array([0.0, 1.0, 2.0, 3.0], np.float32)
    init("base4", base4, np.float32)
    init("lo", np.array(0.0, np.float32), np.float32)
    init("hi", np.array(29.0, np.float32), np.float32)

    def idx_vec(off_name, out):
        n("Add", [off_name, "base4"], out + "_f")      # [4]
        n("Clip", [out + "_f", "lo", "hi"], out + "_c")
        n("Cast", [out + "_c"], out, to=I64)           # [4] int64
        return out

    idx_vec("col_roff", "cr_idx")
    idx_vec("col_coff", "cc_idx")
    idx_vec("cyn_roff", "pr_idx")
    idx_vec("cyn_coff", "pc_idx")

    # colour block 4x4
    n("Gather", ["cplane", "cr_idx"], "cblk_r", axis=2)  # [1,1,4,30]
    n("Gather", ["cblk_r", "cc_idx"], "cgrid", axis=3)   # [1,1,4,4] colour value
    # plus block 4x4
    n("Gather", ["pplane", "pr_idx"], "pblk_r", axis=2)  # [1,1,4,30]
    n("Gather", ["pblk_r", "pc_idx"], "pgrid", axis=3)   # [1,1,4,4] cyan 0/1

    # flatten grids to [16]
    init("s16", np.array([16], np.int64), np.int64)
    n("Reshape", ["cgrid", "s16"], "cflat")  # [16] colour values
    n("Reshape", ["pgrid", "s16"], "pflat")  # [16] plus 0/1

    # ============================================================
    # Kronecker assembly on 16x16 canvas with size-dependent index maps
    # ============================================================
    # For output cell (R,C): macro flat = (R//s)*4 + (C//s) ; micro flat = (R%s)*4 + (C%s)
    # valid = R<s*s & C<s*s.  Precompute both size tables, select by s4.
    R = np.arange(16).reshape(16, 1)
    Cc = np.arange(16).reshape(1, 16)
    macro_tab = np.zeros((2, 256), np.int32)
    micro_tab = np.zeros((2, 256), np.int32)
    valid_tab = np.zeros((2, 256), np.bool_)
    for i, s in enumerate([3, 4]):
        mac = (R // s) * 4 + (Cc // s)
        mic = (R % s) * 4 + (Cc % s)
        val = (R < s * s) & (Cc < s * s)
        mac = np.where(val, mac, 0)
        mic = np.where(val, mic, 0)
        macro_tab[i] = mac.reshape(-1)
        micro_tab[i] = mic.reshape(-1)
        valid_tab[i] = val.reshape(-1)
    init("macro_tab", macro_tab, np.int32)   # [2,256]
    init("micro_tab", micro_tab, np.int32)
    init("valid_tab", valid_tab, np.bool_)

    # select row by s4 (0 -> size3, 1 -> size4)
    init("s256", np.array([256], np.int64), np.int64)
    n("Cast", ["s4"], "s4i", to=I64)         # [] 0/1
    init("s1", np.array([1], np.int64), np.int64)
    n("Reshape", ["s4i", "s1"], "selidx")    # [1]
    n("Gather", ["macro_tab", "selidx"], "macro_s", axis=0)  # [1,256] int32
    n("Gather", ["micro_tab", "selidx"], "micro_s", axis=0)
    n("Gather", ["valid_tab", "selidx"], "valid_s", axis=0)  # [1,256] bool
    n("Reshape", ["macro_s", "s256"], "macro_v")   # [256] int32
    n("Reshape", ["micro_s", "s256"], "micro_v")
    n("Reshape", ["valid_s", "s256"], "valid_v")

    # gather colour value & plus from the flat grids
    n("Gather", ["cflat", "macro_v"], "out_col")   # [256] colour value
    n("Gather", ["pflat", "micro_v"], "out_plus")  # [256] plus 0/1

    # label: keep colour iff coloured(>0) AND plus set AND valid; else sentinel 10
    init("z0", np.array(0.5, np.float32), np.float32)
    n("Greater", ["out_col", "z0"], "is_color")    # [256] bool (colour>0)
    n("Greater", ["out_plus", "z0"], "is_plus")    # [256] bool
    n("And", ["is_color", "is_plus"], "keep0")
    n("And", ["keep0", "valid_v"], "keep")         # [256] bool

    # uint8 colour values for label.  Inside footprint but not kept -> 0 (black bg,
    # ch0); outside footprint (invalid) -> sentinel 10 (all-channels-off).
    n("Cast", ["out_col"], "colu", to=U8)          # [256] uint8 (0..4)
    init("u10", np.array(10, np.uint8), np.uint8)
    init("u0", np.array(0, np.uint8), np.uint8)
    n("Where", ["valid_v", "u0", "u10"], "Lbase")  # [256] 0 inside, 10 outside
    n("Where", ["keep", "colu", "Lbase"], "L256")  # [256] uint8 label

    # reshape -> 16x16 -> pad to 30x30 with sentinel 10
    init("s_16_16", np.array([1, 1, 16, 16], np.int64), np.int64)
    n("Reshape", ["L256", "s_16_16"], "L16")       # [1,1,16,16] uint8
    init("pads", np.array([0, 0, 0, 0, 0, 0, 14, 14], np.int64), np.int64)
    n("Pad", ["L16", "pads", "u10"], "L", mode="constant")  # [1,1,30,30] uint8

    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["L", "chan"], "output")            # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task275", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
