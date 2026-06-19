"""task198 (ARC-AGI 83302e8f) — "permeable line-grid: mark cells adjacent to a gap".

Rule (from the generator — closed-form, NOT a flood-fill):
  A size x size cell grid; each cell is minisize x minisize pixels, cells separated by
  single 1-px lines of `color`.  pitch p = minisize+1, actual_size = size*p-1 (<=29).
  Input: black canvas, grid lines drawn in `color`, then some line pixels punched back to
  black ("permeable points").  A permeable point on a VERTICAL line (c%p==p-1, r%p!=p-1)
  connects cells (R,C) and (R,C+1).  One on a HORIZONTAL line (r%p==p-1, c%p!=p-1) connects
  (R,C) and (R+1,C).  Output: green canvas, same lines; each permeable point becomes
  YELLOW(4); a cell interior is YELLOW(4) if ANY of its 4 walls has a gap, else GREEN(3).
  This is depth-1 (each gap marks exactly the two cells it separates) — no transitive flood.

Encoding (plane-lean): collapse to scalars/vectors + a tiny S x S cell-mark; route the
  10-channel one-hot into the FREE output.  Every full-grid working plane declared as
  small a dtype as its producer/consumers allow (uint8 carriers, bool masks, fp16 only for
  the gap/upsample MatMuls), and the final colour-index plane L is uint8 (900B) feeding a
  uint8 Equal -> bool output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I32 = TensorProto.INT32
I64 = TensorProto.INT64

N = 30
S = 7   # max cells per axis (size in 3..7)


def build(task):
    inits, nodes = [], []

    def init2(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def nn(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- colour scalar ----
    nn("ReduceSum", ["input"], "cnt0", axes=[2, 3], keepdims=1)
    mask0 = np.ones((1, 10, 1, 1), np.float32); mask0[0, 0] = 0.0
    init2("chmask", mask0, np.float32)
    nn("Mul", ["cnt0", "chmask"], "cnt")
    nn("ArgMax", ["cnt"], "color_i64", axis=1, keepdims=1)
    nn("Cast", ["color_i64"], "color_u8", to=U8)

    # ---- in-grid black plane: slice ch0 (fp32) then threshold to bool ----
    # ch0==1 exactly on in-grid black (off-grid is all-zero -> ch0=0).
    init2("s0", np.array([0], np.int64), np.int64)
    init2("s1", np.array([1], np.int64), np.int64)
    init2("ax1", np.array([1], np.int64), np.int64)
    nn("Slice", ["input", "s0", "s1", "ax1"], "ch0")              # [1,1,30,30] f32
    init2("HALF", np.array(0.5, np.float32), np.float32)
    nn("Greater", ["ch0", "HALF"], "isblack_b")                   # bool [1,1,30,30]

    # ---- pitch p and width via 1-D profiles (no full planes) ----
    rampR = np.arange(N, dtype=np.float16).reshape(1, 1, N, 1)
    rampC = np.arange(N, dtype=np.float16).reshape(1, 1, 1, N)
    init2("rampR", rampR, np.float16)
    init2("rampC", rampC, np.float16)
    init2("ONEH", np.array(1.0, np.float16), np.float16)

    nn("ReduceSum", ["input"], "colset", axes=[1, 2], keepdims=1)  # [1,1,1,30] f32
    init2("ZEROF", np.array(0.0, np.float32), np.float32)
    nn("Greater", ["colset", "ZEROF"], "colany_b")
    nn("Cast", ["colany_b"], "colany", to=F16)
    nn("Mul", ["colany", "rampC"], "cc")
    nn("ReduceMax", ["cc"], "Wm1", axes=[3], keepdims=1)            # actual_size-1

    # per-row nonblack count = total occupied - black, per row
    nn("ReduceSum", ["input"], "totrow", axes=[1, 3], keepdims=1)   # [1,1,30,1] f32
    nn("ReduceSum", ["ch0"], "blkrow", axes=[3], keepdims=1)        # [1,1,30,1] f32
    nn("Sub", ["totrow", "blkrow"], "rownb32")
    nn("Cast", ["rownb32"], "rownb", to=F16)
    init2("THR", np.array(6.5, np.float16), np.float16)
    nn("Greater", ["rownb", "THR"], "isline_r_b")                  # [1,1,30,1]
    init2("BIG", np.array(99.0, np.float16), np.float16)
    nn("Where", ["isline_r_b", "rampR", "BIG"], "liner_or_big")
    nn("ReduceMin", ["liner_or_big"], "pm1", axes=[2], keepdims=1)  # p-1
    nn("Add", ["pm1", "ONEH"], "pF")                              # p

    # ---- positional row/col line masks (vectors) ----
    nn("Mod", ["rampR", "pF"], "rmodp", fmod=1)
    nn("Mod", ["rampC", "pF"], "cmodp", fmod=1)
    nn("Equal", ["rmodp", "pm1"], "rline_b")                      # [1,1,30,1] bool
    nn("Equal", ["cmodp", "pm1"], "cline_b")                      # [1,1,1,30] bool
    nn("Not", ["rline_b"], "rnotline_b")
    nn("Not", ["cline_b"], "cnotline_b")

    # ---- single shared in-grid-black plane (fp16, 1800B) ----
    # Gv = isblack & cline(c) & rnotline(r) ; Gh = isblack & rline(r) & cnotline(c).
    # Separable contraction sharing ONE isblack16 plane:
    #   vrow[R,c] = (RselV @ isblack16)[R,c] ; vg1 = vrow*cline(c) ; Vgap = vg1 @ Csel
    #   hrow[R,c] = (RselL @ isblack16)[R,c] ; hg1 = hrow*cnot(c)  ; Hgap = hg1 @ Csel
    nn("Cast", ["isblack_b"], "isblack16", to=F16)               # [1,1,30,30] f16 (shared)

    # ---- cell indices + selector matrices (fp16) ----
    nn("Div", ["rampR", "pF"], "rdiv"); nn("Floor", ["rdiv"], "Ridx")
    nn("Div", ["rampC", "pF"], "cdiv"); nn("Floor", ["cdiv"], "Cidx")
    RidxK = np.arange(S, dtype=np.float16).reshape(1, 1, S, 1)
    init2("RidxK", RidxK, np.float16)
    init2("to1130", np.array([1, 1, 1, N], np.int64), np.int64)
    nn("Reshape", ["Ridx", "to1130"], "Ridx_r")
    nn("Equal", ["Ridx_r", "RidxK"], "Rsel_b")                   # [1,1,S,30] (R,r)
    nn("Cast", ["Rsel_b"], "Rsel", to=F16)
    CidxK = np.arange(S, dtype=np.float16).reshape(1, 1, 1, S)
    init2("CidxK", CidxK, np.float16)
    init2("to1301", np.array([1, 1, N, 1], np.int64), np.int64)
    nn("Reshape", ["Cidx", "to1301"], "Cidx_c")
    nn("Equal", ["Cidx_c", "CidxK"], "Csel_b")                   # [1,1,30,S] (c,C)
    nn("Cast", ["Csel_b"], "Csel", to=F16)

    # row-selectors with the row line/non-line filter folded in (tiny [1,1,S,30] f16)
    init2("to1113", np.array([1, 1, 1, N], np.int64), np.int64)
    nn("Reshape", ["rnotline_b", "to1113"], "rnot_r_b")         # [1,1,1,30]
    nn("Cast", ["rnot_r_b"], "rnot_r", to=F16)
    nn("Reshape", ["rline_b", "to1113"], "rln_r_b")            # [1,1,1,30]
    nn("Cast", ["rln_r_b"], "rln_r", to=F16)
    nn("Mul", ["Rsel", "rnot_r"], "RselV")                      # [1,1,S,30] (non-line rows)
    nn("Mul", ["Rsel", "rln_r"], "RselL")                       # [1,1,S,30] (line rows)
    nn("Cast", ["cline_b"], "cline16", to=F16)                  # [1,1,1,30]
    nn("Cast", ["cnotline_b"], "cnot16", to=F16)                # [1,1,1,30]

    # ---- downsample gaps to cell space (only [1,1,S,*] f16 intermediates) ----
    nn("MatMul", ["RselV", "isblack16"], "vrow")               # [1,1,S,30]
    nn("Mul", ["vrow", "cline16"], "vg1")
    nn("MatMul", ["vg1", "Csel"], "VgapS")                     # [1,1,S,S]
    nn("MatMul", ["RselL", "isblack16"], "hrow")               # [1,1,S,30]
    nn("Mul", ["hrow", "cnot16"], "hg1")
    nn("MatMul", ["hg1", "Csel"], "HgapS")                     # [1,1,S,S]
    init2("ZEROH", np.array(0.0, np.float16), np.float16)
    nn("Greater", ["VgapS", "ZEROH"], "Vg_b")
    nn("Greater", ["HgapS", "ZEROH"], "Hg_b")
    nn("Cast", ["Vg_b"], "Vg", to=F16)
    nn("Cast", ["Hg_b"], "Hg", to=F16)

    # ---- cell yellow Y[R,C] = Vg[R,C] | Vg[R,C-1] | Hg[R,C] | Hg[R-1,C] ----
    init2("pad_left", np.array([0, 0, 0, 1, 0, 0, 0, 0], np.int64), np.int64)
    init2("ZP", np.array(0.0, np.float16), np.float16)
    nn("Pad", ["Vg", "pad_left", "ZP"], "Vg_padL", mode="constant")
    init2("sl0", np.array([0], np.int64), np.int64)
    init2("slS", np.array([S], np.int64), np.int64)
    init2("ax3", np.array([3], np.int64), np.int64)
    nn("Slice", ["Vg_padL", "sl0", "slS", "ax3"], "VgL")
    init2("pad_top", np.array([0, 0, 1, 0, 0, 0, 0, 0], np.int64), np.int64)
    nn("Pad", ["Hg", "pad_top", "ZP"], "Hg_padT", mode="constant")
    init2("ax2", np.array([2], np.int64), np.int64)
    nn("Slice", ["Hg_padT", "sl0", "slS", "ax2"], "HgT")
    nn("Max", ["Vg", "VgL"], "y1")
    nn("Max", ["Hg", "HgT"], "y2")
    nn("Max", ["y1", "y2"], "Ycell")                            # [1,1,S,S] {0,1}

    # interior colour in cell space (uint8): Yellow(4)/Green(3)
    nn("Greater", ["Ycell", "ZEROH"], "Ycell_b")               # [1,1,S,S] bool
    init2("YELU2", np.array(4, np.uint8), np.uint8)
    init2("GRNU", np.array(3, np.uint8), np.uint8)
    nn("Where", ["Ycell_b", "YELU2", "GRNU"], "interiorCell")  # [1,1,S,S] uint8

    # ---- upsample interior colour via double-Gather (uint8, no fp16 full plane) ----
    init2("CLO", np.array(0.0, np.float16), np.float16)
    init2("CHI", np.array(float(S - 1), np.float16), np.float16)
    nn("Clip", ["Ridx", "CLO", "CHI"], "Ridxc")
    nn("Clip", ["Cidx", "CLO", "CHI"], "Cidxc")
    nn("Cast", ["Ridxc"], "Ridx32", to=I32)
    nn("Cast", ["Cidxc"], "Cidx32", to=I32)
    init2("flat30", np.array([N], np.int64), np.int64)
    nn("Reshape", ["Ridx32", "flat30"], "Ridx_idx")           # [30]
    nn("Reshape", ["Cidx32", "flat30"], "Cidx_idx")           # [30]
    nn("Gather", ["interiorCell", "Ridx_idx"], "ipR", axis=2)  # [1,1,30,S] uint8
    nn("Gather", ["ipR", "Cidx_idx"], "interiorL", axis=3)     # [1,1,30,30] uint8

    # ---- line colour plane: permeable point -> 4 else colour ----
    init2("YELU", np.array(4, np.uint8), np.uint8)
    nn("Where", ["isblack_b", "YELU", "color_u8"], "lineL")    # uint8 (900B)

    # ---- compose: online ? lineL : interiorL, then in-grid gate (off-grid -> 99) ----
    nn("Or", ["rline_b", "cline_b"], "online_b")              # bool [1,1,30,30]
    nn("Where", ["online_b", "lineL", "interiorL"], "gridL")  # uint8 (900B)
    nn("Greater", ["rampR", "Wm1"], "rog"); nn("Not", ["rog"], "ringrid_b")
    nn("Greater", ["rampC", "Wm1"], "cog"); nn("Not", ["cog"], "cingrid_b")
    nn("And", ["ringrid_b", "cingrid_b"], "ingrid_b")
    init2("SENT", np.array(99, np.uint8), np.uint8)
    nn("Where", ["ingrid_b", "gridL", "SENT"], "L")           # uint8 (900B)

    # ---- route 10-ch expansion to FREE output ----
    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init2("chan", chan, np.uint8)
    nn("Equal", ["L", "chan"], "output")                       # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task198", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
