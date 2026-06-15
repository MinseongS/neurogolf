"""Task 195 (ARC-AGI 80af3007) — fractal self-tiling of a 3x3 sprite.

Rule (from the generator): a 3x3 binary sprite S (conway_sprite, always covers
every sprite row and column) is upscaled 3x (each on-cell -> a 3x3 gray block)
and dropped at a random (rowoffset, coloffset) on a ~16-19 wide canvas; only the
gray colour 5 ever appears.  The OUTPUT is always a 9x9 grid equal to the
Kronecker product  kron(S, S) * 5 :
    output[3i+r, 3j+c] = 5  iff  S[i,j] AND S[r,c]   (else 0).
The output one-hot covers ONLY the top-left 9x9 footprint; cells outside it are
all-channels-off.

Recovery (offset-free, generalising):
  * The input's nonzero bounding box is ALWAYS exactly 9x9 (the sprite covers
    every row/col), so r0 = first occupied row, c0 = first occupied col.
  * row/col presence are 1-D Conv reductions of the colour-5 channel (kept as
    tiny [1,1,30,1] / [1,1,1,30] profiles -- never a 2-D occupancy plane).
  * S[i,j] = occ[r0+3i, c0+3j].  We gather those 3 rows then 3 cols straight out
    of the colour-5 occupancy and keep a 3x3 sprite.
  * label L (9x9, uint8) = 5 where kron(S,S), else 0; the free BOOL output is
    Equal(L, arange[1,10,1,1]) (opset 11).  kron(S,S)[u,v] = S[u//3,u%3] *
    S[v//3,v%3] is assembled by gather+broadcast, never a colour image.

Memory floor: the lone 30x30 plane is the colour-5 occupancy occ
(ReduceMax over channels, fp16 = 1800 B); everything else is <=300 B (1-D
profiles, 3x3 sprite, 9x9 label).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
B = TensorProto.BOOL


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- colour-5 occupancy on a small working canvas ----
    # The upscaled sprite (9x9 + offset) always lies within rows 0..15, cols
    # 0..17 (width<=19, height<=17, offset>=1; verified 30k fresh: last live
    # row<=15, col<=17), so a 16x18 channel-5 slice captures every live cell.
    # Slice keeps fp32 -> 16*18*4 = 1152 B.
    HR, WC = 16, 18
    init("ss", np.array([0, 5, 0, 0], np.int64), np.int64)
    init("se", np.array([1, 6, HR, WC], np.int64), np.int64)
    init("sax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "ss", "se", "sax"], "occ")  # [1,1,HR,WC] fp32 (1152B)

    # ---- 1-D presence profiles (tiny) ----
    n("ReduceMax", ["occ"], "rowocc_f", axes=[3], keepdims=1)  # [1,1,HR,1]
    n("ReduceMax", ["occ"], "colocc_f", axes=[2], keepdims=1)  # [1,1,1,WC]

    # r0 = first occupied row, c0 = first occupied col (scalars).
    # rowidx = present ? r : 99 ; r0 = min.
    Irow = np.arange(HR, dtype=np.float32).reshape(1, 1, HR, 1)
    Icol = np.arange(WC, dtype=np.float32).reshape(1, 1, 1, WC)
    init("Irow", Irow, np.float32)
    init("Icol", Icol, np.float32)
    init("Big", np.array(99.0, np.float32), np.float32)
    init("Half", np.array(0.5, np.float32), np.float32)
    n("Greater", ["rowocc_f", "Half"], "rpres")     # [1,1,30,1] bool
    n("Greater", ["colocc_f", "Half"], "cpres")
    n("Where", ["rpres", "Irow", "Big"], "rcand")   # [1,1,30,1]
    n("Where", ["cpres", "Icol", "Big"], "ccand")
    n("ReduceMin", ["rcand"], "r0", keepdims=1)     # [1,1,1,1] = first row
    n("ReduceMin", ["ccand"], "c0", keepdims=1)     # [1,1,1,1] = first col

    # ---- gather the 3 sprite rows then 3 sprite cols from occ ----
    # row indices r0 + [0,3,6]; col indices c0 + [0,3,6].  Build as fp32, clip,
    # cast to int64 for Gather (ORT Clip rejects int64 -> clip in float first).
    init("step", np.array([0.0, 3.0, 6.0], np.float32), np.float32)  # [3]
    # r0 is [1,1,1,1]; reshape to scalar [1] then add step.
    init("sc1", np.array([1], np.int64), np.int64)
    n("Reshape", ["r0", "sc1"], "r0s")              # [1]
    n("Reshape", ["c0", "sc1"], "c0s")              # [1]
    n("Add", ["r0s", "step"], "ridx_f")             # [3] fp32
    n("Add", ["c0s", "step"], "cidx_f")             # [3] fp32
    init("lo", np.array(0.0, np.float32), np.float32)
    init("hir", np.array(float(HR - 1), np.float32), np.float32)
    init("hic", np.array(float(WC - 1), np.float32), np.float32)
    n("Clip", ["ridx_f", "lo", "hir"], "ridx_c")    # [3]
    n("Clip", ["cidx_f", "lo", "hic"], "cidx_c")
    n("Cast", ["ridx_c"], "ridx", to=TensorProto.INT64)  # [3]
    n("Cast", ["cidx_c"], "cidx", to=TensorProto.INT64)

    # Gather rows (axis 2) then cols (axis 3) from occ -> S [1,1,3,3] fp32.
    n("Gather", ["occ", "ridx"], "occr", axis=2)    # [1,1,3,30] fp32
    n("Gather", ["occr", "cidx"], "S", axis=3)      # [1,1,3,3] fp32

    # ---- build kron(S,S) on a 9x9 canvas ----
    # kron(S,S)[u,v] = Sflat[macro] * Sflat[micro] with constant index maps
    #   macro = (u//3)*3 + (v//3),  micro = (u%3)*3 + (v%3)   (u,v in 0..8).
    # Gather Sflat (9 values) with each flattened [9,9] index map.
    init("s9", np.array([9], np.int64), np.int64)
    n("Reshape", ["S", "s9"], "Sflat")              # [9]
    u = np.arange(9).reshape(9, 1)
    v = np.arange(9).reshape(1, 9)
    macro = ((u // 3) * 3 + (v // 3)).astype(np.int64)   # [9,9]
    micro = ((u % 3) * 3 + (v % 3)).astype(np.int64)     # [9,9]
    init("macro", macro, np.int64)
    init("micro", micro, np.int64)
    # work in bool: kron = Smac AND Smic (ORT Mul rejects uint8/bool, And is fine)
    n("Greater", ["Sflat", "Half"], "Sb")           # [9] bool (sprite on/off)
    n("Gather", ["Sb", "macro"], "Smac")            # [9,9] bool
    n("Gather", ["Sb", "micro"], "Smic")            # [9,9] bool
    n("And", ["Smac", "Smic"], "Kb")                # [9,9] bool (kron)

    # ---- label map L (9x9, uint8): 5 where kron, else 0 ----
    init("u5", np.array(5, np.uint8), np.uint8)
    init("u0", np.array(0, np.uint8), np.uint8)
    n("Where", ["Kb", "u5", "u0"], "Lsm")           # [9,9] uint8 (81B)
    init("Ls", np.array([1, 1, 9, 9], np.int64), np.int64)
    n("Reshape", ["Lsm", "Ls"], "Lsm4")             # [1,1,9,9] uint8

    # pad 9x9 label -> 30x30 with off-grid sentinel 10 (cells outside footprint
    # become all-channels-off since 10 matches no channel index 0..9).
    init("u10", np.array(10, np.uint8), np.uint8)
    init("pads", np.array([0, 0, 0, 0, 0, 0, 21, 21], np.int64), np.int64)
    n("Pad", ["Lsm4", "pads", "u10"], "L", mode="constant")  # [1,1,30,30] uint8

    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["L", "chan"], "output")             # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task195", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
