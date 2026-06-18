"""task185 (ARC-AGI 7837ac64) — decode a line-grid stamp into its 3x3 colour key.

Rule (generator task_7837ac64.py):
  The input is a `size x size` cell line-grid drawn with `linecolor` lines at spacing
  `sp` (sp in {2,3,4}, size in {6,7,10}); period p = sp+1, line rows/cols at index
  i*p + (p-1) (i = 0..size-1).  A 3x3 colour key `colors` (1..9, plus 0 = blank, all
  same-colour pixels diagonally connected, no empty 3x3 margins) is stamped onto the
  line INTERSECTIONS: cell (row,col) with colour c paints c at the 2x2 block of
  intersections (brow+row+1..+2, bcol+col+1..+2).  OUTPUT is the 3x3 `colors` grid.

Decode (verified 400/400 against the generator):
  Let S[i,j] = colour at intersection (i,j) (linecolor / off-grid -> 0).  Output cell
  (i,j) is colour c iff the 2x2 block S[a:a+2,b:b+2] (a=r0+i, b=c0+j) is uniformly one
  NONZERO colour c, where (r0,c0) is the top-left of the bounding box of all 2x2-mono
  blocks.  i.e. m2[a,b] = (the four S cells of the 2x2 are equal AND nonzero); v2 = that
  colour; output = v2 cropped to the m2 bounding box (always exactly 3x3).

Encoding (collapse to BITMAP resolution, task080/task159 lattice lever):
  colf = Conv(input, k-ramp) -> the ONE fp32 30x30 entry plane.  p = first full-line row
  index + 1.  Downsample colf at line indices i*p+(p-1) (clamped, off-grid masked) ->
  10x10 fp16 subgrid S; subtract linecolor (= S[0,0], the always-plain corner
  intersection) -> {0, colour}.  On the tiny 10x10, build the four 2x2-corner planes by
  ONE pad-to-12x12 + fixed Slices; m2 = all-equal & >0 ; v2 = colour where m2.  r0/c0 =
  ArgMax of the row-any / col-any of m2.  Gather a 3x3 window of v2 at (r0..r0+2,
  c0..c0+2) -> the 3x3 colour key.  Pad to 30x30 (sentinel 99) and finish with
  Equal(L_uint8, arange) into the FREE BOOL output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I32 = TensorProto.INT32
I64 = TensorProto.INT64

W = 30
K = 10     # subgrid side (size <= 10)


def build(task):
    inits, nodes = [], []
    _np = {F16: np.float16, F32: np.float32, BOOL: np.bool_, U8: np.uint8,
           I32: np.int32, I64: np.int64}

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=_np[dtype]), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- colour-index plane colf [1,1,30,30] fp32 (the ONE entry plane) ----
    init("KW", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), F32)
    n("Conv", ["input", "KW"], "colf32")            # sum_k k*input_k

    # ---- per-row non-bg count (no-pad 1x30 conv over channels 1..9) --------
    rk = np.zeros((1, 10, 1, W), np.float32); rk[0, 1:, 0, :] = 1.0
    init("ROWK", rk, F32)
    n("Conv", ["input", "ROWK"], "rowsum")          # [1,1,30,1] fp32 per-row count

    # ---- period p = first full-line row index + 1 --------------------------
    n("ReduceMax", ["rowsum"], "Amax", axes=[2, 3], keepdims=1)   # active width A
    init("HALFF", np.array(0.5, np.float32), F32)
    n("Sub", ["Amax", "HALFF"], "WTHR")
    n("Greater", ["rowsum", "WTHR"], "isrow_b")     # full-line rows
    n("Cast", ["isrow_b"], "isrowf", to=F16)
    n("ArgMax", ["isrowf"], "pm1_i64", axis=2, keepdims=0)   # [1,1,1] = p-1
    n("Cast", ["pm1_i64"], "pm1_i32", to=I32)
    init("ONE_I", np.array(1, np.int32), I32)
    n("Add", ["pm1_i32", "ONE_I"], "p_i32a")
    init("flat1", np.array([1], np.int64), I64)
    n("Reshape", ["p_i32a", "flat1"], "p_i32")      # [1] = p

    # ---- line indices  li = i*p + (p-1) , clamp <=29 -----------------------
    init("ar10f", np.arange(K, dtype=np.float32), F32)
    n("Cast", ["p_i32"], "pf0", to=F32)             # [1]
    n("Mul", ["ar10f", "pf0"], "ipf")               # i*p
    init("ONEF", np.array(1.0, np.float32), F32)
    n("Sub", ["pf0", "ONEF"], "pm1f")               # p-1
    n("Add", ["ipf", "pm1f"], "lif")                # i*p + (p-1)  [10]
    init("HI29F", np.array(float(W - 1), np.float32), F32)
    n("Min", ["lif", "HI29F"], "licf")              # clamp <=29
    n("Cast", ["licf"], "lidx", to=I32)             # [10]

    # ---- downsample colf at line indices -> S [1,1,10,10] ------------------
    n("Gather", ["colf32", "lidx"], "Sg", axis=2)   # [1,1,10,30]
    n("Gather", ["Sg", "lidx"], "Sraw32", axis=3)   # [1,1,10,10] fp32 colour idx
    n("Cast", ["Sraw32"], "Sraw", to=F16)

    # ---- off-grid validity: line i valid iff li < H (=A here uses col count;
    #      grid is square so per-row width A == per-col height) --------------
    n("Cast", ["Amax"], "AmaxF", to=F32)
    n("Less", ["lif", "AmaxF"], "valid_b")          # [10] broadcast -> [1,1,1,10]
    n("Cast", ["valid_b"], "validf", to=F16)
    init("v_rsh", np.array([1, 1, K, 1], np.int64), I64)
    init("v_csh", np.array([1, 1, 1, K], np.int64), I64)
    n("Reshape", ["validf", "v_rsh"], "vrow")
    n("Reshape", ["validf", "v_csh"], "vcol")
    n("Mul", ["vrow", "vcol"], "vmask")             # [1,1,10,10]
    n("Mul", ["Sraw", "vmask"], "Sv")               # off-grid -> 0

    # ---- subtract linecolor.  Recover it from a NON-intersection line cell:
    #      colf[p-1, 0] is on the first line row but column 0 is never an
    #      intersection (intersections sit at col i*p+(p-1)), so it is always
    #      the plain linecolor.  (S[0,0] fails when brow=bcol=0 stamps the
    #      corner intersection.) -----------------------------------------------
    n("Reshape", ["pm1_i32", "flat1"], "pm1v")      # [1] = p-1
    n("Gather", ["colf32", "pm1v"], "lcrow", axis=2)  # [1,1,1,30]
    init("z00", np.array([0], np.int64), I64)
    n("Gather", ["lcrow", "z00"], "lc32", axis=3)   # [1,1,1,1] linecolor fp32
    n("Cast", ["lc32"], "lc", to=F16)
    n("Equal", ["Sv", "lc"], "islc_b")
    init("ZH", np.array(0.0, np.float16), F16)
    n("Where", ["islc_b", "ZH", "Sv"], "S")         # [1,1,10,10] {0, colour}

    # ---- four 2x2-corner planes via ONE pad-to-12x12 + fixed Slices --------
    # block (a,b) corners: S[a,b], S[a+1,b], S[a,b+1], S[a+1,b+1].
    init("padS", np.array([0, 0, 0, 0, 0, 0, 2, 2], np.int64), I64)  # pad bottom/right by 2
    n("Pad", ["S", "padS", "ZH"], "Sp")             # [1,1,12,12]
    init("SLAX", np.array([2, 3], np.int64), I64)

    def win(s0, s1, tag):
        init(f"ws_{tag}", np.array([s0, s1], np.int64), I64)
        init(f"we_{tag}", np.array([s0 + K, s1 + K], np.int64), I64)
        return n("Slice", ["Sp", f"ws_{tag}", f"we_{tag}", "SLAX"], tag)

    A00 = win(0, 0, "A00")    # = S (top-left of each 2x2)
    A10 = win(1, 0, "A10")
    A01 = win(0, 1, "A01")
    A11 = win(1, 1, "A11")

    # ---- m2 = all four equal AND first nonzero ; v2 = the colour -----------
    n("Equal", [A00, A10], "e1_b")
    n("Equal", [A00, A01], "e2_b")
    n("Equal", [A00, A11], "e3_b")
    n("And", ["e1_b", "e2_b"], "ea_b")
    n("And", ["ea_b", "e3_b"], "eq_b")              # all four equal
    n("Greater", [A00, "ZH"], "nz_b")               # top-left nonzero
    n("And", ["eq_b", "nz_b"], "m2_b")              # [1,1,10,10] bool
    n("Cast", ["m2_b"], "m2f", to=F16)

    # ---- r0/c0 = ArgMax of m2 row-any / col-any ----------------------------
    n("ReduceMax", ["m2f"], "rowany", axes=[3], keepdims=1)   # [1,1,10,1]
    n("ReduceMax", ["m2f"], "colany", axes=[2], keepdims=1)   # [1,1,1,10]
    n("ArgMax", ["rowany"], "r0_i64", axis=2, keepdims=0)     # [1,1,1]
    n("ArgMax", ["colany"], "c0_i64", axis=3, keepdims=0)     # [1,1,1]
    n("Cast", ["r0_i64"], "r0_i32", to=I32)
    n("Cast", ["c0_i64"], "c0_i32", to=I32)
    n("Reshape", ["r0_i32", "flat1"], "r0")         # [1]
    n("Reshape", ["c0_i32", "flat1"], "c0")         # [1]

    # ---- v2 = S where m2 (the per-block colour; top-left cell of the 2x2) ---
    n("Where", ["m2_b", A00, "ZH"], "v2")           # [1,1,10,10] fp16

    # ---- gather the 3x3 window v2[r0..r0+2, c0..c0+2] ----------------------
    init("ar3", np.array([0, 1, 2], np.int32), I32)
    n("Add", ["r0", "ar3"], "ridx")                 # [3]
    n("Add", ["c0", "ar3"], "cidx")                 # [3]
    n("Gather", ["v2", "ridx"], "vr", axis=2)       # [1,1,3,10]
    n("Gather", ["vr", "cidx"], "v33", axis=3)      # [1,1,3,3] fp16 colour key

    # ---- route to 30x30 BOOL output ---------------------------------------
    n("Cast", ["v33"], "L8", to=U8)                 # [1,1,3,3] uint8
    init("padO", np.array([0, 0, 0, 0, 0, 0, W - 3, W - 3], np.int64), I64)
    init("S99", np.array(99, np.uint8), U8)
    n("Pad", ["L8", "padO", "S99"], "Lpad")         # [1,1,30,30] uint8 (outside=99)
    init("arangeU", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), U8)
    n("Equal", ["Lpad", "arangeU"], "output")       # [1,10,30,30] BOOL

    used = {i for node in nodes for i in node.input}
    inits = [t for t in inits if t.name in used]

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task185", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
