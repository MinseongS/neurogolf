"""task174 (ARC-AGI 72ca375d) — extract the symmetric box, move to origin.

Rule (from the generator, verified):
  Three monochrome "boxes" (connected creatures) are placed on a 10x10 grid, each in
  its OWN colour channel.  Box idx 0 is forced to be BOTH vertically-mirror-symmetric
  AND 180-rotationally symmetric about its own bbox; boxes 1,2 are forced to be NEITHER.
  The OUTPUT is box-0's shape, cropped to its bbox and translated to the top-left origin,
  in box-0's colour (channel-0 fills the holes inside the HxW bbox; everything outside
  the bbox is all-channels-off).

  UNIQUE discriminator (verified 0 bad / 8000 fresh): box-0 is the ONLY present colour
  whose bbox-cropped shape equals its own horizontal (column) mirror.  (Symmetric-only
  already discriminates because boxes 1,2 are NEITHER symmetric nor rotational.)

Plane-free encoding (column-signature palindrome — NO full 10-channel reflection plane):
  For colour channel k, the COLUMN SIGNATURE sig_k[c] = sum_r 2^r * M_k[r,c] is a bitmask
  of which rows column c fills.  The shape is horizontally symmetric  <=>  sig_k[] is a
  PALINDROME about its column-bbox centre a_k = cmin+cmax, i.e. sig_k[c]==sig_k[a_k-c] for
  every c in [cmin,cmax] (verified exact 0/8000).  All this runs on tiny [1,10,1,10] vectors.

  - sig = MatMul(rowweight[1,1,1,30], input[1,10,30,30]) -> [1,10,1,30]  (off the FREE input,
    no full 10x10x10 plane ever materialises), sliced to the 10-col active region.
  - reflect via GatherElements on idx=clip(a-c): refl[k,c]=sig_k[clip(a_k-c)].
  - is_sym_k = (sig==refl on the column-span) ; special = is_sym AND present AND k!=0.
  - k = ArgMax(special).
  - decode box from sig DIRECTLY (no box plane): gather sig of channel k, shift cols to
    origin, divide by 2^rmin to shift rows to origin, then box[r,c]=floor(sig'[c]/2^r) mod 2
    on a 5x5 tile (box bbox is always <=5x5).
  - label L (uint8): box0colour on box cells, 0 on inside-bbox holes, sentinel 10 outside;
    Pad to 30x30, Equal(L, arange[0..9]) -> BOOL output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64

N = 30      # full canvas
W = 10      # active region side (grid is 10x10, content top-left)
K = 5       # box bbox is always <= 5x5
NB = 999.0  # big sentinel for min, -NB for max


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, list(ins), [out], **attrs))
        return out

    # ====================================================================
    # 1. column signatures sig_k[c] = sum_r 2^r M_k[r,c]   (off the FREE input)
    # ====================================================================
    roww = (2.0 ** np.arange(N)).astype(np.float32)
    roww[W:] = 0.0                                  # only rows 0..9 carry content
    init("rowwt", roww.reshape(1, 1, 1, N), np.float32)   # [1,1,1,30]
    n("MatMul", ["rowwt", "input"], "sig30")        # [1,10,1,30] f32
    init("c_s", np.array([0], np.int64), np.int64)
    init("c_e", np.array([W], np.int64), np.int64)
    init("c_ax", np.array([3], np.int64), np.int64)
    n("Slice", ["sig30", "c_s", "c_e", "c_ax"], "sig")   # [1,10,1,10] f32 (400B)

    # ====================================================================
    # 2. column bbox: cmin,cmax per channel ; a = cmin+cmax
    # ====================================================================
    init("ZERO", np.array(0.0, np.float32), np.float32)
    n("Greater", ["sig", "ZERO"], "colocc")         # [1,10,1,10] bool
    # axis machinery in fp16 (col indices 0..9 are fp16-exact); halves these planes.
    init("cramp", np.arange(W, dtype=np.float16).reshape(1, 1, 1, W), np.float16)
    init("PB", np.array(NB, np.float16), np.float16)
    init("NBv", np.array(-NB, np.float16), np.float16)
    n("Where", ["colocc", "cramp", "PB"], "cmin_src")   # [1,10,1,10] fp16
    n("ReduceMin", ["cmin_src"], "cmin", axes=[3], keepdims=1)   # [1,10,1,1] fp16
    n("Where", ["colocc", "cramp", "NBv"], "cmax_src")
    n("ReduceMax", ["cmax_src"], "cmax", axes=[3], keepdims=1)   # [1,10,1,1] fp16
    n("Add", ["cmin", "cmax"], "a")                 # [1,10,1,1] fp16 reflection axis

    # (row bbox of the SPECIAL box is derived from its column signatures after
    #  selection — see step 6 — so no per-channel row plane is built here.)

    # ====================================================================
    # 4. reflect sig about a (GatherElements idx = clip(a-c, 0, 9))
    # ====================================================================
    n("Sub", ["a", "cramp"], "amc")                 # [1,10,1,10] fp16 = a-c
    init("CLO", np.array(0.0, np.float16), np.float16)
    init("CHI", np.array(float(W - 1), np.float16), np.float16)
    n("Clip", ["amc", "CLO", "CHI"], "amc_cl")      # fp16
    n("Cast", ["amc_cl"], "ridx", to=TensorProto.INT32)   # [1,10,1,10] int32 idx
    n("GatherElements", ["sig", "ridx"], "refl", axis=3)   # [1,10,1,10] f32

    # symmetric column c  <=>  sig[c]==refl[c]  for every IN-SPAN column.
    # out-of-span columns: colocc=False there, and sig=0; refl gathers a clipped
    # in-span column whose sig may be != 0, so MASK the comparison to in-span cols.
    n("Equal", ["sig", "refl"], "eqcol")            # [1,10,1,10] bool
    # in-span: cmin<=c<=cmax  ==  colocc OR ... no; the palindrome must hold on the
    # FULL [cmin,cmax] span, but interior empty cols have sig=0 on both sides so eq
    # holds there too. So require eq on every col with cmin<=c<=cmax.
    n("Not", [n("Less", ["cramp", "cmin"], "lt_min")], "ge_min")   # c>=cmin
    n("Not", [n("Greater", ["cramp", "cmax"], "gt_max")], "le_max")  # c<=cmax
    n("And", ["ge_min", "le_max"], "inspan")        # [1,10,1,10] bool
    # symmetric iff (NOT inspan) OR eqcol, for all c  ==  no in-span mismatch
    n("Not", ["eqcol"], "neqcol")
    n("And", ["inspan", "neqcol"], "mismatch")      # [1,10,1,10] bool
    n("Cast", ["mismatch"], "mismatch_f", to=F32)
    n("ReduceSum", ["mismatch_f"], "nmis", axes=[3], keepdims=1)   # [1,10,1,1]
    n("Equal", ["nmis", "ZERO"], "is_sym")          # [1,10,1,1] bool

    # present (any column occupied)  <=>  cmax >= 0 (else cmax = -NB) ; and not channel 0
    init("NEG1", np.array(-1.0, np.float16), np.float16)
    n("Greater", ["cmax", "NEG1"], "present")       # [1,10,1,1] bool
    ch0kill = np.zeros((1, 10, 1, 1), np.bool_); ch0kill[0, 0, 0, 0] = True
    init("ch0kill", ch0kill, np.bool_)
    n("Not", ["ch0kill"], "notch0")
    n("And", ["present", "notch0"], "valid")
    n("And", ["is_sym", "valid"], "is_box0")        # unique True channel

    # ====================================================================
    # 5. k = ArgMax(is_box0) over channels
    # ====================================================================
    n("Cast", ["is_box0"], "is_box0_f", to=F32)     # [1,10,1,1]
    n("ArgMax", ["is_box0_f"], "k4", axis=1, keepdims=1)   # [1,1,1,1] int64
    init("shp1", np.array([1], np.int64), np.int64)
    n("Reshape", ["k4", "shp1"], "k1")              # [1] int64

    # gather sig, cmin, cmax of channel k
    n("Gather", ["sig", "k1"], "sigk", axis=1)      # [1,1,1,10] f32
    n("Gather", ["cmin", "k1"], "cmink", axis=1)    # [1,1,1,1]

    # box colour = k (channel index)
    n("Cast", ["k4"], "colf", to=F32)               # [1,1,1,1] colour value
    n("Cast", ["colf"], "col_u8", to=U8)            # uint8 colour

    # ====================================================================
    # 6. shift sig columns to origin: gather cols cmin..cmin+4
    # ====================================================================
    # pad sigk with K trailing zero columns so off-grid gathers read 0
    init("sigpad", np.array([0, 0, 0, 0, 0, 0, 0, K], np.int64), np.int64)  # right pad axis3
    n("Pad", ["sigk", "sigpad", "ZERO"], "sigk_p", mode="constant")  # [1,1,1,W+K]
    init("kramp", np.arange(K, dtype=np.float16), np.float16)   # [5] fp16
    n("Reshape", ["cmink", "shp1"], "cmink_s")      # [1] fp16
    n("Add", ["kramp", "cmink_s"], "cidx_f")        # [5] fp16
    init("KLO", np.array(0.0, np.float16), np.float16)
    init("KHI", np.array(float(W + K - 1), np.float16), np.float16)
    n("Clip", ["cidx_f", "KLO", "KHI"], "cidx_cl")
    n("Cast", ["cidx_cl"], "cidx", to=I64)          # [5] int64
    n("Gather", ["sigk_p", "cidx"], "sig_sh", axis=3)  # [1,1,1,5] f32 (col-shifted sigs)

    # ---- decode box on rows 0..9 (col-origin) via Mod bit-extraction --------
    # box10[r,c] = bit_r(sig_sh[c]) = (sig_sh mod 2^(r+1)) >= 2^r
    p2hi = (2.0 ** (np.arange(W) + 1)).astype(np.float32).reshape(1, 1, W, 1)  # [1,1,10,1]
    p2lo = (2.0 ** np.arange(W)).astype(np.float32).reshape(1, 1, W, 1)
    init("p2hi", p2hi, np.float32)
    init("p2lo", p2lo, np.float32)
    n("Mod", ["sig_sh", "p2hi"], "modr", fmod=1)    # [1,1,10,5] f32 (bcast over rows)
    n("Less", ["modr", "p2lo"], "ltlo")
    n("Not", ["ltlo"], "box10")                     # [1,1,10,5] bool box (global rows)

    # ---- row bbox from box10 row-occupancy ; rmin -> Gather rows to origin ---
    n("Cast", ["box10"], "box10_f", to=F32)
    n("ReduceSum", ["box10_f"], "rowcnt", axes=[3], keepdims=1)   # [1,1,10,1]
    n("Greater", ["rowcnt", "ZERO"], "rowocc")      # [1,1,10,1] bool
    init("rramp", np.arange(W, dtype=np.float16).reshape(1, 1, W, 1), np.float16)
    n("Where", ["rowocc", "rramp", "PB"], "rmin_src")   # fp16
    n("ReduceMin", ["rmin_src"], "rmink", axes=[2], keepdims=1)   # [1,1,1,1] fp16
    # pad box10_f with K trailing zero rows so off-grid row gathers read 0
    init("boxpad", np.array([0, 0, 0, 0, 0, 0, K, 0], np.int64), np.int64)  # bottom pad axis2
    n("Pad", ["box10_f", "boxpad", "ZERO"], "box10_p", mode="constant")  # [1,1,W+K,5]
    n("Reshape", ["rmink", "shp1"], "rmink_s")      # [1]
    n("Add", ["kramp", "rmink_s"], "ridx_f")        # [5]
    n("Clip", ["ridx_f", "KLO", "KHI"], "ridx_cl")
    n("Cast", ["ridx_cl"], "ridx2", to=I64)         # [5]
    n("Gather", ["box10_p", "ridx2"], "boxf", axis=2)   # [1,1,5,5] f32 origin box
    n("Greater", ["boxf", "ZERO"], "boxmask")       # [1,1,5,5] bool box pixels

    # ====================================================================
    # 7. inside-bbox mask: r<H and c<W where H,W = box bbox extent at origin
    #    H-1 = max occupied row of boxmask ; W-1 = max occupied col.
    # ====================================================================
    n("ReduceSum", ["boxf"], "bcolcnt", axes=[2], keepdims=1)   # [1,1,1,5] col occ
    n("Greater", ["bcolcnt", "ZERO"], "bcolocc")    # [1,1,1,5] bool
    krow = np.arange(K, dtype=np.float16).reshape(1, 1, K, 1)
    kcol = np.arange(K, dtype=np.float16).reshape(1, 1, 1, K)
    init("krow", krow, np.float16)
    init("kcol", kcol, np.float16)
    # boxrowocc (rows): from boxf
    n("ReduceSum", ["boxf"], "browcnt", axes=[3], keepdims=1)   # [1,1,5,1]
    n("Greater", ["browcnt", "ZERO"], "browocc")    # [1,1,5,1] bool
    # Hm1 = max r with browocc ; Wm1 = max c with bcolocc
    n("Where", ["browocc", "krow", "NBv"], "Hm1_src")
    n("ReduceMax", ["Hm1_src"], "Hm1", axes=[2], keepdims=1)    # [1,1,1,1]
    n("Where", ["bcolocc", "kcol", "NBv"], "Wm1_src")
    n("ReduceMax", ["Wm1_src"], "Wm1", axes=[3], keepdims=1)    # [1,1,1,1]
    n("Not", [n("Greater", ["krow", "Hm1"], "rgt")], "rin")   # r<=Hm1
    n("Not", [n("Greater", ["kcol", "Wm1"], "cgt")], "cin")   # c<=Wm1
    n("And", ["rin", "cin"], "inside")              # [1,1,5,5] bool

    # ====================================================================
    # 8. label map L (5x5) -> pad to 30x30 -> Equal -> output
    # ====================================================================
    init("u0", np.array(0, np.uint8), np.uint8)
    init("u10", np.array(10, np.uint8), np.uint8)
    n("Where", ["boxmask", "col_u8", "u0"], "Lbox")  # box cells -> colour, else 0
    n("Where", ["inside", "Lbox", "u10"], "Lin")     # outside bbox -> 10
    init("padpads", np.array([0, 0, 0, 0, 0, 0, N - K, N - K], np.int64), np.int64)
    n("Pad", ["Lin", "padpads", "u10"], "L", mode="constant")   # [1,1,30,30] u8
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")             # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, N, N])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, N, N])
    g = helper.make_graph(nodes, "task174", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
