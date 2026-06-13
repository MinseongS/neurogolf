"""Task 382 (ARC f15e1fac): red dots sit on one edge, cyan dots on the
perpendicular edge; the output replicates the cyan edge-pattern down the grid,
each row shifted sideways by the running count of red rows passed so far.  The
whole figure may be horizontally flipped and rotated/reflected by `gravity`
(0..3 = optional transpose and/or vertical flip).

Approach: carry the figure as ONE float16 tensor M (red=2, cyan=8 -- so every
canvas intermediate is 2 bytes and the conditional Transpose/Gather/Where
selectors all work).  Recover orientation flags -- transpose if cyans lie on a
single column, vertical flip if the cyan line is the bottom edge, horizontal flip
if the red line is the right edge -- and canonicalise M to the (flip=0,gravity=0)
frame with a conditional Transpose plus size-aware reversals done by Gather on
data-dependent index vectors rev[k]=size-1-k (each flip is folded into a tiny
conditional gather-index vector so no canvas-sized Where is needed; the same
indices undo the flip later since flips are involutions).  In that frame the rule
is a per-row right shift of the top cyan pattern p by the lower-triangular prefix
count S of red rows, computed in ONE Gather: Out[r,c]=p[(c+1)-S[r]] into a p
padded with a leading zero (out-of-range -> that zero).  Reds are left in place
(the original red mask is reused), so only the cyan layer is round-tripped.
Background/red/cyan are routed into the 10-channel output by a 1x1 Conv.  Every
canvas intermediate is float16 (2 bytes) or bool (1 byte).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..builders import _model

I8 = TensorProto.INT8
I32 = TensorProto.INT32
I64 = TensorProto.INT64
BOOL = TensorProto.BOOL
F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16


def build(task):
    inits, nodes = [], []
    _np = {I8: np.int8, I32: np.int32, I64: np.int64, BOOL: np.bool_,
           F32: np.float32, F16: np.float16}

    def init(name, arr, dtype=None):
        a = np.asarray(arr)
        if dtype is None:
            dtype = a.dtype if a.dtype != np.float64 else np.float32
        else:
            dtype = _np.get(dtype, dtype)
        inits.append(numpy_helper.from_array(np.ascontiguousarray(a, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, list(ins), [out], **attrs))
        return out

    rng = np.arange(30, dtype=np.int64)
    init("Irow", rng.reshape(1, 30), I64)
    init("Icol", rng.reshape(1, 30), I64)
    init("rngFr", np.arange(30, dtype=np.float16).reshape(1, 1, 30, 1), F16)
    init("rngFc", np.arange(30, dtype=np.float16).reshape(1, 1, 1, 30), F16)
    init("rngFc1", (np.arange(30, dtype=np.float16) + 1).reshape(1, 1, 1, 30), F16)
    init("Ltri", np.tril(np.ones((30, 30), np.float16)), F16)
    init("one64", np.array(1, np.int64), I64)
    init("h8", np.array([[[[8]]]], np.float16), F16)        # cyan value
    init("zeroF16", np.array([[[[0]]]], np.float16), F16)
    init("h5", np.array([[[[5]]]], np.float16), F16)        # threshold 2<5<8
    init("h1", np.array([[[[1]]]], np.float16), F16)        # threshold 0<1<2

    # ---- 1. figure M(f16)=2*red+8*cyan ; occupancy occ(f16) ----
    Wsel = np.zeros((1, 10, 1, 1), np.float32)
    Wsel[0, 2, 0, 0] = 2.0
    Wsel[0, 8, 0, 0] = 8.0
    init("Wsel", Wsel)
    n("Conv", ["input", "Wsel"], "Mf")                      # f32 [1,1,30,30]
    n("Cast", ["Mf"], "M", to=F16)
    n("ReduceSum", ["input"], "occf", axes=[1], keepdims=1)  # f32 grid rectangle

    # ---- 2. transpose flag: cyans occupy a single COLUMN ----
    # column has cyan iff max value over rows == 8 (>5)
    n("ReduceMax", ["M"], "colmax", axes=[2], keepdims=1)   # f16 [1,1,1,30]
    n("Greater", ["colmax", "h5"], "cycB")
    n("Cast", ["cycB"], "cyc", to=F16)
    n("ReduceSum", ["cyc"], "ncyc", axes=[3], keepdims=1)
    n("Cast", ["ncyc"], "ncycI", to=I64)
    n("Equal", ["ncycI", "one64b"], "tpB")
    init("one64b", np.array([[[[1]]]], np.int64), I64)

    def cond_transpose(src, dst):
        n("Transpose", [src], src + "_t", perm=[0, 1, 3, 2])
        n("Where", ["tpB", src + "_t", src], dst)

    cond_transpose("M", "M1")

    # original width/height from occ; canonical W/H swap on transpose (scalar)
    n("ReduceMax", ["occf"], "ocr", axes=[2], keepdims=1)    # [1,1,1,30]
    n("ReduceSum", ["ocr"], "oWf", axes=[3], keepdims=1)
    n("Cast", ["oWf"], "oW", to=I64)
    n("ReduceMax", ["occf"], "ocl", axes=[3], keepdims=1)    # [1,1,30,1]
    n("ReduceSum", ["ocl"], "oHf", axes=[2], keepdims=1)
    n("Cast", ["oHf"], "oH", to=I64)
    # W (canonical) = transposed ? oH : oW ;  H = transposed ? oW : oH
    n("Where", ["tpB", "oH", "oW"], "W")
    n("Where", ["tpB", "oW", "oH"], "H")

    # reversal index vectors rev[k]=size-1-k if k<size else k  (within grid)
    init("sh1", np.array([1], np.int64))
    n("Reshape", ["H", "sh1"], "Hs")
    n("Reshape", ["W", "sh1"], "Ws")
    n("Sub", ["Hs", "one64"], "Hm1")
    n("Sub", ["Ws", "one64"], "Wm1")
    n("Sub", ["Hm1", "Irow"], "revR0")
    n("Sub", ["Wm1", "Icol"], "revC0")
    n("Less", ["Irow", "Hs"], "inR")
    n("Less", ["Icol", "Ws"], "inC")
    n("Where", ["inR", "revR0", "Irow"], "revR")            # [1,30]
    n("Where", ["inC", "revC0", "Icol"], "revC")

    # ---- 3. vflip flag, then fold it into the gather index (no canvas Where) ----
    n("ReduceMax", ["M1"], "rowmax", axes=[3], keepdims=1)  # [1,1,30,1]
    n("Greater", ["rowmax", "h5"], "cy1B")
    n("Cast", ["cy1B"], "cy1H", to=F16)
    n("Mul", ["cy1H", "rngFr"], "cyri")
    n("ReduceMax", ["cyri"], "cymax", axes=[2], keepdims=1)
    n("Greater", ["cymax", "zeroF16"], "vfBs")              # scalar [1,1,1,1]
    n("Reshape", ["vfBs", "sh1b"], "vfB")                   # [1] bool
    init("sh1b", np.array([1], np.int64))
    n("Where", ["vfB", "revR", "Irow"], "gidxR0")           # [1,30] gather idx
    init("sh30", np.array([30], np.int64))
    n("Reshape", ["gidxR0", "sh30"], "gidxR")
    n("Gather", ["M1", "gidxR"], "M2", axis=2)              # vflip applied

    # ---- 4. hflip flag, folded into gather index ----
    n("Greater", ["M2", "zeroF16"], "m2pos")
    n("Less", ["M2", "h5"], "m2lt5")
    n("And", ["m2pos", "m2lt5"], "rd2B")
    n("Cast", ["rd2B"], "rd2H", to=F16)
    n("ReduceMax", ["rd2H"], "rdc", axes=[2], keepdims=1)
    n("Mul", ["rdc", "rngFc"], "rdci")
    n("ReduceMax", ["rdci"], "rdmax", axes=[3], keepdims=1)
    n("Greater", ["rdmax", "zeroF16"], "hfBs")
    n("Reshape", ["hfBs", "sh1b"], "hfB")
    n("Where", ["hfB", "revC", "Icol"], "gidxC0")
    n("Reshape", ["gidxC0", "sh30"], "gidxC")
    n("Gather", ["M2", "gidxC"], "Mc", axis=3)              # hflip -> CANONICAL

    # ---- 5. canonical solve ----
    n("Greater", ["Mc", "zeroF16"], "mcpos")
    n("Less", ["Mc", "h5"], "mclt5")
    n("And", ["mcpos", "mclt5"], "crB")      # canonical red (bool)
    n("Greater", ["Mc", "h5"], "ccB")        # canonical cyan (bool)

    # S[r] = prefix count of red rows
    n("Cast", ["crB"], "crH", to=F16)
    n("ReduceMax", ["crH"], "rr", axes=[3], keepdims=1)     # f16 [1,1,30,1]
    init("sh301", np.array([30, 1], np.int64))
    n("Reshape", ["rr", "sh301"], "rrm")
    n("MatMul", ["Ltri", "rrm"], "Sm")                      # f16 [30,1]
    init("sh4r", np.array([1, 1, 30, 1], np.int64))
    n("Reshape", ["Sm", "sh4r"], "S")        # f16 [1,1,30,1]

    # cyan pattern p[c]; pad zero at index 0; shift indices +1 and Clip(min=0)
    n("Cast", ["ccB"], "ccH", to=F16)
    n("ReduceMax", ["ccH"], "pcolH", axes=[2], keepdims=1)  # f16 [1,1,1,30]
    n("Cast", ["pcolH"], "pcol8", to=I8)
    n("Reshape", ["pcol8", "sh30"], "p30")
    init("z8", np.array([0], np.int8), I8)
    n("Concat", ["z8", "p30"], "p31", axis=0)               # [31] int8 (idx0 = 0)

    # Dp[r,c] = (c+1) - S[r] ; map <1 (orig c-S<0) to 0 (-> padded zero at idx 0)
    n("Sub", ["rngFc1", "S"], "Dp")                         # f16 [1,1,30,30]
    n("Less", ["Dp", "h1"], "negB")                         # Dp<1
    n("Where", ["negB", "zeroF16", "Dp"], "Dcf")            # f16
    n("Cast", ["Dcf"], "Dc", to=I32)                        # [1,1,30,30] indices
    n("Gather", ["p31", "Dc"], "ccshift", axis=0)           # [1,1,30,30] int8
    n("Cast", ["ccshift"], "ccsB", to=BOOL)

    # in-grid mask
    init("sh4c", np.array([1, 1, 1, 30], np.int64))
    init("sh4rr", np.array([1, 1, 30, 1], np.int64))
    n("Reshape", ["inC", "sh4c"], "cmask")
    n("Reshape", ["inR", "sh4rr"], "rmask")
    n("And", ["cmask", "rmask"], "inB")
    n("And", ["ccsB", "inB"], "ccoB")                       # canonical cyan out

    # canonical CYAN-only figure (reds are unchanged by the rule, so the original
    # red mask is reused directly and need not be round-tripped)
    n("Where", ["ccoB", "h8", "zeroF16"], "Oc")             # cyan->8

    # ---- 6. undo (reverse order): hflip & vflip reuse the same conditional
    # gather indices (flips are involutions); transpose stays a conditional Where.
    n("Gather", ["Oc", "gidxC"], "O1", axis=3)              # undo hflip
    n("Gather", ["O1", "gidxR"], "O2", axis=2)              # undo vflip
    n("Transpose", ["O2"], "O2t", perm=[0, 1, 3, 2])
    n("Where", ["tpB", "O2t", "O2"], "O")                   # undo transpose

    # ---- 7. assemble channels via 1x1 Conv into output ----
    # red is the ORIGINAL red mask (rule leaves reds in place); cyan from O
    n("Greater", ["M", "zeroF16"], "mpos")
    n("Less", ["M", "h5"], "mlt5")
    n("And", ["mpos", "mlt5"], "fRB")        # original red
    n("Greater", ["O", "h5"], "fCB")         # cyan from solved figure
    n("Or", ["fRB", "fCB"], "colB")
    n("Not", ["colB"], "ncolB")
    n("Cast", ["occf"], "occB", to=BOOL)
    n("And", ["occB", "ncolB"], "bgB")
    # concat the bool masks then Cast once (cheaper than 3 separate f32 masks)
    n("Concat", ["bgB", "fRB", "fCB"], "stackB", axis=1)    # [1,3,30,30] bool
    n("Cast", ["stackB"], "stack", to=F32)                  # [1,3,30,30] f32
    Wout = np.zeros((10, 3, 1, 1), np.float32)
    Wout[0, 0, 0, 0] = 1.0
    Wout[2, 1, 0, 0] = 1.0
    Wout[8, 2, 0, 0] = 1.0
    init("Wout", Wout)
    n("Conv", ["stack", "Wout"], "output")

    return _model(nodes, inits)
