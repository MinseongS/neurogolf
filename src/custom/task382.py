"""Task 382 (ARC f15e1fac): red dots on one edge, cyan dots on the perpendicular
edge; the output replicates the cyan edge-pattern down the grid, each row shifted
sideways by the running count of red rows passed so far.  The figure may be
horizontally flipped and rotated/reflected by `gravity` (0..3 = optional vertical
flip and/or transpose).

CROP-TO-ACTIVE optimisation: the generator bounds the grid to <=20x20 and the
harness anchors it at the top-left corner, so the entire working pipeline runs on
a 20x20 canvas (every fp16 plane 800B instead of 1800B, fp32 1600B, int32 1600B,
bool 400B).  The single fp32 conv entry plane is sliced to 20x20 and cast to fp16
immediately; the final uint8 label map is built at 20x20 and Pad'd (sentinel 10)
back to 30x30 just before the free BOOL Equal output.

Same canonicalise-solve-uncanonicalise logic as before: transpose if cyans lie on
a single column, vertical flip if the cyan line is the bottom edge, horizontal flip
if the red line is the right edge.  Flips fold into tiny conditional gather-index
vectors (involutions, reused to undo).  In the canonical frame Out[r,c]=p[(c+1)-S[r]]
where p is the top cyan pattern and S the lower-triangular prefix count of red rows.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

I8 = TensorProto.INT8
I32 = TensorProto.INT32
I64 = TensorProto.INT64
BOOL = TensorProto.BOOL
F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16

WK = 20  # working canvas (generator bounds grid to <=20x20, anchored top-left)


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

    rng = np.arange(WK, dtype=np.int64)
    init("Irow", rng.reshape(1, WK), I64)
    init("Icol", rng.reshape(1, WK), I64)
    init("rngFc1", (np.arange(WK, dtype=np.float16) + 1).reshape(1, 1, 1, WK), F16)
    init("Ltri", np.tril(np.ones((WK, WK), np.float16)), F16)
    init("one64", np.array(1, np.int64), I64)
    init("h8", np.array([[[[8]]]], np.float16), F16)        # cyan value
    init("zeroF16", np.array([[[[0]]]], np.float16), F16)
    init("h5", np.array([[[[5]]]], np.float16), F16)        # threshold 2<5<8
    init("h1", np.array([[[[1]]]], np.float16), F16)        # threshold 0<1<2

    # crop helpers (Slice [1,1,30,30] -> [1,1,WK,WK])
    init("cstart", np.array([0, 0], np.int64), I64)
    init("cend", np.array([WK, WK], np.int64), I64)
    init("caxes", np.array([2, 3], np.int64), I64)

    # ---- 1. figure M(f16)=2*red+8*cyan on the cropped active region ----
    Wsel = np.zeros((1, 10, 1, 1), np.float32)
    Wsel[0, 2, 0, 0] = 2.0
    Wsel[0, 8, 0, 0] = 8.0
    init("Wsel", Wsel)
    n("Conv", ["input", "Wsel"], "Mf30")                    # f32 [1,1,30,30] (entry)
    n("Slice", ["Mf30", "cstart", "cend", "caxes"], "Mf")   # f32 [1,1,WK,WK]
    n("Cast", ["Mf"], "M", to=F16)
    # The grid is a solid WxH rectangle anchored top-left, so the in-grid mask is
    # just (row<H) AND (col<W) (built later as inB).  Recover H,W from the free
    # channel-0 background plane (==1 on every in-grid cell except red/cyan dots;
    # combined with M>0 it covers the whole rectangle).  Reduce along each axis to
    # 1-D occupancy, then to scalar W,H -- no 2-D occupancy plane retained.
    init("ch0start", np.array([0, 0, 0], np.int64), I64)
    init("ch0end", np.array([1, WK, WK], np.int64), I64)
    init("ch0axes", np.array([1, 2, 3], np.int64), I64)
    n("Slice", ["input", "ch0start", "ch0end", "ch0axes"], "bg0")  # f32 [1,1,WK,WK]
    n("Cast", ["bg0"], "bg0B", to=BOOL)
    n("Greater", ["M", "zeroF16"], "mNZ")                   # red/cyan present (bool)
    n("Or", ["bg0B", "mNZ"], "occB")                        # orig-frame in-grid mask
    # W/H from bg0 alone: every in-grid row/col has a background cell, so the bg0
    # profile equals the full grid extent (no extra fp16 occupancy plane needed).

    # ---- 2. transpose flag: cyans occupy a single COLUMN ----
    n("ReduceMax", ["M"], "colmax", axes=[2], keepdims=1)   # f16 [1,1,1,WK]
    n("Greater", ["colmax", "h5"], "cycB")
    n("Cast", ["cycB"], "cyc", to=F16)
    n("ReduceSum", ["cyc"], "ncyc", axes=[3], keepdims=1)
    n("Cast", ["ncyc"], "ncycI", to=I64)
    init("one64b", np.array([[[[1]]]], np.int64), I64)
    n("Equal", ["ncycI", "one64b"], "tpB")

    def cond_transpose(src, dst):
        n("Transpose", [src], src + "_t", perm=[0, 1, 3, 2])
        n("Where", ["tpB", src + "_t", src], dst)

    cond_transpose("M", "M1")

    # original width/height from occ; canonical W/H swap on transpose (scalar)
    n("ReduceMax", ["bg0"], "ocr", axes=[2], keepdims=1)    # [1,1,1,WK]
    n("ReduceSum", ["ocr"], "oWf", axes=[3], keepdims=1)
    n("Cast", ["oWf"], "oW", to=I64)
    n("ReduceMax", ["bg0"], "ocl", axes=[3], keepdims=1)    # [1,1,WK,1]
    n("ReduceSum", ["ocl"], "oHf", axes=[2], keepdims=1)
    n("Cast", ["oHf"], "oH", to=I64)
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
    n("Where", ["inR", "revR0", "Irow"], "revR")            # [1,WK]
    n("Where", ["inC", "revC0", "Icol"], "revC")

    # ---- 3. vflip flag, then fold it into the gather index (no canvas Where) ----
    init("rngFr", np.arange(WK, dtype=np.float16).reshape(1, 1, WK, 1), F16)
    n("ReduceMax", ["M1"], "rowmax", axes=[3], keepdims=1)  # [1,1,WK,1]
    n("Greater", ["rowmax", "h5"], "cy1B")
    n("Cast", ["cy1B"], "cy1H", to=F16)
    n("Mul", ["cy1H", "rngFr"], "cyri")
    n("ReduceMax", ["cyri"], "cymax", axes=[2], keepdims=1)
    n("Greater", ["cymax", "zeroF16"], "vfBs")              # scalar [1,1,1,1]
    init("sh1b", np.array([1], np.int64))
    n("Reshape", ["vfBs", "sh1b"], "vfB")                   # [1] bool
    n("Where", ["vfB", "revR", "Irow"], "gidxR0")           # [1,WK] gather idx
    init("shWK", np.array([WK], np.int64))
    n("Reshape", ["gidxR0", "shWK"], "gidxR")
    n("Gather", ["M1", "gidxR"], "M2", axis=2)              # vflip applied

    # ---- 4. hflip flag, folded into gather index ----
    init("rngFc", np.arange(WK, dtype=np.float16).reshape(1, 1, 1, WK), F16)
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
    n("Reshape", ["gidxC0", "shWK"], "gidxC")
    n("Gather", ["M2", "gidxC"], "Mc", axis=3)              # hflip -> CANONICAL

    # ---- 5. canonical solve ----
    n("Greater", ["Mc", "zeroF16"], "mcpos")
    n("Less", ["Mc", "h5"], "mclt5")
    n("And", ["mcpos", "mclt5"], "crB")      # canonical red (bool)
    n("Greater", ["Mc", "h5"], "ccB")        # canonical cyan (bool)

    # S[r] = prefix count of red rows
    n("Cast", ["crB"], "crH", to=F16)
    n("ReduceMax", ["crH"], "rr", axes=[3], keepdims=1)     # f16 [1,1,WK,1]
    init("shWK1", np.array([WK, 1], np.int64))
    n("Reshape", ["rr", "shWK1"], "rrm")
    n("MatMul", ["Ltri", "rrm"], "Sm")                      # f16 [WK,1]
    init("sh4r", np.array([1, 1, WK, 1], np.int64))
    n("Reshape", ["Sm", "sh4r"], "S")        # f16 [1,1,WK,1]

    # cyan pattern p[c]; pad zero at index 0; shift indices +1 and Clip(min=0)
    n("Cast", ["ccB"], "ccH", to=F16)
    n("ReduceMax", ["ccH"], "pcolH", axes=[2], keepdims=1)  # f16 [1,1,1,WK]
    n("Cast", ["pcolH"], "pcol8", to=I8)
    n("Reshape", ["pcol8", "shWK"], "pWK")
    init("z8", np.array([0], np.int8), I8)
    n("Concat", ["z8", "pWK"], "p31", axis=0)               # [WK+1] int8 (idx0 = 0)

    # Dp[r,c] = (c+1) - S[r] ; map <1 (orig c-S<0) to 0 (-> padded zero at idx 0)
    n("Sub", ["rngFc1", "S"], "Dp")                         # f16 [1,1,WK,WK]
    n("Less", ["Dp", "h1"], "negB")                         # Dp<1
    n("Where", ["negB", "zeroF16", "Dp"], "Dcf")            # f16
    n("Cast", ["Dcf"], "Dc", to=I32)                        # [1,1,WK,WK] indices
    n("Gather", ["p31", "Dc"], "ccshift", axis=0)           # [1,1,WK,WK] int8
    n("Cast", ["ccshift"], "ccsB", to=BOOL)

    # in-grid mask
    init("sh4c", np.array([1, 1, 1, WK], np.int64))
    init("sh4rr", np.array([1, 1, WK, 1], np.int64))
    n("Reshape", ["inC", "sh4c"], "cmask")
    n("Reshape", ["inR", "sh4rr"], "rmask")
    n("And", ["cmask", "rmask"], "inB")
    n("And", ["ccsB", "inB"], "ccoB")                       # canonical cyan out

    # canonical CYAN-only figure (reds unchanged by the rule; reuse original mask)
    n("Where", ["ccoB", "h8", "zeroF16"], "Oc")             # cyan->8

    # ---- 6. undo (reverse order): hflip & vflip reuse the gather indices;
    # transpose stays a conditional Where.
    n("Gather", ["Oc", "gidxC"], "O1", axis=3)              # undo hflip
    n("Gather", ["O1", "gidxR"], "O2", axis=2)              # undo vflip
    n("Transpose", ["O2"], "O2t", perm=[0, 1, 3, 2])
    n("Where", ["tpB", "O2t", "O2"], "O")                   # undo transpose

    # ---- 7. assemble into a 20x20 uint8 label map, Pad to 30x30, free Equal ----
    # background 0, red 2 (original positions), cyan 8 (solved); off-grid -> 10.
    n("Less", ["M", "h5"], "mlt5")
    n("And", ["mNZ", "mlt5"], "fRB")         # original red (bool); mNZ from step 1
    n("Greater", ["O", "h5"], "fCB")         # cyan from solved figure (bool)
    init("u0", np.array(0, np.uint8), np.uint8)
    init("u2", np.array(2, np.uint8), np.uint8)
    init("u8", np.array(8, np.uint8), np.uint8)
    init("u10", np.array(10, np.uint8), np.uint8)
    n("Where", ["occB", "u0", "u10"], "Lg")  # 0 in-grid (orig-frame rectangle) else 10
    n("Where", ["fCB", "u8", "Lg"], "Lc")    # cyan = 8
    n("Where", ["fRB", "u2", "Lc"], "L20")   # red = 2 (overrides) [1,1,WK,WK] uint8

    # Pad 20x20 -> 30x30 with sentinel 10 (off-grid matches no channel)
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - WK, 30 - WK], np.int64), I64)
    n("Pad", ["L20", "pads", "u10"], "L", mode="constant")  # [1,1,30,30] uint8

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")      # -> free BOOL output

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
