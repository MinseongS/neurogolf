"""task368 (ARC-AGI e76a88a6) — "recolour every gray sprite using the one coloured sprite".

Rule (from the generator):
  The grid (always 10x10) holds `num_sprites` (3-4) identical solid H x W rectangles
  (H,W in {3,4}, never both 4) placed at random non-overlapping positions.  Each
  rectangle is the SAME 2-colour pattern P (drawn from a 2-colour palette, gray excluded).
  Exactly ONE sprite (the first drawn) is shown in its real colours; every other sprite is
  shown all-gray (colour 5).  OUTPUT = redraw EVERY sprite (gray ones and the coloured one)
  in the real pattern P, aligned to each sprite's own top-left corner.

Key structure used:
  Every sprite is a solid H x W rectangle, so for any occupied cell (r,c) its offset within
  its own sprite is recoverable LOCALLY as a run-length:
    dr = consecutive occupied cells upward (incl. self) - 1   (0..3)
    dc = consecutive occupied cells leftward (incl. self) - 1 (0..3)
  These are computed by the product-chain of shifted occupancy (resets at gaps; sprites are
  separated, so a run never spans two sprites).  The colour at offset (dr,dc) is the same for
  every sprite and equals P[dr,dc], which the ONE coloured sprite reveals.

  Binary partition of the 2 palette colours: lo = lower channel index, hi = higher.  We learn
  tableHi[dr,dc] = "is the coloured sprite's cell at offset (dr,dc) the hi colour" as a 4x4
  histogram via a double-MatMul over offset one-hots of the (sparse) hi-colour cells:
        tableHi = (dr_oh * isHi) @ dc_oh^T        ([4,N]@[N,4] -> [4,4])
  Then per occupied cell:  outHi = Gather(tableHi_flat[16], key=dr*4+dc).
  Colours are recovered as one-hot [1,10,1,1] vectors hi_oh / lo_oh; the final op routes the
  10-channel expansion into the FREE output:
        output = Where(occ, Where(outHi, hi_oh, lo_oh), input)

All work is on the 10x10 canvas; only the final Pad lifts the condition/value planes to 30x30
(the grid is always 10x10, so off-canvas is all-background and the Where keeps `input` there).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
I32 = TensorProto.INT32
I64 = TensorProto.INT64
U8 = TensorProto.UINT8

S = 10  # grid is always 10x10
N = S * S


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ------------------------------------------------- colour-index plane (Conv)
    # colf = Sum_k k * input_k   (1x1 Conv, weight = arange[10]).  bg=0, gray=5,
    # palette colours = their channel index.  -> [1,1,30,30] fp32, slice 10x10.
    convw = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("convw", convw, np.float32)
    n("Conv", ["input", "convw"], "colf30")                  # [1,1,30,30] f32
    init("c_s", np.array([0, 0], np.int64), np.int64)
    init("c_e", np.array([S, S], np.int64), np.int64)
    init("c_ax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["colf30", "c_s", "c_e", "c_ax"], "colf")     # [1,1,10,10] f32

    # occ = (colf > 0) as fp16 working canvas
    init("ZERO32", np.array(0.0, np.float32), np.float32)
    n("Greater", ["colf", "ZERO32"], "occ_b0")              # [1,1,10,10] bool
    n("Cast", ["occ_b0"], "occ", to=F16)                     # fp16 {0,1}

    # ----------------------------------------------- dr/dc run-length offsets
    # ONE Pad of occ by 3 on the top, then three Slices give occ(up1/up2/up3);
    # likewise one left Pad by 3 then three Slices for occ(left1/2/3).
    init("ZP", np.array(0.0, np.float16), np.float16)
    init("padT", np.array([0, 0, 3, 0, 0, 0, 0, 0], np.int64), np.int64)
    n("Pad", ["occ", "padT", "ZP"], "occT", mode="constant")     # [1,1,13,10]
    init("padL", np.array([0, 0, 0, 3, 0, 0, 0, 0], np.int64), np.int64)
    n("Pad", ["occ", "padL", "ZP"], "occL", mode="constant")     # [1,1,10,13]

    def vslice(off, tag):  # rows [3-off : 13-off] of occT -> occ(up=off)
        init(f"vs_{tag}", np.array([3 - off], np.int64), np.int64)
        init(f"ve_{tag}", np.array([13 - off], np.int64), np.int64)
        init(f"va_{tag}", np.array([2], np.int64), np.int64)
        n("Slice", ["occT", f"vs_{tag}", f"ve_{tag}", f"va_{tag}"], f"up_{tag}")
        return f"up_{tag}"

    def hslice(off, tag):  # cols [3-off : 13-off] of occL -> occ(left=off)
        init(f"hs_{tag}", np.array([3 - off], np.int64), np.int64)
        init(f"he_{tag}", np.array([13 - off], np.int64), np.int64)
        init(f"ha_{tag}", np.array([3], np.int64), np.int64)
        n("Slice", ["occL", f"hs_{tag}", f"he_{tag}", f"ha_{tag}"], f"lf_{tag}")
        return f"lf_{tag}"

    u1, u2, u3 = vslice(1, "1"), vslice(2, "2"), vslice(3, "3")
    n("Mul", ["occ", u1], "d1")
    n("Mul", ["d1", u2], "d2")
    n("Mul", ["d2", u3], "d3")
    n("Sum", ["d1", "d2", "d3"], "dr")            # [1,1,10,10] fp16 {0..3}

    l1, l2, l3 = hslice(1, "1"), hslice(2, "2"), hslice(3, "3")
    n("Mul", ["occ", l1], "c1")
    n("Mul", ["c1", l2], "c2")
    n("Mul", ["c2", l3], "c3")
    n("Sum", ["c1", "c2", "c3"], "dc")            # [1,1,10,10] fp16 {0..3}

    # key = dr*4 + dc   [1,1,10,10] fp16 (0..15)
    init("FOUR", np.array(4.0, np.float16), np.float16)
    n("Mul", ["dr", "FOUR"], "dr4")
    n("Add", ["dr4", "dc"], "key")
    n("Cast", ["key"], "key_i", to=I32)           # indices for Gather

    # ------------------------------------------------ recover palette colours
    # pres[1,10,1,1] = per-channel presence; zero ch0 and ch5 -> exactly 2 ones
    n("ReduceMax", ["input"], "pres", axes=[2, 3], keepdims=1)   # [1,10,1,1] f32
    colmask = np.ones((1, 10, 1, 1), np.float32)
    colmask[0, 0] = 0.0
    colmask[0, 5] = 0.0
    init("colmask", colmask, np.float32)
    n("Mul", ["pres", "colmask"], "pres2")        # [1,10,1,1] f32 (two ones)

    # above[k] = sum_{j>k} pres2[j] ; below[k] = sum_{j<k} pres2[j]
    # via MatMul on the channel axis: reshape pres2 -> [10,1]
    init("p_shape", np.array([10, 1], np.int64), np.int64)
    n("Reshape", ["pres2", "p_shape"], "p_col")   # [10,1]
    Usup = np.zeros((10, 10), np.float32)          # strictly upper: j>k
    Ulow = np.zeros((10, 10), np.float32)          # strictly lower: j<k
    for k in range(10):
        for j in range(10):
            if j > k:
                Usup[k, j] = 1.0
            if j < k:
                Ulow[k, j] = 1.0
    init("Usup", Usup, np.float32)
    init("Ulow", Ulow, np.float32)
    n("MatMul", ["Usup", "p_col"], "above_c")     # [10,1]
    n("MatMul", ["Ulow", "p_col"], "below_c")     # [10,1]
    # hi_oh = pres2 * (above==0) ; lo_oh = pres2 * (below==0)
    init("v_shape", np.array([1, 10, 1, 1], np.int64), np.int64)
    n("Reshape", ["above_c", "v_shape"], "above")
    n("Reshape", ["below_c", "v_shape"], "below")
    init("ZEROF", np.array(0.0, np.float32), np.float32)
    n("Equal", ["above", "ZEROF"], "above0")       # bool
    n("Equal", ["below", "ZEROF"], "below0")       # bool
    n("Cast", ["above0"], "above0f", to=F32)
    n("Cast", ["below0"], "below0f", to=F32)
    n("Mul", ["pres2", "above0f"], "hi_oh")        # [1,10,1,1] f32 one-hot (hi)
    n("Mul", ["pres2", "below0f"], "lo_oh")        # [1,10,1,1] f32 one-hot (lo)

    # hi_idx / lo_idx scalars (colour channel index 0..9) via Sum k*oh
    arvec = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("arange10", arvec, np.float32)
    n("Mul", ["hi_oh", "arange10"], "hi_kw")
    n("ReduceSum", ["hi_kw"], "hi_idx32", axes=[1], keepdims=1)   # [1,1,1,1] f32
    n("Mul", ["lo_oh", "arange10"], "lo_kw")
    n("ReduceSum", ["lo_kw"], "lo_idx32", axes=[1], keepdims=1)   # [1,1,1,1] f32
    n("Cast", ["hi_idx32"], "hi_idx", to=F16)
    n("Cast", ["lo_idx32"], "lo_idx", to=F16)

    # isHi cell plane = (colf == hi_idx)  -> the hi-colour cells of the canvas
    n("Equal", ["colf", "hi_idx32"], "isHi_b")     # [1,1,10,10] bool
    n("Cast", ["isHi_b"], "isHi", to=F16)          # [1,1,10,10] fp16 {0,1}

    # ----------------------------------- forward histogram tableHi[4,4]
    # dr_masked = dr where isHi else -1 (sentinel): only hi-colour cells keep an
    # offset that can match {0,1,2,3}; everything else maps to -1 (never matched).
    #   dr_masked = dr*isHi - (1 - isHi)
    n("Mul", ["dr", "isHi"], "dr_hi")              # [1,1,10,10] fp16
    init("ONEh", np.array(1.0, np.float16), np.float16)
    n("Sub", ["ONEh", "isHi"], "notHi_cell")
    n("Sub", ["dr_hi", "notHi_cell"], "dr_masked")  # hi: dr, else -1
    # dr_ohw[d,i] = (dr_masked[i]==d)   [4,N] fp16  (already isHi-weighted)
    init("flat_shape", np.array([1, N], np.int64), np.int64)
    init("col_shape", np.array([N, 1], np.int64), np.int64)
    n("Reshape", ["dr_masked", "flat_shape"], "drm_f")   # [1,N]
    n("Reshape", ["dc", "col_shape"], "dc_col")          # [N,1]
    init("ar4r", np.array([[0.0, 1.0, 2.0, 3.0]], np.float16), np.float16)   # [1,4]
    init("ar4c", np.array([[0.0], [1.0], [2.0], [3.0]], np.float16), np.float16)  # [4,1]
    n("Equal", ["drm_f", "ar4c"], "dr_ohw_b")      # [4,N] bool
    n("Cast", ["dr_ohw_b"], "dr_ohw", to=F16)      # [4,N] fp16
    n("Equal", ["dc_col", "ar4r"], "dc_ohT_b")     # [N,4] bool (already transposed)
    n("Cast", ["dc_ohT_b"], "dc_ohT", to=F16)      # [N,4] fp16
    # tableHi = dr_ohw @ dc_ohT  -> [4,4]
    n("MatMul", ["dr_ohw", "dc_ohT"], "tableHi")   # [4,4]
    init("t_shape", np.array([16], np.int64), np.int64)
    n("Reshape", ["tableHi", "t_shape"], "tableHi_flat")  # [16]

    # outHi = Gather(tableHi_flat, key_i)  -> shape of key = [1,1,10,10]
    n("Gather", ["tableHi_flat", "key_i"], "outHi_raw", axis=0)  # [1,1,10,10] f32 {0,1}

    # label colour-index plane L (10x10):
    #   L = occ * ( outHi*hi_idx + (1-outHi)*lo_idx )    (0 on background)
    init("ONEf", np.array(1.0, np.float16), np.float16)
    n("Sub", ["ONEf", "outHi_raw"], "notHi")       # [1,1,10,10] fp16
    n("Mul", ["outHi_raw", "hi_idx"], "hipart")    # broadcast scalar
    n("Mul", ["notHi", "lo_idx"], "lopart")
    n("Add", ["hipart", "lopart"], "Lraw")         # [1,1,10,10] f32 colour idx
    n("Mul", ["Lraw", "occ"], "Lmask")             # 0 on bg
    n("Cast", ["Lmask"], "Lu8", to=U8)             # [1,1,10,10] uint8

    # pad to 30x30 with sentinel 99 off-grid (-> all-channels-off, matching the
    # benchmark where off-grid cells are 0 in EVERY channel, not ch0=1).
    init("pad30", np.array([0, 0, 0, 0, 0, 0, 30 - S, 30 - S], np.int64), np.int64)
    init("SENT", np.array(99, np.uint8), np.uint8)
    n("Pad", ["Lu8", "pad30", "SENT"], "L30", mode="constant")  # [1,1,30,30] uint8
    arange_u8 = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("arange_u8", arange_u8, np.uint8)
    n("Equal", ["L30", "arange_u8"], "output")     # [1,10,30,30] BOOL (free)

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task368", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
