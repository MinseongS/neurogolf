"""task301 (ARC-AGI beb8660c) — "sort the length-coded color bars into a right-justified triangle".

Rule (from tasks/task_beb8660c.py):
  Input grid is W x H, where W = num_colors (3..9) and H = W + gap (gap 0..3).
  Each color k (cyan/8 is always the longest) appears as ONE horizontal bar of
  a DISTINCT length L (lengths are 1..W, one per color) at a random row/col.
  i.e. the pixel-count of color k == its bar length L_k, and these are distinct.
  Output: a right-justified staircase triangle. The color whose bar length is L
  fills output row (L-1+gap), right-aligned over columns [W-L .. W-1] (a run of
  length L). Output rows 0..gap-1 are empty. Equivalently a cell (r,c) inside
  the grid is occupied iff  r + c >= H-1 , and its color = the color whose bar
  length == (r - gap + 1).

Recovery (all scalars / tiny planes; no [1,10,30,30] / 30x30 materialization):
  cnt[k]   = ReduceSum(input, axes=[2,3])  -> [1,10,1,1]   (bar length of color k)
  rowin    = ReduceMax(input, axes=[1,3])  -> [1,1,30,1]   (in-grid rows: 1 for r<H)
  colin    = ReduceMax(input, axes=[1,2])  -> [1,1,1,30]   (in-grid cols: 1 for c<W)
  H = sum(rowin), W = sum(colin), gap = H - W.
  On a small RH=12 x RW=9 canvas (grid always sits at top-left):
    lenOfRow[r] = r - gap + 1
    rowcolor[r] = sum_k k * (cnt[k] == lenOfRow[r])          (per-row color)
    occupied(r,c) = (r + c >= H-1) AND rowin[r] AND colin[c]
    ingrid(r,c)   = rowin[r] AND colin[c]
    L(r,c) = ingrid ? (occupied ? rowcolor[r] : 0) : 99      (uint8 label map)
  Pad L to 30x30 with 99 (off-grid stays 99 -> never equals any color 0..9).
  output = Equal(L_uint8, arange[0..9][1,10,1,1])  -> BOOL   (the FREE 10-ch tensor)
  Off-grid cells = 99 != any k  => all channels 0 (correct: off-grid is all-zero).
  In-grid empty cells = 0       => channel 0 set (correct background).

Working on the 12x9 active canvas shrinks every full plane ~9x vs the 30x30
formulation (the prior 15.84 net carried five ~900-1200B 30x30 planes).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I64 = TensorProto.INT64

RH = 12  # working rows (max H = num_colors(9) + gap(3))
RW = 9   # working cols (max W = num_colors = 9)


def build(task):
    inits, nodes, vis = [], [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def vi(name, dtype, shape):
        vis.append(helper.make_tensor_value_info(name, dtype, shape))

    def n(op, ins, out, dtype=None, shape=None, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        if dtype is not None:
            vi(out, dtype, shape)
        return out

    # ---- per-channel counts (= bar lengths) -------------------------------
    n("ReduceSum", ["input"], "cnt", F32, [1, 10, 1, 1], axes=[2, 3], keepdims=1)

    # ---- in-grid row / col occupancy (H, W recovery) ----------------------
    n("ReduceMax", ["input"], "rowin30", F32, [1, 1, 30, 1], axes=[1, 3], keepdims=1)
    n("ReduceMax", ["input"], "colin30", F32, [1, 1, 1, 30], axes=[1, 2], keepdims=1)
    init("z_s", np.array([0], np.int64), np.int64)
    init("rh_e", np.array([RH], np.int64), np.int64)
    init("ax2", np.array([2], np.int64), np.int64)
    n("Slice", ["rowin30", "z_s", "rh_e", "ax2"], "rowinS", F32, [1, 1, RH, 1])
    init("rw_e", np.array([RW], np.int64), np.int64)
    init("ax3", np.array([3], np.int64), np.int64)
    n("Slice", ["colin30", "z_s", "rw_e", "ax3"], "colinS", F32, [1, 1, 1, RW])

    # H = sum(rowin30), W = sum(colin30)  (scalars)
    n("ReduceSum", ["rowin30"], "Hf", F32, [1, 1, 1, 1], axes=[2, 3], keepdims=1)
    n("ReduceSum", ["colin30"], "Wf", F32, [1, 1, 1, 1], axes=[2, 3], keepdims=1)
    n("Sub", ["Hf", "Wf"], "gapf", F32, [1, 1, 1, 1])
    init("ONE", np.array(1.0, np.float32), np.float32)
    n("Sub", ["Hf", "ONE"], "Hm1", F32, [1, 1, 1, 1])

    # ---- lenOfRow[r] = r - gap + 1  ([1,1,RH,1]) --------------------------
    rramp = np.arange(RH, dtype=np.float32).reshape(1, 1, RH, 1)
    init("rramp", rramp, np.float32)
    n("Sub", ["rramp", "gapf"], "r_minus_gap", F32, [1, 1, RH, 1])
    n("Add", ["r_minus_gap", "ONE"], "lenOfRow", F32, [1, 1, RH, 1])

    # ---- rowcolor[r] = sum_k k * (cnt[k] == lenOfRow[r]) ------------------
    n("Equal", ["cnt", "lenOfRow"], "match", BOOL, [1, 10, RH, 1])  # broadcast
    n("Cast", ["match"], "matchf", F16, [1, 10, RH, 1], to=F16)
    kvec = np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1)
    init("kvec", kvec, np.float16)
    n("Mul", ["matchf", "kvec"], "kmatch", F16, [1, 10, RH, 1])
    n("ReduceSum", ["kmatch"], "rowcolor", F16, [1, 1, RH, 1], axes=[1], keepdims=1)

    # ---- occupied(r,c) = (r + c >= H-1) AND rowin AND colin --------------
    # work in fp16 (all values are small integers -> exact, half the bytes).
    rramp16 = np.arange(RH, dtype=np.float16).reshape(1, 1, RH, 1)
    cramp16 = np.arange(RW, dtype=np.float16).reshape(1, 1, 1, RW)
    init("rramp16", rramp16, np.float16)
    init("cramp16", cramp16, np.float16)
    n("Add", ["rramp16", "cramp16"], "rcsum", F16, [1, 1, RH, RW])  # broadcast r+c
    n("Cast", ["Hm1"], "Hm1h", F16, [1, 1, 1, 1], to=F16)
    # ge: rcsum >= Hm1  ==  Not(rcsum < Hm1)
    n("Less", ["rcsum", "Hm1h"], "lt", BOOL, [1, 1, RH, RW])
    n("Not", ["lt"], "ge", BOOL, [1, 1, RH, RW])

    # ingrid = rowin AND colin (bool)
    n("Cast", ["rowinS"], "rowinb", BOOL, [1, 1, RH, 1], to=BOOL)
    n("Cast", ["colinS"], "colinb", BOOL, [1, 1, 1, RW], to=BOOL)
    n("And", ["rowinb", "colinb"], "ingrid", BOOL, [1, 1, RH, RW])
    n("And", ["ge", "ingrid"], "occupied", BOOL, [1, 1, RH, RW])

    # ---- label map L (small, fp16) ---------------------------------------
    init("ZERO16", np.array(0.0, np.float16), np.float16)
    n("Where", ["occupied", "rowcolor", "ZERO16"], "occval", F16, [1, 1, RH, RW])
    init("S99", np.array(99.0, np.float16), np.float16)
    n("Where", ["ingrid", "occval", "S99"], "Lf", F16, [1, 1, RH, RW])
    n("Cast", ["Lf"], "Lu8", U8, [1, 1, RH, RW], to=U8)

    # ---- pad to 30x30 with 99 --------------------------------------------
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - RH, 30 - RW], np.int64), np.int64)
    init("P99", np.array(99, np.uint8), np.uint8)
    n("Pad", ["Lu8", "pads", "P99"], "L30", U8, [1, 1, 30, 30], mode="constant")

    # ---- output = Equal(L, arange[0..9]) : FREE [1,10,30,30] bool ---------
    arange = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("arange", arange, np.uint8)
    nodes.append(helper.make_node("Equal", ["L30", "arange"], ["output"]))

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task301", [x], [y], inits, value_info=vis)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
