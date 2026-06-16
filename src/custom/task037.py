"""task037 (ARC-AGI 1f876c06) — connect the two same-colored endpoints of each 45° diagonal.

Rule (from generator task_1f876c06.py, size=10 fixed):
  3..6 diagonal segments are drawn on a 10x10 grid, each at 45° (direction cdiff=+1
  down-right or -1 down-left), length 3..7, a DISTINCT color per segment.  The INPUT
  shows ONLY the two endpoints of each segment (start (r,c) and end (r+L-1,c+(L-1)*cdiff),
  both painted that color).  The OUTPUT fills the entire diagonal between (and incl.) the
  two endpoints with the color.  Segments never overlap a cell (generator's bitmap guard).

  Verified exactly (0/2000 fresh): for each color channel k, a cell is filled iff it lies
  between two color-k endpoints along EITHER the main diagonal (r-c const) OR the anti
  diagonal (r+c const):
      mainfill = prefixOR_upleft(ch) AND suffixOR_downright(ch)     (dir (1,1))
      antifill = prefixOR_upright(ch) AND suffixOR_downleft(ch)     (dir (1,-1))
      fill     = mainfill OR antifill
  On the WRONG-direction diagonal the two endpoints sit on different diagonal lines so no
  fill is produced -> a plain OR of the two directions is exact.

Encoding (route 10-ch expansion into FREE output via Equal(L_uint8, arange)):
  - Slice channels 1..9 of the input on the 10x10 active region: ch [1,9,10,10] f16.
  - KEY GEOMETRY BOUND: the max endpoint distance is 6 (length<=7), so each directional
    prefix/suffix-OR is a BOUNDED 7-cell diagonal aggregate = ONE 7x7 diagonal Conv +
    threshold (>0), no doubling-shift chain and no [100,100] reachability matrix.
  - Reshape ch to [9,1,10,10] (colour channels -> BATCH) so a single 7x7 kernel serves every
    channel.  TWO kernels (K_main = diag, K_anti = anti-diag), each reused for the two
    opposite directions by swapping the asymmetric SAME pad side:
      up-left  Conv(K_main,pad TL) , down-right Conv(K_main,pad BR)  -> AND = main fill
      up-right Conv(K_anti,pad TR) , down-left  Conv(K_anti,pad BL)  -> AND = anti fill
      fill = main OR anti   (the two directions are disjoint -> plain OR is exact)
  - collapse the 9 BATCH channels to a colour-index L via a [1,9] colour-weight MatMul
    (fills are disjoint -> the weighted sum is the unique colour 1..9, 0 elsewhere).
  - Pad L to 30x30 with sentinel 255 (off-grid never matches any colour) -> uint8.
  - output = Equal(L_u8[1,1,30,30], arange[0..9][1,10,1,1]) -> BOOL [1,10,30,30] = FREE output.
    In-grid bg cells have L=0 -> ch0=1; off-grid L=255 -> all channels 0 (matches the all-zero
    off-grid target).

  pts 14.84, mem 25700, params 146, fresh 200/200.  Dominant intermediate = the four fp16
  Conv outputs (1800B each, [9,1,10,10]) + the fp32 channel slice (3600B).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I64 = TensorProto.INT64

N = 10  # active grid is always 10x10


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- slice colour channels 1..9 on the 10x10 active region -------------
    init("sl_s", np.array([1, 0, 0], np.int64), np.int64)
    init("sl_e", np.array([10, N, N], np.int64), np.int64)
    init("sl_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "sl_s", "sl_e", "sl_ax"], "ch_f32")   # [1,9,10,10] f32
    n("Cast", ["ch_f32"], "ch", to=F16)                        # [1,9,10,10] f16 {0,1}

    # ---- DIAGONAL prefix/suffix-OR via a SHARED 7x7 diagonal Conv -----------
    # Max endpoint distance is 6 (length<=7), so "an endpoint lies up-left within 6 along
    # the main diagonal" = a 7x7 cross-correlation with K[i,i]=1 (i=0..6) under asymmetric
    # SAME padding (top=left=6), thresholded >0.  Treat the 9 colour channels as the BATCH
    # (reshape [1,9,10,10]->[9,1,10,10]) so ONE [1,1,7,7] kernel serves every channel.
    # The other 3 directions reuse the SAME kernel by flipping axes (Slice step -1):
    #   up-left   : ch
    #   down-right: flip rows & cols
    #   up-right  : flip cols
    #   down-left : flip rows
    # put the 9 colour channels on the BATCH axis so a single [1,1,7,7] kernel serves all.
    init("b91", np.array([9, 1, N, N], np.int64), np.int64)
    n("Reshape", ["ch", "b91"], "chb")                   # [9,1,10,10] f16
    init("ZH", np.array(0.0, np.float16), np.float16)

    # Four 7x7 diagonal kernels (offset t=0..6), each with the matching asymmetric SAME
    # padding so the conv aggregates the 7 cells lying in that direction (no flips needed).
    #   up-left   : K[t,t]=1, pad top=left=6   (reads (r-t,c-t))
    #   down-right: K[6-t,6-t]=1, pad bot=right=6
    #   up-right  : K[t,6-t]=1, pad top=right=6
    #   down-left : K[6-t,t]=1, pad bot=left=6
    def kern(coords):
        K = np.zeros((1, 1, 7, 7), np.float16)
        for (i, j) in coords:
            K[0, 0, i, j] = 1.0
        return K
    # ul reads (r-t,c-t): K[t,t], pad top/left ; dr reads (r+t,c+t): K[t,t], pad bot/right
    # ur reads (r-t,c+t): K[t,6-t], pad top/right ; dl reads (r+t,c-t): K[t,6-t], pad bot/left
    init("K_main", kern([(t, t) for t in range(7)]), np.float16)       # ul & dr
    init("K_anti", kern([(t, 6 - t) for t in range(7)]), np.float16)   # ur & dl

    def prefix_bool(kw, pads, tag):
        n("Conv", ["chb", kw], f"cv_{tag}", pads=pads)   # [9,1,10,10]
        n("Greater", [f"cv_{tag}", "ZH"], f"pb_{tag}")   # bool
        return f"pb_{tag}"

    ul = prefix_bool("K_main", [6, 6, 0, 0], "ul")
    dr_ = prefix_bool("K_main", [0, 0, 6, 6], "dr")
    ur = prefix_bool("K_anti", [6, 0, 0, 6], "ur")
    dl = prefix_bool("K_anti", [0, 6, 6, 0], "dl")

    n("And", [ul, dr_], "mainf_b")                       # main: up-left AND down-right
    n("And", [ur, dl], "antif_b")                        # anti: up-right AND down-left
    n("Or", ["mainf_b", "antif_b"], "fill_b")            # [9,1,10,10] bool
    n("Cast", ["fill_b"], "fill16", to=F16)              # [9,1,10,10] f16

    # collapse the 9 channels (now BATCH) to a colour-index plane.
    # reshape [9,1,10,10]->[9,100], weight row [1,9]*colour -> [1,100] -> [1,1,10,10]
    init("flat9", np.array([9, N * N], np.int64), np.int64)
    n("Reshape", ["fill16", "flat9"], "fill_flat")                # [9,100]
    init("colw", np.arange(1, 10, dtype=np.float16).reshape(1, 9), np.float16)
    n("MatMul", ["colw", "fill_flat"], "Lflat")                   # [1,100]
    init("L_shape", np.array([1, 1, N, N], np.int64), np.int64)
    n("Reshape", ["Lflat", "L_shape"], "L_f16")                   # [1,1,10,10]
    n("Cast", ["L_f16"], "L_u8", to=U8)                           # [1,1,10,10] uint8 (0..9)

    # ---- pad L to 30x30 with sentinel 255 (off-grid matches no colour) ------
    init("Lpads", np.array([0, 0, 0, 0, 0, 0, 30 - N, 30 - N], np.int64), np.int64)
    init("SENT", np.array(255, np.uint8), np.uint8)
    n("Pad", ["L_u8", "Lpads", "SENT"], "L30", mode="constant")  # [1,1,30,30] u8

    # ---- output = Equal(L, arange[0..9]) -> BOOL [1,10,30,30] (FREE) --------
    init("arange", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L30", "arange"], "output")           # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task037", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
