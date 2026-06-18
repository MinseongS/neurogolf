"""task041 (ARC-AGI 22168020) — fill the chevron interior down to each column's pixel.

Rule (from generator, size=10 fixed; verified EXACT 266/266 on arc-gen+train+test):
  ~6 non-overlapping "downward-V / chevron" shapes, each a DISTINCT colour.  Each shape's two
  top tips share one APEX ROW; arms slope down-inward to a 2-px base.  The OUTPUT fills, in
  every column a shape occupies, the contiguous run from that shape's apex row down to the
  column's lowest coloured pixel:

      out[r,c] = k   iff   apex_k <= r <= bottom_k(c)
      apex_k     = min row containing colour k
      bottom_k(c)= max row of colour k in column c
  Colours distinct per shape  ->  purely per-colour (no connectivity / flood-fill).

  EQUIVALENT single-plane form (also verified 266/266), used here to avoid any [1,9,10,10]
  per-colour work plane:
      nb(r,c)  = colour of the NEAREST coloured pixel at-or-below (r,c) in column c
      out[r,c] = nb        if  nb>0 AND r >= apex_{nb}
               = bg        otherwise
  (the apex test also kills the bleed across a vertical GAP between two same-column shapes:
   a gap cell's nearest-below pixel belongs to the LOWER shape whose apex sits below it.)

Encoding (pay ONE fp32 entry plane, then run everything on tiny [1,1,10,10] planes):
  - Slice colour channels 1..9 on the 10x10 active region -> ch [1,9,10,10] f32 (3600B entry,
    serves BOTH the colour-index collapse AND the per-colour apex vector).
  - colf = Conv(ch, W[1,9,1,1]=[1..9]) -> [1,1,10,10] f32 colour-index plane (one-hot -> sum).
  - V = where(occ, colf + 100*(N-1-r), 0)  : pack colour (units) + "height from bottom"
    (hundreds) so a suffix-MAX down each column selects the NEAREST-below pixel.  f16, <2048 exact.
  - M = suffix-max over rows>=r  via 4 shift+Max doublings (s=1,2,4,8) ; nb = Mod(M,100).
  - apex vector: rowany_k = ReduceMax_c ch ; score_k = ReduceMax_r(rowany_k*(N-1-r)) ;
    apex_k = (N-1)-score_k ; prepend apex[0]=999 (bg never satisfies r>=apex).  Gather by nb.
  - fill = (nb>0) AND (r >= apex_nb)  ;  L = nb*fill  -> [1,1,10,10] colour-index.
  - Pad L to 30x30 with sentinel 255 ; output = Equal(L_u8, arange[0..9]) -> BOOL (FREE output).
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

    # ---- entry: slice colour channels 1..9 on the 10x10 region (3600B fp32) ----
    init("sl_s", np.array([1, 0, 0], np.int64), np.int64)
    init("sl_e", np.array([10, N, N], np.int64), np.int64)
    init("sl_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "sl_s", "sl_e", "sl_ax"], "ch")        # [1,9,10,10] f32 {0,1}

    # ---- colf = colour-index plane (1x1 conv collapse) [1,1,10,10] f32 ---------
    init("colw", np.arange(1, 10, dtype=np.float32).reshape(1, 9, 1, 1), np.float32)
    n("Conv", ["ch", "colw"], "colf32")                         # [1,1,10,10] f32
    n("Cast", ["colf32"], "colf", to=F16)                       # f16

    # ---- V = where(occ, colf + 100*(N-1-r), 0) --------------------------------
    init("ZH", np.array(0.0, np.float16), np.float16)
    n("Greater", ["colf", "ZH"], "occ")                         # bool [1,1,10,10]
    # height-from-bottom ramp 100*(N-1-r), shape [1,1,N,1]
    frombot = (100.0 * (N - 1 - np.arange(N))).reshape(1, 1, N, 1)
    init("FROMBOT", frombot, np.float16)
    n("Add", ["colf", "FROMBOT"], "Vfull")                      # f16 (broadcast)
    init("ZHplane", np.array(0.0, np.float16), np.float16)
    n("Where", ["occ", "Vfull", "ZHplane"], "V")                # f16 [1,1,10,10]

    # ---- M = suffix-max over rows>=r  via ONE MaxPool ([N,1] kernel, pad bottom) -
    # out[r,c] = max over input rows [r .. r+N-1] (clipped) = max over all rows >= r.
    n("MaxPool", ["V"], "M", kernel_shape=[N, 1], pads=[0, 0, N - 1, 0],
      strides=[1, 1])                                            # [1,1,N,N] f16
    init("HUND", np.array(100.0, np.float16), np.float16)
    n("Mod", ["M", "HUND"], "nb", fmod=1)                       # f16 nearest-below colour
    n("Cast", ["nb"], "nb_i32", to=I32)                         # int32 index for Gather

    # ---- apex vector: apex_k = (N-1) - max_r(rowany_k * (N-1-r)) ---------------
    n("ReduceMax", ["ch"], "rowany_f32", axes=[3], keepdims=1)  # [1,9,10,1] f32 presence/row
    n("Cast", ["rowany_f32"], "rowany", to=F16)                 # [1,9,10,1] f16 (180B)
    # multiply by (N-1-r) ramp [1,1,N,1] then reduce over rows
    ramp16 = (N - 1 - np.arange(N)).reshape(1, 1, N, 1).astype(np.float16)
    init("RAMP16", ramp16, np.float16)
    n("Mul", ["rowany", "RAMP16"], "rowscore")                  # [1,9,10,1] f16
    n("ReduceMax", ["rowscore"], "score", axes=[2], keepdims=0)  # [1,9,1] f16
    # apex_1..9 = (N-1) - score   -> reshape to [9]
    init("NM1", np.array(float(N - 1), np.float16), np.float16)
    n("Sub", ["NM1", "score"], "apex9_3d")                      # [1,9,1] f16
    init("flat9", np.array([9], np.int64), np.int64)
    n("Reshape", ["apex9_3d", "flat9"], "apex9")                # [9]
    # prepend apex[0] = 999 (bg index never satisfies r>=apex)
    init("APEX0", np.array([999.0], np.float16), np.float16)
    n("Concat", ["APEX0", "apex9"], "apex_vec", axis=0)         # [10] f16
    n("Gather", ["apex_vec", "nb_i32"], "apex_plane")           # [1,1,10,10] f16

    # ---- fill = (nb>0) AND (r >= apex_nb) ; L = nb*fill ------------------------
    rrow = np.arange(N, dtype=np.float16).reshape(1, 1, N, 1)
    init("RROW", rrow, np.float16)
    # r >= apex  <=>  NOT(r < apex)
    n("Less", ["RROW", "apex_plane"], "r_lt_apex")              # bool [1,1,10,10]
    n("Not", ["r_lt_apex"], "r_ge_apex")
    n("Greater", ["M", "ZH"], "nb_pos")                         # M>0 <=> nb pixel exists (f16)
    n("And", ["nb_pos", "r_ge_apex"], "fill_b")                 # bool
    n("Cast", ["fill_b"], "fill16", to=F16)                     # f16 [1,1,10,10]
    n("Mul", ["nb", "fill16"], "L_f16")                         # f16 colour index
    n("Cast", ["L_f16"], "L_u8", to=U8)                         # uint8 [1,1,10,10]

    # ---- pad L to 30x30 with sentinel 255 ; Equal(arange) -> BOOL output ------
    init("Lpads", np.array([0, 0, 0, 0, 0, 0, 30 - N, 30 - N], np.int64), np.int64)
    init("SENT", np.array(255, np.uint8), np.uint8)
    n("Pad", ["L_u8", "Lpads", "SENT"], "L30", mode="constant")  # [1,1,30,30] u8

    init("arange", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L30", "arange"], "output")                     # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task041", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
