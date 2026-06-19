"""task238 (ARC-AGI 9aec4887) — box + recolored sprite overlay.

Rule (verified exact, see generator):
  Input holds TWO separated objects on a background grid:
    * a CYAN (8) conway sprite (length L blob, bbox EXACTLY LxL).
    * a hollow BOX frame (L+2)x(L+2) drawn with 4 DISTINCT non-cyan colours,
      one per side: top=colors[0] right=colors[1] bottom=colors[2] left=colors[3]
      (each side a length-L run, corners empty).
  Output = the (L+2)x(L+2) box frame with the cyan sprite copied into the
  interior (+1,+1) and each cyan cell recoloured by its sprite-coord quadrant:
        d0 = r - c , d1 = (L-1) - r - c
        d0<0 & d1>0 -> top    ; d0<0 & d1<0 -> right
        d0>0 & d1<0 -> bottom ; d0>0 & d1>0 -> left
        on a diagonal (d0==0 or d1==0) -> stays cyan (8)
  Empty interior cells -> background.

Recovery (all scalars from per-channel 1-D occupancy, no flood-fill):
  L, sprite (sr,sc) from the cyan-channel bbox.
  top/bottom = horizontal box channels (rowspan 0) min/max row.
  left/right = vertical   box channels (colspan 0) min/max col.
  Sprite pattern = a 7x7 window of channel 8 gathered at (sr,sc).
  Output label assembled on a 7x7 fp16 work canvas (L+2<=7) with sentinel 10
  outside the frame; Equal(L_u8, arange[0..9]) routes the 10-ch one-hot into the
  FREE bool output.

Re-golf vs prior (15.34): bbox min via ArgMax(occ) directly (no where-src plane),
max via Mul(occ,ramp)+ReduceMax; entire 7x7 value canvas runs in fp16 (half mem).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
I64 = TensorProto.INT64
B = TensorProto.BOOL

WORK = 7  # L+2 max = 7


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- per-channel occupancy profiles -------------------------------------
    n("ReduceMax", ["input"], "rowocc", axes=[3], keepdims=1)  # [1,10,30,1] f32
    n("ReduceMax", ["input"], "colocc", axes=[2], keepdims=1)  # [1,10,1,30] f32

    # mins = first occupied index = ArgMax over the {0,1} occupancy profile
    n("ArgMax", ["rowocc"], "rmin_i", axis=2, keepdims=1)      # [1,10,1,1] i64
    n("ArgMax", ["colocc"], "cmin_i", axis=3, keepdims=1)
    n("Cast", ["rmin_i"], "rmin", to=F32)                      # [1,10,1,1] f32
    n("Cast", ["cmin_i"], "cmin", to=F32)

    # per-channel COLUMN COUNT (# distinct occupied columns) classifies sides
    # and gives cyan L directly (no max-index / ramp plane needed):
    #   vertical box side  (Lx1) -> colcount == 1
    #   horizontal box side (1xL) and cyan (LxL) -> colcount == L
    n("ReduceSum", ["colocc"], "colcnt", axes=[3], keepdims=1)  # [1,10,1,1] f32

    # presence per channel
    init("half", np.array(0.5, np.float32), np.float32)
    n("ReduceMax", ["rowocc"], "pres", axes=[2], keepdims=1)   # [1,10,1,1] f32
    n("Greater", ["pres", "half"], "present")                  # bool

    # valid box channels: present, not ch0, not ch8
    notmask = np.ones((1, 10, 1, 1), np.bool_)
    notmask[0, 0, 0, 0] = False
    notmask[0, 8, 0, 0] = False
    init("notbox", notmask, np.bool_)
    n("And", ["present", "notbox"], "valid")                   # bool

    # vertical sides: colcount == 1 ; horizontal sides: valid AND not vertical.
    init("threehalf", np.array(1.5, np.float32), np.float32)
    n("Less", ["colcnt", "threehalf"], "cnt1")                 # colcount < 1.5 -> ==1
    n("And", ["cnt1", "valid"], "isV")
    n("Not", ["isV"], "notV")
    n("And", ["valid", "notV"], "isH")

    BIG = 1000.0
    init("PBIGf", np.array(BIG, np.float32), np.float32)
    init("NBIGf", np.array(-BIG, np.float32), np.float32)

    # top = argmin row over horizontal channels ; bottom = argmax
    n("Where", ["isH", "rmin", "PBIGf"], "topkey")             # [1,10,1,1]
    n("Where", ["isH", "rmin", "NBIGf"], "botkey")
    n("Where", ["isV", "cmin", "PBIGf"], "leftkey")
    n("Where", ["isV", "cmin", "NBIGf"], "rightkey")

    n("ArgMin", ["topkey"], "top_i", axis=1, keepdims=1)       # [1,1,1,1] i64
    n("ArgMax", ["botkey"], "bot_i", axis=1, keepdims=1)
    n("ArgMin", ["leftkey"], "left_i", axis=1, keepdims=1)
    n("ArgMax", ["rightkey"], "right_i", axis=1, keepdims=1)

    # colour scalars as fp16 (values 0-9, exact)
    n("Cast", ["top_i"], "topf", to=F16)                       # [1,1,1,1] f16
    n("Cast", ["bot_i"], "botf", to=F16)
    n("Cast", ["left_i"], "leftf", to=F16)
    n("Cast", ["right_i"], "rightf", to=F16)

    # ---- cyan (ch8) scalars: L, sr, sc --------------------------------------
    init("shp10", np.array([1, 10], np.int64), np.int64)
    init("idx8", np.array([8], np.int64), np.int64)
    n("Reshape", ["rmin", "shp10"], "rmin10")
    n("Reshape", ["cmin", "shp10"], "cmin10")
    n("Reshape", ["colcnt", "shp10"], "colcnt10")
    n("Gather", ["rmin10", "idx8"], "sr2", axis=1)             # [1,1]
    n("Gather", ["cmin10", "idx8"], "sc2", axis=1)
    n("Gather", ["colcnt10", "idx8"], "Lc2", axis=1)           # cyan colcount = L
    init("shp1", np.array([1], np.int64), np.int64)
    n("Reshape", ["sr2", "shp1"], "sr")                        # [1] f32
    n("Reshape", ["sc2", "shp1"], "sc")
    n("Reshape", ["Lc2", "shp1"], "Lf32")                      # L [1] f32
    init("onef32", np.array(1.0, np.float32), np.float32)
    n("Sub", ["Lf32", "onef32"], "Lm1f32")                     # L-1  [1] f32
    n("Cast", ["Lm1f32"], "Lm1", to=F16)                       # L-1 f16

    # ---- sprite window: 7x7 of channel 8 at (sr,sc) -------------------------
    init("idx9", np.array([9], np.int64), np.int64)
    init("axc1", np.array([1], np.int64), np.int64)
    n("Slice", ["input", "idx8", "idx9", "axc1"], "cyplane")   # [1,1,30,30] f32

    # window starts at (sr-1, sc-1); canvas (R,C) maps to sprite (R-1, C-1)
    baseW = np.arange(WORK, dtype=np.float32) - 1.0
    init("baseW", baseW, np.float32)
    init("c0", np.array(0.0, np.float32), np.float32)
    init("c29", np.array(29.0, np.float32), np.float32)
    n("Add", ["baseW", "sr"], "ridx_f")
    n("Clip", ["ridx_f", "c0", "c29"], "ridx_cl")
    n("Cast", ["ridx_cl"], "ridx", to=I64)                     # [7]
    n("Add", ["baseW", "sc"], "cidx_f")
    n("Clip", ["cidx_f", "c0", "c29"], "cidx_cl")
    n("Cast", ["cidx_cl"], "cidx", to=I64)

    n("Gather", ["cyplane", "ridx"], "Vr", axis=2)             # [1,1,7,30] f32
    n("Gather", ["Vr", "cidx"], "spr", axis=3)                 # [1,1,7,7] f32
    n("Greater", ["spr", "half"], "sprb")                      # [1,1,7,7] bool

    # ---- 7x7 output canvas coords (fp16) ------------------------------------
    Rr = np.arange(WORK, dtype=np.float16).reshape(1, 1, WORK, 1)
    Cc = np.arange(WORK, dtype=np.float16).reshape(1, 1, 1, WORK)
    init("Rr", Rr, np.float16)                                 # [1,1,7,1] f16
    init("Cc", Cc, np.float16)                                 # [1,1,1,7] f16
    init("onef16", np.array(1.0, np.float16), np.float16)

    # interior coords r=R-1, c=C-1
    n("Sub", ["Rr", "onef16"], "rint")                         # [1,1,7,1] f16
    n("Sub", ["Cc", "onef16"], "cint")                         # [1,1,1,7] f16
    n("Sub", ["rint", "cint"], "d0")                           # [1,1,7,7] f16
    n("Add", ["rint", "cint"], "rpc")                          # r+c
    n("Sub", ["Lm1", "rpc"], "d1")                             # (L-1)-r-c

    init("zero16", np.array(0.0, np.float16), np.float16)
    n("Less", ["d0", "zero16"], "d0neg")
    n("Greater", ["d0", "zero16"], "d0pos")
    n("Less", ["d1", "zero16"], "d1neg")
    n("Greater", ["d1", "zero16"], "d1pos")

    n("And", ["d0neg", "d1pos"], "qtop")
    n("And", ["d0neg", "d1neg"], "qright")
    n("And", ["d0pos", "d1neg"], "qbot")
    n("And", ["d0pos", "d1pos"], "qleft")

    # sprite-cell colour value (fp16) by quadrant; default cyan 8
    init("cyan16", np.array(8.0, np.float16), np.float16)
    n("Where", ["qtop", "topf", "cyan16"], "v1")               # [1,1,7,7] f16
    n("Where", ["qright", "rightf", "v1"], "v2")
    n("Where", ["qbot", "botf", "v2"], "v3")
    n("Where", ["qleft", "leftf", "v3"], "sprcol")

    # ---- border colours on the 7x7 canvas (fp16) ----------------------------
    # P = L+1 (= Lm1 + 2)
    init("two16", np.array(2.0, np.float16), np.float16)
    n("Add", ["Lm1", "two16"], "P")                            # P = L+1

    # masks (fp16 Equal not supported -> use range tests)
    n("Less", ["Rr", "onef16"], "isR0")                        # R<1 -> R==0
    n("Less", ["Cc", "onef16"], "isC0")
    # R==P : P-0.5 < R < P+0.5
    init("half16", np.array(0.5, np.float16), np.float16)
    n("Sub", ["P", "half16"], "Pm")
    n("Add", ["P", "half16"], "Pp")
    n("Greater", ["Rr", "Pm"], "RgtPm")
    n("Less", ["Rr", "Pp"], "RltPp")
    n("And", ["RgtPm", "RltPp"], "isRP")
    n("Greater", ["Cc", "Pm"], "CgtPm")
    n("Less", ["Cc", "Pp"], "CltPp")
    n("And", ["CgtPm", "CltPp"], "isCP")
    # 1<=C<=L : C>0 and C<P
    n("Greater", ["Cc", "half16"], "Cge1")
    n("Less", ["Cc", "P"], "CleL")
    n("And", ["Cge1", "CleL"], "Cin")
    n("Greater", ["Rr", "half16"], "Rge1")
    n("Less", ["Rr", "P"], "RleL")
    n("And", ["Rge1", "RleL"], "Rin")

    # gate interior to actual interior region AND sprite presence in one mask
    n("And", ["Rin", "Cin"], "interiorm")
    n("And", ["interiorm", "sprb"], "interiorkeep")
    n("Where", ["interiorkeep", "sprcol", "zero16"], "interior")

    n("And", ["isR0", "Cin"], "mtop")
    n("And", ["isRP", "Cin"], "mbot")
    n("And", ["isC0", "Rin"], "mleft")
    n("And", ["isCP", "Rin"], "mright")

    n("Where", ["mtop", "topf", "interior"], "b1")
    n("Where", ["mright", "rightf", "b1"], "b2")
    n("Where", ["mbot", "botf", "b2"], "b3")
    n("Where", ["mleft", "leftf", "b3"], "valplane")

    # ---- off-frame sentinel: R>P or C>P -> 10 -------------------------------
    n("Greater", ["Rr", "P"], "Rgt")
    n("Greater", ["Cc", "P"], "Cgt")
    n("Or", ["Rgt", "Cgt"], "offframe")
    init("ten16", np.array(10.0, np.float16), np.float16)
    n("Where", ["offframe", "ten16", "valplane"], "Lf")        # [1,1,7,7] f16
    n("Cast", ["Lf"], "Lu8", to=U8)                            # [1,1,7,7] u8

    # ---- pad to 30x30 with sentinel 10, then Equal -> one-hot bool ----------
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    init("u10", np.array(10, np.uint8), np.uint8)
    n("Pad", ["Lu8", "padpads", "u10"], "Lfull", mode="constant")  # [1,1,30,30] u8
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["Lfull", "chan"], "output")                    # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task238", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
