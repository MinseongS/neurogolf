"""task238 (ARC-AGI 9aec4887) — box + recolored sprite overlay.

Rule (verified 0/2000 fresh in Python on the recovery logic):
  The input holds TWO separated objects on a background grid:
    * a CYAN (colour 8) "conway sprite": a length x length blob whose every row
      and column carries >=1 pixel (so its bbox is EXACTLY length x length).
    * a hollow BOX frame of size (length+2) x (length+2) drawn with 4 DISTINCT
      non-cyan colours, one per side:
        top    = colors[0]  (horizontal run, the upper one)
        right  = colors[1]  (vertical   run, the right one)
        bottom = colors[2]  (horizontal run, the lower one)
        left   = colors[3]  (vertical   run, the left one)
      Each side is a length-long run (corners empty).
  Output is the (length+2) x (length+2) box frame, with the cyan sprite copied
  into the interior (offset +1,+1) and each cyan cell RECOLOURED by the quadrant
  of its sprite coord (r,c) (0-indexed, length L):
        d0 = r - c , d1 = (L-1) - r - c
        d0<0 & d1>0 -> top    (colors[0])
        d0<0 & d1<0 -> right  (colors[1])
        d0>0 & d1<0 -> bottom (colors[2])
        d0>0 & d1>0 -> left   (colors[3])
        on a diagonal (d0==0 or d1==0) -> stays cyan (8)
  Empty interior cells -> background.

Recovery (all scalars from per-channel 1-D occupancy, no flood-fill):
  L, sprite (sr,sc) from the cyan channel bbox.
  top/bottom = horizontal box channels (rowspan 0) min/max row.
  left/right = vertical   box channels (colspan 0) min/max col.
  Sprite pattern = a 7x7 window of channel 8 gathered at (sr,sc).
  The output label plane is assembled on a 7x7 work canvas (length+2<=7) with a
  sentinel (10) outside the (L+2)x(L+2) frame; Equal(L_u8, arange[0..9]) routes
  the 10-channel one-hot into the FREE bool output.
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

WORK = 7  # length+2 max = 7


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    BIG = 1000.0

    # ---- per-channel occupancy profiles -------------------------------------
    n("ReduceMax", ["input"], "rowocc", axes=[3], keepdims=1)  # [1,10,30,1]
    n("ReduceMax", ["input"], "colocc", axes=[2], keepdims=1)  # [1,10,1,30]

    init("half", np.array(0.5, np.float32), np.float32)
    n("Greater", ["rowocc", "half"], "rowb")                   # [1,10,30,1] bool
    n("Greater", ["colocc", "half"], "colb")                   # [1,10,1,30] bool

    rr = np.arange(30, dtype=np.float16).reshape(1, 1, 30, 1)
    rc = np.arange(30, dtype=np.float16).reshape(1, 1, 1, 30)
    init("rr", rr, np.float16)
    init("rc", rc, np.float16)
    init("PBIG", np.array(BIG, np.float16), np.float16)
    init("NBIG", np.array(-BIG, np.float16), np.float16)

    n("Where", ["rowb", "rr", "PBIG"], "rmin_src")
    n("ReduceMin", ["rmin_src"], "rmin16", axes=[2], keepdims=1)  # [1,10,1,1] f16
    n("Where", ["rowb", "rr", "NBIG"], "rmax_src")
    n("ReduceMax", ["rmax_src"], "rmax16", axes=[2], keepdims=1)
    n("Where", ["colb", "rc", "PBIG"], "cmin_src")
    n("ReduceMin", ["cmin_src"], "cmin16", axes=[3], keepdims=1)
    n("Where", ["colb", "rc", "NBIG"], "cmax_src")
    n("ReduceMax", ["cmax_src"], "cmax16", axes=[3], keepdims=1)

    n("Cast", ["rmin16"], "rmin", to=F32)                      # [1,10,1,1]
    n("Cast", ["rmax16"], "rmax", to=F32)
    n("Cast", ["cmin16"], "cmin", to=F32)
    n("Cast", ["cmax16"], "cmax", to=F32)

    # presence per channel
    n("ReduceMax", ["rowocc"], "pres", axes=[2], keepdims=1)   # [1,10,1,1] f32
    n("Greater", ["pres", "half"], "present")                  # bool

    # spans
    n("Sub", ["rmax", "rmin"], "rspan")                        # [1,10,1,1]
    n("Sub", ["cmax", "cmin"], "cspan")

    # valid box channels: present, not ch0, not ch8
    notmask = np.ones((1, 10, 1, 1), np.bool_)
    notmask[0, 0, 0, 0] = False
    notmask[0, 8, 0, 0] = False
    init("notbox", notmask, np.bool_)
    n("And", ["present", "notbox"], "valid")                   # [1,10,1,1] bool

    init("halff", np.array(0.5, np.float32), np.float32)
    # horizontal sides: rowspan 0  (rspan < 0.5)
    n("Less", ["rspan", "halff"], "rspan0")
    n("And", ["rspan0", "valid"], "isH")
    # vertical sides: colspan 0
    n("Less", ["cspan", "halff"], "cspan0")
    n("And", ["cspan0", "valid"], "isV")

    init("PBIGf", np.array(BIG, np.float32), np.float32)
    init("NBIGf", np.array(-BIG, np.float32), np.float32)

    # top = argmin row over horizontal channels ; bottom = argmax
    n("Where", ["isH", "rmin", "PBIGf"], "topkey")             # [1,10,1,1]
    n("Where", ["isH", "rmin", "NBIGf"], "botkey")
    n("Where", ["isV", "cmin", "PBIGf"], "leftkey")
    n("Where", ["isV", "cmin", "NBIGf"], "rightkey")

    # ArgMin/ArgMax along channel axis -> channel index = colour value
    n("ArgMin", ["topkey"], "top_i", axis=1, keepdims=1)       # [1,1,1,1] i64
    n("ArgMax", ["botkey"], "bot_i", axis=1, keepdims=1)
    n("ArgMin", ["leftkey"], "left_i", axis=1, keepdims=1)
    n("ArgMax", ["rightkey"], "right_i", axis=1, keepdims=1)

    n("Cast", ["top_i"], "topf", to=F32)                       # [1,1,1,1]
    n("Cast", ["bot_i"], "botf", to=F32)
    n("Cast", ["left_i"], "leftf", to=F32)
    n("Cast", ["right_i"], "rightf", to=F32)

    # ---- cyan (ch8) scalars: L, sr, sc --------------------------------------
    init("shp10", np.array([1, 10], np.int64), np.int64)
    init("idx8", np.array([8], np.int64), np.int64)
    n("Reshape", ["rmin", "shp10"], "rmin10")
    n("Reshape", ["cmin", "shp10"], "cmin10")
    n("Reshape", ["rmax", "shp10"], "rmax10")
    n("Gather", ["rmin10", "idx8"], "sr2", axis=1)             # [1,1]
    n("Gather", ["cmin10", "idx8"], "sc2", axis=1)
    n("Gather", ["rmax10", "idx8"], "smaxr2", axis=1)
    init("shp1", np.array([1], np.int64), np.int64)
    n("Reshape", ["sr2", "shp1"], "sr")                        # [1] f32
    n("Reshape", ["sc2", "shp1"], "sc")
    n("Reshape", ["smaxr2", "shp1"], "smaxr")
    init("onef", np.array(1.0, np.float32), np.float32)
    n("Sub", ["smaxr", "sr"], "Lm1")                           # L-1  [1]
    n("Add", ["Lm1", "onef"], "L")                             # L    [1]

    # ---- sprite window: 7x7 of channel 8 at (sr,sc) -------------------------
    n("Slice", ["input", "idx8", "idx9", "axc1"], "cyplane")   # [1,1,30,30]
    init("idx9", np.array([9], np.int64), np.int64)
    init("axc1", np.array([1], np.int64), np.int64)

    # window starts at (sr-1, sc-1) so canvas (R,C) maps to sprite (R-1, C-1)
    baseW = np.arange(WORK, dtype=np.float32) - 1.0
    init("baseW", baseW, np.float32)                           # [-1,0,1,..,5]
    init("c0", np.array(0.0, np.float32), np.float32)
    init("c29", np.array(29.0, np.float32), np.float32)
    n("Add", ["baseW", "sr"], "ridx_f")
    n("Clip", ["ridx_f", "c0", "c29"], "ridx_cl")
    n("Cast", ["ridx_cl"], "ridx", to=I64)                     # [7]
    n("Add", ["baseW", "sc"], "cidx_f")
    n("Clip", ["cidx_f", "c0", "c29"], "cidx_cl")
    n("Cast", ["cidx_cl"], "cidx", to=I64)

    n("Gather", ["cyplane", "ridx"], "Vr", axis=2)             # [1,1,7,30]
    n("Gather", ["Vr", "cidx"], "spr", axis=3)                 # [1,1,7,7] f32
    n("Greater", ["spr", "half"], "sprb")                      # [1,1,7,7] bool

    # ---- 7x7 output canvas coords -------------------------------------------
    Rr = np.arange(WORK, dtype=np.float32).reshape(1, 1, WORK, 1)
    Cc = np.arange(WORK, dtype=np.float32).reshape(1, 1, 1, WORK)
    init("Rr", Rr, np.float32)                                 # [1,1,7,1]
    init("Cc", Cc, np.float32)                                 # [1,1,1,7]

    # interior coords r=R-1, c=C-1
    n("Sub", ["Rr", "onef"], "rint")                           # [1,1,7,1]
    n("Sub", ["Cc", "onef"], "cint")                           # [1,1,1,7]
    # broadcast to [1,1,7,7]: d0 = r - c
    n("Sub", ["rint", "cint"], "d0")                           # [1,1,7,7]
    # d1 = (L-1) - r - c
    n("Add", ["rint", "cint"], "rpc")                          # r + c  [1,1,7,7]
    n("Sub", ["Lm1", "rpc"], "d1")                             # (L-1) - r - c

    init("zerof", np.array(0.0, np.float32), np.float32)
    n("Less", ["d0", "zerof"], "d0neg")                        # d0<0
    n("Greater", ["d0", "zerof"], "d0pos")                     # d0>0
    n("Less", ["d1", "zerof"], "d1neg")
    n("Greater", ["d1", "zerof"], "d1pos")

    n("And", ["d0neg", "d1pos"], "qtop")                       # -> top
    n("And", ["d0neg", "d1neg"], "qright")
    n("And", ["d0pos", "d1neg"], "qbot")
    n("And", ["d0pos", "d1pos"], "qleft")

    # sprite-cell colour value (float) by quadrant; default cyan 8
    init("cyanf", np.array(8.0, np.float32), np.float32)
    n("Where", ["qtop", "topf", "cyanf"], "v1")                # [1,1,7,7]
    n("Where", ["qright", "rightf", "v1"], "v2")
    n("Where", ["qbot", "botf", "v2"], "v3")
    n("Where", ["qleft", "leftf", "v3"], "sprcol")             # interior cell colour
    # only where the sprite has a cyan pixel
    n("Where", ["sprb", "sprcol", "zerof"], "interior0")       # [1,1,7,7] f32

    # ---- border colours on the 7x7 canvas -----------------------------------
    # P = L+1 (last index of frame). top row R==0,1<=C<=L ; bottom R==P ;
    # left C==0,1<=R<=L ; right C==P.
    n("Add", ["L", "onef"], "P")                               # P = L+1  [1]

    # masks (broadcast)
    n("Equal", ["Rr", "zerof_b"], "isR0")                      # R==0 [1,1,7,1] bool
    init("zerof_b", np.array(0.0, np.float32), np.float32)
    n("Equal", ["Cc", "zerof_b"], "isC0")                      # C==0 [1,1,1,7]
    # R==P (broadcast P[1] over [1,1,7,1])
    n("Equal", ["Rr", "P"], "isRP")                            # [1,1,7,1]
    n("Equal", ["Cc", "P"], "isCP")                            # [1,1,1,7]
    # 1<=C<=L  i.e. C>=1 and C<=L  -> C>0 and C<P
    n("Greater", ["Cc", "zerof_b"], "Cge1")
    n("Less", ["Cc", "P"], "CleL")                             # C<P  -> C<=L
    n("And", ["Cge1", "CleL"], "Cin")                          # [1,1,1,7]
    n("Greater", ["Rr", "zerof_b"], "Rge1")
    n("Less", ["Rr", "P"], "RleL")
    n("And", ["Rge1", "RleL"], "Rin")                          # [1,1,7,1]

    # gate interior to the actual interior region
    n("And", ["Rin", "Cin"], "interiorm")                      # [1,1,7,7] bool
    n("Where", ["interiorm", "interior0", "zerof"], "interior")

    n("And", ["isR0", "Cin"], "mtop")                          # [1,1,7,7] bool
    n("And", ["isRP", "Cin"], "mbot")
    n("And", ["isC0", "Rin"], "mleft")
    n("And", ["isCP", "Rin"], "mright")

    # build border value plane (priority: borders are mutually exclusive)
    n("Where", ["mtop", "topf", "interior"], "b1")             # [1,1,7,7]
    n("Where", ["mright", "rightf", "b1"], "b2")
    n("Where", ["mbot", "botf", "b2"], "b3")
    n("Where", ["mleft", "leftf", "b3"], "valplane")           # full colour value

    # ---- off-frame sentinel: R>P or C>P -> 10 -------------------------------
    n("Greater", ["Rr", "P"], "Rgt")                           # R>P [1,1,7,1]
    n("Greater", ["Cc", "P"], "Cgt")                           # C>P [1,1,1,7]
    n("Or", ["Rgt", "Cgt"], "offframe")                        # [1,1,7,7] bool
    init("tenf", np.array(10.0, np.float32), np.float32)
    n("Where", ["offframe", "tenf", "valplane"], "Lf")         # [1,1,7,7] f32
    n("Cast", ["Lf"], "Lu8", to=U8)                            # [1,1,7,7] u8

    # ---- pad to 30x30 with sentinel 10, then Equal -> one-hot bool ----------
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    init("u10", np.array(10, np.uint8), np.uint8)
    n("Pad", ["Lu8", "padpads", "u10"], "Lfull", mode="constant")  # [1,1,30,30]
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["Lfull", "chan"], "output")                    # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task238", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
