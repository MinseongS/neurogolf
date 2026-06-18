"""task138 (ARC-AGI 5daaa586) — boxed rays, cropped to the box.

Rule (from the ARC-GEN generator, verified fresh 200/200 in numpy):
  Input is an H x W grid (H,W ~ 10..26) holding a rectangular "box" made of four
  FULL lines: a left & right vertical line (columns `left`,`right`, each spanning
  every row) and a top & bottom horizontal line (rows `up`,`down`, each spanning
  every column).  The four lines carry colours colors[0..3] = (left,right,up,down)
  drawn in a random `draworder` (so the four corners take the colour of the
  last-drawn line through them).  Scattered single-cell pixels of one `drawcolor`
  (== exactly one of the four line colours, distinct colours otherwise) sit on the
  grid.  Each drawcolor pixel emits a RAY in a single global direction: toward the
  LEFT wall if drawcolor==colors[0], RIGHT if colors[1], UP if colors[2], DOWN if
  colors[3]; the ray paints drawcolor from the pixel up to (not into) the wall.
  Output is the box region [up..down] x [left..right] moved to the top-left of a
  fresh canvas: the four edge lines, the drawcolor pixels and their rays.

Encoding (ONNX, opset 11, label-map + Equal):
  1.  colf = sum_k k*input_k -> the one fp32 value plane [1,1,30,30] (3600B); cast
      to fp16 (colh) and run every downstream full-plane op in fp16 (half cost).
  2.  occupancy occ = colf>0 ; per-row / per-col occupancy COUNTS recover H,W and
      the line rows/cols exactly (a full line has count == H (cols) / W (rows); no
      scattered-pixel line ever reaches that count -- 0 collisions over 5000
      instances), giving scalars left/right/up/down via ramp+Where+ReduceMin/Max.
  3.  line colours read from NON-corner line cells (mask vertical lines to interior
      ROWS, horizontal lines to interior COLS, via 1-D per-axis max profiles so no
      extra full plane); drawcolor = ReduceMax(colf on the strict interior).
      Direction = Equal(drawcolor, each line colour).
  4.  ray = directional prefix/suffix OR of the seed mask: ONE triangular MatMul
      per axis (seed @ Mc horizontal, Mr @ seed vertical), triangle chosen by the
      recovered direction; masked to the strict interior.  Only TWO triangle inits
      are needed (upper UT and lower LT) since MrU==UT and MrD==LT.
  5.  value plane V = Where(ray, drawcolor, colh) (rays never reach the walls, so
      the lines incl. baked-in corner overlaps survive).
  6.  crop+shift V to the top-left by Gather(axis2, arange+up) then Gather(axis3,
      arange+left); cells with row>=oh or col>=ow get sentinel 10.
  7.  output = Equal(L, arange[0..9]) -> free BOOL one-hot.
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

N = 30


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- 1. value plane colf = sum_k k*input_k (one fp32 entry plane) --------
    init("kW", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    n("Conv", ["input", "kW"], "colf")                     # [1,1,30,30] f32
    n("Cast", ["colf"], "colh", to=F16)                    # fp16 value plane
    init("half16", np.array(0.5, np.float16), np.float16)
    n("Greater", ["colh", "half16"], "occ")                # bool [1,1,30,30]
    n("Cast", ["occ"], "occh", to=F16)                     # fp16 {0,1}

    # ---- 2. recover H,W and line rows/cols (fp16 small profiles) -------------
    n("ReduceSum", ["occh"], "nzcol", axes=[2], keepdims=1)  # [1,1,1,30] (#rows)
    n("ReduceSum", ["occh"], "nzrow", axes=[3], keepdims=1)  # [1,1,30,1] (#cols)
    n("ReduceMax", ["occh"], "rowany", axes=[3], keepdims=1)  # [1,1,30,1]
    n("ReduceMax", ["occh"], "colany", axes=[2], keepdims=1)  # [1,1,1,30]
    n("ReduceSum", ["rowany"], "Hf", axes=[2], keepdims=1)  # [1,1,1,1] = H (fp16)
    n("ReduceSum", ["colany"], "Wf2", axes=[3], keepdims=1)  # [1,1,1,1] = W

    # full vertical line col -> nzcol==H ; full horizontal row -> nzrow==W
    n("Equal", ["nzcol", "Hf"], "iscol")                   # [1,1,1,30] bool
    n("Equal", ["nzrow", "Wf2"], "isrow")                  # [1,1,30,1] bool

    rampc16 = np.arange(N, dtype=np.float16).reshape(1, 1, 1, N)
    rampr16 = np.arange(N, dtype=np.float16).reshape(1, 1, N, 1)
    init("rampc", rampc16, np.float16)
    init("rampr", rampr16, np.float16)
    init("PBIG", np.array(1000.0, np.float16), np.float16)
    init("NBIG", np.array(-1000.0, np.float16), np.float16)

    n("Where", ["iscol", "rampc", "PBIG"], "lcand")
    n("ReduceMin", ["lcand"], "leftf", axes=[3], keepdims=1)  # [1,1,1,1] fp16
    n("Where", ["iscol", "rampc", "NBIG"], "rcand")
    n("ReduceMax", ["rcand"], "rightf", axes=[3], keepdims=1)
    n("Where", ["isrow", "rampr", "PBIG"], "ucand")
    n("ReduceMin", ["ucand"], "upf", axes=[2], keepdims=1)
    n("Where", ["isrow", "rampr", "NBIG"], "dcand")
    n("ReduceMax", ["dcand"], "downf", axes=[2], keepdims=1)

    # ---- 3. interior mask, line colours & drawcolor --------------------------
    # strict interior = (left<c<right) AND (up<r<down)
    n("Less", ["leftf", "rampc"], "gtL")                   # [1,1,1,30] bool
    n("Less", ["rampc", "rightf"], "ltR")
    n("And", ["gtL", "ltR"], "incol")                      # [1,1,1,30]
    n("Less", ["upf", "rampr"], "gtU")                     # [1,1,30,1]
    n("Less", ["rampr", "downf"], "ltD")
    n("And", ["gtU", "ltD"], "inrow")                      # [1,1,30,1]
    n("And", ["incol", "inrow"], "interior")               # [1,1,30,30] bool

    n("Equal", ["rampc", "leftf"], "isLcol")               # [1,1,1,30] bool
    n("Equal", ["rampc", "rightf"], "isRcol")
    n("Equal", ["rampr", "upf"], "isUrow")                 # [1,1,30,1] bool
    n("Equal", ["rampr", "downf"], "isDrow")

    init("zero16", np.array(0.0, np.float16), np.float16)
    # Per-column max of colf restricted to interior ROWS (one fp16 plane),
    # then read left/right colour by 1-D selection; per-row max restricted to
    # interior COLS for up/down.  Corner overlaps are excluded (they sit on
    # exterior rows/cols of the perpendicular line).
    n("Where", ["inrow", "colh", "zero16"], "colf_ir")     # [1,1,30,30] fp16
    n("ReduceMax", ["colf_ir"], "colmax", axes=[2], keepdims=1)  # [1,1,1,30]
    n("Where", ["incol", "colh", "zero16"], "colf_ic")     # [1,1,30,30] fp16
    n("ReduceMax", ["colf_ic"], "rowmax", axes=[3], keepdims=1)  # [1,1,30,1]

    n("Where", ["isLcol", "colmax", "zero16"], "Lcv")      # [1,1,1,30]
    n("ReduceMax", ["Lcv"], "lc", axes=[3], keepdims=1)    # [1,1,1,1]
    n("Where", ["isRcol", "colmax", "zero16"], "Rcv")
    n("ReduceMax", ["Rcv"], "rc", axes=[3], keepdims=1)
    n("Where", ["isUrow", "rowmax", "zero16"], "Ucv")      # [1,1,30,1]
    n("ReduceMax", ["Ucv"], "uc", axes=[2], keepdims=1)
    n("Where", ["isDrow", "rowmax", "zero16"], "Dcv")
    n("ReduceMax", ["Dcv"], "dc", axes=[2], keepdims=1)

    # drawcolor = max colf over the strict interior (all interior fg == drawcolor)
    n("Where", ["interior", "colh", "zero16"], "intv")     # [1,1,30,30] fp16
    n("ReduceMax", ["intv"], "draw", axes=[2, 3], keepdims=1)  # [1,1,1,1]

    n("Equal", ["draw", "lc"], "isLeft")                   # [1,1,1,1] bool
    n("Equal", ["draw", "rc"], "isRight")
    n("Equal", ["draw", "uc"], "isUp")
    n("Equal", ["draw", "dc"], "isDown")
    n("Or", ["isLeft", "isRight"], "isHor")

    # ---- 4. seed mask & directional ray --------------------------------------
    n("Equal", ["colh", "draw"], "isdraw")                 # [1,1,30,30] bool
    n("And", ["isdraw", "interior"], "seedb")              # interior drawcolor px
    n("Cast", ["seedb"], "seed", to=F16)                   # fp16 {0,1}

    I = np.arange(N).reshape(N, 1)
    J = np.arange(N).reshape(1, N)
    UT = (I <= J).astype(np.float16)                       # [i,j]=(i<=j)
    LT = (I >= J).astype(np.float16)                       # [i,j]=(i>=j)
    init("UT", UT, np.float16)
    init("LT", LT, np.float16)
    # horizontal: right -> cell c gets seeds c0<=c -> Mc[c0,c]=(c0<=c)=UT;
    #             left  -> c0>=c -> Mc=LT.
    n("Where", ["isRight", "UT", "LT"], "Mc")              # [N,N] fp16
    # vertical: down -> cell r gets seeds r0<=r -> Mr[r,r0]=(r0<=r)=LT;
    #           up   -> r0>=r -> Mr=UT.
    n("Where", ["isDown", "LT", "UT"], "Mr")               # [N,N] fp16
    n("MatMul", ["seed", "Mc"], "rayH")                    # [1,1,30,30] fp16
    n("MatMul", ["Mr", "seed"], "rayV")                    # [1,1,30,30] fp16
    n("Where", ["isHor", "rayH", "rayV"], "rayc")
    n("Greater", ["rayc", "half16"], "rayp")               # bool
    n("And", ["rayp", "interior"], "ray")                  # clip to interior

    # ---- 5. value plane V = Where(ray, drawcolor, colh) ----------------------
    n("Where", ["ray", "draw", "colh"], "V")               # [1,1,30,30] fp16

    # ---- 6. crop+shift V to top-left, sentinel outside box -------------------
    init("base", np.arange(N, dtype=np.float32), np.float32)  # [N]
    init("shp1", np.array([1], np.int64), np.int64)
    init("c0", np.array(0.0, np.float32), np.float32)
    init("c29", np.array(float(N - 1), np.float32), np.float32)

    n("Cast", ["upf"], "upf32", to=F32)
    n("Reshape", ["upf32", "shp1"], "up1")                 # [1]
    n("Add", ["base", "up1"], "ridxf")                     # [N]
    n("Clip", ["ridxf", "c0", "c29"], "ridxc")
    n("Cast", ["ridxc"], "ridx", to=I64)
    n("Cast", ["leftf"], "leftf32", to=F32)
    n("Reshape", ["leftf32", "shp1"], "lf1")
    n("Add", ["base", "lf1"], "cidxf")
    n("Clip", ["cidxf", "c0", "c29"], "cidxc")
    n("Cast", ["cidxc"], "cidx", to=I64)

    n("Gather", ["V", "ridx"], "Vr", axis=2)               # [1,1,30,30] fp16
    n("Gather", ["Vr", "cidx"], "Vs", axis=3)              # shifted to origin

    # keep mask: row<oh AND col<ow (oh=down-up+1, ow=right-left+1)
    init("one16", np.array(1.0, np.float16), np.float16)
    n("Sub", ["downf", "upf"], "ohm1")
    n("Add", ["ohm1", "one16"], "oh")
    n("Sub", ["rightf", "leftf"], "owm1")
    n("Add", ["owm1", "one16"], "ow")
    n("Less", ["rampr", "oh"], "rkeep")                    # [1,1,30,1] bool
    n("Less", ["rampc", "ow"], "ckeep")                    # [1,1,1,30] bool
    n("And", ["rkeep", "ckeep"], "keep")                   # [1,1,30,30] bool

    n("Cast", ["Vs"], "Lin", to=U8)                        # integer-exact colours
    init("u10", np.array(10, np.uint8), np.uint8)
    n("Where", ["keep", "Lin", "u10"], "L")                # [1,1,30,30] uint8

    # ---- 7. final Equal into the free BOOL output ----------------------------
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                    # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task138", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
