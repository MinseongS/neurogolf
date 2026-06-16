"""task378 (ARC-AGI ec883f72) — "antenna man": add 4 diagonal rays to a bullseye.

Rule (from the generator):
  Input = a concentric "bullseye": an OUTER ring of colors[1] (rows [row-2,row+height+1],
  cols [col-2,col+width+1]), a MIDDLE ring of black, and a solid INNER rectangle of colors[0]
  (rows [row,row+height-1], cols [col,col+width-1]).  OUTPUT = INPUT plus 4 diagonal "antenna"
  rays of colors[0] emanating OUTWARD from the 4 corners of the OUTER ring:
      from (r0,c0)=top-left    : direction (-1,-1)  -> cells r<r0 on r-c == r0-c0
      from (r0,c1)=top-right   : direction (-1,+1)  -> cells r<r0 on r+c == r0+c1
      from (r1,c0)=bottom-left : direction (+1,-1)  -> cells r>r1 on r+c == r1+c0
      from (r1,c1)=bottom-right: direction (+1,+1)  -> cells r>r1 on r-c == r1-c1
  where (r0,c0,r1,c1) is the bounding box of the OUTER ring = the bbox of ALL non-black
  content (colors[1] is the outermost).  Rays are clipped to the size x size grid.

Encoding (closed-form geometric reconstruction, route 10-ch into FREE output via Where):
  colf = sum_k k*input_k  (1x1 Conv) -> the only [1,1,30,30] fp32 plane (non-bg occupancy).
  nonbg row/col occupancy = ReduceMax(colf) over cols / rows -> tiny [1,1,30,1]/[1,1,1,30].
  (r0,r1) = (min,max) occupied row; (c0,c1) = (min,max) occupied col  (scalars via ramps).
  center = ((r0+r1)//2,(c0+c1)//2) lies inside the inner rect -> Gather colors[0] one-hot.
  antenna_mask (2-D) from row/col ramps:
     top = (R<r0) AND (Equal(R-C, r0-c0) OR Equal(R+C, r0+c1))
     bot = (R>r1) AND (Equal(R-C, r1-c1) OR Equal(R+C, r1+c0))
     mask = (top OR bot) AND in_grid     (in_grid = rowin AND colin, the size x size region)
  output = Where(mask, colors0_onehot[1,10,1,1], input)  -> FREE [1,10,30,30] bool output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64

N = 30
W = 12  # active canvas: grid is size x size, size in [6,12], top-left of the 30x30


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- per-channel row/col presence (tiny tensors) ----------------------
    # The TRUE outer-ring corners = inner-rect (colors[0]) bbox expanded by 2.
    # Inner rect is clipped LATER than the outer ring, so on every on-grid side
    # the visible inner bbox edge is the true edge -> ray origins are exact, and
    # clipping pushes every off-grid corner's diagonal fully off the grid.
    init("ZERO", np.array(0.0, np.float32), np.float32)
    init("BIG16", np.array(1000.0, np.float16), np.float16)
    n("ReduceMax", ["input"], "chrow32", axes=[3], keepdims=1)  # [1,10,30,1] fp32
    n("ReduceMax", ["input"], "chcol32", axes=[2], keepdims=1)  # [1,10,1,30] fp32
    # presence (bool), then slice the cheap bool to the active W rows/cols
    n("Greater", ["chrow32", "ZERO"], "chrow_b30")  # [1,10,30,1] bool
    n("Greater", ["chcol32", "ZERO"], "chcol_b30")  # [1,10,1,30] bool
    init("w_s", np.array([0], np.int64), np.int64)
    init("w_e", np.array([W], np.int64), np.int64)
    init("ax2", np.array([2], np.int64), np.int64)
    init("ax3", np.array([3], np.int64), np.int64)
    n("Slice", ["chrow_b30", "w_s", "w_e", "ax2"], "chrow_b")  # [1,10,W,1] bool
    n("Slice", ["chcol_b30", "w_s", "w_e", "ax3"], "chcol_b")  # [1,10,1,W] bool

    rampR = np.arange(W, dtype=np.float16).reshape(1, 1, W, 1)
    rampC = np.arange(W, dtype=np.float16).reshape(1, 1, 1, W)
    init("rampR", rampR, np.float16)   # [1,1,W,1] fp16
    init("rampC", rampC, np.float16)   # [1,1,1,W] fp16
    init("ZERO16", np.array(0.0, np.float16), np.float16)

    # per-channel bbox edges -> [1,10,1,1] fp16 (ramp values <= W-1, exact)
    # rmax = max present ramp ; rmin = min present ramp (absent rows -> BIG)
    n("Where", ["chrow_b", "rampR", "ZERO16"], "chrow_r")        # [1,10,W,1]
    n("ReduceMax", ["chrow_r"], "rmax_k", axes=[2], keepdims=1)  # [1,10,1,1]
    n("Where", ["chrow_b", "rampR", "BIG16"], "chrow_minsrc")    # [1,10,W,1]
    n("ReduceMin", ["chrow_minsrc"], "rmin_k", axes=[2], keepdims=1)
    n("Where", ["chcol_b", "rampC", "ZERO16"], "chcol_c")        # [1,10,1,W]
    n("ReduceMax", ["chcol_c"], "cmax_k", axes=[3], keepdims=1)
    n("Where", ["chcol_b", "rampC", "BIG16"], "chcol_minsrc")    # [1,10,1,W]
    n("ReduceMin", ["chcol_minsrc"], "cmin_k", axes=[3], keepdims=1)

    # present_k (bool) per channel
    n("ReduceMax", ["chrow32"], "pres_kf", axes=[2], keepdims=1)  # [1,10,1,1] fp32
    n("Greater", ["pres_kf", "ZERO"], "pres_b")  # bool

    # span_k = (rmax-rmin)+(cmax-cmin). Absent or bg(ch0) -> +BIG (excluded).
    n("Sub", ["rmax_k", "rmin_k"], "rspan_k")
    n("Sub", ["cmax_k", "cmin_k"], "cspan_k")
    n("Add", ["rspan_k", "cspan_k"], "span_k")  # [1,10,1,1] fp16
    init("BIGV", np.array(1000.0, np.float16).reshape(1, 1, 1, 1), np.float16)
    n("Where", ["pres_b", "span_k", "BIGV"], "span_pres")  # [1,10,1,1]
    ch0pen = np.zeros((1, 10, 1, 1), np.float16); ch0pen[0, 0, 0, 0] = 2000.0
    init("CH0PEN", ch0pen, np.float16)
    n("Add", ["span_pres", "CH0PEN"], "span_f")  # [1,10,1,1] fp16

    # inner channel = argmin span -> Equal-to-min one-hot
    n("ReduceMin", ["span_f"], "minspan", axes=[1], keepdims=1)  # [1,1,1,1]
    n("Equal", ["span_f", "minspan"], "is_inner")   # [1,10,1,1] bool
    n("Cast", ["is_inner"], "is_inner_f", to=F16)   # [1,10,1,1] fp16 {0,1}

    # gather inner bbox edges = sum_k edge_k * is_inner_k
    n("Mul", ["rmin_k", "is_inner_f"], "rmin_sel")
    n("ReduceSum", ["rmin_sel"], "inr0", axes=[1], keepdims=1)  # [1,1,1,1] fp16
    n("Mul", ["rmax_k", "is_inner_f"], "rmax_sel")
    n("ReduceSum", ["rmax_sel"], "inr1", axes=[1], keepdims=1)
    n("Mul", ["cmin_k", "is_inner_f"], "cmin_sel")
    n("ReduceSum", ["cmin_sel"], "inc0", axes=[1], keepdims=1)
    n("Mul", ["cmax_k", "is_inner_f"], "cmax_sel")
    n("ReduceSum", ["cmax_sel"], "inc1", axes=[1], keepdims=1)

    # outer-ring corners = inner bbox +/- 2 (fp16 scalars, values <= ~33)
    init("TWO16", np.array(2.0, np.float16), np.float16)
    n("Sub", ["inr0", "TWO16"], "r0")  # outer top
    n("Add", ["inr1", "TWO16"], "r1")  # outer bot
    n("Sub", ["inc0", "TWO16"], "c0")  # outer left
    n("Add", ["inc1", "TWO16"], "c1")  # outer right

    # colors[0] one-hot fp32 (for the FREE Where value branch dtype = input fp32)
    n("Cast", ["is_inner_f"], "color0_oh", to=F32)  # [1,10,1,1] fp32 {0,1}

    # ---- diagonal constants (scalars, fp16) -------------------------------
    n("Sub", ["r0", "c0"], "d_tl")  # r0-c0
    n("Sub", ["r1", "c1"], "d_br")  # r1-c1
    n("Add", ["r0", "c1"], "s_tr")  # r0+c1
    n("Add", ["r1", "c0"], "s_bl")  # r1+c0

    # ---- 2-D planes built on the W x W active canvas (top-left) -----------
    n("Sub", ["rampR", "rampC"], "RmC")  # [1,1,W,W] fp16
    n("Add", ["rampR", "rampC"], "RpC")  # [1,1,W,W] fp16

    # ---- in-grid 1-D occupancy (size x size top-left region), W-length -----
    # rowin[r] = any channel present in row r (over all channels incl. bg ch0)
    n("ReduceMax", ["chrow32"], "rowin_f30", axes=[1], keepdims=1)  # [1,1,30,1] fp32
    n("ReduceMax", ["chcol32"], "colin_f30", axes=[1], keepdims=1)  # [1,1,1,30] fp32
    n("Greater", ["rowin_f30", "ZERO"], "rowin30")  # [1,1,30,1] bool
    n("Greater", ["colin_f30", "ZERO"], "colin30")  # [1,1,1,30] bool
    n("Slice", ["rowin30", "w_s", "w_e", "ax2"], "rowin")  # [1,1,W,1] bool
    n("Slice", ["colin30", "w_s", "w_e", "ax3"], "colin")  # [1,1,1,W] bool

    # ---- ray masks: keep row-only conds 1-D until the final broadcast -----
    n("Less", ["rampR", "r0"], "isTop")     # [1,1,W,1] bool (r<r0)
    n("Greater", ["rampR", "r1"], "isBot")  # [1,1,W,1] bool (r>r1)
    n("And", ["isTop", "rowin"], "topRow")  # [1,1,W,1] bool
    n("And", ["isBot", "rowin"], "botRow")  # [1,1,W,1] bool
    n("Equal", ["RmC", "d_tl"], "eq_tl")    # [1,1,W,W] bool
    n("Equal", ["RpC", "s_tr"], "eq_tr")
    n("Equal", ["RmC", "d_br"], "eq_br")
    n("Equal", ["RpC", "s_bl"], "eq_bl")
    n("Or", ["eq_tl", "eq_tr"], "diagTop")  # [1,1,W,W] bool
    n("Or", ["eq_br", "eq_bl"], "diagBot")
    n("And", ["diagTop", "topRow"], "rayTop")  # broadcast [1,1,W,1] -> [1,1,W,W]
    n("And", ["diagBot", "botRow"], "rayBot")
    n("Or", ["rayTop", "rayBot"], "ray")       # [1,1,W,W] bool
    n("And", ["ray", "colin"], "maskW")        # broadcast [1,1,1,W] -> in-grid cols

    # ---- pad mask from W x W to 30 x 30 (Pad rejects bool -> via uint8) ----
    n("Cast", ["maskW"], "maskW_u8", to=TensorProto.UINT8)  # [1,1,W,W] uint8
    init("pads", np.array([0, 0, 0, 0, 0, 0, N - W, N - W], np.int64), np.int64)
    init("ZEROU8", np.array(0, np.uint8), np.uint8)
    n("Pad", ["maskW_u8", "pads", "ZEROU8"], "mask30_u8", mode="constant")  # [1,1,30,30] uint8
    n("Cast", ["mask30_u8"], "mask", to=BOOL)  # [1,1,30,30] bool

    # ---- output = Where(mask, color0, input) : FREE [1,10,30,30] -----------
    n("Where", ["mask", "color0_oh", "input"], "output")

    x = helper.make_tensor_value_info("input", F32, [1, 10, N, N])
    y = helper.make_tensor_value_info("output", F32, [1, 10, N, N])
    g = helper.make_graph(nodes, "task378", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
