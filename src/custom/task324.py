"""task324 (ARC-AGI d07ae81c) — draw both 45 deg diagonals through each seed dot,
recolouring by the underlying striped background.

Rule (from the generator, verified 6000/6000 + 20000/20000 on the detectors):
  Background is a two-colour stripe pattern: base colour bg0 = bgcolors[0] fills
  the grid, stripe colour bg1 = bgcolors[1] fills a set of full horizontal stripes
  (rows in brows) and full vertical stripes (cols in bcols).  A few "dot" seeds
  (2..3 cells, exactly two distinct dot colours colors[0]/colors[1]) sit on the
  grid.  In the OUTPUT both diagonals (main r-c=const AND anti r+c=const) through
  every seed are recoloured:  a diagonal cell over the BASE background becomes
  colors[0], a diagonal cell over a STRIPE becomes colors[1].  The seed cells are
  fixed points (their input value already equals the correct output colour), so:

      out[r,c] = colors[0]  if on_diag and input[r,c]==bg0
                 colors[1]  if on_diag and input[r,c]==bg1
                 input[r,c] otherwise

Exact closed-form detectors (no flood-fill / connectivity):
  * bg0 (base): the two background colours each occupy >=5 cells; the dot colours
    occupy <=3 cells each.  The top-left 2x2 block (rows 0..1, cols 0..1) is NEVER
    on a stripe (the first stripe index is >=2) so the unique background colour
    appearing there is bg0.  bg1 = the OTHER background colour.
  * seeds = in-grid cells that are neither bg0 nor bg1.
  * stripe row r = an in-grid row with ZERO bg0 cells (a full stripe row is all
    bg1); stripe col c = an in-grid col with ZERO bg0 cells.
  * colors[1] = max seed value among seeds sitting on a stripe row/col;
    colors[0] = max seed value among the remaining (base) seeds.
  * on_diag[r,c] = (seed exists on the main diagonal r-c) OR (on the anti diagonal
    r+c).  Computed by spreading the seed plane along each diagonal with one full
    diagonal-line Conv (SAME pad), thresholded >0 -- no scan, no NonZero.

Encoding (route the 10-ch expansion into the FREE bool output via Equal):
  Work on the 20x20 active canvas (grid size <=20).  Build a uint8 label plane L
  (output colour 0..9 in-grid, sentinel 255 off-grid), Pad to 30x30, final
  output = Equal(L, arange[0..9]) -> BOOL [1,10,30,30].
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

N = 20  # active grid side bound (width,height in 10..20)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ===== colour-index plane =============================================
    # colf = sum_k k * input_k via a 1x1 Conv (NO [1,10,30,30] intermediate).
    init("convk", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    n("Conv", ["input", "convk"], "colf30")                     # [1,1,30,30] f32
    init("kw", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)

    # slice to the 20x20 active canvas, then work in fp16 (colour idx 0..9 exact)
    init("sl_s", np.array([0, 0], np.int64), np.int64)
    init("sl_e", np.array([N, N], np.int64), np.int64)
    init("sl_ax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["colf30", "sl_s", "sl_e", "sl_ax"], "g32")      # [1,1,20,20] f32
    n("Cast", ["g32"], "g", to=F16)                             # [1,1,20,20] f16
    init("zero16", np.array(0.0, np.float16), np.float16)

    # ===== in-grid rectangle = (row < H) AND (col < W), H/W from 1-D occupancy
    n("ReduceMax", ["input"], "rowocc", axes=[1, 3], keepdims=1)  # [1,1,30,1]
    n("ReduceMax", ["input"], "colocc", axes=[1, 2], keepdims=1)  # [1,1,1,30]
    n("ReduceSum", ["rowocc"], "Hf", axes=[2], keepdims=1)        # [1,1,1,1] = H
    n("ReduceSum", ["colocc"], "Wf", axes=[3], keepdims=1)        # [1,1,1,1] = W
    init("rowramp", np.arange(N, dtype=np.float32).reshape(1, 1, N, 1), np.float32)
    init("colramp", np.arange(N, dtype=np.float32).reshape(1, 1, 1, N), np.float32)
    n("Less", ["rowramp", "Hf"], "rin")                          # [1,1,20,1] bool
    n("Less", ["colramp", "Wf"], "cin")                          # [1,1,1,20] bool
    n("And", ["rin", "cin"], "ingrid")                           # [1,1,20,20] bool

    # ===== background colours bg0, bg1 ====================================
    # per-colour total count
    n("ReduceSum", ["input"], "cnt", axes=[2, 3], keepdims=1)   # [1,10,1,1] f32
    init("five", np.array(5.0, np.float32), np.float32)
    n("Greater", ["cnt", "five"], "isbg_b")                     # bool [1,10,1,1]
    n("Cast", ["isbg_b"], "isbg", to=F32)                       # {0,1}

    # colours present in the top-left 2x2 block
    init("b_s", np.array([0, 0], np.int64), np.int64)
    init("b_e", np.array([2, 2], np.int64), np.int64)
    init("b_ax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["input", "b_s", "b_e", "b_ax"], "blk")          # [1,10,2,2]
    n("ReduceMax", ["blk"], "inblk", axes=[2, 3], keepdims=1)   # [1,10,1,1] {0,1}
    # bg0 one-hot = isbg AND inblk  (unique)
    n("Mul", ["isbg", "inblk"], "bg0oh")                        # [1,10,1,1]
    # bg1 one-hot = isbg AND NOT bg0oh
    n("Sub", ["isbg", "bg0oh"], "bg1oh")                        # [1,10,1,1]

    # scalar colour values bg0, bg1 = sum_k k * onehot  (-> fp16 for Equal with g)
    def colourval(oh, name):
        n("Mul", [oh, "kw"], name + "_p")
        n("ReduceSum", [name + "_p"], name + "32", axes=[1, 2, 3], keepdims=1)
        n("Cast", [name + "32"], name, to=F16)                  # [1,1,1,1] f16
        return name
    colourval("bg0oh", "bg0")
    colourval("bg1oh", "bg1")

    # ===== seed mask = in-grid AND not bg0 AND not bg1 ====================
    n("Equal", ["g", "bg0"], "is_bg0")                          # bool [1,1,20,20]
    n("Equal", ["g", "bg1"], "is_bg1")                          # bool
    n("Or", ["is_bg0", "is_bg1"], "is_bg")
    n("Not", ["is_bg"], "not_bg")
    n("And", ["ingrid", "not_bg"], "seed_b")                    # bool [1,1,20,20]

    # ===== stripe rows / cols (bg0 count == 0 among in-grid cells) =========
    # bg0m = is_bg0 (f16) ; a stripe row/col is an in-grid row/col with 0 bg0.
    n("Cast", ["is_bg0"], "bg0m", to=F16)                       # {0,1} f16
    n("ReduceSum", ["bg0m"], "bg0row", axes=[3], keepdims=1)    # [1,1,20,1] count/row
    n("ReduceSum", ["bg0m"], "bg0col", axes=[2], keepdims=1)    # [1,1,1,20] count/col
    n("Equal", ["bg0row", "zero16"], "row_nobg0")
    n("Equal", ["bg0col", "zero16"], "col_nobg0")
    n("And", ["row_nobg0", "rin"], "striperow")                # [1,1,20,1] bool
    n("And", ["col_nobg0", "cin"], "stripecol")                # [1,1,1,20] bool
    n("Or", ["striperow", "stripecol"], "onstripe")            # [1,1,20,20] bool

    # ===== colors[0], colors[1] ===========================================
    # col1 = max seed value over seeds on a stripe ; col0 = max over base seeds.
    # Where(mask, g, 0) keeps the masked seed values (1 plane each, no cast+mul).
    n("And", ["seed_b", "onstripe"], "seed_str_b")
    n("Not", ["onstripe"], "notstr")
    n("And", ["seed_b", "notstr"], "seed_base_b")
    n("Where", ["seed_str_b", "g", "zero16"], "gstr")
    n("Where", ["seed_base_b", "g", "zero16"], "gbase")
    n("ReduceMax", ["gstr"], "col1", axes=[1, 2, 3], keepdims=1)   # [1,1,1,1] f16
    n("ReduceMax", ["gbase"], "col0", axes=[1, 2, 3], keepdims=1)  # [1,1,1,1] f16

    # ===== on_diag via diagonal-line Convs ================================
    n("Cast", ["seed_b"], "seed16", to=F16)                     # [1,1,20,20] f16
    # Two reach-R diagonal passes (R=10) cover the full reach 19 with a much
    # smaller kernel (21x21 vs 39x39) -> ~5x fewer Conv params.  The intermediate
    # sum is only thresholded (>0) downstream so partial sums are harmless.
    R = 10
    KS = 2 * R + 1                                              # 21
    Kmain = np.zeros((1, 1, KS, KS), np.float16)
    Kanti = np.zeros((1, 1, KS, KS), np.float16)
    for i in range(KS):
        Kmain[0, 0, i, i] = 1.0          # main diagonal r-c const
        Kanti[0, 0, i, KS - 1 - i] = 1.0  # anti diagonal r+c const
    init("Kmain", Kmain, np.float16)
    init("Kanti", Kanti, np.float16)
    n("Conv", ["seed16", "Kmain"], "cvm1", pads=[R, R, R, R])   # reach +/-10
    n("Conv", ["cvm1", "Kmain"], "cvm", pads=[R, R, R, R])      # reach +/-20
    n("Conv", ["seed16", "Kanti"], "cva1", pads=[R, R, R, R])
    n("Conv", ["cva1", "Kanti"], "cva", pads=[R, R, R, R])
    n("Greater", ["cvm", "zero16"], "onm")
    n("Greater", ["cva", "zero16"], "ona")
    n("Or", ["onm", "ona"], "ondiag_raw")                       # bool [1,1,20,20]
    n("And", ["ondiag_raw", "ingrid"], "ondiag")               # restrict in-grid

    # ===== label plane L (uint8) ==========================================
    # default: input colour g (uint8); override diagonal-over-bg0 -> col0,
    #          diagonal-over-bg1 -> col1; off-grid -> sentinel 255.
    n("Cast", ["g"], "g_u8", to=U8)
    n("Cast", ["col0"], "col0_u8", to=U8)
    n("Cast", ["col1"], "col1_u8", to=U8)
    n("And", ["ondiag", "is_bg0"], "diag0")   # diagonal cell over base
    n("And", ["ondiag", "is_bg1"], "diag1")   # diagonal cell over stripe
    n("Where", ["diag0", "col0_u8", "g_u8"], "L1")
    n("Where", ["diag1", "col1_u8", "L1"], "L2")
    # off-grid sentinel 255
    init("sent", np.array(255, np.uint8), np.uint8)
    n("Where", ["ingrid", "L2", "sent"], "L")  # [1,1,20,20] uint8

    # pad to 30x30 (sentinel) and final Equal
    init("Lpads", np.array([0, 0, 0, 0, 0, 0, 30 - N, 30 - N], np.int64), np.int64)
    init("sentpad", np.array(255, np.uint8), np.uint8)
    n("Pad", ["L", "Lpads", "sentpad"], "L30", mode="constant")  # [1,1,30,30] u8
    init("arange", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L30", "arange"], "output")                     # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task324", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
