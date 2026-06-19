"""task324 (ARC-AGI d07ae81c) — draw both 45 deg diagonals through each seed dot,
recolouring by the underlying striped background.

Rule (from the generator, verified 3000/3000 on the detectors):
  Background is a two-colour stripe pattern: base colour bg0 = bgcolors[0] fills
  the grid, stripe colour bg1 = bgcolors[1] fills a set of full horizontal stripes
  (rows in brows) and full vertical stripes (cols in bcols).  A few "dot" seeds
  (2..3 cells, exactly two distinct dot colours colors[0]/colors[1]) sit on the
  grid.  In the OUTPUT both diagonals (main r-c=const AND anti r+c=const) through
  every seed are recoloured:  a diagonal cell over the BASE background becomes
  colors[0], a diagonal cell over a STRIPE becomes colors[1].  The seed cells are
  fixed points (their input value already equals the correct output colour), so:

      out[r,c] = colors[0]  if on_diag and bg(r,c)==bg0 (base)
                 colors[1]  if on_diag and bg(r,c)==bg1 (stripe)
                 input[r,c] otherwise

Exact closed-form detectors (no flood-fill / connectivity):
  * bg0 (base): the first stripe row/col index is >=2, so the top-left 2x2 block
    is NEVER on a stripe.  Background colours occupy >=5 cells; dot colours <=3.
    bg0 = the unique colour that is BOTH a background colour (count>=5) AND
    present in the top-left 2x2 block.  bg1 = the OTHER background colour.
  * seeds = in-grid cells that are neither bg0 nor bg1.
  * stripe row r = an in-grid row with ZERO bg0 cells; stripe col similarly.
  * colors[1] = max seed value among seeds on a stripe; colors[0] = max among the
    remaining (base) seeds.  (Generator guarantees both kinds of seed exist.)
  * on_diag[r,c] = a seed lies on the main (r-c) OR anti (r+c) diagonal of (r,c),
    obtained by spreading the seed plane along each diagonal with a SAME-padded
    diagonal Conv (applied twice for full +-19 reach), thresholded >0.

Encoding (route the 10-ch expansion into the FREE bool output via Equal):
  The colour-index plane is produced by a DILATED (dil=10) 2x2 Conv whose only
  nonzero weight is the [.,.,0,0] arange tap: on the 30x30 input this yields a
  [1,1,20,20] plane (30 - (2-1)*10 = 20) -- i.e. colf already cropped to the
  20x20 active canvas in ONE op, so every downstream plane is 20x20 (counted at
  400B uint8 / 800B fp16 / 1600B fp32 instead of the 30x30 size).
  Build a uint8 label L (output colour 0..9 in-grid, sentinel 255 off-grid),
  Pad to 30x30, output = Equal(L, arange[0..9]) -> BOOL [1,10,30,30].
"""

import numpy as np
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

    # ===== colour-index plane (DILATED conv crops to 20x20 in one op) ======
    # colf = sum_k k * input_k.  2x2 kernel, only [.,.,0,0] tap = arange, dil=10:
    # output side = 30 - (2-1)*10 = 20 -> [1,1,20,20] f32 directly.
    convk = np.zeros((1, 10, 2, 2), np.float32)
    convk[:, :, 0, 0] = np.arange(10, dtype=np.float32).reshape(10, 1)[:, 0]
    init("convk", convk, np.float32)
    init("kw", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    n("Conv", ["input", "convk"], "g32", dilations=[10, 10])     # [1,1,20,20] f32
    n("Cast", ["g32"], "g", to=F16)                              # [1,1,20,20] f16
    init("zero16", np.array(0.0, np.float16), np.float16)

    # ===== in-grid rectangle = (row < H) AND (col < W) =====================
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
    n("ReduceSum", ["input"], "cnt", axes=[2, 3], keepdims=1)    # [1,10,1,1] f32
    init("five", np.array(5.0, np.float32), np.float32)
    n("Greater", ["cnt", "five"], "isbg_b")                      # bool [1,10,1,1]
    n("Cast", ["isbg_b"], "isbg", to=F32)                        # {0,1}

    init("b_s", np.array([0, 0], np.int64), np.int64)
    init("b_e", np.array([2, 2], np.int64), np.int64)
    init("b_ax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["input", "b_s", "b_e", "b_ax"], "blk")           # [1,10,2,2]
    n("ReduceMax", ["blk"], "inblk", axes=[2, 3], keepdims=1)    # [1,10,1,1] {0,1}
    n("Mul", ["isbg", "inblk"], "bg0oh")                         # bg0 one-hot
    n("Sub", ["isbg", "bg0oh"], "bg1oh")                         # bg1 one-hot

    def colourval(oh, name):
        n("Mul", [oh, "kw"], name + "_p")
        n("ReduceSum", [name + "_p"], name + "32", axes=[1, 2, 3], keepdims=1)
        n("Cast", [name + "32"], name, to=F16)                   # [1,1,1,1] f16
        return name
    colourval("bg0oh", "bg0")
    colourval("bg1oh", "bg1")

    # ===== seed mask = in-grid AND not bg0 AND not bg1 ====================
    n("Equal", ["g", "bg0"], "is_bg0")                           # bool [1,1,20,20]
    n("Equal", ["g", "bg1"], "is_bg1")
    n("Or", ["is_bg0", "is_bg1"], "is_bg")
    n("Not", ["is_bg"], "not_bg")
    n("And", ["ingrid", "not_bg"], "seed_b")                     # bool [1,1,20,20]

    # ===== stripe rows / cols (in-grid row/col with 0 bg0 cells) ===========
    n("Cast", ["is_bg0"], "bg0m", to=F16)                        # {0,1} f16
    n("ReduceSum", ["bg0m"], "bg0row", axes=[3], keepdims=1)     # [1,1,20,1]
    n("ReduceSum", ["bg0m"], "bg0col", axes=[2], keepdims=1)     # [1,1,1,20]
    n("Equal", ["bg0row", "zero16"], "row_nobg0")
    n("Equal", ["bg0col", "zero16"], "col_nobg0")
    n("And", ["row_nobg0", "rin"], "striperow")                 # [1,1,20,1] bool
    n("And", ["col_nobg0", "cin"], "stripecol")                 # [1,1,1,20] bool
    n("Or", ["striperow", "stripecol"], "onstripe")            # [1,1,20,20] bool

    # ===== colors[0], colors[1] ===========================================
    n("And", ["seed_b", "onstripe"], "seed_str_b")
    n("Not", ["onstripe"], "notstr")
    n("And", ["seed_b", "notstr"], "seed_base_b")
    n("Where", ["seed_str_b", "g", "zero16"], "gstr")
    n("Where", ["seed_base_b", "g", "zero16"], "gbase")
    n("ReduceMax", ["gstr"], "col1", axes=[1, 2, 3], keepdims=1)   # [1,1,1,1] f16
    n("ReduceMax", ["gbase"], "col0", axes=[1, 2, 3], keepdims=1)  # [1,1,1,1] f16

    # ===== on_diag via ONE X-shaped diagonal Conv =========================
    # Single 39x39 kernel = both diagonals, SAME pad 19 -> +-19 reach covers the
    # whole 20x20 canvas in ONE conv (one fp16 plane), thresholded >0.
    n("Cast", ["seed_b"], "seed16", to=F16)                     # [1,1,20,20] f16
    KS = 2 * N - 1                                              # 39
    Kx = np.zeros((1, 1, KS, KS), np.float16)
    for i in range(KS):
        Kx[0, 0, i, i] = 1.0
        Kx[0, 0, i, KS - 1 - i] = 1.0
    init("Kx", Kx, np.float16)
    p = N - 1                                                   # 19
    n("Conv", ["seed16", "Kx"], "cvx", pads=[p, p, p, p])       # [1,1,20,20] f16
    n("Greater", ["cvx", "zero16"], "ondiag")                  # bool [1,1,20,20]

    # ===== label plane L (uint8) ==========================================
    # Every on-diagonal cell (base->col0, stripe->col1) is recoloured; seeds are
    # fixed points of this rule so recolouring them too is exact.  Non-diagonal
    # cells keep their input colour.  Off-grid -> sentinel 255 (matches nothing).
    n("Cast", ["g"], "g_u8", to=U8)
    n("Cast", ["col0"], "col0_u8", to=U8)
    n("Cast", ["col1"], "col1_u8", to=U8)
    n("Where", ["onstripe", "col1_u8", "col0_u8"], "inner")    # diag colour
    n("Where", ["ondiag", "inner", "g_u8"], "L2")              # [1,1,20,20] uint8
    init("sent", np.array(255, np.uint8), np.uint8)
    n("Where", ["ingrid", "L2", "sent"], "L")                   # [1,1,20,20] uint8

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
