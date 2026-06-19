"""task396 (ARC-AGI fcb5c309) — crop the LARGEST hollow box & recolour it to the static colour.

Rule (from the generator, verified 0/400 fresh in numpy + isolated ONNX fresh):
  A WxH grid (12..18 each) holds 2-3 hollow rectangular boxes drawn in colour `colors[0]`
  (1px outline, black interior) plus scattered single-pixel "static" in colour `colors[1]`
  (some static lands INSIDE the boxes). `wides`/`talls` are sorted DESCENDING so box 0 is the
  LARGEST box (max width AND max height). OUTPUT = a tall0 x wide0 grid showing box 0 with every
  NON-BLACK cell of its region (outline + interior static) painted the STATIC colour c1; the
  black interior stays black:
      output[r][c] = c1  if  input[brow0+r][bcol0+c] != 0  else  0.

Closed-form recovery (no flood-fill / per-channel planes / run-conv army):
  * colf = sum_k k*input_k  (1x1 conv) = per-cell colour index.
  * Horizontal same-colour adjacency pairs eqh[r,c] = (colf[r,c]==colf[r,c+1]) & (colf>0).
    A solid run of W cells has W-1 consecutive eqh pairs.  Box edges are the only long solid
    runs (largest box has the longest), so the GLOBAL max horizontal pair-run = wide0-1.
    RUN-LENGTH-ENDING-HERE per row via CUMSUM-RESET (replaces the k=2..7 conv army):
        cs = CumSum(eqh, axis); reset_val = where(eq, -BIG, cs); rl = cs - prefixmax(reset_val)
    (one-sided full-length MaxPool = the prefix-max, ZERO params).  Pad a leading zero so the
    pre-first-reset region is well-defined.  maxH = max(rl); wide0 = maxH+1.
  * The cells with rl==maxH are the RIGHT-ENDS of box-0 horizontal edges -> end_col;
    bcol0 = min(end_col) - maxH.   Vertical analogously -> tall0, brow0.
  * box colour c0 = colf at (brow0,bcol0); static colour c1 = the other present non-bg colour.
  * Gather-shift colf30 to (brow0,bcol0), crop WORK x WORK, in-grid mask r<tall0 & c<wide0,
    paint non-black -> c1, emit Equal(L, arange[0..9]) -> FREE BOOL one-hot output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64

A = 18      # active canvas (grids fit in 18x18 at the top-left)
WORK = 8    # box side in 3..8 -> output window at most 8x8


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ------------------------------------------------------------------ #
    # 1. colour-index plane colf30 = sum_k k*input_k  (1x1 conv) f32
    #    + an 18x18 fp32 working slice colf
    # ------------------------------------------------------------------ #
    init("wsel", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    n("Conv", ["input", "wsel"], "colf30")                # [1,1,30,30] f32
    init("a_s", np.array([0, 0], np.int64), np.int64)
    init("a_e", np.array([A, A], np.int64), np.int64)
    init("a_ax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["colf30", "a_s", "a_e", "a_ax"], "colff")  # [1,1,18,18] f32
    n("Cast", ["colff"], "colf", to=F16)                   # [1,1,18,18] f16

    # ------------------------------------------------------------------ #
    # 2. adjacency-pair maps (fp16 working planes)
    # ------------------------------------------------------------------ #
    init("s0", np.array([0], np.int64), np.int64)
    init("sA1", np.array([A - 1], np.int64), np.int64)
    init("s1", np.array([1], np.int64), np.int64)
    init("sA", np.array([A], np.int64), np.int64)
    init("ax3", np.array([3], np.int64), np.int64)
    init("zerof", np.array(0.0, np.float32), np.float32)
    init("zeroh", np.array(0.0, np.float16), np.float16)
    init("BIGh", np.array(99.0, np.float16), np.float16)
    init("negBIGh", np.array(-99.0, np.float16), np.float16)
    init("oneh", np.array(1.0, np.float16), np.float16)
    init("zeru8", np.array(0, np.uint8), np.uint8)

    # horizontal: eqh = (colf[:, :-1]==colf[:, 1:]) & colf[:, :-1]>0
    n("Slice", ["colf", "s0", "sA1", "ax3"], "hL")        # [1,1,18,17] f16
    n("Slice", ["colf", "s1", "sA", "ax3"], "hR")
    n("Equal", ["hL", "hR"], "heq")
    n("Greater", ["hL", "zeroh"], "hpos")
    n("And", ["heq", "hpos"], "eqhb")                     # bool [1,1,18,17]
    n("Cast", ["eqhb"], "eqh", to=TensorProto.UINT8)      # 0/1 uint8

    # ------------------------------------------------------------------ #
    # 3. HORIZONTAL run-length-ending-here via cumsum-reset + one-sided MaxPool.
    #    Pad a leading zero so the pre-first-reset region is sane; after the
    #    leading pad, plane index along the axis == end coordinate.
    #    CumSum needs fp32 -> ONE fp32 cumsum plane, then fp16.
    #    (Only the horizontal direction is needed; tall0 is recovered by a
    #    cheap 1-D probe down box-0's left edge column.)
    # ------------------------------------------------------------------ #
    init("padh", np.array([0, 0, 0, 1, 0, 0, 0, 0], np.int64), np.int64)
    n("Pad", ["eqh", "padh", "zeru8"], "eqhp", mode="constant")    # [1,1,18,18] u8
    n("Cast", ["eqhp"], "eqhpf", to=F32)                          # f32 for cumsum
    n("CumSum", ["eqhpf", "ax3"], "cshf")                         # f32 [1,1,18,18]
    n("Cast", ["cshf"], "csh", to=F16)
    n("Cast", ["eqhp"], "eqhpb", to=BOOL)                        # 0/1 u8 -> bool
    n("Where", ["eqhpb", "negBIGh", "csh"], "rsth")              # f16
    n("MaxPool", ["rsth"], "pmh", kernel_shape=[1, A], pads=[0, A - 1, 0, 0],
      strides=[1, 1])                                            # prefix-max f16
    n("Sub", ["csh", "pmh"], "rlh")                              # run-of-pairs ending here
    n("ReduceMax", ["rlh"], "maxHk", axes=[2, 3], keepdims=1)    # = wide0-1 (f16 scalar)
    n("Add", ["maxHk", "oneh"], "wide0h")                       # [1,1,1,1] f16
    n("Cast", ["wide0h"], "wide0", to=F32)

    # ------------------------------------------------------------------ #
    # 4. box-0 position from the horizontal run map.
    #    bcol0 = (min col with per-col max-run == maxH) - maxH
    #    brow0 =  min row with per-row max-run == maxH      (the TOP edge row)
    # ------------------------------------------------------------------ #
    init("rampc", np.arange(A, dtype=np.float16).reshape(1, 1, 1, A), np.float16)
    init("rampr", np.arange(A, dtype=np.float16).reshape(1, 1, A, 1), np.float16)
    n("ReduceMax", ["rlh"], "colmaxH", axes=[2], keepdims=1)     # [1,1,1,18] f16
    n("Equal", ["colmaxH", "maxHk"], "ishw")                     # [1,1,1,18] bool
    n("Where", ["ishw", "rampc", "BIGh"], "hcol")               # [1,1,1,18] f16
    n("ReduceMin", ["hcol"], "endcolk", axes=[3], keepdims=1)
    n("Sub", ["endcolk", "maxHk"], "bcol0h")
    n("Cast", ["bcol0h"], "bcol0", to=F32)                      # [1,1,1,1] f32
    n("ReduceMax", ["rlh"], "rowmaxH", axes=[3], keepdims=1)     # [1,1,18,1] f16
    n("Equal", ["rowmaxH", "maxHk"], "isrw")                     # [1,1,18,1] bool
    n("Where", ["isrw", "rampr", "BIGh"], "rrow")               # [1,1,18,1] f16
    n("ReduceMin", ["rrow"], "brow0h", axes=[2], keepdims=1)    # [1,1,1,1] f16 (= brow0)
    n("Cast", ["brow0h"], "brow0", to=F32)

    # ------------------------------------------------------------------ #
    # 5. box colour c0 = colf30 at (brow0,bcol0).
    # ------------------------------------------------------------------ #
    init("shp1", np.array([1], np.int64), np.int64)
    n("Reshape", ["brow0", "shp1"], "brow0_1")
    n("Cast", ["brow0_1"], "brow0_i", to=I64)
    n("Reshape", ["bcol0", "shp1"], "bcol0_1")
    n("Cast", ["bcol0_1"], "bcol0_i", to=I64)
    n("Gather", ["colf", "brow0_i"], "c0row", axis=2)             # [1,1,1,18] f16
    n("Gather", ["c0row", "bcol0_i"], "c0cell", axis=3)           # [1,1,1,1] f16

    # ------------------------------------------------------------------ #
    # 6. static colour c1 = present non-bg colour != c0.
    # ------------------------------------------------------------------ #
    n("ReduceMax", ["input"], "present", axes=[2, 3], keepdims=1)  # [1,10,1,1] f32
    chramp = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("chramp", chramp, np.float32)
    chramph = np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1)
    init("chramph", chramph, np.float16)
    n("Equal", ["chramph", "c0cell"], "is_c0")                    # f16==f16
    n("Greater", ["present", "zerof"], "presb")
    n("Not", ["is_c0"], "not_c0")
    ch0b = np.zeros((1, 10, 1, 1), np.bool_); ch0b[0, 0, 0, 0] = True
    init("ch0b", ch0b, np.bool_)
    n("Not", ["ch0b"], "not_ch0")
    n("And", ["presb", "not_c0"], "tmp_a")
    n("And", ["tmp_a", "not_ch0"], "c1mask")
    n("Where", ["c1mask", "chramp", "zerof"], "c1src")
    n("ArgMax", ["c1src"], "c1_i", axis=1, keepdims=1)            # int64 [1,1,1,1]
    n("Cast", ["c1_i"], "c1u", to=TensorProto.UINT8)             # [1,1,1,1] uint8

    # ------------------------------------------------------------------ #
    # 6b. tall0 = run of c0 down box-0's LEFT-edge column (col bcol0) from brow0.
    #     1-D probe on a tiny [1,1,18,1] column -> first non-c0 row >= brow0.
    # ------------------------------------------------------------------ #
    n("Gather", ["colf", "bcol0_i"], "colvec", axis=3)           # [1,1,18,1] f16
    n("Equal", ["colvec", "c0cell"], "isc0col")                  # [1,1,18,1] bool
    n("Not", ["isc0col"], "notc0col")
    n("Less", ["rampr", "brow0h"], "aboverow")                   # row < brow0
    n("Not", ["aboverow"], "atbelow")                            # row >= brow0
    n("And", ["notc0col", "atbelow"], "stoprow")                 # non-c0 at/below top
    init("Aend", np.array(float(A), np.float16), np.float16)      # box may reach bottom
    n("Where", ["stoprow", "rampr", "Aend"], "stopidx")          # [1,1,18,1] f16
    n("ReduceMin", ["stopidx"], "endrowk", axes=[2], keepdims=1)  # first stop row
    n("Sub", ["endrowk", "brow0h"], "tall0h")                    # = tall0 (f16 scalar)
    n("Cast", ["tall0h"], "tall0", to=F32)

    # ------------------------------------------------------------------ #
    # 7. shift colf to box top-left & crop WORK x WORK.
    # ------------------------------------------------------------------ #
    init("baseW", np.arange(WORK, dtype=np.float32), np.float32)
    n("Reshape", ["brow0", "shp1"], "brow0_s")
    n("Add", ["baseW", "brow0_s"], "ridx_f")
    init("c0clip", np.array(0.0, np.float32), np.float32)
    init("c17clip", np.array(float(A - 1), np.float32), np.float32)
    n("Clip", ["ridx_f", "c0clip", "c17clip"], "ridx_cl")
    n("Cast", ["ridx_cl"], "ridx", to=I64)
    n("Reshape", ["bcol0", "shp1"], "bcol0_s")
    n("Add", ["baseW", "bcol0_s"], "cidx_f")
    n("Clip", ["cidx_f", "c0clip", "c17clip"], "cidx_cl")
    n("Cast", ["cidx_cl"], "cidx", to=I64)
    n("Gather", ["colf", "ridx"], "Vr", axis=2)                   # [1,1,WORK,18] f16
    n("Gather", ["Vr", "cidx"], "Vs", axis=3)                     # [1,1,WORK,WORK] f16

    # ------------------------------------------------------------------ #
    # 8. in-grid mask (r<tall0, c<wide0) on the WORK x WORK canvas.
    # ------------------------------------------------------------------ #
    init("wr", np.arange(WORK, dtype=np.float32).reshape(1, 1, WORK, 1), np.float32)
    init("wc", np.arange(WORK, dtype=np.float32).reshape(1, 1, 1, WORK), np.float32)
    n("Less", ["wr", "tall0"], "rmask")                           # [1,1,WORK,1]
    n("Less", ["wc", "wide0"], "cmask")                           # [1,1,1,WORK]
    n("And", ["rmask", "cmask"], "boxmask")                       # [1,1,WORK,WORK]

    # ------------------------------------------------------------------ #
    # 9. label L: c1 where (in-grid & Vs!=0), 0 in-grid black, -1 outside;
    #    expand to one-hot in the FREE BOOL output.
    # ------------------------------------------------------------------ #
    init("halfh2", np.array(0.5, np.float16), np.float16)
    n("Greater", ["Vs", "halfh2"], "nz")                          # non-black cell (f16)
    n("And", ["nz", "boxmask"], "paint")                          # bool [1,1,WORK,WORK]
    n("Where", ["paint", "c1u", "zeru8"], "Lin")                  # uint8: c1 or 0
    init("sent", np.array(99, np.uint8), np.uint8)                # outside sentinel (>9)
    n("Where", ["boxmask", "Lin", "sent"], "Lw")                  # uint8
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    n("Pad", ["Lw", "padpads", "sent"], "L30", mode="constant")   # [1,1,30,30] uint8
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L30", "chan"], "output")                         # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task396", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
