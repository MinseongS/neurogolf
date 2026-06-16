"""Task 107 (469497ad): zoom-upscale a 5x5 grid by a variable factor and
overlay red (color 2) corner-ray decorations around the 2x2 "box".

Rule (from the ARC-GEN generator): the input is always 5x5; the output is
(5*factor)x(5*factor) where factor = (#distinct colors in the last row) + 1
(factor in 2..6).  The output is the kron upscale of the input by factor*factor,
overlaid with red at four diagonal corner-rays of length `factor` emanating from
the corners of the (upscaled) 2x2 box.  Red is only ever drawn on background
cells (verified over 3000 fresh instances: 0 conflicts).

Lean encoding (no 30x30 fp32/int32 index plane; no red lookup table):
  * colour-label of the 5x5 input  lab5[1,1,5,5]  (Slice input->[1,10,5,5],
    1x1 Conv with kvec=arange(10)); padded to [1,1,6,6] with a sentinel 10 on
    the extra row/col (row/col 5).
  * SENTINEL-VIA-GATHER: upscale index gidx = clip(floor(arange30/f), 0, 5).
    In-grid rows R<5f give floor in 0..4; out-of-grid R>=5f give floor>=5 ->
    clipped to 5 -> gathers the sentinel-10 row/col.  So the double Gather
    (axis=2 then axis=3) of lab6 yields a [1,1,30,30] label that is ALREADY 10
    (= no-channel) outside the 5*factor region -- no separate in-grid Where.
  * red mask ARITHMETIC (no table):  red = (RmC==(row-col)*f OR
    RpC==(row+col+2)*f-1) AND R in [trow-f+1,trow] U [brow,brow+f-1], where
    trow=row*f-1, brow=(row+2)*f.  TL&BR rays share one main diagonal, TR&BL
    share one anti-diagonal.  Computed on a 24x24 fp16 canvas (red never
    exceeds coord 23) then padded to 30x30.  Verified exact vs the generator.
  * L = Where(red, 2, label) ; final Equal(L, arange[0..9]) -> free BOOL output.

Scalars:
  * factor-2 = 4 - sum_i dot(onehot[i], onehot[i+1]) over the last-row cells.
  * row = 1 - occ(0,1), col = 1 - occ(2,0)   (box at (0,1)/(1,0)/(1,1)).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    I32 = TensorProto.INT32
    F32 = TensorProto.FLOAT
    init("c4", np.array(4, np.int32), np.int32)

    # Slice operand initializers (opset 11: starts/ends/axes are inputs)
    init("lr_s", np.array([4, 0], np.int64), np.int64)
    init("lr_e", np.array([5, 5], np.int64), np.int64)
    init("lr_ax", np.array([2, 3], np.int64), np.int64)
    init("a_s", np.array([0], np.int64), np.int64)
    init("a_e", np.array([4], np.int64), np.int64)
    init("b_s", np.array([1], np.int64), np.int64)
    init("b_e", np.array([5], np.int64), np.int64)
    init("col_ax", np.array([3], np.int64), np.int64)
    init("p01_s", np.array([0, 1], np.int64), np.int64)
    init("p01_e", np.array([1, 2], np.int64), np.int64)
    init("p_ax", np.array([2, 3], np.int64), np.int64)
    init("p20_s", np.array([2, 0], np.int64), np.int64)
    init("p20_e", np.array([3, 1], np.int64), np.int64)
    init("ch_s", np.array([1], np.int64), np.int64)
    init("ch_e", np.array([10], np.int64), np.int64)
    init("ch_ax", np.array([1], np.int64), np.int64)
    init("g5_s", np.array([0, 0], np.int64), np.int64)
    init("g5_e", np.array([5, 5], np.int64), np.int64)
    init("g5_ax", np.array([2, 3], np.int64), np.int64)

    # ----- factor-2 = 4 - sum_i dot(lr[i], lr[i+1]) -----
    n("Slice", ["input", "lr_s", "lr_e", "lr_ax"], "lr")      # [1,10,1,5]
    n("Slice", ["lr", "a_s", "a_e", "col_ax"], "lra")         # cols 0..3
    n("Slice", ["lr", "b_s", "b_e", "col_ax"], "lrb")         # cols 1..4
    n("Mul", ["lra", "lrb"], "lrm")                     # [1,10,1,4]
    n("ReduceSum", ["lrm"], "dotsum_f", keepdims=0)     # scalar f32
    n("Cast", ["dotsum_f"], "dotsum", to=I32)
    n("Sub", ["c4", "dotsum"], "fidx")                  # int32 scalar in 0..4

    # ----- box position scalars: row = 1 - occ(0,1), col = 1 - occ(2,0) -----
    # box at (0,1)->occ01=1; (1,0)->occ20=1; (1,1)->both 0.
    n("Slice", ["input", "p01_s", "p01_e", "p_ax"], "p01")    # [1,10,1,1]
    n("Slice", ["p01", "ch_s", "ch_e", "ch_ax"], "p01c")      # channels 1..9
    n("ReduceSum", ["p01c"], "occ01_f", keepdims=0)           # scalar f32 0/1
    n("Slice", ["input", "p20_s", "p20_e", "p_ax"], "p20")
    n("Slice", ["p20", "ch_s", "ch_e", "ch_ax"], "p20c")
    n("ReduceSum", ["p20c"], "occ20_f", keepdims=0)           # scalar f32 0/1
    init("c1f", np.array(1.0, np.float32), np.float32)
    n("Sub", ["c1f", "occ01_f"], "rowf")               # scalar float row in {0,1}
    n("Sub", ["c1f", "occ20_f"], "colf")               # scalar float col in {0,1}

    # ----- factor f (float scalar) -----
    init("c2f", np.array(2.0, np.float32), np.float32)
    n("Cast", ["fidx"], "fidx_f", to=TensorProto.FLOAT)
    n("Add", ["fidx_f", "c2f"], "fscal")                # scalar float = factor

    # ----- arithmetic red mask (no table). Red = on main-diag RmC==(row-col)*f
    #       OR anti-diag RpC==(row+col+2)*f-1, AND R in the two ray ranges
    #       [trow-f+1, trow] (above box) or [brow, brow+f-1] (below box), where
    #       trow=row*f-1, brow=(row+2)*f. Verified exact vs the generator. -----
    # Red lives only in rows/cols < 24 (max coord (row+2)*f+(f-1) <= 23), so the
    # diagonal/range planes use a 24x24 fp16 canvas (half the bytes of 30x30,
    # and only 24 wide), then red_b is padded back to 30x30.
    RS = 24
    init("R24", np.arange(RS, dtype=np.float32).reshape(1, 1, RS, 1), np.float32)
    init("C24", np.arange(RS, dtype=np.float32).reshape(1, 1, 1, RS), np.float32)
    init("c1fs", np.array(1.0, np.float32), np.float32)
    n("Sub", ["rowf", "colf"], "rmc")                  # row - col
    n("Add", ["rowf", "colf"], "rpc")                  # row + col
    n("Mul", ["rmc", "fscal"], "md")                   # main-diag const = (row-col)*f
    init("c2fs", np.array(2.0, np.float32), np.float32)
    n("Add", ["rpc", "c2fs"], "rpc2")                  # row+col+2
    n("Mul", ["rpc2", "fscal"], "ad0")                 # (row+col+2)*f
    n("Sub", ["ad0", "c1fs"], "ad")                    # anti-diag const = (row+col+2)*f-1
    # diagonal planes in fp16 (R-C in [-23,23], R+C in [0,46], integer-exact).
    F16 = TensorProto.FLOAT16
    n("Cast", ["R24"], "R16", to=F16)                  # [1,1,24,1] fp16
    n("Cast", ["C24"], "C16", to=F16)                  # [1,1,1,24] fp16
    n("Cast", ["md"], "md16", to=F16)
    n("Cast", ["ad"], "ad16", to=F16)
    n("Sub", ["R16", "C16"], "RmC")                    # [1,1,24,24] fp16 = R-C
    n("Add", ["R16", "C16"], "RpC")                    # [1,1,24,24] fp16 = R+C
    n("Equal", ["RmC", "md16"], "ondm")                # bool main diag
    n("Equal", ["RpC", "ad16"], "onda")                # bool anti diag
    n("Or", ["ondm", "onda"], "ondiag")                # [1,1,24,24] bool
    # row ranges: trow=row*f-1, brow=(row+2)*f  (use R24 ramp)
    n("Mul", ["rowf", "fscal"], "rowf_f")
    n("Sub", ["rowf_f", "c1fs"], "trow")               # trow scalar
    n("Add", ["rowf", "c2fs"], "rowp2")
    n("Mul", ["rowp2", "fscal"], "brow")               # brow scalar
    n("Sub", ["trow", "fscal"], "trow_lo0")
    n("Add", ["trow_lo0", "c1fs"], "trow_lo")          # trow-f+1
    n("Add", ["brow", "fscal"], "brow_hi0")
    n("Sub", ["brow_hi0", "c1fs"], "brow_hi")          # brow+f-1
    # above-box range: trow_lo <= R <= trow ; Not(Less)==>= (no GreaterOrEqual)
    n("Less", ["R24", "trow_lo"], "lt_tlo")
    n("Not", ["lt_tlo"], "ge_tlo")                     # R >= trow_lo
    n("Add", ["trow", "c1fs"], "trow_p1")
    n("Less", ["R24", "trow_p1"], "le_trow")           # R < trow+1  == R <= trow
    n("And", ["ge_tlo", "le_trow"], "rng_top")         # [1,1,24,1] bool
    # below-box range: brow <= R <= brow_hi
    n("Less", ["R24", "brow"], "lt_brow")
    n("Not", ["lt_brow"], "ge_brow")                   # R >= brow
    n("Add", ["brow_hi", "c1fs"], "brow_hi_p1")
    n("Less", ["R24", "brow_hi_p1"], "le_bhi")         # R <= brow_hi
    n("And", ["ge_brow", "le_bhi"], "rng_bot")         # [1,1,24,1] bool
    n("Or", ["rng_top", "rng_bot"], "rrange")          # [1,1,24,1] bool
    n("And", ["ondiag", "rrange"], "red24")            # [1,1,24,24] bool
    # pad red mask back to [1,1,30,30] (ORT Pad rejects bool -> via uint8)
    n("Cast", ["red24"], "red24u", to=TensorProto.UINT8)
    init("rpad", np.array([0, 0, 0, 0, 0, 0, 30 - RS, 30 - RS], np.int64),
         np.int64)
    init("rpv", np.array(0, np.uint8), np.uint8)
    n("Pad", ["red24u", "rpad", "rpv"], "red30u", mode="constant")  # [1,1,30,30]
    n("Cast", ["red30u"], "red_b", to=TensorProto.BOOL)             # [1,1,30,30]

    # ----- separable upscale indices gidx = clip(floor(arange30 / f), 0, 5).
    #       Out-of-grid R>=5f gives floor>=5 -> clip 5 -> gathers the padded
    #       sentinel row/col (value 10), so NO separate in-grid Where is needed.
    Ar = np.arange(30, dtype=np.float32)
    init("Ar", Ar, np.float32)                          # [30]
    n("Div", ["Ar", "fscal"], "Adiv")                   # [30]
    n("Floor", ["Adiv"], "Afl")                         # [30] float floor(R/f)
    init("c0f", np.array(0.0, np.float32), np.float32)
    init("c5fc", np.array(5.0, np.float32), np.float32)
    n("Clip", ["Afl", "c0f", "c5fc"], "Acl")            # [30] float clipped 0..5
    n("Cast", ["Acl"], "gidx", to=I32)                  # [30] int32 0..5

    # ----- colour-label of the 5x5 input  lab5[1,1,5,5] u8, padded to 6x6
    #       with sentinel 10 (the out-of-grid value, matches no channel) -----
    n("Slice", ["input", "g5_s", "g5_e", "g5_ax"], "in5")     # [1,10,5,5]
    init("kvec", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1),
         np.float32)
    n("Conv", ["in5", "kvec"], "lab5_f", kernel_shape=[1, 1])  # [1,1,5,5] f32
    n("Cast", ["lab5_f"], "lab5", to=TensorProto.UINT8)        # [1,1,5,5] u8
    init("lpad", np.array([0, 0, 0, 0, 0, 0, 1, 1], np.int64), np.int64)
    init("v10", np.array(10, np.uint8), np.uint8)
    n("Pad", ["lab5", "lpad", "v10"], "lab6", mode="constant")  # [1,1,6,6] u8

    # ----- upscale: gather rows then cols (indices in 0..5) -----
    n("Gather", ["lab6", "gidx"], "up_r", axis=2)       # [1,1,30,6] u8
    n("Gather", ["up_r", "gidx"], "L0", axis=3)         # [1,1,30,30] u8 (w/ sentinel)

    # ----- overlay red (color 2); red_b is false outside grid, so safe -----
    init("c2u", np.array(2, np.uint8), np.uint8)
    n("Where", ["red_b", "c2u", "L0"], "L")             # [1,1,30,30] u8
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                 # free BOOL

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task107", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
