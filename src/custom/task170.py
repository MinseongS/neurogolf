"""task170 (ARC-AGI 6ecd11f4) — mask the colour-box by the magnified sprite shape.

Rule (from the generator):
  Input (H,W in 21..28) holds two objects:
    * a "megasprite": a size x size conway shape (size in {3,4}) drawn in a single
      colour `scolor`, magnified by `smag` (3..5), placed at (srow=1, scol).  Every
      row and every column of the size x size shape is occupied.
    * a "colour box": a size x size patch at (brow=H-size-1, bcol) where each cell
      (r,c) holds a distinct colour `colors[r*size+c]` (colours 1..9).
  Output is the size x size grid (top-left aligned):
      output[r][c] = colors[r*size+c]   if (r,c) is a sprite pixel, else 0.
  i.e. the colour box MASKED by the (downsampled) sprite shape.

Recovery (all closed-form scalars, no flood fill):
  colf = sum_k k*input_k                         single [1,1,30,30] f32 plane
  sc   = argmax over channels 1..9 of pixel count (the magnified mono-colour blob)
  bbot = bottom-most colored row                 (box bottom = H-2)
  bcol = leftmost colored col on row bbot ; size = its run width ; brow = bbot-size+1
  sprite mask = (colf==sc) restricted to rows < brow
    srow,scol = its top-left ; smag = (col-span)/size  (col extent never clipped)
  box colours : Gather colf at rows brow+{0..3}, cols bcol+{0..3}
  sprite test : Gather colf at rows srow+smag*{0..3}, cols scol+smag*{0..3}; ==sc
  output[r,c] = box[r,c] if (sprite present AND r<size AND c<size) else 0
  one-hot the 4x4 label, Pad straight into the FREE 30x30 output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
I64 = TensorProto.INT64
B = TensorProto.BOOL


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- 1. colour-index plane colf = sum_k k * input_k ---------------------
    init("cw", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    n("Conv", ["input", "cw"], "colf")              # [1,1,30,30] f32 (only big plane)

    init("zero", np.array(0.0, np.float32), np.float32)

    # ---- 2. sprite colour sc = argmax channel count (ch0 excluded) ----------
    n("ReduceSum", ["input"], "ccnt", axes=[2, 3], keepdims=1)   # [1,10,1,1] f32
    init("ch0mask", np.array([0] + [1] * 9, np.float32).reshape(1, 10, 1, 1),
         np.float32)
    n("Mul", ["ccnt", "ch0mask"], "ccnt0")          # drop background channel
    n("ArgMax", ["ccnt0"], "sc_i", axis=1, keepdims=1)           # [1,1,1,1] int64
    init("shp1", np.array([1], np.int64), np.int64)
    n("Reshape", ["sc_i", "shp1"], "sc1")           # [1] int64
    n("Cast", ["sc1"], "scf", to=F32)               # [1] f32 = sc

    # ramps -------------------------------------------------------------------
    init("rowramp", np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1),
         np.float32)                                  # [1,1,30,1]
    init("colramp", np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30),
         np.float32)                                  # [1,1,1,30]
    init("BIG", np.array(99.0, np.float32), np.float32)

    # ---- 3. bbot = bottom-most colored row ----------------------------------
    # rowmax = per-row max colour (>0 iff the row has any coloured cell); no extra
    # full bool plane (reduce directly off colf).
    n("ReduceMax", ["colf"], "rowmax", axes=[3], keepdims=1)       # [1,1,30,1] f32
    n("Greater", ["rowmax", "zero"], "rowhasb")                    # [1,1,30,1] bool
    n("Where", ["rowhasb", "rowramp", "zero"], "rowhasr")         # [1,1,30,1]
    n("ReduceMax", ["rowhasr"], "bbot", axes=[2], keepdims=1)      # [1,1,1,1] f32
    n("Cast", ["bbot"], "bbot_i", to=I64)
    n("Reshape", ["bbot_i", "shp1"], "bbot1")                     # [1] int64

    # ---- 4. box row at bbot -> bcol, size, brow -----------------------------
    n("Gather", ["colf", "bbot1"], "boxrow_g", axis=2)            # [1,1,1,30] f32
    n("Greater", ["boxrow_g", "zero"], "boxrow")                  # [1,1,1,30] bool
    # bcol = min col where boxrow ; bcolmax = max col where boxrow
    n("Where", ["boxrow", "colramp", "BIG"], "bcol_cand")         # [1,1,1,30]
    n("ReduceMin", ["bcol_cand"], "bcolf", axes=[3], keepdims=1)  # [1,1,1,1]
    n("Cast", ["boxrow"], "boxrowf", to=F32)
    n("Mul", ["boxrowf", "colramp"], "bcolmax_cand")
    n("ReduceMax", ["bcolmax_cand"], "bcolmaxf", axes=[3], keepdims=1)
    # size = bcolmax - bcol + 1
    n("Sub", ["bcolmaxf", "bcolf"], "size_m1")
    init("onef", np.array(1.0, np.float32), np.float32)
    n("Add", ["size_m1", "onef"], "sizef")                       # [1,1,1,1] f32
    # brow = bbot - size + 1 = bbot - size_m1
    n("Sub", ["bbot", "size_m1"], "browf")                       # [1,1,1,1] f32

    # ---- 5. sprite mask restricted to rows < brow ---------------------------
    # spv = 1.0 where (colf==sc AND row<brow) else 0.0  -> ONE extra full plane.
    n("Equal", ["colf", "scf"], "eqsc")             # [1,1,30,30] bool (full plane A)
    n("Less", ["rowramp", "browf"], "toprows")      # [1,1,30,1] bool
    n("Cast", ["toprows"], "toprowsf", to=F32)      # [1,1,30,1] f32 {0,1}
    n("Where", ["eqsc", "toprowsf", "zero"], "spv")  # [1,1,30,30] f32 (full plane B)
    # tiny row / col presence profiles
    n("ReduceMax", ["spv"], "spc", axes=[2], keepdims=1)         # [1,1,1,30] f32
    n("ReduceMax", ["spv"], "spr", axes=[3], keepdims=1)         # [1,1,30,1] f32
    # srow = min row where spr>0
    n("Greater", ["spr", "zero"], "sprb")
    n("Where", ["sprb", "rowramp", "BIG"], "srow_cand")
    n("ReduceMin", ["srow_cand"], "srowf", axes=[2], keepdims=1)  # [1,1,1,1]
    # scol = min col where spc>0 ; spcmax = max col where spc>0
    n("Greater", ["spc", "zero"], "spcb")
    n("Where", ["spcb", "colramp", "BIG"], "scol_cand")
    n("ReduceMin", ["scol_cand"], "scolf", axes=[3], keepdims=1)
    n("Where", ["spcb", "colramp", "zero"], "spcmax_cand")
    n("ReduceMax", ["spcmax_cand"], "spcmaxf", axes=[3], keepdims=1)
    # smag = (spcmax - scol + 1) / size
    n("Sub", ["spcmaxf", "scolf"], "spcspan_m1")
    n("Add", ["spcspan_m1", "onef"], "spcspan")
    n("Div", ["spcspan", "sizef"], "smagf")                      # exact integer ratio
    # truncate to integer via cast round-trip (value already integral)
    n("Cast", ["smagf"], "smag_i", to=I64)
    n("Cast", ["smag_i", ], "smag_int_f", to=F32)               # clean integer f32

    # ---- 6. build the four gather index vectors (length 4) -------------------
    init("ramp4", np.arange(4, dtype=np.float32).reshape([4]), np.float32)  # [4]
    # box rows = brow + ramp4 ; box cols = bcol + ramp4
    init("shpsc", np.array([1], np.int64), np.int64)
    n("Reshape", ["browf", "shpsc"], "brow_s")      # [1]
    n("Reshape", ["bcolf", "shpsc"], "bcol_s")      # [1]
    n("Reshape", ["srowf", "shpsc"], "srow_s")      # [1]
    n("Reshape", ["scolf", "shpsc"], "scol_s")      # [1]
    n("Reshape", ["smag_int_f", "shpsc"], "smag_s")  # [1]

    n("Add", ["ramp4", "brow_s"], "boxr_f")         # [4] f32
    n("Add", ["ramp4", "bcol_s"], "boxc_f")         # [4]
    # sprite rows = srow + smag*ramp4 ; cols = scol + smag*ramp4
    n("Mul", ["ramp4", "smag_s"], "step_f")         # [4]
    n("Add", ["step_f", "srow_s"], "spr_f")         # [4]
    n("Add", ["step_f", "scol_s"], "spc_f")         # [4]
    # clamp sprite indices to <=29 (clipped bottom blocks read off-grid bg=0!=sc)
    init("c29", np.array(29.0, np.float32), np.float32)
    n("Min", ["spr_f", "c29"], "spr_fc")
    n("Min", ["spc_f", "c29"], "spc_fc")

    for nm in ("boxr_f", "boxc_f", "spr_fc", "spc_fc"):
        n("Cast", [nm], nm + "_i", to=I64)

    # ---- 7. gather the 4x4 colour-box block ---------------------------------
    n("Gather", ["colf", "boxr_f_i"], "box_rows", axis=2)   # [1,1,4,30]
    n("Gather", ["box_rows", "boxc_f_i"], "box4", axis=3)   # [1,1,4,4] f32 colours

    # ---- 8. gather the 4x4 sprite-presence block ----------------------------
    n("Gather", ["colf", "spr_fc_i"], "sp_rows", axis=2)    # [1,1,4,30]
    n("Gather", ["sp_rows", "spc_fc_i"], "sp4", axis=3)     # [1,1,4,4] f32
    n("Equal", ["sp4", "scf"], "present")                   # [1,1,4,4] bool

    # ---- 9. in-bounds mask r<size AND c<size --------------------------------
    init("ramp4r", np.arange(4, dtype=np.float32).reshape(1, 1, 4, 1), np.float32)
    init("ramp4c", np.arange(4, dtype=np.float32).reshape(1, 1, 1, 4), np.float32)
    n("Less", ["ramp4r", "sizef"], "rin")      # [1,1,4,1] bool
    n("Less", ["ramp4c", "sizef"], "cin")      # [1,1,1,4] bool
    n("And", ["rin", "cin"], "inbox")          # [1,1,4,4] bool
    n("And", ["present", "inbox"], "keep")      # [1,1,4,4] bool

    # ---- 10. label = box4 where keep else 0, one-hot, pad into free output ---
    n("Cast", ["box4"], "box4u", to=U8)        # colours 0..9
    init("u0", np.array(0, np.uint8), np.uint8)
    n("Where", ["keep", "box4u", "u0"], "Lout")             # [1,1,4,4] uint8
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["Lout", "chan"], "ohb")        # [1,10,4,4] bool
    # zero out cells outside the size x size output grid (off-box -> all channels 0)
    n("And", ["ohb", "inbox"], "ohb2")         # [1,10,4,4] bool (broadcast inbox)
    n("Cast", ["ohb2"], "oh", to=U8)           # Pad needs non-bool
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 26, 26], np.int64), np.int64)
    n("Pad", ["oh", "padpads", "u0"], "output", mode="constant")  # [1,10,30,30] u8

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", U8, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task170", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
