"""task188 (ARC-AGI 7b7f7511) — un-duplicate: output = the single top-left tile.

Rule (from the generator): the input is a `height x width` colour tile DUPLICATED
once along one axis — vertically (grid = 2h x w, output = top half) or
horizontally (grid = h x 2w, output = left half).  `width, height` are each
`randint(2,4)`, so the *duplicated* dimension is always even and in {4,6,8} while
the *non-duplicated* dimension is in {2,3,4}.  The output is exactly the unique
tile = the top-left `height x width` block of the input; everything outside that
block is background (all-channels-off).

Axis decision (R = occupied rows, C = occupied cols, both deducible from the
one-hot extent):
  * R > 4              -> R is the duplicated dim   -> VERTICAL  (keep rows < R/2)
  * C > 4              -> C is the duplicated dim   -> HORIZONTAL (keep cols < C/2)
  * R == 4 and C < 4   -> only R can be 2h          -> VERTICAL
  * C == 4 and R < 4   -> only C can be 2w          -> HORIZONTAL
  * R == 4 and C == 4  -> genuinely ambiguous; decide by which axis actually
                          tiles (top-half == bottom-half => vertical, else
                          horizontal).  ~0.09% of all grids tile BOTH ways at 4x4
                          and are then non-deterministic in the generator (the
                          `vert` flag is a free coin flip) — an irreducible wall
                          shared by the public net.
  (max(R,C) >= 4 always, since the duplicated dim is 2h >= 4.)

So  vert = (R>4) OR (R==4 AND C<4) OR (R==4 AND C==4 AND top==bottom).
    keep_rows = vert ? R/2 : R ;  keep_cols = vert ? C : C/2.
    output = input masked to (row < keep_rows) AND (col < keep_cols).

Memory: the only sizeable intermediate is the bool keep-mask [1,1,30,30] = 900 B;
the final `output = Where(mask, input, 0)` is FREE (named `output`).  Everything
else is fp32 scalars / 1-D profiles.  All values are tiny integers => float32-exact.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
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

    # ---- grid extent R (occupied rows) and C (occupied cols) ----------------
    # rowocc[r] = any channel set in row r (background ch0 counts -> whole grid).
    n("ReduceMax", ["input"], "rowocc", axes=[1, 3], keepdims=1)  # [1,1,30,1] f32
    n("ReduceMax", ["input"], "colocc", axes=[1, 2], keepdims=1)  # [1,1,1,30] f32
    n("ReduceSum", ["rowocc"], "R", axes=[2], keepdims=1)         # [1,1,1,1] f32
    n("ReduceSum", ["colocc"], "C", axes=[3], keepdims=1)         # [1,1,1,1] f32

    init("F35", np.array(3.5, np.float32), np.float32)   # threshold for ">4"/"<4"
    init("F45", np.array(4.5, np.float32), np.float32)

    # bigR = R > 4 ; R4 = R == 4 (i.e. 3.5 < R < 4.5) ; Cl4 = C < 4
    n("Greater", ["R", "F45"], "bigR")                   # R > 4  bool
    n("Greater", ["R", "F35"], "Rge4")
    n("Less", ["R", "F45"], "Rle4")
    n("And", ["Rge4", "Rle4"], "R4")                     # R == 4  bool
    n("Greater", ["C", "F45"], "bigC")                   # C > 4  bool
    n("Greater", ["C", "F35"], "Cge4")
    n("Less", ["C", "F35"], "Cl4")                       # C < 4  bool
    n("Less", ["C", "F45"], "Cle4")
    n("And", ["Cge4", "Cle4"], "C4")                     # C == 4 bool

    # ---- vtile (only meaningful when R==4 & C==4): top 2 rows == bottom 2 rows
    # over cols 0..3.  Slice is valid at any shape; we only USE it under eq4. ----
    init("ts0", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("te", np.array([1, 10, 2, 4], np.int64), np.int64)   # rows 0:2, cols 0:4
    init("ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "ts0", "te", "ax"], "topblk")        # [1,10,2,4] f32
    init("bs0", np.array([0, 0, 2, 0], np.int64), np.int64)
    init("be", np.array([1, 10, 4, 4], np.int64), np.int64)   # rows 2:4, cols 0:4
    n("Slice", ["input", "bs0", "be", "ax"], "botblk")        # [1,10,2,4] f32
    n("Sub", ["topblk", "botblk"], "tdiff")
    n("Abs", ["tdiff"], "tabs")
    n("ReduceSum", ["tabs"], "tsum", keepdims=1)              # [1,1,1,1] f32
    init("HALF", np.array(0.5, np.float32), np.float32)
    n("Less", ["tsum", "HALF"], "vtile")                     # top==bottom bool

    # ---- vert = bigR OR (R4 AND Cl4) OR (R4 AND C4 AND vtile) ----------------
    n("And", ["R4", "Cl4"], "vA")
    n("And", ["R4", "C4"], "eq4")
    n("And", ["eq4", "vtile"], "vB")
    n("Or", ["bigR", "vA"], "vOr1")
    n("Or", ["vOr1", "vB"], "vert")                          # [1,1,1,1] bool
    n("Cast", ["vert"], "vf", to=F32)                        # 1.0 if vertical

    # ---- keep_rows = R*(1 - 0.5*vf) ; keep_cols = C*(0.5 + 0.5*vf) -----------
    init("ONE", np.array(1.0, np.float32), np.float32)
    n("Mul", ["vf", "HALF"], "halfvf")                       # 0.5*vf
    n("Sub", ["ONE", "halfvf"], "rfac")                      # 1 - 0.5*vf
    n("Mul", ["R", "rfac"], "keepR")                         # rows to keep
    n("Add", ["HALF", "halfvf"], "cfac")                     # 0.5 + 0.5*vf
    n("Mul", ["C", "cfac"], "keepC")                         # cols to keep

    # ---- row/col index ramps and masks (separable) --------------------------
    ramp = np.arange(30, dtype=np.float32)
    init("rrow", ramp.reshape(1, 1, 30, 1), np.float32)      # [1,1,30,1]
    init("rcol", ramp.reshape(1, 1, 1, 30), np.float32)      # [1,1,1,30]
    n("Less", ["rrow", "keepR"], "rmask")                    # [1,1,30,1] bool
    n("Less", ["rcol", "keepC"], "cmask")                    # [1,1,1,30] bool
    n("And", ["rmask", "cmask"], "keepmask")                 # [1,1,30,30] bool 900B

    # ---- output = input where in the top-left tile, else 0 ------------------
    init("ZEROF", np.array(0.0, np.float32), np.float32)
    n("Where", ["keepmask", "input", "ZEROF"], "output")     # [1,10,30,30] f32 FREE

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task188", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
