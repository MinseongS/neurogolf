"""task396 (ARC-AGI fcb5c309) — crop the LARGEST hollow box & recolour it to the static colour.

Rule (from the generator task_fcb5c309.py, verified 0/20000 fresh):
  A WxH grid (12..18 each) holds 2-3 hollow rectangular boxes drawn in colour `colors[0]`
  (a 1px outline, black interior) plus scattered single-pixel "static" in colour `colors[1]`
  (some static lands INSIDE the boxes).  `wides` and `talls` are each sorted DESCENDING, so
  box 0 is the LARGEST box (max width AND max height).  The OUTPUT is a tall0 x wide0 grid
  showing box 0, but every NON-BLACK cell of box 0's region (its outline AND the static
  pixels inside it) is painted with the STATIC colour `colors[1]`; the black interior stays
  black.  Verified exactly:
      output[r][c] = colors[1]  if  input[brow0+r][bcol0+c] != 0  else  0,
                     for  0<=r<tall0, 0<=c<wide0.

  Closed-form recovery (NO flood-fill / connectivity / per-channel planes):
    * colf = sum_k k*input_k (1x1 conv) is the per-cell colour index.
    * "same-colour horizontal pair"  eqh[r,c] = (colf[r,c]==colf[r,c+1] and colf[r,c]>0).
      A horizontal RUN of length L has L-1 consecutive eqh pairs.  Box edges are full solid
      runs; random static almost never forms long equal runs, so the GLOBAL max run is box
      0's edge (it is the largest box).  runH[r,c] = #{k in 2..7 : a run of >=k pairs starts
      here} = max(pairs_run-1, 0); wide0 = max(runH)+2.  Vertical analogously -> tall0.
    * box-0 top-left = topmost-then-leftmost start cell of a maximal horizontal pair-run.
    * box colour c0 = colf at (brow0,bcol0); static colour c1 = the OTHER present non-bg
      colour (c1 is the only colour we actually emit).

Encoding (route the 10-ch expansion into the FREE BOOL output; tiny working planes):
  Everything runs on the single colf plane.  The run-length convs run on an 18x18 fp16
  slice (grids fit in 18x18 at the top-left).  Gather-shift colf to the box top-left, mask
  to the tall0 x wide0 window (sentinel -1 outside), recolour non-black -> c1, and emit
  output = Equal(L, arange[0..9]) -> free BOOL one-hot.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64

A = 18     # active canvas (grids are <=18x18 at the top-left)
WORK = 8   # box width/height in 3..8 -> output window is at most 8x8


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ======================================================================
    # 1. colour-index plane colf = sum_k k*input_k  (1x1 conv) [1,1,30,30] f32
    # ======================================================================
    init("wsel", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    n("Conv", ["input", "wsel"], "colf30")                # [1,1,30,30] f32
    # 18x18 fp16 working slice
    init("a_s", np.array([0, 0], np.int64), np.int64)
    init("a_e", np.array([A, A], np.int64), np.int64)
    init("a_ax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["colf30", "a_s", "a_e", "a_ax"], "colf18f")  # [1,1,18,18] f32
    n("Cast", ["colf18f"], "colf", to=F16)                # [1,1,18,18] f16

    # ======================================================================
    # 2. same-colour adjacency pairs on colf (single plane).
    #    eqh[r,c] = colf[r,c]==colf[r,c+1] AND colf[r,c]>0   [1,1,18,17] bool
    # ======================================================================
    init("s0", np.array([0], np.int64), np.int64)
    init("sA1", np.array([A - 1], np.int64), np.int64)
    init("s1", np.array([1], np.int64), np.int64)
    init("sA", np.array([A], np.int64), np.int64)
    init("ax3", np.array([3], np.int64), np.int64)
    init("ax2", np.array([2], np.int64), np.int64)
    init("zeroh", np.array(0.0, np.float16), np.float16)
    # horizontal
    n("Slice", ["colf", "s0", "sA1", "ax3"], "hL")        # [1,1,18,17]
    n("Slice", ["colf", "s1", "sA", "ax3"], "hR")
    n("Equal", ["hL", "hR"], "heq")                       # bool
    n("Greater", ["hL", "zeroh"], "hpos")
    n("And", ["heq", "hpos"], "eqhb")                     # bool [1,1,18,17]
    n("Cast", ["eqhb"], "eqh", to=F16)
    # vertical
    n("Slice", ["colf", "s0", "sA1", "ax2"], "vU")        # [1,1,17,18]
    n("Slice", ["colf", "s1", "sA", "ax2"], "vD")
    n("Equal", ["vU", "vD"], "veq")
    n("Greater", ["vU", "zeroh"], "vpos")
    n("And", ["veq", "vpos"], "eqvb")
    n("Cast", ["eqvb"], "eqv", to=F16)

    # ======================================================================
    # 3. run lengths.  For k=2..7: a k-wide valid conv on the pair map ==k means
    #    >=k consecutive pairs start here.  Sum -> runH=max(pairs_run-1,0) per
    #    start cell.  wide0 = max(runH)+2 ; tall0 = max(runV)+2.
    # ======================================================================
    runH_terms, runV_terms = [], []
    for k in range(2, 7 + 1):
        init(f"kh{k}", np.ones((1, 1, 1, k), np.float16), np.float16)
        init(f"kv{k}", np.ones((1, 1, k, 1), np.float16), np.float16)
        init(f"kf{k}", np.array(float(k), np.float16), np.float16)
        n("Conv", ["eqh", f"kh{k}"], f"ch{k}", pads=[0, 0, 0, k - 1])  # [1,1,18,17]
        n("Equal", [f"ch{k}", f"kf{k}"], f"ph{k}")
        n("Cast", [f"ph{k}"], f"phf{k}", to=F16)
        runH_terms.append(f"phf{k}")
        n("Conv", ["eqv", f"kv{k}"], f"cv{k}", pads=[0, 0, k - 1, 0])  # [1,1,17,18]
        n("Equal", [f"cv{k}", f"kf{k}"], f"pv{k}")
        n("Cast", [f"pv{k}"], f"pvf{k}", to=F16)
        runV_terms.append(f"pvf{k}")

    n("Sum", runH_terms, "runH")  # [1,1,18,17] f16
    n("Sum", runV_terms, "runV")  # [1,1,17,18] f16
    n("ReduceMax", ["runH"], "rHmax", axes=[2, 3], keepdims=1)  # [1,1,1,1] f16
    n("ReduceMax", ["runV"], "rVmax", axes=[2, 3], keepdims=1)
    init("two", np.array(2.0, np.float16), np.float16)
    n("Add", ["rHmax", "two"], "wide0h")
    n("Add", ["rVmax", "two"], "tall0h")
    n("Cast", ["wide0h"], "wide0", to=F32)                # [1,1,1,1] f32
    n("Cast", ["tall0h"], "tall0", to=F32)

    # ======================================================================
    # 4. box-0 top-left = topmost-then-leftmost start of a maximal h pair-run.
    # ======================================================================
    n("Equal", ["runH", "rHmax"], "HRwin0")               # bool
    n("Greater", ["runH", "zeroh"], "runHpos")
    n("And", ["HRwin0", "runHpos"], "HRwinb")             # [1,1,18,17] bool
    n("Cast", ["HRwinb"], "HRwin", to=F16)
    n("ReduceMax", ["HRwin"], "rowhas", axes=[3], keepdims=1)  # [1,1,18,1] f16
    init("rampr", np.arange(A, dtype=np.float16).reshape(1, 1, A, 1), np.float16)
    init("BIG", np.array(99.0, np.float16), np.float16)
    init("halfh", np.array(0.5, np.float16), np.float16)
    n("Greater", ["rowhas", "halfh"], "rowhasb")
    n("Where", ["rowhasb", "rampr", "BIG"], "rowidx")     # [1,1,18,1] f16
    n("ReduceMin", ["rowidx"], "brow0h", axes=[2], keepdims=1)  # [1,1,1,1] f16
    n("Cast", ["brow0h"], "brow0", to=F32)
    # leftmost col in that row
    init("shp1", np.array([1], np.int64), np.int64)
    n("Reshape", ["brow0h", "shp1"], "brow0_1")
    n("Cast", ["brow0_1"], "brow0_i", to=I64)
    n("Gather", ["HRwin", "brow0_i"], "row_sel", axis=2)  # [1,1,1,17] f16
    init("rampc", np.arange(A - 1, dtype=np.float16).reshape(1, 1, 1, A - 1),
         np.float16)
    n("Greater", ["row_sel", "halfh"], "row_selb")
    n("Where", ["row_selb", "rampc", "BIG"], "colidx")    # [1,1,1,17] f16
    n("ReduceMin", ["colidx"], "bcol0h", axes=[3], keepdims=1)
    n("Cast", ["bcol0h"], "bcol0", to=F32)

    # box colour c0 = colf at (brow0,bcol0)  (gather the row, then the col)
    n("Reshape", ["bcol0h", "shp1"], "bcol0_1")
    n("Cast", ["bcol0_1"], "bcol0_i", to=I64)
    n("Gather", ["colf30", "brow0_i"], "c0row", axis=2)   # [1,1,1,30] f32
    n("Gather", ["c0row", "bcol0_i"], "c0cell", axis=3)   # [1,1,1,1] f32

    # ======================================================================
    # 5. static colour c1 = the present non-bg colour != c0.
    # ======================================================================
    n("ReduceMax", ["input"], "present", axes=[2, 3], keepdims=1)  # [1,10,1,1] f32
    chramp = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("chramp", chramp, np.float32)
    init("zerof", np.array(0.0, np.float32), np.float32)
    n("Equal", ["chramp", "c0cell"], "is_c0")             # [1,10,1,1] bool
    n("Greater", ["present", "zerof"], "presb")
    n("Not", ["is_c0"], "not_c0")
    ch0b = np.zeros((1, 10, 1, 1), np.bool_)
    ch0b[0, 0, 0, 0] = True
    init("ch0b", ch0b, np.bool_)
    n("Not", ["ch0b"], "not_ch0")
    n("And", ["presb", "not_c0"], "tmp_a")
    n("And", ["tmp_a", "not_ch0"], "c1mask")              # bool [1,10,1,1]
    n("Where", ["c1mask", "chramp", "zerof"], "c1src")
    n("ArgMax", ["c1src"], "c1_i", axis=1, keepdims=1)
    n("Cast", ["c1_i"], "c1f", to=F16)                    # [1,1,1,1] f16

    # ======================================================================
    # 6. shift colf30 to the box top-left & crop a WORK x WORK window.
    # ======================================================================
    init("baseW", np.arange(WORK, dtype=np.float32), np.float32)
    n("Reshape", ["brow0", "shp1"], "brow0_s")
    n("Add", ["baseW", "brow0_s"], "ridx_f")
    init("c0clip", np.array(0.0, np.float32), np.float32)
    init("c29clip", np.array(29.0, np.float32), np.float32)
    n("Clip", ["ridx_f", "c0clip", "c29clip"], "ridx_cl")
    n("Cast", ["ridx_cl"], "ridx", to=I64)
    n("Reshape", ["bcol0", "shp1"], "bcol0_s")
    n("Add", ["baseW", "bcol0_s"], "cidx_f")
    n("Clip", ["cidx_f", "c0clip", "c29clip"], "cidx_cl")
    n("Cast", ["cidx_cl"], "cidx", to=I64)
    n("Gather", ["colf30", "ridx"], "Vr", axis=2)         # [1,1,WORK,30] f32
    n("Gather", ["Vr", "cidx"], "Vs", axis=3)             # [1,1,WORK,WORK] f32

    # ======================================================================
    # 7. in-grid mask (r<tall0, c<wide0) on the WORK x WORK canvas.
    # ======================================================================
    init("wr", np.arange(WORK, dtype=np.float32).reshape(1, 1, WORK, 1), np.float32)
    init("wc", np.arange(WORK, dtype=np.float32).reshape(1, 1, 1, WORK), np.float32)
    n("Less", ["wr", "tall0"], "rmask")                   # [1,1,WORK,1] bool
    n("Less", ["wc", "wide0"], "cmask")                   # [1,1,1,WORK] bool
    n("And", ["rmask", "cmask"], "boxmask")               # [1,1,WORK,WORK] bool

    # ======================================================================
    # 8. label L: c1 where (in-grid AND Vs!=0), 0 elsewhere in-grid, -1 outside.
    # ======================================================================
    init("halff", np.array(0.5, np.float32), np.float32)
    n("Greater", ["Vs", "halff"], "nz")                   # non-black cell
    n("And", ["nz", "boxmask"], "paint")
    init("zerolab", np.array(0.0, np.float16), np.float16)
    n("Where", ["paint", "c1f", "zerolab"], "Lin")        # [1,1,WORK,WORK] f16
    init("neg1", np.array(-1.0, np.float16), np.float16)
    n("Where", ["boxmask", "Lin", "neg1"], "Lw")
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    n("Pad", ["Lw", "padpads", "neg1"], "L30", mode="constant")  # [1,1,30,30] f16
    init("chan", np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1), np.float16)
    n("Equal", ["L30", "chan"], "output")                 # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task396", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
