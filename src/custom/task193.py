"""task193 (ARC-AGI 7f4411dc) — "remove the isolated static pixels, keep the boxes".

Rule (from the generator):
  The grid contains several solid rectangular boxes (`color`), each at least 2x2
  (wides/talls drawn from [2,5]), plus a sprinkling of single "static" pixels of
  the same `color`.  The static pixels are isolated single cells (the generator
  calls common.remove_neighbors so no two static pixels are adjacent, and it
  refuses any static pixel that would connect two shapes or touch a box border in
  more than one place).  OUTPUT = INPUT with every static pixel removed -> only the
  solid boxes survive.

  Per cell the deterministic local rule is:
    keep(r,c)  iff  cell (r,c) belongs to at least one fully-filled 2x2 square.
  A box cell (box >=2x2) is always part of a filled 2x2; an isolated static pixel
  (1x1, never adjacent to another static pixel) can never form a filled 2x2 even if
  it touches a box on one side.  Verified exact over 3000 fresh instances.

Encoding (floor-break, route the 10-ch expansion into the FREE output):
  occ[1,1,30,30] = 1 - input[:,0:1] (non-background occupancy; the grid is the
                   colour `color` plus background, so ch0==0 <=> a coloured cell).
  block = Conv(occ, 2x2 ones)  with bottom/right pad -> count of filled cells in the
          2x2 block whose TOP-LEFT is (r,c); block is "full" iff count >= 3.5 (==4).
  keep  = Conv(blockfull, 2x2 ones) with top/left pad -> a cell is kept iff ANY of
          the 4 blocks covering it (TL at (r,c),(r-1,c),(r,c-1),(r-1,c-1)) is full.
  output = Where(keep_cond, input, bg_onehot)   ([1,1,30,30] cond, [1,10,1,1] value
          [1,0,..,0] background, FREE [1,10,30,30] input) -> removed cells become
          background, kept cells keep their input colour.  The 10-channel tensor is
          only the free output.  Dominant intermediate: the [1,1,30,30] fp16 planes.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- occupancy occ = non-background AND in-grid ------------------------
    # off-grid cells are all-zero across channels (ch0==0), so 1-ch0 would mark
    # them occupied; gate by in-grid = (max over channels) > 0.
    # colf = sum_k k*input_k via a 1x1 Conv (10->1): >0 exactly on coloured cells
    # (background ch0 contributes weight 0).  Doubles as the occupancy plane and
    # avoids a channel-0 Slice + Sub.
    # ONE 1x1 Conv (10->1) packs every signal into a single f32 plane g:
    #   weights w = [50, 1, 2, ..., 9]  ->  background(ch0)=50, coloured cell=k(1..9),
    #   off-grid (all channels 0)=0.  All three cases are read off g by thresholds:
    #     occ (coloured) = 0.5 < g < 49.5 ;  off-grid = g < 0.5 .
    wpack = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    wpack[0, 0, 0, 0] = 50.0
    init("WPACK", wpack, np.float32)
    n("Conv", ["input", "WPACK"], "g30", kernel_shape=[1, 1])  # [1,1,30,30] f32
    # crop g to the 20x20 active region (size in [7,20] -> coords 0..19) so every
    # downstream plane is 20x20.  cells outside 20x20 are always off-grid -> handled
    # by padding the final select-cond with True (select input -> all-zero there).
    init("c20_s", np.array([0, 0], np.int64), np.int64)
    init("c20_e", np.array([20, 20], np.int64), np.int64)
    init("c20_ax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["g30", "c20_s", "c20_e", "c20_ax"], "g")      # [1,1,20,20] f32
    init("HALFH", np.array(0.5, np.float16), np.float16)
    init("HALF32", np.array(0.5, np.float32), np.float32)
    init("BG_THR", np.array(49.5, np.float32), np.float32)
    n("Greater", ["g", "HALF32"], "ge_b")                    # g>0.5  (in-grid)
    n("Less", ["g", "BG_THR"], "lt_b")                       # g<49.5 (not background)
    n("And", ["ge_b", "lt_b"], "occ_b")                      # [1,1,20,20] coloured
    n("Less", ["g", "HALF32"], "offgrid_b")                  # [1,1,20,20] g<0.5 off-grid
    n("Cast", ["occ_b"], "occ", to=F16)                       # [1,1,20,20] f16 {0,1}

    # ---- block = Conv(occ, 2x2 ones), pad bottom/right ---------------------
    # count of filled cells in the 2x2 whose TOP-LEFT is (r,c).
    Wk = np.ones((1, 1, 2, 2), np.float16)
    init("Wsum", Wk, np.float16)
    # pad bottom=1 right=1 so output stays 20x20 with TL anchoring.
    n("Conv", ["occ", "Wsum"], "blockcnt",
      kernel_shape=[2, 2], pads=[0, 0, 1, 1])                 # [1,1,20,20] f16
    init("THREE5", np.array(3.5, np.float16), np.float16)
    n("Greater", ["blockcnt", "THREE5"], "blockfull_b")       # [1,1,20,20] bool
    n("Cast", ["blockfull_b"], "blockfull", to=F16)

    # ---- keep = Conv(blockfull, 2x2 ones), pad top/left --------------------
    # cell (r,c) kept iff any of the 4 covering blocks (TL at (r,c),(r-1,c),
    # (r,c-1),(r-1,c-1)) is full -> dilate the block map by a 2x2 anchored bottom-right.
    init("Wsum2", Wk, np.float16)
    n("Conv", ["blockfull", "Wsum2"], "keepcnt",
      kernel_shape=[2, 2], pads=[1, 1, 0, 0])                 # [1,1,20,20] f16
    n("Greater", ["keepcnt", "HALFH"], "keep_b")              # [1,1,20,20] bool

    # ---- off-grid handling: force keep on off-grid cells -------------------
    # selecting `input` (all-zero off-grid) yields the all-zero padded target.
    # selcond = keep OR off-grid  on the 20x20 region; cells >=20 are always
    # off-grid so we pad selcond with True (-> select input).
    n("Or", ["keep_b", "offgrid_b"], "selcond20_b")           # [1,1,20,20] bool
    n("Cast", ["selcond20_b"], "selcond20_u8", to=TensorProto.UINT8)
    init("spads", np.array([0, 0, 0, 0, 0, 0, 10, 10], np.int64), np.int64)
    init("ONEU8", np.array(1, np.uint8), np.uint8)
    n("Pad", ["selcond20_u8", "spads", "ONEU8"], "selcond30_u8", mode="constant")
    n("Cast", ["selcond30_u8"], "selcond_b", to=BOOL)         # [1,1,30,30] bool

    # ---- single Where -> FREE [1,10,30,30] output --------------------------
    bg = np.zeros((1, 10, 1, 1), np.float32)
    bg[0, 0, 0, 0] = 1.0
    init("bg_onehot", bg, np.float32)
    n("Where", ["selcond_b", "input", "bg_onehot"], "output")

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task193", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
