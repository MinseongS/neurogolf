"""Task 232 (97999447): horizontal alternating comet trails.

Rule (from ARC-GEN generator 97999447): the input is a W x H grid (W,H in
7..14) containing a few single coloured pixels, one per (distinct) row, each in
the left half (col <= width//2), colours random non-gray.  For every coloured
pixel at (row, col, color) the output paints a horizontal trail from col to the
right grid edge: cell c gets `color` when c has the same parity as col, else
gray (5).  Cells left of col, rows with no pixel, and off-grid cells stay
background (0).  Each row is independent and holds at most one pixel.

Per-cell label map + final Equal (opset 11 BOOL output) so no [1,10,30,30]
one-hot stack is ever materialised.  For output cell (r,c):

    active   = has_pixel[r] AND (c >= start_col[r])
    label    = color[r]  if parity(c)==parity(start_col[r])
               5 (gray)  otherwise           (when active)
             = 0         when in-grid but not active
             = 10        when off-grid (never matches a channel)

start_col[r], color[r], has_pixel[r] are recovered as 1-D per-row scalars from
ONE coloured-value plane colf = sum_k k*input_k (>0 exactly at coloured cells:
background and off-grid both reduce to 0).  Everything 2-D below is built on a
14x14 working canvas (grid <= 14x14) from broadcast 1-D vectors, then L is
padded back to 30x30 with the off-grid sentinel before the final Equal.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

WORK = 14  # working canvas side: max grid dim = 14


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- constants ----
    init("half", np.array(0.5, np.float32), np.float32)
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    # column index ramp on the 14-wide canvas (for parity + c>=start)
    init("col_w", np.arange(WORK, dtype=np.float32).reshape(1, 1, 1, WORK),
         np.float32)
    init("row_w", np.arange(WORK, dtype=np.float32).reshape(1, 1, WORK, 1),
         np.float32)
    # Per-channel column weight matrix W[1,10,30,2] for ONE batched MatMul over
    # the FREE fp32 input (contracts the column axis directly):
    #   col 0 -> k (constant per channel)  => MatMul gives k*rowsum_k
    #   col 1 -> c*(k>0)                    => MatMul gives sum_c c*input[k,r,c]
    # background channel 0 contributes 0 to both (k=0 and the (k>0) gate).
    _w = np.zeros((10, 30, 2), np.float32)
    for k in range(10):
        _w[k, :, 0] = k                       # color weight
        if k >= 1:
            _w[k, :, 1] = np.arange(30)       # start-col weight
    init("wmat", _w.reshape(1, 10, 30, 2), np.float32)
    # split the [.,.,.,2] result back into the two per-row vectors
    init("c0", np.array([0], np.int64), np.int64)
    init("c1", np.array([1], np.int64), np.int64)
    init("c2", np.array([2], np.int64), np.int64)
    init("ax3", np.array([3], np.int64), np.int64)
    # crop the 30-long per-row 1-D vectors to 14 (axis 2)
    init("s0", np.array([0], np.int64), np.int64)
    init("s14", np.array([WORK], np.int64), np.int64)
    init("ax2", np.array([2], np.int64), np.int64)
    # label constants
    init("v5", np.array(5, np.uint8), np.uint8)     # gray
    init("v0", np.array(0, np.uint8), np.uint8)     # in-grid background
    init("v10", np.array(10, np.uint8), np.uint8)   # off-grid sentinel
    init("padval", np.array(10, np.uint8), np.uint8)
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)

    # ---- per-row scalars WITHOUT any 30x30 plane (one batched MatMul) ----
    n("MatMul", ["input", "wmat"], "cs30")                      # [1,10,30,2]
    n("ReduceSum", ["cs30"], "cs", axes=[1], keepdims=1)        # [1,1,30,2]
    n("Slice", ["cs", "s0", "s14", "ax2"], "cs14")             # [1,1,14,2]
    n("Slice", ["cs14", "c0", "c1", "ax3"], "color_r")         # [1,1,14,1] color
    n("Slice", ["cs14", "c1", "c2", "ax3"], "start_r")         # [1,1,14,1] start_col
    # has_pixel[r] = color[r] > 0
    n("Greater", ["color_r", "half"], "hasb")                   # [1,1,14,1] bool

    # ---- masks (all broadcast to [1,1,14,14]) ----
    # in-grid rectangle: row<H & col<W, H/W from 1-D occupancy of the colf plane
    # (any coloured cell marks its row/col; but background rows have no marker,
    #  so derive H,W from the FULL grid occupancy instead -> ReduceMax of input)
    n("ReduceMax", ["input"], "rowocc30", axes=[1, 3], keepdims=1)  # [1,1,30,1]
    n("ReduceMax", ["input"], "colocc30", axes=[1, 2], keepdims=1)  # [1,1,1,30]
    n("ReduceSum", ["rowocc30"], "Hf", keepdims=0)                  # scalar H
    n("ReduceSum", ["colocc30"], "Wf", keepdims=0)                  # scalar W
    n("Less", ["row_w", "Hf"], "rrect")            # [1,1,14,1] bool
    n("Less", ["col_w", "Wf"], "crect")            # [1,1,1,14] bool
    n("And", ["rrect", "crect"], "ingrid")         # [1,1,14,14] bool

    # active = has_pixel[r] & (col >= start_col[r]) & in-grid
    # col >= start  <=>  not(col < start)
    n("Less", ["col_w", "start_r"], "lt_start")         # [1,1,14,14] bool (col<start)
    n("Not", ["lt_start"], "ge_start")                  # col>=start
    n("And", ["hasb", "ge_start"], "act0")
    n("And", ["act0", "ingrid"], "active")              # [1,1,14,14] bool

    # parity match: parity(col) == parity(start_col[r])
    # parity = value mod 2, via floor; use Mod (allowed) on f32 (exact ints)
    init("modn", np.array(2.0, np.float32), np.float32)
    n("Mod", ["col_w", "modn"], "cpar", fmod=1)         # [1,1,1,14] f32 {0,1}
    n("Mod", ["start_r", "modn"], "spar", fmod=1)       # [1,1,14,1] f32 {0,1}
    n("Equal", ["cpar", "spar"], "parmatch")            # [1,1,14,14] bool

    # ---- colour index per cell (uint8): color[r] where active ----
    n("Cast", ["color_r"], "color_u", to=TensorProto.UINT8)   # [1,1,14,1]

    # ---- label map L (uint8 14x14): build inside-out ----
    # base: in-grid -> 0, off-grid -> 10
    n("Where", ["ingrid", "v0", "v10"], "L_base")             # [1,1,14,14]
    # active & parity-match -> color ; active & !match -> gray(5)
    n("Where", ["parmatch", "color_u", "v5"], "L_paint")      # broadcast color_u
    n("Where", ["active", "L_paint", "L_base"], "L14")        # final 14x14 label

    # pad to 30x30 with off-grid sentinel, then final Equal -> free BOOL output
    n("Pad", ["L14", "padpads", "padval"], "L", mode="constant")  # [1,1,30,30]
    n("Equal", ["L", "chan"], "output")

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
