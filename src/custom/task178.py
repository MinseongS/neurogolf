"""Task 178 (ARC-AGI 746b3537): run-length the colored bands into a line.

Rule: the input is a grid of solid colour bands of varying thickness, stacked
along one axis (rows for xpose=0, cols for xpose=1). The output lists the
distinct band colours in order, each collapsed to a single cell, as a single
column (xpose=0) or single row (xpose=1).

Key structural facts from the generator:
  * width 1..5, 3..5 colours, thicks 1..3  -> band axis length = sum(thicks) <=15,
    cross axis <=5, output length = #colours <=5.
  * CONSECUTIVE band colours are forced distinct, so every change of colour along
    the band axis starts a new output slot, and #runs == #colours.
  * For xpose=1 (vertical bands) the FIRST ROW already contains all band colours
    in order; for xpose=0 the FIRST COLUMN does. So we only need to read line 0.

Pipeline (per axis; rows shown, cols symmetric with axes swapped):
  line     = input[:, :, 0:1, 0:15]                 first row, 15 cells  [1,10,1,15]
  prev     = line shifted right by one (pad a zero col at front)
  change   = sum_ch |line - prev| > 0               colour differs from left nbr
  valid    = max_ch line > 0                          cell is in-grid (has colour)
  runstart = change & valid                           starts a new band
  cum      = CumSum(runstart) (1-indexed)             band number of each cell
  assign[c,s] = (cum[c]==s+1) & runstart[c]           which slot each runstart fills
  compact  = line_matrix[10,15] @ assign[15,5] = [10,5]   one-hot colour per slot
then place compact at row/col 0 of a 5x5 block, select axis by which axis has more
runs, and Pad the 5x5 block out to 30x30 (the final Pad output IS the free output).

All math is integer-valued; we run it in fp16 (counts <=15, exact) except CumSum
(ORT rejects fp16 CumSum) which runs on a tiny [1,1,15,1] fp32 plane. The final
candidate/select planes are uint8 (output is scored as >0).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL
LEN = 15   # max band-axis length (sum thicks)
SLOTS = 5  # max #colours / output length


def build(task):
    inits, nodes, vinfos = [], [], []

    def init(name, arr, dtype=np.float32):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    def vi(name, dtype, shape):
        vinfos.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    # shared constants
    init("zero16", np.array(0.0, np.float16), np.float16)
    init("half16", np.array(0.5, np.float16), np.float16)
    init("slots16", (np.arange(SLOTS, dtype=np.float16) + 1.0).reshape(1, 1, 1, SLOTS), np.float16)
    init("ax_w", np.array(3, np.int64), np.int64)  # cumsum axis for row line (axis 3)
    init("ax_h", np.array(2, np.int64), np.int64)  # cumsum axis for col line (axis 2)
    init("colorvec", np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1), np.float16)

    def pipeline(tag, line_axis, slot_axis):
        """line_axis: spatial axis the band runs along (3=cols for rows-line,
        2=rows for cols-line). slot_axis matches for the assign Equal."""
        if line_axis == 3:                # horizontal line: row 0, 15 cols
            starts, ends = [0, 0, 0, 0], [1, 10, 1, LEN]
            line_sh = [1, 10, 1, LEN]
            red_sh = [1, 1, 1, LEN]
            pad_prev = [0, 0, 0, 1, 0, 0, 0, 0]          # pad one col at front
            red_slice_last = [1, 1, 1, LEN - 1]
            cum_axis = "ax_w"
            out_block_sh = [10, 1, SLOTS]                # compact -> rows-of-output
        else:                              # vertical line: col 0, 15 rows
            starts, ends = [0, 0, 0, 0], [1, 10, LEN, 1]
            line_sh = [1, 10, LEN, 1]
            red_sh = [1, 1, LEN, 1]
            pad_prev = [0, 0, 1, 0, 0, 0, 0, 0]          # pad one row at top
            red_slice_last = [1, 1, LEN - 1, 1]
            cum_axis = "ax_h"
            out_block_sh = [10, SLOTS, 1]

        s0 = init(f"st_{tag}", np.array(starts, np.int64), np.int64)
        e0 = init(f"en_{tag}", np.array(ends, np.int64), np.int64)
        ax = init(f"sax_{tag}", np.array([0, 1, 2, 3], np.int64), np.int64)
        n("Slice", ["input", s0, e0, ax], f"line32_{tag}")
        vi(f"line32_{tag}", F32, line_sh)
        n("Cast", [f"line32_{tag}"], f"line_{tag}", to=F16)
        vi(f"line_{tag}", F16, line_sh)

        # colour-index per cell: ci = sum_ch ch*line, folded into ONE no-pad Conv
        # with the colorvec [1,10,1,1] weight (collapses the 10-channel axis with
        # no [1,10,1,15] intermediate). Output is the small [1,1,*,*] index plane.
        n("Conv", [f"line_{tag}", "colorvec"], f"ci_{tag}")
        vi(f"ci_{tag}", F16, red_sh)
        # prev_ci = ci shifted by one along band axis (front cell -> 0): on the
        # SMALL [1,1,*,*] plane, so slice+pad is nearly free.
        sl_s = init(f"sls_{tag}", np.array([0, 0, 0, 0], np.int64), np.int64)
        sl_e = init(f"sle_{tag}", np.array(red_slice_last, np.int64), np.int64)
        n("Slice", [f"ci_{tag}", sl_s, sl_e, ax], f"cicut_{tag}")
        vi(f"cicut_{tag}", F16, red_slice_last)
        n("Pad", [f"cicut_{tag}", init(f"pp_{tag}", np.array(pad_prev, np.int64), np.int64),
                  "zero16"], f"cprev_{tag}", mode="constant")
        vi(f"cprev_{tag}", F16, red_sh)
        # change = ci differs from prev_ci
        n("Sub", [f"ci_{tag}", f"cprev_{tag}"], f"cd_{tag}")
        n("Abs", [f"cd_{tag}"], f"cda_{tag}")
        vi(f"cd_{tag}", F16, red_sh)
        vi(f"cda_{tag}", F16, red_sh)
        n("Greater", [f"cda_{tag}", "half16"], f"change_{tag}")
        vi(f"change_{tag}", BOOL, red_sh)
        # valid = cell is in-grid (has any colour)
        n("Greater", [f"ci_{tag}", "half16"], f"valid_{tag}")
        vi(f"valid_{tag}", BOOL, red_sh)
        n("And", [f"change_{tag}", f"valid_{tag}"], f"rsb_{tag}")
        vi(f"rsb_{tag}", BOOL, red_sh)
        n("Cast", [f"rsb_{tag}"], f"rs_{tag}", to=F16)
        vi(f"rs_{tag}", F16, red_sh)

        # cumsum must be fp32 -> tiny plane
        n("Cast", [f"rs_{tag}"], f"rs32_{tag}", to=F32)
        vi(f"rs32_{tag}", F32, red_sh)
        n("CumSum", [f"rs32_{tag}", cum_axis], f"cum_{tag}")
        vi(f"cum_{tag}", F32, red_sh)
        n("Cast", [f"cum_{tag}"], f"cum16_{tag}", to=F16)
        vi(f"cum16_{tag}", F16, red_sh)

        # reshape cum and valid to [1,1,LEN,1] so the slot axis is last (broadcast
        # against slots16). assign[c,s] = (cum[c]==s+1) AND in-grid(c): every cell
        # of band s+1 contributes its colour to slot s (background cells are
        # excluded by `valid`, so output channel 0 stays empty). Output is scored
        # as >0 so the thickness-weighted (non-one-hot) magnitude is fine.
        rsh = init(f"rsh_{tag}", np.array([1, 1, LEN, 1], np.int64), np.int64)
        n("Reshape", [f"cum16_{tag}", rsh], f"cumc_{tag}")
        vi(f"cumc_{tag}", F16, [1, 1, LEN, 1])
        n("Reshape", [f"valid_{tag}", rsh], f"validc_{tag}")
        vi(f"validc_{tag}", BOOL, [1, 1, LEN, 1])
        n("Equal", [f"cumc_{tag}", "slots16"], f"eqs_{tag}")
        vi(f"eqs_{tag}", BOOL, [1, 1, LEN, SLOTS])
        n("And", [f"eqs_{tag}", f"validc_{tag}"], f"assignb_{tag}")
        vi(f"assignb_{tag}", BOOL, [1, 1, LEN, SLOTS])
        n("Cast", [f"assignb_{tag}"], f"assign4_{tag}", to=F16)
        vi(f"assign4_{tag}", F16, [1, 1, LEN, SLOTS])

        # compact[ch,slot] = sum_c line[ch,c]*assign[c,slot] via a 4D MatMul that
        # contracts the band axis. line must be [1,10,1,LEN]; for the col line
        # ([1,10,LEN,1]) reshape it (a 1-dim transpose).
        if line_axis == 3:
            line_mm = f"line_{tag}"
        else:
            lmsh = init(f"lmsh_{tag}", np.array([1, 10, 1, LEN], np.int64), np.int64)
            n("Reshape", [f"line_{tag}", lmsh], f"linemm_{tag}")
            vi(f"linemm_{tag}", F16, [1, 10, 1, LEN])
            line_mm = f"linemm_{tag}"
        # [1,10,1,LEN] @ [1,1,LEN,SLOTS] -> [1,10,1,SLOTS]
        n("MatMul", [line_mm, f"assign4_{tag}"], f"compact_{tag}")
        vi(f"compact_{tag}", F16, [1, 10, 1, SLOTS])

        # reshape compact [1,10,1,SLOTS] to [1,10,1,SLOTS] (rows, no-op) or
        # [1,10,SLOTS,1] (cols)
        bsh = init(f"bsh_{tag}", np.array([1] + out_block_sh, np.int64), np.int64)
        n("Reshape", [f"compact_{tag}", bsh], f"block_{tag}")
        block_full = [1] + out_block_sh
        vi(f"block_{tag}", F16, block_full)
        # convert to uint8 for the select/pad path
        n("Cast", [f"block_{tag}"], f"blocku_{tag}", to=U8)
        vi(f"blocku_{tag}", U8, block_full)
        # pad to [1,10,5,5]
        if out_block_sh[1] == 1:    # [1,10,1,5] -> pad rows to 5
            pads = [0, 0, 0, 0, 0, 0, SLOTS - 1, 0]
        else:                        # [1,10,5,1] -> pad cols to 5
            pads = [0, 0, 0, 0, 0, 0, 0, SLOTS - 1]
        p = init(f"p55_{tag}", np.array(pads, np.int64), np.int64)
        zu = init(f"zu_{tag}", np.array(0, np.uint8), np.uint8)
        n("Pad", [f"blocku_{tag}", p, zu], f"block55_{tag}", mode="constant")
        vi(f"block55_{tag}", U8, [1, 10, SLOTS, SLOTS])

        # run count scalar for axis selection (sum runstarts) - reuse rs
        n("ReduceSum", [f"rs_{tag}"], f"rcnt_{tag}", axes=[1, 2, 3], keepdims=1)
        vi(f"rcnt_{tag}", F16, [1, 1, 1, 1])
        return f"block55_{tag}", f"rcnt_{tag}"

    blk_r, rc_r = pipeline("r", line_axis=3, slot_axis=3)  # horizontal bands -> col output
    blk_c, rc_c = pipeline("c", line_axis=2, slot_axis=2)  # vertical bands -> row output

    # horizontal bands => first column carries colours => col line ("c") has the runs.
    # vertical bands => first row carries colours => row line ("r") has the runs.
    # whichever line has MORE runs is the band axis.
    n("Greater", [rc_r, rc_c], "use_r")
    vi("use_r", BOOL, [1, 1, 1, 1])
    n("Where", ["use_r", blk_r, blk_c], "inner")
    vi("inner", U8, [1, 10, SLOTS, SLOTS])
    # pad 5x5 inner to 30x30 -> free output
    pf = init("pad_full", np.array([0, 0, 0, 0, 0, 0, 30 - SLOTS, 30 - SLOTS], np.int64), np.int64)
    zu2 = init("zu2", np.array(0, np.uint8), np.uint8)
    n("Pad", ["inner", pf, zu2], "output", mode="constant")

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", U8, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits, value_info=vinfos)
    m = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 11)])
    return m
