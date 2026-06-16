"""task383 (ARC-AGI f1cefba8) — barnacle stripes through a coloured box.

Rule (from the generator): the grid holds ONE axis-aligned box at (brow,bcol),
size tall x wide, with a 2-px OUTER ring in colour C0 and a SOLID inner block in
colour C1.  A handful of "barnacle" markers are placed as single C1 pixels ON
the C0 ring (the inner ring line: rows brow+1 / brow+tall-2, cols bcol+1 /
bcol+wide-2).  Each marker projects a STRIPE perpendicular to the ring it sits
on:
  * a marker on the TOP/BOTTOM ring -> a full COLUMN stripe through its column;
  * a marker on the LEFT/RIGHT ring -> a full ROW stripe through its row.
The output draws the clean box PLUS, for every stripe line:
  * INSIDE the box, the whole crossed column/row becomes C0 (the inner C1 flips
    to C0, ring stays C0);
  * OUTSIDE the box, the crossed column/row is filled C1 across the full grid
    extent (only along the perpendicular direction; a stripe column only paints
    the rows outside the box, a stripe row only the cols outside the box).

Everything is SEPARABLE into 1-D row/col vectors, so no [1,10,30,30] or
[1,1,30,30] box plane is ever built.  Per-cell label among {bg=0, C0, C1}:
    bothbox = ibr[r] & ibc[c]                  (ibr=rowhas, ibc=colhas)
    trig    = (ringrow[r]|SR[r]) | (ringcol[c]|SC[c])
    isC0    = bothbox & trig
    isC1    = (bothbox & ~trig) | (~bothbox & (SC[c]|SR[r]))
where ibr/ibc are the box-occupancy profiles, ringrow/ringcol = ibr/ibc eroded
by 2 (MinPool window 5) then XOR, and SC/SR are stripe-column/row flags found by
reducing (colf==C1) over the ring rows/cols.

Scalars:
    C1 = max colour over the inner block (solid C1, markers only live on ring);
    C0 = max colour over non-bg cells whose colour != C1.

Output = Where(isC0, onehot(C0), Where(isC1, onehot(C1), bg_onehot)) routed into
the FREE bool output; onehot(C0/C1) are [1,10,1,1] one-hots built by Equal on the
recovered scalar colours.

Dominant intermediates: the one fp32 colour-index plane colf [1,1,30,30] (3600B,
the irreducible 10->1 entry) and the fp16 isC1 plane (1800B); everything else is
1-D [1,1,30,1]/[1,1,1,30] vectors or [1,10,1,1] scalars.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
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

    # ---- colour-index plane colf = sum_k k*input_k via a 1x1 Conv (no [1,10,..]
    # intermediate) -> [1,1,30,30] fp32 (the one irreducible entry plane). -------
    cw = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("cw", cw, np.float32)
    n("Conv", ["input", "cw"], "colf32")            # [1,1,30,30] f32 (entry plane)
    # cast to fp16 immediately; all downstream full-canvas ops run in fp16
    # (colours/indices < 100 are fp16-exact).  task377 lever.
    n("Cast", ["colf32"], "colf", to=F16)           # [1,1,30,30] f16 (1800B)

    # ---- in-grid mask (separable, NO 30x30 plane): ig_row[r]=1 iff row r has any
    # channel (incl. bg) set = ReduceMax(input, axes=[1,3]); ig_col via [1,2]. ---
    n("ReduceMax", ["input"], "ig_row", axes=[1, 3], keepdims=1)  # [1,1,30,1] f32
    n("ReduceMax", ["input"], "ig_col", axes=[1, 2], keepdims=1)  # [1,1,1,30] f32

    # ---- box occupancy profiles --------------------------------------------
    # non-bg cell <=> colf > 0 ; box rows/cols are contiguous & solid so
    # ibr[r] = rowhas[r] = any nonbg in row r ; ibc = colhas.
    init("half", np.array(0.5, np.float32), np.float32)
    init("half16", np.array(0.5, np.float16), np.float16)
    init("zero16f", np.array(0.0, np.float16), np.float16)
    n("ReduceMax", ["colf"], "rowmax", axes=[3], keepdims=1)   # [1,1,30,1] f16
    n("ReduceMax", ["colf"], "colmax", axes=[2], keepdims=1)   # [1,1,1,30] f16
    n("Greater", ["rowmax", "half16"], "ibr")                  # [1,1,30,1] bool
    n("Greater", ["colmax", "half16"], "ibc")                  # [1,1,1,30] bool
    n("Cast", ["ibr"], "ibr_f", to=F16)
    n("Cast", ["ibc"], "ibc_f", to=F16)

    # ---- erode the profiles by 2 (MinPool window 5) to get inner rows/cols ----
    # MinPool = -MaxPool(-x).  pads=2 each side, kernel 5, stride 1 -> same length.
    n("Neg", ["ibr_f"], "nibr")
    n("MaxPool", ["nibr"], "mibr", kernel_shape=[5, 1], pads=[2, 0, 2, 0],
      strides=[1, 1])
    n("Neg", ["mibr"], "innerr_f")                             # [1,1,30,1] f32 {0,1}
    n("Neg", ["ibc_f"], "nibc")
    n("MaxPool", ["nibc"], "mibc", kernel_shape=[1, 5], pads=[0, 2, 0, 2],
      strides=[1, 1])
    n("Neg", ["mibc"], "innerc_f")                             # [1,1,1,30] f32 {0,1}
    n("Greater", ["innerr_f", "half16"], "innerr")             # bool
    n("Greater", ["innerc_f", "half16"], "innerc")
    # ringrow = ibr & ~innerr ; ringcol = ibc & ~innerc
    n("Not", ["innerr"], "ninnerr")
    n("And", ["ibr", "ninnerr"], "ringrow")                    # [1,1,30,1] bool
    n("Not", ["innerc"], "ninnerc")
    n("And", ["ibc", "ninnerc"], "ringcol")                    # [1,1,1,30] bool

    # ---- recover colour scalars C1 (inner) and C0 (ring) ---------------------
    # innermask2d = innerr & innerc.  C1 = max colf over inner cells.
    init("neg1", np.array(-1.0, np.float16), np.float16)
    n("And", ["innerr", "innerc"], "innermask")                # [1,1,30,30] bool
    # C1 = max colf over inner cells (Where fuses mask+select, no extra Cast plane)
    n("Where", ["innermask", "colf", "neg1"], "colf_inner")    # [1,1,30,30] f16
    n("ReduceMax", ["colf_inner"], "C1", axes=[2, 3], keepdims=1)  # [1,1,1,1] f16
    # C0 = max colf over cells with colf != C1 (bg/off-grid contribute 0 < C0,
    # inner C1 cells are excluded) -> picks the ring colour with NO nonbg gate.
    n("Equal", ["colf", "C1"], "isC1")                         # [1,1,30,30] bool
    n("Not", ["isC1"], "notC1")
    n("Where", ["notC1", "colf", "neg1"], "colf_c0")           # [1,1,30,30] f16
    n("ReduceMax", ["colf_c0"], "C0", axes=[2, 3], keepdims=1)  # [1,1,1,1] f16

    # ---- stripe detection: SC[c] = ibc[c] & any(isC1 & ringrow over rows) ----
    # SR[r] = ibr[r] & any(isC1 & ringcol over cols).  Where fuses the mask.
    n("Cast", ["ringrow"], "ringrow_f16", to=F16)              # [1,1,30,1]
    n("Cast", ["ringcol"], "ringcol_f16", to=F16)              # [1,1,1,30]
    n("Where", ["isC1", "ringrow_f16", "zero16f"], "c1ring_r")  # [1,1,30,30] f16
    n("ReduceMax", ["c1ring_r"], "sc_raw", axes=[2], keepdims=1)  # [1,1,1,30] f16
    n("Where", ["isC1", "ringcol_f16", "zero16f"], "c1ring_c")  # [1,1,30,30] f16
    n("ReduceMax", ["c1ring_c"], "sr_raw", axes=[3], keepdims=1)  # [1,1,30,1] f16
    n("Greater", ["sc_raw", "half16"], "sc_hit")               # [1,1,1,30] bool
    n("Greater", ["sr_raw", "half16"], "sr_hit")               # [1,1,30,1] bool
    n("And", ["sc_hit", "ibc"], "SC")                          # [1,1,1,30] bool
    n("And", ["sr_hit", "ibr"], "SR")                          # [1,1,30,1] bool

    # ---- per-cell label L among {bg=0, C0, C1, off-grid=99} -------------------
    # Build from 1-D row/col vectors with the MINIMUM number of 30x30 planes:
    #   inbox: L = (Ar|Ac) ? C0 : C1     (Ar=ringrow|SR, Ac=ringcol|SC)
    #   outbox in-grid: L = (SC|SR) ? C1 : 0
    #   off-grid: L = 99
    # All combining ops below produce one 30x30 plane each; keep it to ~7 total.
    n("Or", ["ringrow", "SR"], "Ar")                           # [1,1,30,1] bool
    n("Or", ["ringcol", "SC"], "Ac")                           # [1,1,1,30] bool
    n("And", ["ibr", "ibc"], "bothbox")                        # [1,1,30,30] bool
    n("Or", ["Ar", "Ac"], "AorB")                              # [1,1,30,30] bool
    n("Or", ["SC", "SR"], "scORsr")                            # [1,1,30,30] bool
    # in-grid mask from the cheap fp32 axis-reductions
    n("Greater", ["ig_row", "half"], "igr")                    # [1,1,30,1] bool
    n("Greater", ["ig_col", "half"], "igc")                    # [1,1,1,30] bool
    n("And", ["igr", "igc"], "ingrid")                         # [1,1,30,30] bool

    init("sent99", np.array(99.0, np.float16), np.float16)
    # out-of-box value: ingrid ? (scORsr ? C1 : 0) : 99
    # NB: a stripe col/row would otherwise paint C1 into the off-grid padding,
    # so the in-grid gate is OUTERMOST.
    n("Where", ["scORsr", "C1", "zero16f"], "out_ig")          # [1,1,30,30] f16
    n("Where", ["ingrid", "out_ig", "sent99"], "out_val")      # [1,1,30,30] f16
    # in-box value: AorB ? C0 : C1
    n("Where", ["AorB", "C0", "C1"], "in_val")                 # [1,1,30,30] f16
    # combine by box membership
    n("Where", ["bothbox", "in_val", "out_val"], "L")          # [1,1,30,30] f16
    # one-hot expand into the FREE bool output.
    chan = np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1)
    init("chan", chan, np.float16)
    n("Equal", ["L", "chan"], "output")                        # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task383", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
