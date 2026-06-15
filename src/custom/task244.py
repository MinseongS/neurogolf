"""task244 (ARC-AGI 9f236235) — un-magnify a flipped line-grid.

Rule (from the generator):
  output is a size x size grid (size in {3,4}) of pixels (colors 1..9, 0 = bg).
  create_linegrid(output, magnifier, linecolor) blows each output cell (r,c)
  into a mag x mag solid block (mag = magnifier in 2..5) separated by gridlines
  of `linecolor`; spacing sp = mag+1.  actual_size = size*sp - 1.  Then the whole
  grid is flipped horizontally (each row reversed).

  So output[r][c] = input[ r*sp ][ actual_size-1 - c*sp ]  (input is the flipped
  line-grid).  Both sp and size vary per instance and must be RECOVERED.

Recovery (all exact, validated on 8000 fresh instances):
  * colors 1..9 only appear on real content (linecolor excluded from pixels,
    pixels never 0).  Off-grid cells are color 0.  The full actual_size x
    actual_size grid is bounded by gridlines that reach the last row/col, so
    actual_size-1 = max column index carrying any nonzero color.
  * actual_size uniquely determines (size, sp) EXCEPT actual_size==11, which is
    either (size=3,sp=4) or (size=4,sp=3).  Disambiguate: row index 2 is a full
    horizontal gridline (every cell one nonzero color) iff sp==3.  Detected as
    "max over colors 1..9 of (count of that color in row 2) == actual_size".

Construction:
  recover actual/sp/size as scalars -> build row/col gather-index vectors
  rowidx = arange(4)*sp,  colidx = (actual-1) - arange(4)*sp.  Double-Gather the
  one-hot input (axis 3 then axis 2) -> [1,10,4,4] blocks, zero out the cells
  with r>=size or c>=size, Pad to [1,10,30,30] = output (free).

Largest intermediate: the [1,10,30,4] column-gathered plane (4800B) — inherent
to a two-axis data-dependent gather.  ~8.8KB total vs 31.7KB public net.
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

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, list(ins), [out], **attrs))
        return out

    F = TensorProto.FLOAT
    I64 = TensorProto.INT64

    # ---- recover actual_size -------------------------------------------------
    # Off-grid cells are ALL-ZERO one-hot; in-grid cells (incl. background color
    # 0 -> channel 0) have exactly one channel set.  colprof[c] = max over all
    # channels & rows = 1 iff column c is inside the (contiguous) grid.
    n("ReduceMax", ["input"], "colprof", axes=[1, 2], keepdims=1)  # [1,1,1,30]
    init("onef", np.array([[[[1.0]]]], np.float32), np.float32)
    # actual-1 = max col index that is inside the grid
    ar30 = np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30)
    init("ar30", ar30, np.float32)
    n("Mul", ["colprof", "ar30"], "colpos")
    n("ReduceMax", ["colpos"], "last", axes=[2, 3], keepdims=1)   # [1,1,1,1] = actual-1
    n("Add", ["last", "onef"], "actual_f")                        # actual_size (float)

    # ---- disambiguate sp at actual==11 --------------------------------------
    # row 2 color histogram: sum over cols of input[:, :, 2, :]
    init("r2_starts", np.array([0, 0, 2, 0], np.int64), np.int64)
    init("r2_ends", np.array([1, 10, 3, 30], np.int64), np.int64)
    init("r2_axes", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "r2_starts", "r2_ends", "r2_axes"], "row2")  # [1,10,1,30]
    n("ReduceSum", ["row2"], "r2cnt", axes=[3], keepdims=1)       # [1,10,1,1]
    # zero out channel 0 so only nonzero colors counted, then max over channels
    chmask = np.array([0.] + [1.] * 9, np.float32).reshape(1, 10, 1, 1)
    init("chmask", chmask, np.float32)
    n("Mul", ["r2cnt", "chmask"], "r2cnt_nz")
    n("ReduceMax", ["r2cnt_nz"], "r2max", axes=[1], keepdims=1)   # [1,1,1,1]
    n("Equal", ["r2max", "actual_f"], "disamb_b")                 # bool [1,1,1,1]
    n("Cast", ["disamb_b"], "disamb_f", to=F)                     # 0/1

    # ---- lookup sp & size from actual (+ disamb correction) -----------------
    # tables indexed by actual_size (0..23); ambiguous slot 11 corrected by disamb
    sp_base = np.zeros(24, np.float32)
    size_base = np.zeros(24, np.float32)
    dsp = np.zeros(24, np.float32)      # delta to add when disamb==1
    dsize = np.zeros(24, np.float32)
    m0 = {8: (3, 3), 11: (3, 4), 14: (3, 5), 15: (4, 4),
          17: (3, 6), 19: (4, 5), 23: (4, 6)}
    for a, (s, sp) in m0.items():
        size_base[a] = s
        sp_base[a] = sp
    dsize[11] = 4 - 3      # disamb==1 -> size 4
    dsp[11] = 3 - 4        # disamb==1 -> sp 3
    init("sp_base", sp_base.reshape(24), np.float32)
    init("size_base", size_base.reshape(24), np.float32)
    init("dsp", dsp.reshape(24), np.float32)
    init("dsize", dsize.reshape(24), np.float32)

    # actual as int64 index [1]
    n("Cast", ["actual_f"], "actual_i", to=I64)                   # [1,1,1,1]
    init("flat1", np.array([1], np.int64), np.int64)
    n("Reshape", ["actual_i", "flat1"], "actual_idx")             # [1]

    def table_lookup(base_t, delta_t, out):
        b = n("Gather", [base_t, "actual_idx"], out + "_b")       # [1]
        d = n("Gather", [delta_t, "actual_idx"], out + "_d")
        # disamb_f is [1,1,1,1]; reshape to [1]
        return b, d

    n("Reshape", ["disamb_f", "flat1"], "disamb1")               # [1]
    table_lookup("sp_base", "dsp", "sp")
    n("Mul", ["sp_d", "disamb1"], "sp_dd")
    n("Add", ["sp_b", "sp_dd"], "sp_f")                          # [1] float
    table_lookup("size_base", "dsize", "size")
    n("Mul", ["size_d", "disamb1"], "size_dd")
    n("Add", ["size_b", "size_dd"], "size_f")                    # [1] float

    # ---- build gather index vectors -----------------------------------------
    ar4 = np.arange(4, dtype=np.float32)
    init("ar4", ar4.reshape(4), np.float32)
    # rowidx = arange(4) * sp
    n("Mul", ["ar4", "sp_f"], "rowidx_f")                        # [4]
    n("Cast", ["rowidx_f"], "rowidx", to=I64)
    # colidx = (actual-1) - arange(4)*sp = last - rowidx
    n("Reshape", ["last", "flat1"], "last1")                     # [1]
    n("Sub", ["last1", "rowidx_f"], "colidx_f")                  # [4]
    n("Cast", ["colidx_f"], "colidx", to=I64)

    # ---- double gather of the one-hot input ---------------------------------
    n("Gather", ["input", "colidx"], "g1", axis=3)              # [1,10,30,4]
    n("Gather", ["g1", "rowidx"], "g2", axis=2)                # [1,10,4,4]

    # ---- mask cells with r>=size or c>=size ---------------------------------
    # rowmask[i] = (i < size) ; colmask[j] = (j < size)
    n("Less", ["ar4", "size_f"], "inmask_b")                    # [4] bool
    n("Cast", ["inmask_b"], "inmask_f", to=F)                   # [4]
    init("rm_shape", np.array([4, 1], np.int64), np.int64)
    init("cm_shape", np.array([1, 4], np.int64), np.int64)
    n("Reshape", ["inmask_f", "rm_shape"], "rowmask")          # [4,1]
    n("Reshape", ["inmask_f", "cm_shape"], "colmask")          # [1,4]
    n("Mul", ["rowmask", "colmask"], "mask44")                 # [4,4]
    init("mask_shape", np.array([1, 1, 4, 4], np.int64), np.int64)
    n("Reshape", ["mask44", "mask_shape"], "mask")             # [1,1,4,4]
    n("Mul", ["g2", "mask"], "blocks")                         # [1,10,4,4]

    # ---- pad to [1,10,30,30] = output ---------------------------------------
    pads = np.array([0, 0, 0, 0, 0, 0, 26, 26], np.int64)      # end-pad rows&cols
    init("pads", pads, np.int64)
    init("padval", np.array(0.0, np.float32), np.float32)
    n("Pad", ["blocks", "pads", "padval"], "output", mode="constant")

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task244", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
