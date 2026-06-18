"""task009 (ARC-AGI 06df4c85) — "connect same-colored cells sharing a row/column".

Rule (from generator task_06df4c85.py + common.create_linegrid):
  The pixel grid is a `create_linegrid(bitmap, spacing=2, linecolor)` rendering of an
  underlying bitmap of size n in [6,10]: each bitmap cell (r,c) becomes a 2x2 block at
  pixel rows/cols {3r,3r+1}x{3c,3c+1}; pixel rows/cols == 2 (mod 3) are gridlines (linecolor).
  The transform = `connect_bitmap`: for every pair of SAME-colored bitmap cells that share a
  row, fill the cells between them with that color (likewise columns). Equivalently per color
  v: fill the span between the min and max occupied position along each axis. Verified that
  distinct colors' spans NEVER overlap (max coverage 1/cell over 261 instances), so a span-fill
  per channel composes by union with no precedence concerns.

Encoding (bounded-active-region escape — work at bitmap scale, then upsample):
  bm = input[:,:, ::3, ::3]                                   [1,10,10,10] one-hot bitmap
  Per channel, a span-fill mask along an axis = (prefix-sum>0) AND (suffix-sum>0), prefix/
  suffix being triangular MatMuls (task070 idiom).  Row spans (col axis, right-mul): pref=bm@UpTri,
  suf=bm@LowTri.  Col spans (row axis, left-mul): pref=LowTri@bm, suf=UpTri@bm.
  connected one-hot  fill_oh = rowfill | colfill | (bm>0)     [1,10,10,10] bool
  Upsample to pixels by Gather(axis2 ridx)·Gather(axis3 cidx), ridx[r]=r//3.
  Reinsert gridlines: gridline_pix = (r%3==2 | c%3==2) AND in-grid(ReduceMax over channels>0);
  linecolor one-hot lc = input[:,:,2:3,0:1] (pixel (2,0) is always a gridline since n>=6).
  output = Where(gridline_pix, lc, block_oh)  -> FREE [1,10,30,30] one-hot.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I64 = TensorProto.INT64

N = 10  # bitmap canvas (n in [6,10]; off-grid bitmap cells are all-zero)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- downsample input one-hot to bitmap scale, channels 1..9 ----------
    # slice channels 1..9 only (channel 0 = background is excluded so its
    # full-grid span cannot flood the fill); avoids the chmask Mul too.
    init("ds_s", np.array([1, 0, 0], np.int64), np.int64)
    init("ds_e", np.array([10, 30, 30], np.int64), np.int64)
    init("ds_ax", np.array([1, 2, 3], np.int64), np.int64)
    init("ds_st", np.array([1, 3, 3], np.int64), np.int64)
    n("Slice", ["input", "ds_s", "ds_e", "ds_ax", "ds_st"], "bm_f32")  # [1,9,10,10]
    n("Cast", ["bm_f32"], "bm", to=F16)  # [1,9,10,10] f16

    # ---- triangular prefix/suffix matrices (bitmap scale) ------------------
    LowTri = np.tril(np.ones((N, N), np.float16))  # LowTri[r,k]=1 iff k<=r
    UpTri = np.triu(np.ones((N, N), np.float16))   # UpTri[r,k]=1  iff k>=r
    init("LowTri", LowTri.reshape(1, 1, N, N), np.float16)
    init("UpTri", UpTri.reshape(1, 1, N, N), np.float16)

    # A span-fill mask in one direction = (prefix>0) AND (suffix>0); since both
    # are non-negative, prefix*suffix>0 is the same predicate -> one Greater.
    n("MatMul", ["bm", "UpTri"], "pref_row")  # [1,9,10,10] pref over col axis
    n("MatMul", ["bm", "LowTri"], "suf_row")  # suffix over col axis
    n("Mul", ["pref_row", "suf_row"], "rowspan")  # >0 where in row span
    n("MatMul", ["LowTri", "bm"], "pref_col")  # pref over row axis
    n("MatMul", ["UpTri", "bm"], "suf_col")    # suffix over row axis
    n("Mul", ["pref_col", "suf_col"], "colspan")  # >0 where in col span
    n("Add", ["rowspan", "colspan"], "span")   # >0 where in any span
    init("Z", np.array(0.0, np.float16), np.float16)
    n("Greater", ["span", "Z"], "fill_oh")  # [1,9,10,10] bool connected one-hot

    # ---- connected bitmap as a 1-channel COLOR-INDEX plane (bitmap scale) ---
    # L_bm[r,c] = sum_k k * fill_oh[k]  (>0 only on connected cells; bg cells 0)
    n("Cast", ["fill_oh"], "fill_f", to=F16)              # [1,9,10,10] f16
    idxvec = np.arange(1, 10, dtype=np.float16).reshape(1, 9, 1, 1)  # colours 1..9
    init("idxvec", idxvec, np.float16)
    n("Mul", ["fill_f", "idxvec"], "fill_idx")            # [1,9,10,10] f16
    n("ReduceSum", ["fill_idx"], "L_bm", axes=[1], keepdims=1)  # [1,1,10,10] f16

    # ---- upsample L_bm -> pixel scale via Gather(axis2)·Gather(axis3) -------
    ridx = np.repeat(np.arange(N, dtype=np.int64), 3)  # len 30: [0,0,0,1,...]
    init("ridx", ridx, np.int64)
    n("Gather", ["L_bm", "ridx"], "Lg2", axis=2)   # [1,1,30,10] f16
    n("Gather", ["Lg2", "ridx"], "L_block", axis=3)  # [1,1,30,30] f16

    # ---- gridline static pattern (separable) + in-grid mask ----------------
    rr = (np.arange(30) % 3 == 2)
    init("gprow", rr.reshape(1, 1, 30, 1), np.bool_)  # [1,1,30,1]
    init("gpcol", rr.reshape(1, 1, 1, 30), np.bool_)  # [1,1,1,30]
    n("Or", ["gprow", "gpcol"], "gp")  # [1,1,30,30] bool
    init("ZF", np.array(0.0, np.float32), np.float32)
    n("ReduceMax", ["input"], "ingrid_f", axes=[1], keepdims=1)  # [1,1,30,30] f32
    n("Greater", ["ingrid_f", "ZF"], "ingrid")  # [1,1,30,30] bool

    # ---- linecolor as a scalar index from pixel (2,0) ----------------------
    init("lc_s", np.array([2, 0], np.int64), np.int64)
    init("lc_e", np.array([3, 1], np.int64), np.int64)
    init("lc_ax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["input", "lc_s", "lc_e", "lc_ax"], "lc")  # [1,10,1,1] f32 one-hot
    n("Cast", ["lc"], "lc_f", to=F16)
    idx10 = np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1)
    init("idx10", idx10, np.float16)
    n("Mul", ["lc_f", "idx10"], "lc_idxv")                # [1,10,1,1] f16
    n("ReduceSum", ["lc_idxv"], "lc_idx", axes=[1], keepdims=1)  # [1,1,1,1] f16

    # ---- final color-index plane -> one-hot bool output (FREE) -------------
    #   gridline pixel -> linecolor index ; else -> upsampled block index
    #   off-grid -> sentinel -1 (matches no channel in Equal => all zero)
    n("Where", ["gp", "lc_idx", "L_block"], "L_grid")     # [1,1,30,30] f16
    init("NEG1", np.array(-1.0, np.float16), np.float16)
    n("Where", ["ingrid", "L_grid", "NEG1"], "L_final")   # [1,1,30,30] f16
    arange = np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1)
    init("arange", arange, np.float16)
    n("Equal", ["L_final", "arange"], "output")  # FREE [1,10,30,30] bool

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task009", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
