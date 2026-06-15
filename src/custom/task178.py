"""Task 178 (ARC-AGI 746b3537): run-length the colored bands into a line.

Rule: the input is a grid of solid color bands of varying thickness, stacked
along one axis. The output is the sequence of distinct band colors (top-to-bottom
or left-to-right), each collapsed to a single cell, laid out as a single column
(xpose=0, horizontal bands) or single row (xpose=1, vertical bands).

Orientation handling (the generator transposes BOTH input and output): we detect
the band axis from the data. In xpose=0 every in-grid ROW is solid (one color);
in xpose=1 every in-grid COLUMN is solid. We count solid in-grid rows vs solid
in-grid cols; whichever is larger is the band/output axis. The two pipelines are
symmetric; a single boolean `horiz` selects between them.

Per-axis algorithm (rows shown; cols identical with axes swapped):
  band[r,:]   = per-row one-hot color  (ReduceMax over width)
  ingrid[r]   = row has any colored cell
  runstart[r] = ingrid & (band[r] != band[r-1])
  pos[r]      = (prefix count of runstart, inclusive) - 1   via lower-tri MatMul
  A[r,i]      = runstart[r] * (pos[r]==i)      assignment matrix [30,30]
  outband[i]  = A^T @ band                     [30,10] one-hot per output slot
then scatter outband[i] to canvas position (i,0) for rows / (0,i) for cols.

All math is integer-valued and float32-exact (counts <= 30, positions < 30).
Equality uses the 1-Clip(Abs(d),0,1) trick (opset-10 safe on floats).
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..builders import _model


def build(task):
    inits = []
    nodes = []
    vinfos = []

    def init(name, arr, dtype=np.float32):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    def vi(name, dtype, shape):
        vinfos.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    # ---- shared scalars / constants ----
    init("one", np.array(1.0, np.float32))
    init("half", np.array(0.5, np.float32))
    init("zero", np.array(0.0, np.float32))
    # channel mask: 0 for background channel 0, 1 for colors 1..9
    chmask = np.ones((1, 10, 1, 1), np.float32)
    chmask[0, 0, 0, 0] = 0.0
    init("chmask", chmask)
    # lower-triangular inclusive-prefix matrix L[30,30], L[i,j]=1 if j<=i
    L = np.tril(np.ones((30, 30), np.float32))
    init("L", L)
    # =====================================================================
    # Helper to build one axis pipeline. `axis_reduce` is the spatial axis we
    # reduce over to get per-line color (3 -> per row, 2 -> per col).
    # Produces a [1,10,L,1] (rows) or [1,10,1,L] canvas placed at line 0.
    # =====================================================================
    def line_pipeline(tag, reduce_axis, line_axis):
        # band: per-line one-hot color presence
        n("ReduceMax", ["input"], f"band_{tag}", axes=[reduce_axis], keepdims=1)
        # cnt: per-line per-color count
        n("ReduceSum", ["input"], f"cnt_{tag}", axes=[reduce_axis], keepdims=1)
        if line_axis == 2:  # rows: shapes [1,10,30,1]
            sh = [1, 10, 30, 1]
        else:               # cols: shapes [1,10,1,30]
            sh = [1, 10, 1, 30]
        vi(f"band_{tag}", TensorProto.FLOAT, sh)
        vi(f"cnt_{tag}", TensorProto.FLOAT, sh)

        # colored count per line (sum over colored channels), and max per-color
        n("Mul", [f"cnt_{tag}", "chmask"], f"cntm_{tag}")
        vi(f"cntm_{tag}", TensorProto.FLOAT, sh)
        n("ReduceSum", [f"cntm_{tag}"], f"colored_{tag}", axes=[1], keepdims=1)
        n("ReduceMax", [f"cntm_{tag}"], f"maxc_{tag}", axes=[1], keepdims=1)
        ls = sh.copy(); ls[1] = 1
        vi(f"colored_{tag}", TensorProto.FLOAT, ls)
        vi(f"maxc_{tag}", TensorProto.FLOAT, ls)
        # ingrid = colored > 0
        n("Greater", [f"colored_{tag}", "zero"], f"ingrid_{tag}")
        vi(f"ingrid_{tag}", TensorProto.BOOL, ls)
        # solid = (colored == maxc) & ingrid  (eq via |d|<0.5)
        n("Sub", [f"colored_{tag}", f"maxc_{tag}"], f"sd_{tag}")
        n("Abs", [f"sd_{tag}"], f"sda_{tag}")
        n("Less", [f"sda_{tag}", "half"], f"solideq_{tag}")
        vi(f"sd_{tag}", TensorProto.FLOAT, ls)
        vi(f"sda_{tag}", TensorProto.FLOAT, ls)
        vi(f"solideq_{tag}", TensorProto.BOOL, ls)
        n("And", [f"solideq_{tag}", f"ingrid_{tag}"], f"solid_{tag}")
        vi(f"solid_{tag}", TensorProto.BOOL, ls)
        # count of solid in-grid lines -> scalar
        n("Cast", [f"solid_{tag}"], f"solidf_{tag}", to=TensorProto.FLOAT)
        n("ReduceSum", [f"solidf_{tag}"], f"solidn_{tag}", axes=[2, 3],
          keepdims=1)
        vi(f"solidf_{tag}", TensorProto.FLOAT, ls)
        vi(f"solidn_{tag}", TensorProto.FLOAT, [1, 1, 1, 1])

        # ---- band [1,10,L,1] -> [10,30] -> transpose -> [30,10] (line,channel)
        n("Reshape", [f"band_{tag}", "sh_10_30"], f"bo2_{tag}")  # [10,30]
        vi(f"bo2_{tag}", TensorProto.FLOAT, [10, 30])
        n("Transpose", [f"bo2_{tag}"], f"bo_{tag}", perm=[1, 0])  # [30,10]
        vi(f"bo_{tag}", TensorProto.FLOAT, [30, 10])
        # ingrid as [30,1] float
        n("Cast", [f"ingrid_{tag}"], f"igf_{tag}", to=TensorProto.FLOAT)
        n("Reshape", [f"igf_{tag}", "sh_30_1"], f"ig30_{tag}")   # [30,1]
        vi(f"igf_{tag}", TensorProto.FLOAT, ls)
        vi(f"ig30_{tag}", TensorProto.FLOAT, [30, 1])

        # prev[r] = bo[r-1] (shift down by one along axis0), pad first row 0.
        # Shift via MatMul with shift matrix S[30,30] (S[i,j]=1 if j==i-1).
        n("MatMul", ["Shift", f"bo_{tag}"], f"prev_{tag}")       # [30,10]
        vi(f"prev_{tag}", TensorProto.FLOAT, [30, 10])
        # diff = sum_ch |bo - prev| > 0  -> [30,1]
        n("Sub", [f"bo_{tag}", f"prev_{tag}"], f"dv_{tag}")
        n("Abs", [f"dv_{tag}"], f"dva_{tag}")
        n("ReduceSum", [f"dva_{tag}"], f"dsum_{tag}", axes=[1], keepdims=1)
        vi(f"dv_{tag}", TensorProto.FLOAT, [30, 10])
        vi(f"dva_{tag}", TensorProto.FLOAT, [30, 10])
        vi(f"dsum_{tag}", TensorProto.FLOAT, [30, 1])
        n("Greater", [f"dsum_{tag}", "half"], f"changed_{tag}")  # [30,1] bool
        vi(f"changed_{tag}", TensorProto.BOOL, [30, 1])
        n("Cast", [f"changed_{tag}"], f"changedf_{tag}", to=TensorProto.FLOAT)
        vi(f"changedf_{tag}", TensorProto.FLOAT, [30, 1])
        # runstart = changed * ingrid  -> [30,1]
        n("Mul", [f"changedf_{tag}", f"ig30_{tag}"], f"rs_{tag}")
        vi(f"rs_{tag}", TensorProto.FLOAT, [30, 1])

        # pos = (L @ rs) - 1   -> [30,1]
        n("MatMul", ["L", f"rs_{tag}"], f"pfx_{tag}")            # [30,1]
        n("Sub", [f"pfx_{tag}", "one"], f"pos_{tag}")           # [30,1]
        vi(f"pfx_{tag}", TensorProto.FLOAT, [30, 1])
        vi(f"pos_{tag}", TensorProto.FLOAT, [30, 1])

        # A[r,i] = rs[r] * (pos[r]==i). pos:[30,1] broadcast vs idxrow:[1,30]
        # eq = 1 - Clip(|pos - idxrow|,0,1)  -> [30,30]
        n("Sub", [f"pos_{tag}", "idxrow"], f"pd_{tag}")          # [30,30]
        n("Abs", [f"pd_{tag}"], f"pda_{tag}")
        n("Clip", [f"pda_{tag}"], f"pdc_{tag}", min=0.0, max=1.0)
        n("Sub", ["one", f"pdc_{tag}"], f"eqpos_{tag}")          # [30,30]
        vi(f"pd_{tag}", TensorProto.FLOAT, [30, 30])
        vi(f"pda_{tag}", TensorProto.FLOAT, [30, 30])
        vi(f"pdc_{tag}", TensorProto.FLOAT, [30, 30])
        vi(f"eqpos_{tag}", TensorProto.FLOAT, [30, 30])
        # multiply by runstart (broadcast rs[30,1])
        n("Mul", [f"eqpos_{tag}", f"rs_{tag}"], f"A_{tag}")      # [30,30]
        vi(f"A_{tag}", TensorProto.FLOAT, [30, 30])
        # outband[i,ch] = sum_r A[r,i]*bo[r,ch] = A^T @ bo
        n("Transpose", [f"A_{tag}"], f"At_{tag}", perm=[1, 0])   # [30,30]
        vi(f"At_{tag}", TensorProto.FLOAT, [30, 30])
        n("MatMul", [f"At_{tag}", f"bo_{tag}"], f"outband_{tag}")  # [30,10]
        vi(f"outband_{tag}", TensorProto.FLOAT, [30, 10])
        return f"outband_{tag}", f"solidn_{tag}"

    # shared reshape/index constants
    init("sh_10_30", np.array([10, 30], np.int64), np.int64)
    init("sh_30_1", np.array([30, 1], np.int64), np.int64)
    idxrow = np.arange(30, dtype=np.float32).reshape(1, 30)
    init("idxrow", idxrow)           # [1,30]
    # shift matrix S[i,j]=1 if j==i-1
    S = np.zeros((30, 30), np.float32)
    for i in range(1, 30):
        S[i, i - 1] = 1.0
    init("Shift", S)

    out_r, solidn_r = line_pipeline("r", reduce_axis=3, line_axis=2)  # rows
    out_c, solidn_c = line_pipeline("c", reduce_axis=2, line_axis=3)  # cols

    # horiz = solidn_r > solidn_c   (scalar bool [1,1,1,1])
    n("Greater", [solidn_r, solidn_c], "horiz")
    vi("horiz", TensorProto.BOOL, [1, 1, 1, 1])
    # reshape to scalar for selecting [30,10] tensors
    n("Reshape", ["horiz", "sh_scalar"], "horiz_s")
    init("sh_scalar", np.array([1], np.int64), np.int64)
    vi("horiz_s", TensorProto.BOOL, [1])

    # select the correct outband [30,10]
    n("Where", ["horiz_s", out_r, out_c], "outband")    # [30,10] broadcast
    vi("outband", TensorProto.FLOAT, [30, 10])

    # Build the output canvas. outband[i] is the one-hot color for output slot i.
    # For horizontal bands (column output) place at (row i, col 0):
    #   canvas[ch, i, 0] = outband[i, ch]
    # For vertical bands (row output) place at (row 0, col i):
    #   canvas[ch, 0, i] = outband[i, ch]
    # We build both and select with horiz.
    #
    # outband -> [1,10,30,1]: transpose to [10,30] then reshape.
    n("Transpose", ["outband"], "obT", perm=[1, 0])     # [10,30]
    vi("obT", TensorProto.FLOAT, [10, 30])
    # column layout: [1,10,30,1]  -> pad cols to 30
    n("Reshape", ["obT", "sh_col"], "col_small")        # [1,10,30,1]
    init("sh_col", np.array([1, 10, 30, 1], np.int64), np.int64)
    vi("col_small", TensorProto.FLOAT, [1, 10, 30, 1])
    n("Pad", ["col_small"], "col_canvas", mode="constant", value=0.0,
      pads=[0, 0, 0, 0, 0, 0, 0, 29])                   # width 1 -> 30
    vi("col_canvas", TensorProto.FLOAT, [1, 10, 30, 30])
    # row layout: [1,10,1,30] -> pad rows to 30
    n("Reshape", ["obT", "sh_row"], "row_small")        # [1,10,1,30]
    init("sh_row", np.array([1, 10, 1, 30], np.int64), np.int64)
    vi("row_small", TensorProto.FLOAT, [1, 10, 1, 30])
    n("Pad", ["row_small"], "row_canvas", mode="constant", value=0.0,
      pads=[0, 0, 0, 0, 0, 0, 29, 0])                   # height 1 -> 30
    vi("row_canvas", TensorProto.FLOAT, [1, 10, 30, 30])

    # select layout with horiz (broadcast [1,1,1,1] bool over [1,10,30,30])
    n("Where", ["horiz", "col_canvas", "row_canvas"], "output")

    return _model(nodes, inits, vinfos)
