"""Task 213 (ARC-AGI 8e1813be): collapse colored stripes into a K x K block.

Rule: the input has K equally-spaced solid stripes (every 3rd line), each a
distinct color drawn from {1,2,3,4,6,7,8,9} (never gray=5, never black=0), plus
a (K+2)x(K+2) gray(5)/black(0) box marker overlaid somewhere. The output is a
KxK grid where row r (xpose=0) is entirely the r-th stripe color top-to-bottom;
under xpose the output is transposed so column c is the c-th stripe color.

Because stripe colors exclude 0 and 5, the box never contributes a stripe color.
We mask the input to stripe channels only, then:
  band[line,:] = one-hot stripe color of that line (ReduceMax over the line)
  isstripe[line] = line contains a stripe color
  pos[line]      = (inclusive prefix count of isstripe) - 1   (lower-tri MatMul)
  A[line,i]      = isstripe[line] * (pos[line]==i)
  outband[i,:]   = A^T @ band     -> one-hot color for output slot i
  K              = total stripe count
The KxK block is outband broadcast over the first K columns (xpose=0) or first K
rows (xpose=1). Orientation is detected from the data: a stripe line has one
color spanning many cells, so its max per-color count >= 2; the perpendicular
direction sees each color at most once. horiz = (#such rows > #such cols).

All math is integer-valued and float32-exact. Equality/threshold use the
1-Clip(Abs(d),0,1) and Less tricks (opset-10 safe on floats).
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

    init("one", np.array(1.0, np.float32))
    init("c1p5", np.array(1.5, np.float32))     # threshold for count >= 2
    # stripe-channel mask: 1 for {1,2,3,4,6,7,8,9}, 0 for ch0(black), ch5(gray)
    smask = np.ones((1, 10, 1, 1), np.float32)
    smask[0, 0, 0, 0] = 0.0
    smask[0, 5, 0, 0] = 0.0
    init("smask", smask)
    L = np.tril(np.ones((30, 30), np.float32))
    init("L", L)
    init("sh_10_30", np.array([10, 30], np.int64), np.int64)
    idxrow = np.arange(30, dtype=np.float32).reshape(1, 30)
    init("idxrow", idxrow)                       # [1,30]
    arange30 = np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30)
    init("ar_row", arange30)                     # [1,1,1,30] for col mask
    arange30v = np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1)
    init("ar_col", arange30v)                    # [1,1,30,1] for row mask

    # mask input to stripe channels only
    n("Mul", ["input", "smask"], "gs")
    vi("gs", TensorProto.FLOAT, [1, 10, 30, 30])

    def line_pipeline(tag, reduce_axis, line_axis):
        if line_axis == 2:
            sh = [1, 10, 30, 1]
        else:
            sh = [1, 10, 1, 30]
        # per-line stripe presence (one-hot) and per-line per-color count
        n("ReduceMax", ["gs"], f"band_{tag}", axes=[reduce_axis], keepdims=1)
        n("ReduceSum", ["gs"], f"cnt_{tag}", axes=[reduce_axis], keepdims=1)
        vi(f"band_{tag}", TensorProto.FLOAT, sh)
        vi(f"cnt_{tag}", TensorProto.FLOAT, sh)
        # max per-color count along the line -> [1,1,L,1]/[1,1,1,L]
        ls = sh.copy(); ls[1] = 1
        n("ReduceMax", [f"cnt_{tag}"], f"maxc_{tag}", axes=[1], keepdims=1)
        vi(f"maxc_{tag}", TensorProto.FLOAT, ls)
        # "stripe-dominated" line: max count >= 2
        n("Greater", [f"maxc_{tag}", "c1p5"], f"dom_{tag}")
        vi(f"dom_{tag}", TensorProto.BOOL, ls)
        n("Cast", [f"dom_{tag}"], f"domf_{tag}", to=TensorProto.FLOAT)
        n("ReduceSum", [f"domf_{tag}"], f"domn_{tag}", axes=[2, 3], keepdims=1)
        vi(f"domf_{tag}", TensorProto.FLOAT, ls)
        vi(f"domn_{tag}", TensorProto.FLOAT, [1, 1, 1, 1])

        # band -> [10,30] -> [30,10] (line, channel)
        n("Reshape", [f"band_{tag}", "sh_10_30"], f"bo2_{tag}")
        n("Transpose", [f"bo2_{tag}"], f"bo_{tag}", perm=[1, 0])
        vi(f"bo2_{tag}", TensorProto.FLOAT, [10, 30])
        vi(f"bo_{tag}", TensorProto.FLOAT, [30, 10])

        # isstripe[line] = any stripe channel present -> [30,1]
        n("ReduceSum", [f"bo_{tag}"], f"iss_{tag}", axes=[1], keepdims=1)
        vi(f"iss_{tag}", TensorProto.FLOAT, [30, 1])  # 0 or 1 (one-hot)

        # pos = (L @ iss) - 1   -> [30,1]
        n("MatMul", ["L", f"iss_{tag}"], f"pfx_{tag}")
        n("Sub", [f"pfx_{tag}", "one"], f"pos_{tag}")
        vi(f"pfx_{tag}", TensorProto.FLOAT, [30, 1])
        vi(f"pos_{tag}", TensorProto.FLOAT, [30, 1])

        # K = total stripe count (scalar)
        n("ReduceSum", [f"iss_{tag}"], f"K_{tag}", axes=[0, 1], keepdims=1)
        vi(f"K_{tag}", TensorProto.FLOAT, [1, 1])

        # A[line,i] = iss[line] * (pos[line]==i)
        n("Sub", [f"pos_{tag}", "idxrow"], f"pd_{tag}")          # [30,30]
        n("Abs", [f"pd_{tag}"], f"pda_{tag}")
        n("Clip", [f"pda_{tag}"], f"pdc_{tag}", min=0.0, max=1.0)
        n("Sub", ["one", f"pdc_{tag}"], f"eqpos_{tag}")
        vi(f"pd_{tag}", TensorProto.FLOAT, [30, 30])
        vi(f"pda_{tag}", TensorProto.FLOAT, [30, 30])
        vi(f"pdc_{tag}", TensorProto.FLOAT, [30, 30])
        vi(f"eqpos_{tag}", TensorProto.FLOAT, [30, 30])
        n("Mul", [f"eqpos_{tag}", f"iss_{tag}"], f"A_{tag}")     # [30,30]
        vi(f"A_{tag}", TensorProto.FLOAT, [30, 30])
        n("Transpose", [f"A_{tag}"], f"At_{tag}", perm=[1, 0])
        vi(f"At_{tag}", TensorProto.FLOAT, [30, 30])
        n("MatMul", [f"At_{tag}", f"bo_{tag}"], f"outband_{tag}")  # [30,10]
        vi(f"outband_{tag}", TensorProto.FLOAT, [30, 10])
        return f"outband_{tag}", f"domn_{tag}", f"K_{tag}"

    out_r, domn_r, K_r = line_pipeline("r", reduce_axis=3, line_axis=2)
    out_c, domn_c, K_c = line_pipeline("c", reduce_axis=2, line_axis=3)

    # horiz = (#dominated rows) > (#dominated cols)
    n("Greater", [domn_r, domn_c], "horiz")
    vi("horiz", TensorProto.BOOL, [1, 1, 1, 1])
    init("sh_scalar", np.array([1], np.int64), np.int64)
    n("Reshape", ["horiz", "sh_scalar"], "horiz_s")
    vi("horiz_s", TensorProto.BOOL, [1])

    # select outband [30,10] and K scalar
    n("Where", ["horiz_s", out_r, out_c], "outband")
    vi("outband", TensorProto.FLOAT, [30, 10])
    # K is identical for both orientations (same #stripes); use K_r vs K_c via horiz
    n("Where", ["horiz_s", K_r, K_c], "Ksel")           # [1,1]
    vi("Ksel", TensorProto.FLOAT, [1, 1])
    n("Reshape", ["Ksel", "sh_scalar"], "Kscalar")      # [1]
    vi("Kscalar", TensorProto.FLOAT, [1])

    # outband [30,10] -> obT [10,30]
    n("Transpose", ["outband"], "obT", perm=[1, 0])
    vi("obT", TensorProto.FLOAT, [10, 30])

    # --- horizontal layout: row i filled with outband[i] across cols 0..K-1 ---
    # base_rows[1,10,30,1] = outband per output row.
    init("sh_col", np.array([1, 10, 30, 1], np.int64), np.int64)
    n("Reshape", ["obT", "sh_col"], "rows_base")        # [1,10,30,1]
    vi("rows_base", TensorProto.FLOAT, [1, 10, 30, 1])
    # column validity mask c < K  -> [1,1,1,30]
    n("Less", ["ar_row", "Kscalar"], "colvalid")        # bool [1,1,1,30]
    vi("colvalid", TensorProto.BOOL, [1, 1, 1, 30])
    n("Cast", ["colvalid"], "colvalidf", to=TensorProto.FLOAT)
    vi("colvalidf", TensorProto.FLOAT, [1, 1, 1, 30])
    # broadcast: [1,10,30,1] * [1,1,1,30] -> [1,10,30,30]
    n("Mul", ["rows_base", "colvalidf"], "horiz_canvas")
    vi("horiz_canvas", TensorProto.FLOAT, [1, 10, 30, 30])

    # --- vertical layout: col i filled with outband[i] across rows 0..K-1 ---
    init("sh_row", np.array([1, 10, 1, 30], np.int64), np.int64)
    n("Reshape", ["obT", "sh_row"], "cols_base")        # [1,10,1,30]
    vi("cols_base", TensorProto.FLOAT, [1, 10, 1, 30])
    n("Less", ["ar_col", "Kscalar"], "rowvalid")        # bool [1,1,30,1]
    vi("rowvalid", TensorProto.BOOL, [1, 1, 30, 1])
    n("Cast", ["rowvalid"], "rowvalidf", to=TensorProto.FLOAT)
    vi("rowvalidf", TensorProto.FLOAT, [1, 1, 30, 1])
    n("Mul", ["cols_base", "rowvalidf"], "vert_canvas")
    vi("vert_canvas", TensorProto.FLOAT, [1, 10, 30, 30])

    n("Where", ["horiz", "horiz_canvas", "vert_canvas"], "output")

    return _model(nodes, inits, vinfos)
