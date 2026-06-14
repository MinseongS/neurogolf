"""Task 297 (ARC-AGI bd4472b8): unfold a colour header into stacked bars.

Rule (verified exact on all 265 stored examples + fresh arc-gen). The input is an
N-wide, (2N+2)-tall grid whose row 0 is a header of N distinct non-gray colours
(one per column) and row 1 is a full gray(5) row; the rest is background. The
output keeps rows 0 and 1 and fills the remaining 2N rows with full-width bars:
output row r (r >= 2) is a solid bar of colour header[(r-2) mod N]. Because
r-2 ranges over 0..2N-1, (r-2) mod N is simply (r-2) or (r-2)-N, so the bar in
output row t pulls header column j with  t-2-j in {0, N}.

Construction (no flood-fill, single-Where output, tiny params):
  * N = number of coloured cells in row 0 (sum of channels 1..9 over the row).
  * header one-hot = input row 0  ->  [1,10,1,30] indexed by column j.
  * selector ST[1,1,30,30] indexed [j, t]:  ST[j,t] = 1 iff (t-2-j in {0,N})
    and j < N.  Built from two tiny iota vectors:  P = colidx(dim3) - rowidx(dim2)
    - 2  gives t-2-j, and J = rowidx(dim2) gives j.
  * Cvec[1,10,30,1] = (header @ ST) transposed  -> per output-row colour one-hot.
  * fillmask[1,1,30,30] = (r >= 2) & (r < 2N+2) & (c < N).
  * output = Where(fillmask, Cvec, input)  (fill cells are background in input).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..builders import _model


def build(task):
    inits = []
    nodes = []
    vinfos = []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    def vi(name, dtype, shape):
        vinfos.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    # ---- header one-hot = input row 0 -> [1,10,1,30] -----------------------
    # header[ch,j] = sum_r input[ch,r,j]*e0[r];  (e0[1,1,1,30] @ input) -> row0.
    e0 = np.zeros((1, 1, 1, 30), np.float32)
    e0[0, 0, 0, 0] = 1.0
    init("e0", e0, np.float32)                                # [1,1,1,30]
    n("MatMul", ["e0", "input"], "header")                    # [1,10,1,30] f32
    vi("header", TensorProto.FLOAT, [1, 10, 1, 30])

    # ---- N = #coloured cells in row 0 (scalar int) -------------------------
    # Off-grid canvas cells are all-zero (no background one-hot), and row 0 of
    # the grid holds exactly N coloured (non-zero channel) header cells, so the
    # total one-hot mass of row 0 is exactly N.
    n("ReduceSum", ["header"], "Nf", axes=[1, 2, 3], keepdims=1)  # [1,1,1,1] f32
    vi("Nf", TensorProto.FLOAT, [1, 1, 1, 1])
    n("Cast", ["Nf"], "N", to=TensorProto.INT32)              # [1,1,1,1] i32
    vi("N", TensorProto.INT32, [1, 1, 1, 1])

    # ---- index iota vectors -------------------------------------------------
    iota_r = np.arange(30, dtype=np.int32).reshape(1, 1, 30, 1)
    iota_c = np.arange(30, dtype=np.int32).reshape(1, 1, 1, 30)
    init("Ir", iota_r, np.int32)                              # [1,1,30,1] = r
    init("Ic", iota_c, np.int32)                              # [1,1,1,30] = c
    init("two_i", np.array(2, np.int32), np.int32)
    init("one_i", np.array(1, np.int32), np.int32)

    # ---- per output-row source column srcidx[30] (1-D) ---------------------
    # a = t-2 ; src = a if a<N else a-N  (= (t-2) mod N over the valid band).
    # Out-of-band rows are masked away by fillmask, so any in-range index is ok.
    iotat = np.arange(30, dtype=np.int32)                     # [30]
    init("It", iotat, np.int32)                              # [30] = t
    # N as a flat [1] tensor for clean broadcasting with the [30] band.
    init("shp1", np.array([1], np.int64), np.int64)
    n("Reshape", ["N", "shp1"], "N1")                        # [1] i32
    vi("N1", TensorProto.INT32, [1])
    n("Sub", ["It", "two_i"], "aband")                       # a = t-2  [30]
    vi("aband", TensorProto.INT32, [30])
    n("Less", ["aband", "N1"], "altN")                       # a<N  [30] bool
    vi("altN", TensorProto.BOOL, [30])
    n("Not", ["altN"], "agen")                               # a>=N  [30] bool
    vi("agen", TensorProto.BOOL, [30])
    n("Cast", ["agen"], "ageni", to=TensorProto.INT32)       # [30] i32
    vi("ageni", TensorProto.INT32, [30])
    n("Mul", ["ageni", "N1"], "subN")                        # N where a>=N  [30]
    vi("subN", TensorProto.INT32, [30])
    n("Sub", ["aband", "subN"], "srcidx")                    # src column  [30]
    vi("srcidx", TensorProto.INT32, [30])

    # ---- Cvec[1,10,30,1] = header columns gathered per output row ----------
    n("Gather", ["header", "srcidx"], "Cvr", axis=3)         # [1,10,1,30] f32
    vi("Cvr", TensorProto.FLOAT, [1, 10, 1, 30])
    n("Transpose", ["Cvr"], "Cvec", perm=[0, 1, 3, 2])       # [1,10,30,1] f32
    vi("Cvec", TensorProto.FLOAT, [1, 10, 30, 1])

    # ---- fillmask[1,1,30,30] = (r>=2)&(r<2N+2)&(c<N) -----------------------
    # r>=2 <=> r>1 ; 2N+2 upper bound; r = Ir(dim2), c = Ic(dim3)
    n("Greater", ["Ir", "one_i"], "rge2")                     # [1,1,30,1] bool
    vi("rge2", TensorProto.BOOL, [1, 1, 30, 1])
    # 2N+1 : r < 2N+2 <=> r <= 2N+1 <=> r < 2N+2 ; use Less(Ir, 2N+2)
    n("Add", ["N", "N"], "twoN")                              # [1,1,1,1]
    vi("twoN", TensorProto.INT32, [1, 1, 1, 1])
    n("Add", ["twoN", "two_i"], "twoNp2")                     # 2N+2
    vi("twoNp2", TensorProto.INT32, [1, 1, 1, 1])
    n("Less", ["Ir", "twoNp2"], "rlt")                        # [1,1,30,1] bool
    vi("rlt", TensorProto.BOOL, [1, 1, 30, 1])
    n("And", ["rge2", "rlt"], "rin")                          # [1,1,30,1] bool
    vi("rin", TensorProto.BOOL, [1, 1, 30, 1])
    n("Less", ["Ic", "N"], "cltN")                            # [1,1,1,30] bool
    vi("cltN", TensorProto.BOOL, [1, 1, 1, 30])
    n("And", ["rin", "cltN"], "fillmask")                     # [1,1,30,30] bool
    vi("fillmask", TensorProto.BOOL, [1, 1, 30, 30])

    # ---- output = Where(fillmask, Cvec, input) -----------------------------
    n("Where", ["fillmask", "Cvec", "input"], "output")       # [1,10,30,30] f32

    return _model(nodes, inits, vinfos)
