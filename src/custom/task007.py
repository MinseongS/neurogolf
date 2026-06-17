"""task007 (ARC-AGI 05269061) — periodic diagonal-stripe fill.

Rule (from the generator):
  A `size`x`size` square grid (size=7 in every instance).  There are `num`=3
  stripe colors.  For each cell, diag = r + c and the TRUE color is
  colors[diag % 3].  The OUTPUT is the fully-filled grid:
      output[r][c] = colors[(r + c) % 3].
  The INPUT shows that same coloring ONLY on a handful of complete anti-
  diagonals (`diags`); every other cell is black(0).  Crucially `diags` picks
  exactly one diagonal from each residue class mod 3
  (diags = [choice(range(s, 2*size-1, 3)) for s in range(3)]), so every residue
  class m in {0,1,2} has at least one shown cell, all of color colors[m].

  Therefore the 3 stripe colors are recoverable from the input:
      color[m] = the (unique) nonzero color appearing on cells with (r+c)%3==m,
  and the output is the closed-form periodic fill output[r][c]=color[(r+c)%3].

Encoding (closed-form, tier-S/A; route the 10-ch expansion into the FREE output):
  Work on the fixed 7x7 active grid (W=7), flattened to length P=49 (p=r*7+c).
  Rmat[p, m] = 1 iff (r+c)%3 == m   (residue indicator, [P,3] const).

  1. counts: reshape input[:, :, :7, :7] -> [1,10,49]; MatMul with Rmat[49,3]
       cm = [1,10,3]  where cm[0,k,m] = #shown cells of color k in residue m
       (nonzero only at k = color[m]).  No [1,10,7,7] product plane.
  2. per-residue color index (scalar, length-3):
       cmpos = (cm > 0)            -> [1,10,3] {0,1}
       colidx[m] = sum_k k * cmpos[k,m]   via MatMul kramp[1,10] @ cmpos -> [1,1,3]
       (exactly one k per m is set, so this is exactly color[m]).
  3. color-index plane:  Lflat = Rmat[49,3] @ colidx[3,1] -> [49,1] -> [1,1,7,7]
       L[r,c] = color[(r+c)%3].  Pad to [1,1,30,30] (fp16, 1800B) -> the
       dominant intermediate.
  4. output = Equal(L_pad, arange[1,10,1,1])  -> BOOL [1,10,30,30] (FREE output).
       Off-grid (padded) cells have L=0 and the arange channel-0 entry is 0, so
       Equal would set channel 0 = True off-grid.  The harness compares
       (out>0) against the target one-hot, whose off-grid cells are all 0
       (channel 0 included), so we must NOT light channel-0 off-grid.  We fix
       this by setting the padding sentinel to a value (=99) that matches no
       channel index, so off-grid cells are all-False in every channel.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64

W = 7   # active grid is always 7x7
P = W * W


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- slice input to the 7x7 active grid -> [1,10,7,7] (fp32, from FREE input)
    init("g_s", np.array([0, 0], np.int64), np.int64)
    init("g_e", np.array([W, W], np.int64), np.int64)
    init("g_ax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["input", "g_s", "g_e", "g_ax"], "grid")      # [1,10,7,7] f32

    # ---- SEPARABLE residue contraction (keeps every intermediate at 7-axis,
    #      never builds the [1,10,49] flatten):  (r+c)%3 = (r%3 + c%3)%3.
    # Step a) contract cols by col-residue class j=c%3:
    #   Acol[1,10,7,3] = MatMul(grid[1,10,7,7], CM[7,3]),  CM[c,j]=1 iff c%3==j.
    CM = np.zeros((W, 3), dtype=np.float32)
    for c in range(W):
        CM[c, c % 3] = 1.0
    init("CM", CM, np.float32)                               # [7,3]
    n("MatMul", ["grid", "CM"], "Acol")                      # [1,10,7,3] f32 (210)

    # reshape to [1,10,21] (210 elems) and contract rows+col-class into residue:
    init("shp_a", np.array([1, 10, W * 3], np.int64), np.int64)
    n("Reshape", ["Acol", "shp_a"], "Aflat")                 # [1,10,21] f32
    # T[r*3+j, m] = 1 iff (r%3 + j)%3 == m
    T = np.zeros((W * 3, 3), dtype=np.float32)
    for r in range(W):
        for j in range(3):
            T[r * 3 + j, (r % 3 + j) % 3] = 1.0
    init("T", T, np.float32)                                 # [21,3]
    n("MatMul", ["Aflat", "T"], "cm")                        # [1,10,3] f32

    # ---- cmpos = (cm > 0) -> {0,1} f32 ------------------------------------
    init("ZEROF", np.array(0.0, np.float32), np.float32)
    n("Greater", ["cm", "ZEROF"], "cmpos_b")                 # [1,10,3] bool
    n("Cast", ["cmpos_b"], "cmpos", to=F32)                  # [1,10,3] f32

    # ---- colidx[3] = kramp[1,1,10] @ cmpos[1,10,3] -> reshape [3] -----------
    # exactly one k per residue m is set => colidx[m] = color[m].
    kramp = np.arange(10, dtype=np.float32).reshape(1, 1, 10)
    init("kramp", kramp, np.float32)                         # [1,1,10] f32
    n("MatMul", ["kramp", "cmpos"], "colidx")                # [1,1,3] f32
    init("shp_c", np.array([3], np.int64), np.int64)
    n("Reshape", ["colidx", "shp_c"], "colvec")              # [3] f32

    # ---- L[1,1,7,7] = Gather(colvec, idxmap)  idxmap[r,c]=(r+c)%3 -----------
    # Gather the per-residue color straight to its cells (no scatter MatMul).
    idxmap = np.zeros((1, 1, W, W), dtype=np.int64)
    for r in range(W):
        for c in range(W):
            idxmap[0, 0, r, c] = (r + c) % 3
    init("idxmap", idxmap, np.int64)                         # [1,1,7,7] int64
    n("Gather", ["colvec", "idxmap"], "L16", axis=0)         # [1,1,7,7] f32

    # cast the tiny color-index plane to uint8 (1 byte) before padding to 30x30
    U8 = TensorProto.UINT8
    n("Cast", ["L16"], "L8", to=U8)                          # [1,1,7,7] uint8

    # ---- pad to [1,1,30,30] with sentinel 99 (matches no channel index) ----
    # uint8 carrier => 900B (half of fp16); Pad accepts uint8 (rejects bool).
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), np.int64)
    init("SENT", np.array(99, np.uint8), np.uint8)
    n("Pad", ["L8", "pads", "SENT"], "Lpad", mode="constant")  # [1,1,30,30] uint8

    # ---- output = Equal(Lpad, arange[1,10,1,1]) -> BOOL (FREE [1,10,30,30]) -
    # ORT Equal supports uint8.  arange channel index 0..9; off-grid sentinel
    # 99 matches no channel => off-grid all-False in every channel (correct).
    arange = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("arange", arange, np.uint8)                         # [1,10,1,1] uint8
    n("Equal", ["Lpad", "arange"], "output")                 # [1,10,30,30] bool

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task007", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
