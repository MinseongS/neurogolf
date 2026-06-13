"""Task 205 (confettibox / ARC 8731374e).

Rule: a solid wide x tall rectangle of `boxcolor` sits in a noise grid, with
1-3 interior pixels of a second `color`. Output = the box, with each special
pixel's whole row and whole column flooded with `color`:
    out[r][c] = color  if (r is a cross-row OR c is a cross-col) else boxcolor
output size = box (wide x tall), top-left aligned.

Net pipeline (single-channel [1,1,30,30] fp16 math; every value is a small
integer, exact in fp16 up to 2048):
  integer grid G and grid-mask gm via 1x1 Convs on the one-hot input. Outside
  the grid each cell gets a UNIQUE negative sentinel (-idx) so equal-neighbour
  runs never form there. Detect the solid box as the region whose cells equal
  both a horizontal AND a vertical neighbour-run of length L=6 (the box is
  always >=6; a chance 6x6 noise block is negligible): neighbour-difference
  Convs -> equal maps -> Conv count L-1 consecutive -> dilate Conv -> hcov/vcov
  -> `solid`. Box bounds r0,r1,c0,c1 from row/col occupancy of `solid` via
  ReduceMax of index-weighted occupancy. boxcolor = max(G*solid),
  color = max(G*crosscell). Cross-row / cross-col occupancy is shifted to the
  top-left corner with two Equal-built MatMul shift matrices (only the 1-D
  occupancy vectors are shifted, never the full grid). The corner value grid
  (boxcolor / color, -1 outside the h x w box) is one-hot-decoded into `output`.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ..builders import _model

L = 6  # box side is always >= 6, so a length-6 run is box-only


def build(task):
    inits, nodes = [], []
    F16 = onnx.TensorProto.FLOAT16
    # all canvas math runs in fp16 (every value is a small integer, exact in fp16
    # up to 2048; this halves the byte cost of every intermediate vs fp32)

    def init(name, arr, dtype=np.float16):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    # shared scalars / index planes (fp16)
    init("Half", np.array(0.5, np.float16))
    init("One", np.array(1.0, np.float16))
    init("C30", np.array(30.0, np.float16))
    init("Icol", np.arange(30, dtype=np.float16).reshape(1, 1, 30, 1))   # row index
    init("IcolH", (30.0 - np.arange(30)).astype(np.float16).reshape(1, 1, 30, 1))
    init("Irow", np.arange(30, dtype=np.float16).reshape(1, 1, 1, 30))   # col index
    init("IrowH", (30.0 - np.arange(30)).astype(np.float16).reshape(1, 1, 1, 30))

    # input is fp32 -> cast to fp16 for all interior math
    n("Cast", ["input"], "inp16", to=F16)

    # ---- integer grid G and grid mask gm (1-channel) ----
    init("Wg", np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1))
    n("Conv", ["inp16", "Wg"], "G")                 # [1,1,30,30] integer colors
    init("Wm", np.ones((1, 10, 1, 1), np.float16))
    n("Conv", ["inp16", "Wm"], "gm_f")              # [1,1,30,30] 1 inside grid
    n("Greater", ["gm_f", "Half"], "gm_b")          # bool inside grid
    # Gm: G inside grid, unique negative sentinel outside (kills exterior equal-runs)
    init("IDX", (np.arange(900).reshape(1, 1, 30, 30) + 1).astype(np.float16))
    n("Sub", ["gm_f", "One"], "gm1")                # 0 inside, -1 outside
    n("Mul", ["gm1", "IDX"], "Gm")                  # 0 inside, -idx outside (G is 0 there too)
    n("Add", ["G", "Gm"], "Gms")                    # G inside, -idx outside

    # ---- equal-to-neighbour maps via neighbour-difference Conv (==0) ----
    init("Wdh", np.array([1.0, -1.0], np.float16).reshape(1, 1, 1, 2))
    n("Conv", ["Gms", "Wdh"], "dh")                 # [1,1,30,29] = G[:, :-1]-G[:, 1:]
    n("Abs", ["dh"], "dha"); n("Greater", ["dha", "Half"], "hne_b"); n("Not", ["hne_b"], "Eh_b")
    n("Cast", ["Eh_b"], "Eh", to=F16)               # [1,1,30,29] equal-right
    init("Wdv", np.array([1.0, -1.0], np.float16).reshape(1, 1, 2, 1))
    n("Conv", ["Gms", "Wdv"], "dv")                 # [1,1,29,30]
    n("Abs", ["dv"], "dva"); n("Greater", ["dva", "Half"], "vne_b"); n("Not", ["vne_b"], "Ev_b")
    n("Cast", ["Ev_b"], "Ev", to=F16)               # [1,1,29,30]

    # ---- runs of L-1 consecutive equals, then dilate to cover L cells ----
    Lm = L - 1
    # horizontal run-count over Lm cols (Conv 1x(L-1)); ==L-1 -> start
    init("WhR", np.ones((1, 1, 1, Lm), np.float16))
    n("Conv", ["Eh", "WhR"], "hrun")                # [1,1,30,29-(L-1)+1]=[1,1,30,24]
    init("LmThr", np.array(Lm - 0.5, np.float16))
    n("Greater", ["hrun", "LmThr"], "hs_b"); n("Cast", ["hs_b"], "hs", to=F16)
    # dilate: cover the L cells of each run; Conv ones(1,L) padded so width back to 30
    init("WhD", np.ones((1, 1, 1, L), np.float16))
    n("Conv", ["hs", "WhD"], "hcov_s", pads=[0, L - 1, 0, L - 1])  # 25+5+5-6+1=30
    n("Greater", ["hcov_s", "Half"], "hcov_b")      # bool
    # vertical
    init("WvR", np.ones((1, 1, Lm, 1), np.float16))
    n("Conv", ["Ev", "WvR"], "vrun")                # [1,1,24,30]
    n("Greater", ["vrun", "LmThr"], "vs_b"); n("Cast", ["vs_b"], "vs", to=F16)
    init("WvD", np.ones((1, 1, L, 1), np.float16))
    n("Conv", ["vs", "WvD"], "vcov_s", pads=[L - 1, 0, L - 1, 0])
    n("Greater", ["vcov_s", "Half"], "vcov_b")      # bool
    n("And", ["hcov_b", "vcov_b"], "solid_hv")      # bool
    n("And", ["solid_hv", "gm_b"], "solid_b")       # restrict to grid interior
    n("Cast", ["solid_b"], "solid", to=F16)  # [1,1,30,30]

    # ---- box bounds ----
    n("ReduceMax", ["solid"], "rowocc", axes=[3], keepdims=1)  # [1,1,30,1]
    n("ReduceMax", ["solid"], "colocc", axes=[2], keepdims=1)  # [1,1,1,30]
    n("Mul", ["rowocc", "IcolH"], "r0a"); n("ReduceMax", ["r0a"], "r0m", keepdims=0)
    n("Sub", ["C30", "r0m"], "r0")
    n("Mul", ["rowocc", "Icol"], "r1a"); n("ReduceMax", ["r1a"], "r1", keepdims=0)
    n("Mul", ["colocc", "IrowH"], "c0a"); n("ReduceMax", ["c0a"], "c0m", keepdims=0)
    n("Sub", ["C30", "c0m"], "c0")
    n("Mul", ["colocc", "Irow"], "c1a"); n("ReduceMax", ["c1a"], "c1", keepdims=0)

    # ---- inbox plane ----
    init("NegHalf", np.array(-0.5, np.float16))
    n("Sub", ["Icol", "r0"], "ir_lo"); n("Greater", ["ir_lo", "NegHalf"], "ir1")
    n("Sub", ["r1", "Icol"], "ir_hi"); n("Greater", ["ir_hi", "NegHalf"], "ir2")
    n("And", ["ir1", "ir2"], "inr_b")               # [1,1,30,1] bool
    n("Sub", ["Irow", "c0"], "ic_lo"); n("Greater", ["ic_lo", "NegHalf"], "ic1")
    n("Sub", ["c1", "Irow"], "ic_hi"); n("Greater", ["ic_hi", "NegHalf"], "ic2")
    n("And", ["ic1", "ic2"], "inc_b")               # [1,1,1,30] bool
    n("And", ["inr_b", "inc_b"], "inbox_b")         # [1,1,30,30] bool

    # ---- boxcolor / color (single-channel maxima) ----
    n("Mul", ["G", "solid"], "Gs"); n("ReduceMax", ["Gs"], "boxcolor", keepdims=1)  # scalar [1,1,1,1]
    # boxplane: cells equal to boxcolor (bool, within whole canvas)
    n("Sub", ["G", "boxcolor"], "gd"); n("Abs", ["gd"], "gda")
    n("Greater", ["gda", "Half"], "notbp_b")        # |G-boxcolor|>0.5  == NOT boxcolor
    n("And", ["inbox_b", "notbp_b"], "crosscell_b")  # bool [1,1,30,30]
    n("Cast", ["crosscell_b"], "crosscell", to=F16)
    n("Mul", ["G", "crosscell"], "Gc"); n("ReduceMax", ["Gc"], "color", keepdims=1)  # scalar

    # 1-D cross-row / cross-col occupancy (absolute), then shift to corner
    n("ReduceMax", ["crosscell"], "crossrow", axes=[3], keepdims=1)  # [1,1,30,1]
    n("ReduceMax", ["crosscell"], "crosscol", axes=[2], keepdims=1)  # [1,1,1,30]

    # shift matrices (built from r0,c0)
    Dr = (np.arange(30)[None, :] - np.arange(30)[:, None]).astype(np.int32)  # Dr[i,k]=k-i
    Dc = (np.arange(30)[:, None] - np.arange(30)[None, :]).astype(np.int32)  # Dc[a,j]=a-j
    init("Dr_i", Dr, dtype=np.int32)
    init("Dc_i", Dc, dtype=np.int32)
    n("Cast", ["r0"], "r0_i", to=onnx.TensorProto.INT32)
    n("Cast", ["c0"], "c0_i", to=onnx.TensorProto.INT32)
    n("Equal", ["Dr_i", "r0_i"], "Sr_b"); n("Cast", ["Sr_b"], "Sr", to=F16)  # [30,30]
    n("Equal", ["Dc_i", "c0_i"], "Pc_b"); n("Cast", ["Pc_b"], "Pc", to=F16)
    # corner-coords cross occupancy: crossrow_c[i]=crossrow[i+r0], crosscol_c[j]=crosscol[j+c0]
    n("MatMul", ["Sr", "crossrow"], "crossrow_c")   # [1,1,30,1] tiny
    n("MatMul", ["crosscol", "Pc"], "crosscol_c")   # [1,1,1,30] tiny
    n("Add", ["crossrow_c", "crosscol_c"], "crsum")  # [1,1,30,30]
    n("Greater", ["crsum", "Half"], "cor_b")         # bool: cross at corner cell

    # corner mask: rows < h, cols < w
    n("Sub", ["r1", "r0"], "hm1"); n("Add", ["hm1", "One"], "h")
    n("Sub", ["c1", "c0"], "wm1"); n("Add", ["wm1", "One"], "w")
    n("Sub", ["h", "Icol"], "h_gt"); n("Greater", ["h_gt", "Half"], "cr_b")
    n("Sub", ["w", "Irow"], "w_gt"); n("Greater", ["w_gt", "Half"], "cc_b")
    n("And", ["cr_b", "cc_b"], "incorner_b")         # [1,1,30,30] bool

    # corner value grid: incorner ? (boxcolor + (color-boxcolor)*cross) : -1
    n("Cast", ["incorner_b"], "incorner", to=F16)
    n("And", ["incorner_b", "cor_b"], "cross_c_b"); n("Cast", ["cross_c_b"], "cross_c", to=F16)
    n("Sub", ["color", "boxcolor"], "dcol")
    n("Mul", ["cross_c", "dcol"], "vc")              # (color-box) where cross-in-corner
    n("Add", ["vc", "boxcolor"], "vin")              # box + that
    n("Mul", ["vin", "incorner"], "vin_m")           # 0 outside corner
    n("Sub", ["incorner", "One"], "cneg")            # -1 outside corner, 0 inside
    n("Add", ["vin_m", "cneg"], "Vfinal")            # -1 outside (matches no channel)

    # one-hot decode straight into output
    n("Cast", ["Vfinal"], "Vout_i", to=onnx.TensorProto.INT32)
    init("Cvec_i", np.arange(10, dtype=np.int32).reshape(1, 10, 1, 1), dtype=np.int32)
    n("Equal", ["Vout_i", "Cvec_i"], "out_b")        # [1,10,30,30]
    n("Cast", ["out_b"], "output", to=onnx.TensorProto.FLOAT)

    return _model(nodes, inits)
