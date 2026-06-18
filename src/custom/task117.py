"""task117 (ARC-AGI 4c5c2cf0) — 4-fold reflect the creature legs about the body centre.

Rule (from the generator):
  A fixed 3x3 X-pattern "body" in `color` occupies the 5 cells
  {(0,0),(0,2),(1,1),(2,0),(2,2)} relative to (rowoff,coloff); centre = (rowoff+1,coloff+1).
  A conway sprite "leg" in `legcolor` is stamped into one quadrant in the INPUT; the OUTPUT
  stamps the three reflected quadrants too.  Every leg pixel is reflected about the body
  centre: row R -> 2*(rowoff+1)-R, col C -> 2*(coloff+1)-C.  The body X is symmetric about
  that centre, so:  output = input reflected 4-fold about the centre.

Encoding (Tier B), centre found on a single colour-index plane (no 10-channel spatial plane):
  ONE 1x1 Conv on the FREE input with the runtime weight packw[k]=(k+1)+1000*cnt5_k
  (cnt = ReduceSum(input,[2,3]); cnt5 = (cnt==5)) packs THREE signals into one fp32 plane,
  cropped to 15x15 then cast to fp16:
    in-grid = (P>0)   (bg colour-0 -> 1; off-grid -> 0)
    valid   = (P>=1000)   (the "my colour occurs exactly 5x" tie-break gate)
    colour  L = P mod 1000   (= colour+1; the +1 is matched against arange+1 at the end)
  Body centre = window where the X corners+centre share one nonzero colour:
    S1 = Conv(L, Xker),  S2x5 = Conv(L*L, 5*Xker),  Xker=[[1,0,1],[0,1,0],[1,0,1]]
    centre iff 5*S2 == S1^2 (Cauchy-Schwarz: the 5 cells equal) AND valid.
  Tie-break: a leg conway-sprite is occasionally itself a count-5 X (~0.02%), giving two
  candidates; the BODY is the figure centre, i.e. the candidate row/col nearest the grid
  geometric centre gc=(H-1)/2 (the leg-X is offset to one quadrant). Per-axis nearest-to-gc
  selection is exact (verified 100000/100000).  Reflections: Rmat[R,r]=(2*cr-r==R),
  CmatT[c,C]=(2*cc-c==C) as fp16 MatMuls; Lout=Max(L,Rmat@L,L@CmatT,Rmat@L@CmatT).  Off-grid
  cells (P==0 -> ingrid False) are sentinel'd to all-False; output = Equal(pad,arange+1) -> BOOL.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64

N = 30
W = 15  # active work canvas (grid size is 12..15)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- packed plane: P = sum_k ((k+1) + 1000*cnt5_k)*input_k --------------
    # ONE 1x1 Conv on the FREE input encodes THREE signals from the disjoint magnitude bands:
    #   in-grid  = (P > 0)         (bg colour-0 maps to 1, off-grid stays 0)
    #   valid    = (P >= 1000)     ("my colour occurs exactly 5x")
    #   colour L = (P mod 1000) - 1
    init("Lker32", (np.arange(10, dtype=np.float32) + 1).reshape(1, 10, 1, 1), np.float32)
    n("ReduceSum", ["input"], "cnt", axes=[2, 3], keepdims=1)        # [1,10,1,1] f32
    init("FIVEF", np.array(5.0, np.float32), np.float32)
    n("Equal", ["cnt", "FIVEF"], "cnt5_b")                           # bool [1,10,1,1]
    n("Cast", ["cnt5_b"], "cnt5f", to=F32)
    init("THOUF", np.array(1000.0, np.float32), np.float32)
    n("Mul", ["cnt5f", "THOUF"], "cnt5x1000")                       # [1,10,1,1]
    n("Add", ["Lker32", "cnt5x1000"], "packw")                      # [1,10,1,1] runtime weight
    n("Conv", ["input", "packw"], "Pfull", kernel_shape=[1, 1])     # [1,1,N,N] f32

    # crop to active 15x15
    init("crop1_s", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("crop1_e", np.array([1, 1, W, W], np.int64), np.int64)
    init("crop1_ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["Pfull", "crop1_s", "crop1_e", "crop1_ax"], "P32")   # [1,1,W,W] f32
    n("Cast", ["P32"], "P", to=F16)                                  # [1,1,W,W] fp16

    init("HALFH", np.array(0.5, np.float16), np.float16)
    init("FIVEHUNH", np.array(500.0, np.float16), np.float16)
    init("THOUH", np.array(1000.0, np.float16), np.float16)
    n("Greater", ["P", "FIVEHUNH"], "valid")                        # bool [1,1,W,W]
    n("Greater", ["P", "HALFH"], "ingrid")                         # bool [1,1,W,W] (P>0)
    # L is the colour index SHIFTED by +1 (bg=1, colour k -> k+1); the +1 is undone at the end
    # by matching against arange+1, so no extra Sub plane is needed.
    n("Mod", ["P", "THOUH"], "L", fmod=1)                          # [1,1,W,W] fp16 (k+1)

    n("Mul", ["L", "L"], "Lsq")                                      # [1,1,W,W] fp16

    # ---- X-pattern convs (corners+centre) -----------------------------------
    # centre iff 5*S2 == S1^2 (Cauchy: all 5 X-cells share one value) AND S1>0 (nonzero).
    # Xocc==5 is implied by Cauchy+S1>0 (any empty X-cell breaks the equality), so it is dropped.
    Xk = np.array([[[[1, 0, 1], [0, 1, 0], [1, 0, 1]]]], np.float16)  # [1,1,3,3]
    init("Xker", Xk, np.float16)
    init("Xker5", 5 * Xk, np.float16)  # folds the *5 of the Cauchy test into the conv
    n("Conv", ["L", "Xker"], "S1", kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    n("Conv", ["Lsq", "Xker5"], "S2x5", kernel_shape=[3, 3], pads=[1, 1, 1, 1])  # 5*S2

    n("Mul", ["S1", "S1"], "S1sq")
    n("Equal", ["S2x5", "S1sq"], "cauchy")                           # bool [1,1,W,W]
    # ---- combine with the count-5 colour gate (valid, computed above) -------
    # `valid` (cell colour occurs exactly 5x) already excludes empty/background and the
    # leg colour, so the S1>0 gate is redundant.
    n("And", ["cauchy", "valid"], "win_b")                          # bool [1,1,W,W]

    # ---- centre scalars (cr,cc): collapse the win mask to row/col vectors ---
    # A leg conway-sprite can itself be a count-5 X (~0.02% of instances), giving TWO win
    # candidates. The true body centre is the figure centre, i.e. the candidate row/col
    # CLOSEST to the grid geometric centre gc=(H-1)/2; the stray leg-X is offset to one
    # quadrant. Per-axis nearest-to-gc selection is exact (verified 100000/100000).
    n("Cast", ["win_b"], "win", to=F16)                             # [1,1,W,W] fp16
    n("ReduceMax", ["win"], "rowwin", axes=[3], keepdims=1)         # [1,1,W,1] candidate rows
    n("ReduceMax", ["win"], "colwin", axes=[2], keepdims=1)         # [1,1,1,W] candidate cols
    init("rowramp", np.arange(W, dtype=np.float16).reshape(1, 1, W, 1), np.float16)
    init("colramp", np.arange(W, dtype=np.float16).reshape(1, 1, 1, W), np.float16)
    init("ZEROH", np.array(0.0, np.float16), np.float16)
    init("NEGBIGH", np.array(-1e4, np.float16), np.float16)

    # grid centre gc = (H-1)/2, H = number of in-grid rows (bg fills the grid square => L>0).
    # Reduce L over cols FIRST to a 1-D row profile (no full occupancy plane).
    n("ReduceMax", ["L"], "rowmaxL", axes=[3], keepdims=1)         # [1,1,W,1] max L per row
    n("Greater", ["rowmaxL", "ZEROH"], "rowocc_b")                # bool [1,1,W,1]
    n("Cast", ["rowocc_b"], "rowocc", to=F16)
    n("ReduceSum", ["rowocc"], "Hf", axes=[1, 2, 3], keepdims=1)   # [1,1,1,1] = H
    init("ONEHF", np.array(1.0, np.float16), np.float16)
    n("Sub", ["Hf", "ONEHF"], "Hm1")
    n("Mul", ["Hm1", "HALFH"], "gc")                               # (H-1)/2

    def central(winvec, ramp, axes_red, tag):
        # among candidate indices (winvec>0), pick the one nearest gc (max of -(idx-gc)^2).
        n("Greater", [winvec, "ZEROH"], f"in_{tag}")               # bool candidate mask
        n("Sub", [ramp, "gc"], f"d_{tag}")                         # idx - gc
        n("Mul", [f"d_{tag}", f"d_{tag}"], f"d2_{tag}")
        n("Sub", ["ZEROH", f"d2_{tag}"], f"nd2_{tag}")            # -(idx-gc)^2
        n("Where", [f"in_{tag}", f"nd2_{tag}", "NEGBIGH"], f"sc_{tag}")
        n("ReduceMax", [f"sc_{tag}"], f"best_{tag}", axes=axes_red, keepdims=1)
        n("Equal", [f"sc_{tag}", f"best_{tag}"], f"pick_{tag}")    # 1 at chosen idx
        n("Cast", [f"pick_{tag}"], f"pickf_{tag}", to=F16)
        n("Mul", [f"pickf_{tag}", ramp], f"pr_{tag}")
        n("ReduceSum", [f"pr_{tag}"], f"v_{tag}", axes=axes_red, keepdims=1)
        return f"v_{tag}"

    central("rowwin", "rowramp", [2], "cr")   # -> "v_cr"
    central("colwin", "colramp", [3], "cc")   # -> "v_cc"

    # ---- reflection matrices (all fp16; Equal exact for integers <2048) -----
    # Rmat[R,r] = (2*cr - r == R) ; CmatT[c,C] = (2*cc - c == C).
    init("axW3", np.arange(W, dtype=np.float16).reshape(1, 1, 1, W), np.float16)
    init("axW2", np.arange(W, dtype=np.float16).reshape(1, 1, W, 1), np.float16)
    init("TWOH", np.array(2.0, np.float16), np.float16)
    n("Mul", ["v_cr", "TWOH"], "cr2")
    n("Sub", ["cr2", "axW3"], "rrefl")                              # [1,1,1,W] fp16
    n("Equal", ["rrefl", "axW2"], "Rmat_b")                         # [1,1,W,W]
    n("Cast", ["Rmat_b"], "Rmat", to=F16)
    n("Mul", ["v_cc", "TWOH"], "cc2")
    n("Sub", ["cc2", "axW2"], "crefl")                              # [1,1,W,1] fp16
    n("Equal", ["crefl", "axW3"], "Cmat_b")                         # [1,1,W,W]
    n("Cast", ["Cmat_b"], "CmatT", to=F16)

    # ---- Lout = Max(L, Rmat@L, L@CmatT, Rmat@L@CmatT) -----------------------
    n("MatMul", ["Rmat", "L"], "LR")
    n("MatMul", ["L", "CmatT"], "LC")
    n("MatMul", ["LR", "CmatT"], "LRC")
    nodes.append(helper.make_node("Max", ["L", "LR", "LC", "LRC"], ["Lout16"]))
    n("Cast", ["Lout16"], "Lout", to=U8)                            # [1,1,W,W] u8

    # ---- mask off-grid cells to a sentinel (all-False in the output) --------
    # off-grid cells are all-zero in the input -> Pfull == 0 -> ingrid == False; the within-15
    # cells beyond the true grid size are also off-grid (Pfull==0), so `ingrid` already covers
    # both. The 30x30 pad below adds the sentinel for the outer border.
    init("SENT", np.array(99, np.uint8), np.uint8)
    n("Where", ["ingrid", "Lout", "SENT"], "Lmask")                # [1,1,W,W] u8

    init("pad_amt", np.array([0, 0, 0, 0, 0, 0, N - W, N - W], np.int64), np.int64)
    init("pad_sent", np.array(99, np.uint8), np.uint8)
    n("Pad", ["Lmask", "pad_amt", "pad_sent"], "Lpad", mode="constant")  # [1,1,N,N] u8
    # L is shifted by +1, so channel k matches index k+1.
    init("chan", (np.arange(10, dtype=np.uint8) + 1).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["Lpad", "chan"], "output")                          # [1,10,N,N] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, N, N])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, N, N])
    gph = helper.make_graph(nodes, "task117", [x], [y], inits)
    return helper.make_model(gph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
