"""task333 (ARC-AGI d43fd935) — beams shot toward a green 2x2 box.

Rule (from the generator):
  The grid (always 10x10) holds ONE green(3) 2x2 box at (boxrow,boxcol) and a set
  of coloured pixels.  For each coloured pixel:
    * the pixel itself is always drawn;
    * if the pixel lies in one of the box's two ROWS (row in {boxrow,boxrow+1}),
      it shoots a horizontal beam of its colour TOWARD the box, filling every cell
      between the pixel and the box (stopping at the green edge);
    * if it lies in one of the box's two COLUMNS (col in {boxcol,boxcol+1}),
      it shoots a vertical beam TOWARD the box (stopping at the green edge).
  Per the generator's flags, there is at most ONE pixel on each of the four sides
  (left / right of box rows, above / below box cols), and stray pixels (not in any
  box band) are simply left as single dots.  So the output = input with the four
  beam segments filled in.

Floor-break encoding (single uint8 label map + final Equal; no [1,10,30,30] tensor):
  Work on a 10x10 canvas.  C[r,c] = colour index of `input` (arange Conv, ch0=0).
  Green plane -> per-row / per-col green presence -> box-row mask, box-col mask, and
  the boxrow / boxcol scalars defining the four regions (left<boxcol, right>boxcol+1,
  up<boxrow, down>boxrow+1).  Because each region holds at most ONE coloured pixel,
  a directional fill toward the box is a triangular prefix/suffix MatMul of the
  region-restricted colour map (prefix-sum of a single value = that value):
    leftfill  = prefix over cols (carry rightward) on box rows, masked col<boxcol
    rightfill = suffix over cols (carry leftward)  on box rows, masked col>boxcol+1
    upfill    = prefix over rows (carry downward)  on box cols, masked row<boxrow
    downfill  = suffix over rows (carry upward)    on box cols, masked row>boxrow+1
  The four regions are disjoint, so beam = sum of the four.  L = beam where beam>0
  else C; Pad to [1,1,30,30] with sentinel 10 (off-grid); free BOOL output =
  Equal(L, arange[1,10,1,1]).  All values are small integers, exact in fp32/uint8.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F = TensorProto.FLOAT
H = TensorProto.FLOAT16
U8 = TensorProto.UINT8
I64 = TensorProto.INT64
B = TensorProto.BOOL
N = 30
M = 10  # working canvas (grid is always 10x10)


def build(task):
    inits, nodes, _seen = [], [], set()

    def init(name, arr, dtype):
        if name in _seen:
            return name
        _seen.add(name)
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    init("half", np.array(0.5, np.float16), np.float16)
    init("one", np.array(1.0, np.float16), np.float16)
    init("linev", np.arange(M, dtype=np.float16), np.float16)  # [M]

    # ----- colour label map C[M,M] via 1x1 arange Conv (ch0 weight 0) -------
    # Conv over the full input (3600B fp32) is cheaper than cropping the 10-channel
    # input; cast the 30x30 result to fp16 then crop to the 10x10 active region.
    wc = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("kcolor", wc, np.float32)
    n("Conv", ["input", "kcolor"], "Cc")          # [1,1,30,30] fp32 colour index
    init("c_starts", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("c_ends", np.array([1, 1, M, M], np.int64), np.int64)
    init("c_axes", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["Cc", "c_starts", "c_ends", "c_axes"], "Ccrop")  # [1,1,M,M] fp32
    n("Cast", ["Ccrop"], "Cch", to=H)             # [1,1,M,M] fp16
    n("Squeeze", ["Cch"], "C", axes=[0, 1])       # [M,M] fp16 colour index

    # ----- green plane G[M,M]: green is the ONLY colour-3 cell (the box) -----
    # (the generator excludes green from the beam colours), so G = (C == 3).
    init("g_lo", np.array(2.5, np.float16), np.float16)
    init("g_hi", np.array(3.5, np.float16), np.float16)
    n("Greater", ["C", "g_lo"], "G_gt")
    n("Less", ["C", "g_hi"], "G_lt")
    n("And", ["G_gt", "G_lt"], "G_b")             # [M,M] bool (green)
    n("Cast", ["G_b"], "G", to=H)                 # [M,M] fp16

    # per-row / per-col green presence
    n("ReduceMax", ["G"], "grow", axes=[1], keepdims=0)   # [M] (row has green)
    n("ReduceMax", ["G"], "gcol", axes=[0], keepdims=0)   # [M] (col has green)

    # boxrow = min row with green ; box rows = {boxrow, boxrow+1}; gcol/grow are
    # already the box-row / box-col masks (green spans exactly those 2 rows/cols).
    init("big", np.array(100.0, np.float16), np.float16)

    def lo_index(mask, nm):
        # min index where mask==1 : minimize (index + big*(1-mask))
        n("Sub", ["one", mask], nm + "_inv")
        n("Mul", [nm + "_inv", "big"], nm + "_pen")
        n("Add", ["linev", nm + "_pen"], nm + "_w")
        n("ReduceMin", [nm + "_w"], nm, axes=[0], keepdims=1)  # [1]
        return nm

    lo_index("grow", "boxrow")   # [1]
    lo_index("gcol", "boxcol")   # [1]
    n("Add", ["boxrow", "one"], "boxrow1")
    n("Add", ["boxcol", "one"], "boxcol1")

    # region masks over the line coordinate (linev) ---------------------------
    # left  : col < boxcol      right : col > boxcol+1
    # up    : row < boxrow      down  : row > boxrow+1
    n("Less", ["linev", "boxcol"], "leftcol_b")        # [M]
    n("Greater", ["linev", "boxcol1"], "rightcol_b")   # [M]
    n("Less", ["linev", "boxrow"], "uprow_b")          # [M]
    n("Greater", ["linev", "boxrow1"], "downrow_b")    # [M]

    # ----- restrict C to box rows (horizontal) / box cols (vertical) --------
    n("Unsqueeze", ["grow"], "growR", axes=[1])   # [M,1] row mask
    n("Unsqueeze", ["gcol"], "gcolR", axes=[0])   # [1,M] col mask
    n("Mul", ["C", "growR"], "Crows")             # [M,M] colour only on box rows
    n("Mul", ["C", "gcolR"], "Ccols")             # [M,M] colour only on box cols

    # restrict to the proper side BEFORE the directional fill
    n("Unsqueeze", ["leftcol_b"], "leftcolR", axes=[0])    # [1,M]
    n("Unsqueeze", ["rightcol_b"], "rightcolR", axes=[0])  # [1,M]
    n("Unsqueeze", ["uprow_b"], "uprowR", axes=[1])        # [M,1]
    n("Unsqueeze", ["downrow_b"], "downrowR", axes=[1])    # [M,1]
    n("Cast", ["leftcolR"], "leftcolF", to=H)
    n("Cast", ["rightcolR"], "rightcolF", to=H)
    n("Cast", ["uprowR"], "uprowF", to=H)
    n("Cast", ["downrowR"], "downrowF", to=H)
    n("Mul", ["Crows", "leftcolF"], "Cleft")      # [M,M] left-side pixel colours
    n("Mul", ["Crows", "rightcolF"], "Cright")    # [M,M]
    n("Mul", ["Ccols", "uprowF"], "Cup")          # [M,M]
    n("Mul", ["Ccols", "downrowF"], "Cdown")      # [M,M]

    # ----- directional fills via triangular MatMuls -------------------------
    # cols: prefix (carry rightward) = X @ Utri  where Utri[a,b]=1 if a<=b
    #       suffix (carry leftward)  = X @ Ltri  where Ltri[a,b]=1 if a>=b
    Utri = np.triu(np.ones((M, M), np.float16))   # Utri[a,b]=1 if a<=b
    Ltri = np.tril(np.ones((M, M), np.float16))   # Ltri[a,b]=1 if a>=b
    init("Utri", Utri, np.float16)
    init("Ltri", Ltri, np.float16)
    n("MatMul", ["Cleft", "Utri"], "leftfill0")   # carry rightward over cols
    n("MatMul", ["Cright", "Ltri"], "rightfill0")  # carry leftward over cols
    # rows: prefix (carry downward) = Ltri @ X (Ltri[r,r']=1 if r>=r')
    #       suffix (carry upward)   = Utri @ X
    n("MatMul", ["Ltri", "Cup"], "upfill0")        # carry downward over rows
    n("MatMul", ["Utri", "Cdown"], "downfill0")    # carry upward over rows

    # re-mask each fill to its own region (so a fill never bleeds past the box)
    n("Mul", ["leftfill0", "leftcolF"], "leftfill")
    n("Mul", ["rightfill0", "rightcolF"], "rightfill")
    n("Mul", ["upfill0", "uprowF"], "upfill")
    n("Mul", ["downfill0", "downrowF"], "downfill")

    # ----- combine: beam = sum of disjoint fills ----------------------------
    n("Add", ["leftfill", "rightfill"], "hbeam")
    n("Add", ["upfill", "downfill"], "vbeam")
    n("Add", ["hbeam", "vbeam"], "beam")           # [M,M] float

    # L = beam where beam>0 else C
    n("Greater", ["beam", "half"], "beam_b")       # [M,M]
    n("Where", ["beam_b", "beam", "C"], "Lf")      # [M,M] float colour
    n("Cast", ["Lf"], "L10", to=U8)                # [M,M] uint8

    # ----- Pad to [1,1,30,30] with sentinel 10 (off-grid) -------------------
    init("Mshape", np.array([1, 1, M, M], np.int64), np.int64)
    n("Reshape", ["L10", "Mshape"], "Lr")          # [1,1,M,M]
    init("u10", np.array(10, np.uint8), np.uint8)
    init("padcfg", np.array([0, 0, 0, 0, 0, 0, N - M, N - M], np.int64), np.int64)
    n("Pad", ["Lr", "padcfg", "u10"], "L", mode="constant")  # [1,1,N,N]

    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["L", "chan"], "output")            # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F, [1, 10, N, N])
    y = helper.make_tensor_value_info("output", B, [1, 10, N, N])
    g = helper.make_graph(nodes, "task333", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
